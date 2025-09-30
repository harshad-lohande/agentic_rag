# src/agentic_rag/app/weaviate_config.py

import inspect
import weaviate
from weaviate.classes.config import Configure, VectorDistances, Property, DataType
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from agentic_rag.config import settings
from agentic_rag.logging_config import logger


def _safe_hnsw_config(**kwargs):
    """
    Build an HNSW config filtering out unsupported parameters for the installed
    weaviate-client version. Falls back to minimal config on failure.
    """
    sig = inspect.signature(Configure.VectorIndex.hnsw)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = sorted(set(kwargs.keys()) - accepted)
    if dropped:
        logger.warning(f"HNSW config: dropping unsupported parameters: {dropped}")
    try:
        return Configure.VectorIndex.hnsw(**filtered)
    except Exception as e:
        logger.error(f"HNSW config creation failed with filtered params ({filtered.keys()}): {e}")
        # Minimal fallback
        return Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
            ef_construction=kwargs.get("ef_construction", 128),
            max_connections=kwargs.get("max_connections", 32),
        )


def create_weaviate_vector_store(
    client: weaviate.Client,
    index_name: str,
    embedding_model: HuggingFaceEmbeddings,
    text_key: str = "text",
    enable_hnsw_optimization: bool = True,
) -> WeaviateVectorStore:
    try:
        if enable_hnsw_optimization:
            _configure_hnsw_collection(client, index_name)

        vector_store = WeaviateVectorStore(
            client=client,
            index_name=index_name,
            text_key=text_key,
            embedding=embedding_model,
        )
        logger.info(f"Created Weaviate vector store for index '{index_name}' (HNSW optimization={enable_hnsw_optimization})")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating Weaviate vector store: {e}")
        raise


def _configure_hnsw_collection(client: weaviate.Client, collection_name: str):
    try:
        if client.collections.exists(collection_name):
            logger.info(f"Collection '{collection_name}' already exists; skipping creation.")
            return

        logger.info(f"Creating collection '{collection_name}' with HNSW optimization")
        logger.info(
            f"HNSW target params: ef_construction={settings.HNSW_EF_CONSTRUCTION}, "
            f"ef(search)={getattr(settings, 'HNSW_EF', 'n/a')}, "
            f"max_connections={settings.HNSW_MAX_CONNECTIONS}"
        )

        # Build version‑compatible config
        vector_config = _safe_hnsw_config(
            distance_metric=VectorDistances.COSINE,
            ef_construction=settings.HNSW_EF_CONSTRUCTION,
            max_connections=settings.HNSW_MAX_CONNECTIONS,
            # Opportunistic extras (filtered if unsupported)
            ef=getattr(settings, "HNSW_EF", 64),
            dynamic_ef_min=16,
            dynamic_ef_max=512,
            dynamic_ef_factor=8,
            vector_cache_max_objects=1_000_000,
            flat_search_cutoff=40_000,
            cleanup_interval_seconds=300,
        )

        client.collections.create(
            name=collection_name,
            vector_index_config=vector_config,
            properties=[
                Property(
                    name="text",
                    data_type=DataType.TEXT,
                    description="Document text content",
                ),
                Property(
                    name="source",
                    data_type=DataType.TEXT,
                    description="Source document filename",
                ),
                Property(
                    name="chunk_number",
                    data_type=DataType.INT,
                    description="Chunk number within document",
                ),
            ],
        )
        logger.info(f"Successfully created collection '{collection_name}'")
    except Exception as e:
        logger.error(f"Error configuring HNSW collection '{collection_name}': {e}")
        logger.warning("Falling back to default Weaviate configuration (no custom HNSW params)")


def create_semantic_cache_collection(client: weaviate.Client) -> bool:
    try:
        collection_name = settings.SEMANTIC_CACHE_INDEX_NAME
        if client.collections.exists(collection_name):
            logger.info(f"Semantic cache collection '{collection_name}' already exists")
            return True

        logger.info(f"Creating semantic cache collection '{collection_name}'")

        cache_vector_config = _safe_hnsw_config(
            distance_metric=VectorDistances.COSINE,
            ef_construction=128,
            max_connections=16,
            ef=32,
            dynamic_ef_min=16,
            dynamic_ef_max=64,
            dynamic_ef_factor=4,
            vector_cache_max_objects=100_000,
            flat_search_cutoff=10_000,
            cleanup_interval_seconds=600,
        )

        client.collections.create(
            name=collection_name,
            vector_index_config=cache_vector_config,
            properties=[
                Property(
                    name="query_text",
                    data_type=DataType.TEXT,
                    description="Cached query text",
                ),
                Property(
                    name="cache_id",
                    data_type=DataType.TEXT,
                    description="Cache entry ID for Redis lookup",
                ),
                Property(
                    name="doc_id",
                    data_type=DataType.TEXT,
                    description="Document ID for cleanup purposes",
                ),
                Property(
                    name="created_at",
                    data_type=DataType.TEXT,
                    description="Creation timestamp",
                ),
                Property(
                    name="answer_preview",
                    data_type=DataType.TEXT,
                    description="Preview of cached answer",
                ),
            ],
        )

        logger.info(f"Successfully created semantic cache collection '{collection_name}'")
        return True
    except Exception as e:
        logger.error(f"Error creating semantic cache collection: {e}")
        return False


def get_collection_info(client: weaviate.Client, collection_name: str) -> dict:
    try:
        if not client.collections.exists(collection_name):
            return {"exists": False, "error": "Collection not found"}

        collection = client.collections.get(collection_name)
        config = collection.config.get()
        vic = config.vector_index_config

        info = {
            "exists": True,
            "name": collection_name,
            "vector_index_type": vic.__class__.__name__ if vic else "Unknown",
        }

        # Safely extract attributes if present
        for attr in ["ef_construction", "ef", "max_connections", "distance_metric"]:
            if hasattr(vic, attr):
                info[attr] = getattr(vic, attr)

        return info
    except Exception as e:
        logger.error(f"Error getting collection info for '{collection_name}': {e}")
        return {"exists": False, "error": str(e)}