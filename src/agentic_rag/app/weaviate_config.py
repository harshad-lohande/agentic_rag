# src/agentic_rag/app/weaviate_config.py

import weaviate
from weaviate.classes.config import Configure
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from agentic_rag.config import settings
from agentic_rag.logging_config import logger


def create_weaviate_vector_store(
    client: weaviate.Client,
    index_name: str,
    embedding_model: HuggingFaceEmbeddings,
    text_key: str = "text",
    enable_hnsw_optimization: bool = True
) -> WeaviateVectorStore:
    """
    Create a Weaviate vector store with optimized HNSW parameters.
    
    Args:
        client: Weaviate client instance
        index_name: Name of the index/collection
        embedding_model: Embedding model instance
        text_key: Key for text content
        enable_hnsw_optimization: Whether to apply HNSW optimizations
    
    Returns:
        Configured WeaviateVectorStore instance
    """
    try:
        # Check if collection exists and configure HNSW if needed
        if enable_hnsw_optimization:
            _configure_hnsw_collection(client, index_name)
        
        # Create vector store instance
        vector_store = WeaviateVectorStore(
            client=client,
            index_name=index_name,
            text_key=text_key,
            embedding=embedding_model,
        )
        
        logger.info(f"Created Weaviate vector store for index '{index_name}' with HNSW optimization")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error creating Weaviate vector store: {e}")
        raise


def _configure_hnsw_collection(client: weaviate.Client, collection_name: str):
    """
    Configure HNSW parameters for a Weaviate collection.
    
    Args:
        client: Weaviate client instance
        collection_name: Name of the collection to configure
    """
    try:
        # Check if collection exists
        if client.collections.exists(collection_name):
            logger.info(f"Collection '{collection_name}' already exists, HNSW parameters may already be set")
            return
        
        # Create collection with HNSW configuration
        logger.info(f"Creating collection '{collection_name}' with HNSW optimization")
        logger.info(f"HNSW parameters: efConstruction={settings.HNSW_EF_CONSTRUCTION}, "
                   f"ef={settings.HNSW_EF}, maxConnections={settings.HNSW_MAX_CONNECTIONS}")
        
        # Configure vector index with HNSW parameters
        vector_config = Configure.VectorIndex.hnsw(
            distance_metric="cosine",
            ef_construction=settings.HNSW_EF_CONSTRUCTION,
            ef=settings.HNSW_EF,
            max_connections=settings.HNSW_MAX_CONNECTIONS,
            dynamic_ef_min=16,  # Minimum ef value during search
            dynamic_ef_max=512,  # Maximum ef value during search
            dynamic_ef_factor=8,  # Factor for dynamic ef adjustment
            vector_cache_max_objects=1000000,  # Max objects in vector cache
            flat_search_cutoff=40000,  # Switch to flat search for small datasets
            cleanup_interval_seconds=300,  # Cleanup interval for deleted objects
            pq_enabled=False,  # Product Quantization disabled for better accuracy
        )
        
        # Create collection with optimized configuration
        client.collections.create(
            name=collection_name,
            vector_index_config=vector_config,
            # Additional properties can be configured here
            properties=[
                weaviate.classes.config.Property(
                    name="text",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="Document text content"
                ),
                weaviate.classes.config.Property(
                    name="source",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="Source document filename"
                ),
                weaviate.classes.config.Property(
                    name="chunk_number",
                    data_type=weaviate.classes.config.DataType.INT,
                    description="Chunk number within document"
                ),
            ]
        )
        
        logger.info(f"Successfully created collection '{collection_name}' with HNSW optimization")
        
    except Exception as e:
        logger.error(f"Error configuring HNSW collection '{collection_name}': {e}")
        # Continue with default configuration if HNSW setup fails
        logger.warning("Falling back to default Weaviate configuration")


def create_semantic_cache_collection(client: weaviate.Client) -> bool:
    """
    Create a dedicated collection for semantic caching with optimized HNSW parameters.
    
    Args:
        client: Weaviate client instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        collection_name = settings.SEMANTIC_CACHE_INDEX_NAME
        
        # Check if collection exists
        if client.collections.exists(collection_name):
            logger.info(f"Semantic cache collection '{collection_name}' already exists")
            return True
        
        logger.info(f"Creating semantic cache collection '{collection_name}'")
        
        # Configure vector index optimized for caching (faster queries)
        cache_vector_config = Configure.VectorIndex.hnsw(
            distance_metric="cosine",
            ef_construction=128,  # Lower than main index for faster builds
            ef=32,  # Lower for faster cache lookups
            max_connections=16,  # Lower for cache use case
            dynamic_ef_min=16,
            dynamic_ef_max=64,
            dynamic_ef_factor=4,
            vector_cache_max_objects=100000,
            flat_search_cutoff=10000,
            cleanup_interval_seconds=600,  # More frequent cleanup for cache
            pq_enabled=False,
        )
        
        # Create cache collection
        client.collections.create(
            name=collection_name,
            vector_index_config=cache_vector_config,
            properties=[
                weaviate.classes.config.Property(
                    name="query_text",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="Cached query text"
                ),
                weaviate.classes.config.Property(
                    name="cache_id",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="Cache entry ID for Redis lookup"
                ),
                weaviate.classes.config.Property(
                    name="doc_id",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="Document ID for cleanup purposes"
                ),
                weaviate.classes.config.Property(
                    name="created_at",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="Creation timestamp"
                ),
                weaviate.classes.config.Property(
                    name="answer_preview",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="Preview of cached answer"
                ),
            ]
        )
        
        logger.info(f"Successfully created semantic cache collection '{collection_name}'")
        return True
        
    except Exception as e:
        logger.error(f"Error creating semantic cache collection: {e}")
        return False


def get_collection_info(client: weaviate.Client, collection_name: str) -> dict:
    """
    Get information about a Weaviate collection including HNSW configuration.
    
    Args:
        client: Weaviate client instance
        collection_name: Name of the collection
        
    Returns:
        Dictionary with collection information
    """
    try:
        if not client.collections.exists(collection_name):
            return {"exists": False, "error": "Collection not found"}
        
        collection = client.collections.get(collection_name)
        config = collection.config.get()
        
        # Extract HNSW configuration
        vector_config = config.vector_index_config
        
        info = {
            "exists": True,
            "name": collection_name,
            "vector_index_type": vector_config.__class__.__name__ if vector_config else "Unknown",
        }
        
        # Add HNSW-specific information if available
        if hasattr(vector_config, 'ef_construction'):
            info.update({
                "hnsw_ef_construction": vector_config.ef_construction,
                "hnsw_ef": vector_config.ef,
                "hnsw_max_connections": vector_config.max_connections,
                "distance_metric": vector_config.distance_metric,
            })
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting collection info for '{collection_name}': {e}")
        return {"exists": False, "error": str(e)}