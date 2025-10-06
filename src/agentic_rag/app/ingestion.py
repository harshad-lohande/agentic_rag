# src/agentic_rag/app/ingestion.py

import os
import tempfile
from urllib.parse import urlparse
import weaviate

# Optional: only needed for S3 ingestion
try:
    import boto3
except ImportError:
    boto3 = None

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from agentic_rag.app.document_parser import DocumentParser
from agentic_rag.app.chunking_strategy import chunk_text
from agentic_rag.app.weaviate_config import (
    create_weaviate_vector_store,
    create_semantic_cache_collection,
)
from agentic_rag.config import settings
from agentic_rag.logging_config import logger


def _is_s3_uri(path: str) -> bool:
    try:
        p = urlparse(path)
        return p.scheme == "s3" and bool(p.netloc)
    except Exception:
        return False


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    # s3://bucket/prefix -> (bucket, prefix)
    p = urlparse(s3_uri)
    return p.netloc, p.path.lstrip("/")


def ingest_documents():
    """
    Orchestrates the ingestion pipeline:
    - production + aws + s3://... -> stream S3 objects one-by-one via a temp file, process, then delete
    - development + none -> read from local directory
    """
    logger.info("Starting the ingestion process...")

    try:
        DATA_SOURCE = settings.DATA_TO_INDEX
        WEAVIATE_HOST = settings.WEAVIATE_HOST
        WEAVIATE_PORT = settings.WEAVIATE_PORT
        EMBEDDING_MODEL = settings.EMBEDDING_MODEL
        INDEX_NAME = settings.WEAVIATE_STORAGE_INDEX_NAME

        logger.info(f"Data source: {DATA_SOURCE}")

        # Initialize components
        parser = DocumentParser()
        logger.info("DocumentParser initialized.")
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")

        # Connect to Weaviate
        logger.info("Connecting to Weaviate...")
        client = weaviate.connect_to_local(host=WEAVIATE_HOST, port=WEAVIATE_PORT)
        logger.info("Successfully connected to Weaviate.")

        # Create semantic cache collection if enabled
        if settings.ENABLE_SEMANTIC_CACHE:
            logger.info("Setting up semantic cache collection...")
            create_semantic_cache_collection(client)

        # Vector store
        vector_store = create_weaviate_vector_store(
            client=client,
            index_name=INDEX_NAME,
            embedding_model=embedding_model,
            text_key="text",
            enable_hnsw_optimization=True,
        )

        def _process_file(file_path: str) -> int:
            """
            Core processing: parse -> chunk -> add_documents
            """
            try:
                doc_generator = parser.parse(file_path)
                docs = []
                for parsed_data in doc_generator:
                    chunks = chunk_text(
                        text=parsed_data["text"], embedding_model=embedding_model
                    )
                    for i, chunk in enumerate(chunks):
                        docs.append(
                            Document(
                                page_content=chunk,
                                metadata={
                                    "source": parsed_data["metadata"]["file_name"],
                                    "chunk_number": i + 1,
                                    **parsed_data["metadata"],
                                },
                            )
                        )
                if docs:
                    vector_store.add_documents(docs, by_text=False)
                    logger.info(f"Indexed {len(docs)} chunks from {file_path}")
                    return len(docs)
                return 0
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                return 0

        total_chunks_indexed = 0

        use_s3 = (
            settings.APP_ENVIRONMENT == "production"
            and settings.CLOUD_PROVIDER == "aws"
            and _is_s3_uri(DATA_SOURCE)
        )

        if use_s3:
            if boto3 is None:
                logger.error("boto3 not installed; cannot ingest from S3. Aborting.")
                return

            bucket, prefix = _parse_s3_uri(DATA_SOURCE)
            logger.info(
                f"Ingesting from S3 bucket='{bucket}' prefix='{prefix or '(root)'}'"
            )

            s3 = boto3.client("s3")
            paginator = s3.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

            files_found = 0
            with tempfile.TemporaryDirectory(prefix="ingest_s3_") as tmpdir:
                for page in pages:
                    for obj in page.get("Contents", []):
                        key = obj.get("Key")
                        if not key or key.endswith("/"):
                            continue
                        base = os.path.basename(key) or "object"
                        staged_path = os.path.join(tmpdir, base)
                        try:
                            s3.download_file(bucket, key, staged_path)
                            files_found += 1
                            total_chunks_indexed += _process_file(staged_path)
                        finally:
                            try:
                                if os.path.exists(staged_path):
                                    os.remove(staged_path)
                            except Exception:
                                pass

            if files_found == 0:
                logger.warning(
                    "No files found under the specified S3 prefix. Aborting."
                )
                return
        else:
            # Local directory mode
            data_dir = DATA_SOURCE
            if not os.path.isdir(data_dir):
                logger.warning(f"Local data directory does not exist: {data_dir}")
                return
            file_paths = [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if os.path.isfile(os.path.join(data_dir, f))
            ]
            if not file_paths:
                logger.warning("No files found in the data directory. Aborting.")
                return
            logger.info(f"Found {len(file_paths)} documents to ingest.")
            for fp in file_paths:
                total_chunks_indexed += _process_file(fp)

        if total_chunks_indexed > 0:
            logger.info("--- Ingestion Complete ---")
            logger.info(
                f"Successfully indexed a total of {total_chunks_indexed} chunks into Weaviate class '{INDEX_NAME}'."
            )
        else:
            logger.warning("No chunks were created from the documents. Aborting.")

    except Exception as e:
        logger.error(
            f"An unexpected error occurred during the ingestion process: {e}",
            exc_info=True,
        )
