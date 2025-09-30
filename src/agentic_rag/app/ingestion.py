# src/agentic_rag/app/ingestion.py

import os
import weaviate

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


def ingest_documents():
    """
    Orchestrates the ingestion pipeline in a memory-efficient way:
    1. Streams document content chunk by chunk.
    2. Parses and further chunks the text.
    3. Creates embeddings and stores them in Weaviate.
    """
    logger.info("Starting the ingestion process...")

    try:
        # Define constants
        DATA_DIR = settings.DATA_TO_INDEX
        WEAVIATE_HOST = settings.WEAVIATE_HOST
        WEAVIATE_PORT = settings.WEAVIATE_PORT
        EMBEDDING_MODEL = settings.EMBEDDING_MODEL
        INDEX_NAME = settings.WEAVIATE_STORAGE_INDEX_NAME

        logger.info(f"Data directory is set to: {DATA_DIR}")

        # Initialize the components
        parser = DocumentParser()
        logger.info("DocumentParser initialized.")

        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")

        file_paths = [
            os.path.join(DATA_DIR, f)
            for f in os.listdir(DATA_DIR)
            if os.path.isfile(os.path.join(DATA_DIR, f))
        ]

        if not file_paths:
            logger.warning("No files found in the data directory. Aborting.")
            return

        logger.info(f"Found {len(file_paths)} documents to ingest.")

        # Connect to Weaviate
        logger.info("Connecting to Weaviate...")
        client = weaviate.connect_to_local(host=WEAVIATE_HOST, port=WEAVIATE_PORT)
        logger.info("Successfully connected to Weaviate.")

        # Create semantic cache collection if enabled
        if settings.ENABLE_SEMANTIC_CACHE:
            logger.info("Setting up semantic cache collection...")
            create_semantic_cache_collection(client)

        # Instantiate the Vector Store object with HNSW optimization
        vector_store = create_weaviate_vector_store(
            client=client,
            index_name=INDEX_NAME,
            embedding_model=embedding_model,
            text_key="text",
            enable_hnsw_optimization=True,
        )

        total_chunks_indexed = 0
        for file_path in file_paths:
            try:
                logger.info(f"Processing {file_path}...")

                doc_generator = parser.parse(file_path)

                all_chunks_for_file = []
                for parsed_data in doc_generator:
                    chunks = chunk_text(
                        text=parsed_data["text"], embedding_model=embedding_model
                    )

                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "source": parsed_data["metadata"]["file_name"],
                                "chunk_number": i + 1,
                                **parsed_data["metadata"],
                            },
                        )
                        all_chunks_for_file.append(doc)

                if all_chunks_for_file:
                    vector_store.add_documents(all_chunks_for_file, by_text=False)
                    logger.info(
                        f"Indexed {len(all_chunks_for_file)} chunks from {file_path}"
                    )
                    total_chunks_indexed += len(all_chunks_for_file)

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")

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
