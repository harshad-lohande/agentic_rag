# src/agentic_rag/scripts/delete_indexed_data.py

import weaviate
from agentic_rag.config import settings
from agentic_rag.logging_config import setup_logging, logger

def clear_weaviate_index():
    """
    Connects to Weaviate and deletes all data from the specified index.
    """
    try:
        # Connect to your local Weaviate instance
        client = weaviate.connect_to_local(
            host=settings.WEAVIATE_HOST,
            port=settings.WEAVIATE_PORT
        )

        index_name = settings.INDEX_NAME

        # Check if the collection exists before trying to delete
        if client.collections.exists(index_name):
            logger.info(f"Index '{index_name}' exists. Deleting all objects...")
            # Delete the entire collection/class
            client.collections.delete(index_name)
            logger.info(f"Successfully deleted index '{index_name}'.")
        else:
            logger.warning(f"Index '{index_name}' does not exist. Nothing to delete.")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        if 'client' in locals() and client.is_connected():
            client.close()

def main():
    setup_logging()
    clear_weaviate_index()

if __name__ == '__main__':
    main()