# run_ingestion.py

from agentic_rag.logging_config import setup_logging
from agentic_rag.app.ingestion import ingest_documents

def main():
    try:
        setup_logging()
        ingest_documents()
    except ImportError as e:
        from agentic_rag.logging_config import logger
        logger.error(f"Failed to import ingestion module: {e}")
        logger.error("Please check that the file 'app/ingestion.py' exists and is accessible.")
    except Exception as e:
        from agentic_rag.logging_config import logger
        logger.error(f"An unexpected error occurred during script execution: {e}")

if __name__ == "__main__":
    main()