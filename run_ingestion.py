import sys
import os

# --- Add the project root to the Python path ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# --- Setup Logging ---
from app.logging_config import setup_logging
setup_logging()

# --- Run the ingestion script ---
try:
    from app.ingestion import ingest_documents
    ingest_documents()
except ImportError as e:
    from app.logging_config import logger
    logger.error(f"Failed to import ingestion module: {e}")
    logger.error("Please check that the file 'app/ingestion.py' exists and is accessible.")
except Exception as e:
    from app.logging_config import logger
    logger.error(f"An unexpected error occurred during script execution: {e}")