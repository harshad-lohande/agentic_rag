# src/agentic_rag/scripts/run_ingestion.py

from agentic_rag.logging_config import setup_logging
from agentic_rag.app.ingestion import ingest_documents

def main():
    setup_logging()
    ingest_documents()

if __name__ == "__main__":
    main()