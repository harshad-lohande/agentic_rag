import sys
import os

# --- Add the project root to the Python path ---
# This ensures that the 'app' module can be found
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print(f"--- Project root added to path: {project_root} ---", flush=True)

# --- Now, try to import and run the ingestion script ---
try:
    print("--- Attempting to import ingestion module... ---", flush=True)
    from app.ingestion import ingest_documents
    print("--- Import successful. Starting ingestion... ---", flush=True)
    ingest_documents()
except ImportError as e:
    print(f"--- FAILED TO IMPORT INGESTION MODULE ---", flush=True)
    print(f"Error: {e}", flush=True)
    print("Please check that the file 'app/ingestion.py' exists and is accessible.", flush=True)
except Exception as e:
    print(f"--- AN UNEXPECTED ERROR OCCURRED DURING SCRIPT EXECUTION ---", flush=True)
    print(f"Error: {e}", flush=True)