import os
import weaviate  # <-- Import the weaviate library
from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
# Note: The import path has changed for the Weaviate vector store
from langchain_weaviate.vectorstores import WeaviateVectorStore

from app.document_parser import DocumentParser
from app.chunking_strategy import chunk_text
from config import settings

# Define constants
DATA_DIR = settings.DATA_TO_INDEX
WEAVIATE_HOST = settings.WEAVIATE_HOST
WEAVIATE_PORT = settings.WEAVIATE_PORT
EMBEDDING_MODEL = settings.EMBEDDING_MODEL
INDEX_NAME = settings.INDEX_NAME

def ingest_documents():
    """
    Orchestrates the ingestion pipeline:
    1. Loads documents from the data directory.
    2. Parses and chunks the documents.
    3. Creates embeddings and stores them in Weaviate.
    """
    # Initialize the components
    parser = DocumentParser()
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    file_paths = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
    
    if not file_paths:
        print("No files found in the data directory. Aborting.")
        return

    print(f"Found {len(file_paths)} documents to ingest.")

    all_chunks = []
    for file_path in file_paths:
        try:
            print(f"Processing {file_path}...")
            parsed_data = parser.parse(file_path)
            chunks = chunk_text(parsed_data["text"])
            
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": parsed_data["metadata"]["file_name"],
                        "chunk_number": i + 1,
                        **parsed_data["metadata"]
                    }
                )
                all_chunks.append(doc)
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
    
    if not all_chunks:
        print("No chunks were created from the documents. Aborting.")
        return

    print(f"Created a total of {len(all_chunks)} chunks. Now indexing...")

    #  --- THIS IS THE UPDATED PART ---
    # Create the Weaviate client using the new V4 method
    client = weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT
    )

    # Use the new WeaviateVectorStore class and pass the client
    WeaviateVectorStore.from_documents(
        documents=all_chunks,
        embedding=embedding_model,
        client=client,  # <-- Pass the client object
        index_name=INDEX_NAME,
        by_text=False
    )

    print("--- Ingestion Complete ---")
    print(f"Successfully indexed {len(all_chunks)} chunks into Weaviate class '{INDEX_NAME}'.")