# app/ingestion.py

import os
import weaviate

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
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
    Orchestrates the ingestion pipeline in a memory-efficient way:
    1. Streams document content chunk by chunk.
    2. Parses and further chunks the text.
    3. Creates embeddings and stores them in Weaviate.
    """
    parser = DocumentParser()
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    file_paths = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
    
    if not file_paths:
        print("No files found in the data directory. Aborting.")
        return

    print(f"Found {len(file_paths)} documents to ingest.")

    # Connect to Weaviate
    client = weaviate.connect_to_local(host=WEAVIATE_HOST, port=WEAVIATE_PORT)
    
    # Instantiate the Vector Store object
    vector_store = WeaviateVectorStore(
        client=client,
        index_name=INDEX_NAME,
        text_key="text",
        embedding=embedding_model,
    )

    total_chunks_indexed = 0
    for file_path in file_paths:
        try:
            print(f"Processing {file_path}...")
            
            # The parser now returns a generator
            doc_generator = parser.parse(file_path)
            
            all_chunks_for_file = []
            for parsed_data in doc_generator:
                # Further chunk the text from the parsed data
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
                    all_chunks_for_file.append(doc)
            
            if all_chunks_for_file:
                vector_store.add_documents(all_chunks_for_file, by_text=False)
                print(f"Indexed {len(all_chunks_for_file)} chunks from {file_path}")
                total_chunks_indexed += len(all_chunks_for_file)

        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
    
    if total_chunks_indexed > 0:
        print("\n--- Ingestion Complete ---")
        print(f"Successfully indexed a total of {total_chunks_indexed} chunks into Weaviate class '{INDEX_NAME}'.")
    else:
        print("No chunks were created from the documents. Aborting.")