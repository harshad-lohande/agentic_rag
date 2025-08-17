# app/retriever.py

import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Tuple

from config import settings

def create_retriever() -> Tuple[VectorStoreRetriever, weaviate.Client]:
    """
    Creates and returns retriever from the Weaviate vector store.
    
    This retriever is configured for hybrid search, combining vector similarity
    with keyword-based (BM25) search for more robust results.

    Also returns the client for proper connection closing.
    """
    # 1. Connect to the Weaviate instance
    client = weaviate.connect_to_local(
        host=settings.WEAVIATE_HOST,
        port=settings.WEAVIATE_PORT
    )

    # Instantiate the embedding model that was used for ingestion
    embedding_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    
    # 2. Instantiate the Vector Store object
    vector_store = WeaviateVectorStore(
        client=client,
        index_name=settings.INDEX_NAME,
        text_key="text",
        embedding=embedding_model,
    )

    # 3. Create and return the retriever
    retriever = vector_store.as_retriever(
        search_kwargs={
            'k': 2,          # Number of documents to retrieve
            'alpha': 0.5     # Balance between vector and keyword search (0 = keyword, 1 = vector)
        }
    )
    
    # Return both the retriever and the client
    return retriever, client