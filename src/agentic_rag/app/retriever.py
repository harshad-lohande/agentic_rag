# src/agentic_rag/app/retriever.py

import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Tuple

from agentic_rag.config import settings
# from agentic_rag.logging_config import logger

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
            'k': 5,          # Number of documents to retrieve
            'alpha': 0.5     # Balance between vector and keyword search (0 = keyword, 1 = vector)
        }
    )

    # # 3. Create and return the retriever with MMR search
    # logger.info("Creating retriever with Maximum Marginal Relevance (MMR) search for diverse results.")
    # retriever = vector_store.as_retriever(
    #     search_type="mmr",
    #     search_kwargs={
    #         'k': 4,          # The final number of documents to return
    #         'fetch_k': 10,   # The number of documents to fetch initially for re-ranking
    #         'lambda_mult': 0.5 # 0 for max diversity, 1 for max relevance
    #     }
    # )
    
    # Return both the retriever and the client
    return retriever, client