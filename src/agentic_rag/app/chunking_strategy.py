# src/agentic_rag/app/chunking_strategy.py

from typing import List, Literal, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings


try:
    # Semantic chunking (LangChain Experimental)
    from langchain_experimental.text_splitter import SemanticChunker
except Exception:
    SemanticChunker = None

from agentic_rag.config import settings
from agentic_rag.logging_config import logger
from langchain_huggingface import HuggingFaceEmbeddings


def chunk_text(
    text: str,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
    strategy: Optional[Literal["recursive", "semantic"]] = settings.CHUNKING_STRATEGY,
    embedding_model: Optional[Embeddings] = None,
) -> List[str]:
    """
    Split text into chunks using either recursive (default) or semantic chunking.
    Args:
        text: Input text to split.
        chunk_size: Used by recursive splitter.
        chunk_overlap: Overlap for recursive or buffer for semantic.
        strategy: "recursive" or "semantic". If None, uses settings.CHUNKING_STRATEGY.
        embedding_model: Pre-loaded embedding model for semantic chunking.
    Returns:
        List[str]: chunks
    """
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")

    chosen = strategy

    if chosen == "semantic":
        if SemanticChunker is None:
            logger.warning(
                "SemanticChunker not available. Falling back to RecursiveCharacterTextSplitter."
            )
        else:
            try:
                logger.info("--- Using SemanticChunker for text chunking ---")

                embeddings = (
                    embedding_model
                    if embedding_model
                    else HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
                )

                splitter = SemanticChunker(
                    embeddings=embeddings,
                    breakpoint_threshold_type=settings.SEMANTIC_BREAKPOINT_TYPE,
                    breakpoint_threshold_amount=settings.SEMANTIC_BREAKPOINT_AMOUNT,
                    buffer_size=chunk_overlap,
                    add_start_index=True,
                )
                return splitter.split_text(text)
            except Exception as e:
                logger.error(
                    f"Semantic chunking failed, falling back to recursive. Error: {e}"
                )

    # Fallback or explicit recursive
    logger.info("--- Using RecursiveCharacterTextSplitter for text chunking ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    return text_splitter.split_text(text)
