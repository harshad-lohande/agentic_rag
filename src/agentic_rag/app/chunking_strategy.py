# app/chunking_strategy.py

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Splits a long text into smaller chunks using a recursive character text splitter.

    Args:
        text: The input text to be chunked.
        chunk_size: The maximum size of each chunk (in characters).
        chunk_overlap: The number of characters to overlap between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True, # Helpful for tracking chunk origins
    )
    
    chunks = text_splitter.split_text(text)
    return chunks