# src/agentic_rag/app/fast_compression.py

import numpy as np
from typing import List, Optional
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer, util

from agentic_rag.config import settings
from agentic_rag.logging_config import logger
from agentic_rag.app.model_registry import model_registry


class FastExtractiveDocs:
    """
    Fast extractive document compression using pre-loaded SentenceTransformer.
    
    This replaces the expensive LLM-based compression that was taking 50+ seconds
    with a fast extractive method that completes in milliseconds.
    """
    
    def __init__(self):
        self._sentence_transformer = None
    
    def _get_sentence_transformer(self):
        """Get SentenceTransformer model for sentence-level similarity."""
        if self._sentence_transformer is None:
            # Try to use the same embedding model from registry for consistency
            embedding_model = model_registry.get_embedding_model()
            if embedding_model and hasattr(embedding_model, 'client'):
                # Use the same model as the embedding model if possible
                model_name = settings.EMBEDDING_MODEL
            else:
                # Fallback to a fast sentence transformer model
                model_name = "all-MiniLM-L6-v2"  # Fast and efficient
                
            logger.debug(f"Loading SentenceTransformer for fast compression: {model_name}")
            self._sentence_transformer = SentenceTransformer(model_name)
            
        return self._sentence_transformer
    
    def compress_documents(self, documents: List[Document], query: str, max_sentences: int = 15) -> List[Document]:
        """
        Fast extractive compression using sentence-level similarity.
        
        Args:
            documents: List of documents to compress
            query: User query for relevance scoring
            max_sentences: Maximum number of sentences to keep per document
            
        Returns:
            List of compressed documents with most relevant sentences
        """
        if not documents:
            return documents
            
        logger.info(f"🚀 Fast extractive compression for {len(documents)} documents")
        
        try:
            sentence_transformer = self._get_sentence_transformer()
            compressed_docs = []
            
            # Get query embedding once
            query_embedding = sentence_transformer.encode(query, convert_to_tensor=True)
            
            for doc in documents:
                compressed_content = self._extract_relevant_sentences(
                    doc.page_content, 
                    query_embedding, 
                    sentence_transformer, 
                    max_sentences
                )
                
                # Create new document with compressed content
                compressed_doc = Document(
                    page_content=compressed_content,
                    metadata={
                        **doc.metadata,
                        "compression_method": "fast_extractive",
                        "original_length": len(doc.page_content),
                        "compressed_length": len(compressed_content)
                    }
                )
                compressed_docs.append(compressed_doc)
            
            total_original = sum(len(doc.page_content) for doc in documents)
            total_compressed = sum(len(doc.page_content) for doc in compressed_docs)
            compression_ratio = total_compressed / total_original if total_original > 0 else 1.0
            
            logger.info(f"✅ Fast compression complete: {compression_ratio:.2%} of original size retained")
            return compressed_docs
            
        except Exception as e:
            logger.error(f"❌ Fast compression failed: {e}")
            # Return original documents if compression fails
            return documents
    
    def _extract_relevant_sentences(self, text: str, query_embedding, sentence_transformer, max_sentences: int) -> str:
        """Extract the most relevant sentences from text based on query similarity."""
        
        # Split text into sentences (simple approach)
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= max_sentences:
            return text  # No compression needed
        
        # Get embeddings for all sentences
        sentence_embeddings = sentence_transformer.encode(sentences, convert_to_tensor=True)
        
        # Calculate similarity scores
        similarities = util.cos_sim(query_embedding, sentence_embeddings)[0]
        
        # Get indices of top sentences
        top_indices = similarities.argsort(descending=True)[:max_sentences]
        
        # Sort indices to maintain original order
        top_indices = sorted(top_indices.cpu().numpy())
        
        # Extract top sentences and join them
        relevant_sentences = [sentences[i] for i in top_indices]
        return " ".join(relevant_sentences)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting. Could be enhanced with nltk/spacy for better accuracy."""
        import re
        
        # Simple sentence splitting on common sentence endings
        sentences = re.split(r'[.!?]+', text)
        
        # Clean up and filter out very short sentences
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        return sentences


# Global instance for reuse
fast_extractive_compressor = FastExtractiveDocs()


def fast_compress_documents(documents: List[Document], query: str) -> List[Document]:
    """
    Fast document compression function that can replace expensive LLM-based compression.
    
    This provides a 1000x+ speedup over LLM-based compression while maintaining
    reasonable quality by extracting the most relevant sentences.
    """
    return fast_extractive_compressor.compress_documents(documents, query)