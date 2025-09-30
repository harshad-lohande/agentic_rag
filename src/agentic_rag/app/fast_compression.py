# src/agentic_rag/app/fast_compression.py

from typing import List
from langchain_core.documents import Document

from agentic_rag.config import settings
from agentic_rag.logging_config import logger
from agentic_rag.app.model_registry import model_registry

try:
    from sentence_transformers import SentenceTransformer, util  # type: ignore

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore
    SENTENCE_TRANSFORMERS_AVAILABLE = False


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
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.warning(
                    "SentenceTransformers not available, cannot use semantic similarity"
                )
                return None

            # Prefer the singleton ST from the model registry
            st = model_registry.get_sentence_transformer_for_compression()
            if st is not None:
                self._sentence_transformer = st
                return self._sentence_transformer

            # On-demand fallback: use the configured embedding model (not a different default)
            model_name = settings.FAST_COMPRESSION_MODEL
            logger.debug(
                f"Loading SentenceTransformer on-demand for compression: {model_name}"
            )
            try:
                self._sentence_transformer = SentenceTransformer(model_name)  # type: ignore[name-defined]
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer: {e}")
                return None

        return self._sentence_transformer

    def compress_documents(
        self, documents: List[Document], query: str, max_sentences: int = settings.FAST_COMPRESSION_MAX_SENTENCES
    ) -> List[Document]:
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

        logger.info(f"Fast extractive compression for {len(documents)} documents")

        try:
            sentence_transformer = self._get_sentence_transformer()

            # If sentence transformer is not available, return simple truncated documents
            if sentence_transformer is None:
                logger.warning("Falling back to simple text truncation for compression")
                return self._simple_truncate_documents(
                    documents, max_sentences * 50
                )  # Estimate ~50 chars per sentence

            compressed_docs = []

            # Get query embedding once
            query_embedding = sentence_transformer.encode(query, convert_to_tensor=True)

            for doc in documents:
                compressed_content = self._extract_relevant_sentences(
                    doc.page_content,
                    query_embedding,
                    sentence_transformer,
                    max_sentences,
                )

                # Create new document with compressed content
                compressed_doc = Document(
                    page_content=compressed_content,
                    metadata={
                        **doc.metadata,
                        "compression_method": "fast_extractive",
                        "original_length": len(doc.page_content),
                        "compressed_length": len(compressed_content),
                    },
                )
                compressed_docs.append(compressed_doc)

            total_original = sum(len(doc.page_content) for doc in documents)
            total_compressed = sum(len(doc.page_content) for doc in compressed_docs)
            compression_ratio = (
                (1 - total_compressed / total_original) if total_original > 0 else 1.0
            )

            logger.info(
                f"Fast compression complete: {compression_ratio:.2%} of original size compressed"
            )
            return compressed_docs

        except Exception as e:
            logger.error(f"Fast compression failed: {e}")
            # Return original documents if compression fails
            return documents

    def _simple_truncate_documents(
        self, documents: List[Document], max_chars: int
    ) -> List[Document]:
        """Simple fallback compression when sentence transformers is not available."""
        compressed_docs = []

        for doc in documents:
            content = doc.page_content
            if len(content) > max_chars:
                # Simple truncation - could be enhanced with smarter boundary detection
                truncated_content = content[:max_chars].rsplit(" ", 1)[0] + "..."
            else:
                truncated_content = content

            compressed_doc = Document(
                page_content=truncated_content,
                metadata={
                    **doc.metadata,
                    "compression_method": "simple_truncation",
                    "original_length": len(content),
                    "compressed_length": len(truncated_content),
                },
            )
            compressed_docs.append(compressed_doc)

        return compressed_docs

    def _extract_relevant_sentences(
        self, text: str, query_embedding, sentence_transformer, max_sentences: int
    ) -> str:
        """Extract the most relevant sentences from text based on query similarity."""

        # Split text into sentences (simple approach)
        sentences = self._split_into_sentences(text)

        if len(sentences) <= max_sentences:
            return text  # No compression needed

        try:
            # Get embeddings for all sentences
            sentence_embeddings = sentence_transformer.encode(
                sentences, convert_to_tensor=True
            )

            # Calculate similarity scores
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                similarities = util.cos_sim(query_embedding, sentence_embeddings)[0]

                # Get indices of top sentences
                top_indices = similarities.argsort(descending=True)[:max_sentences]

                # Sort indices to maintain original order
                top_indices = sorted(top_indices.cpu().numpy())

                # Extract top sentences and join them
                relevant_sentences = [sentences[i] for i in top_indices]
                return " ".join(relevant_sentences)
            else:
                # Simple fallback - take first max_sentences
                return " ".join(sentences[:max_sentences])

        except Exception as e:
            logger.warning(f"Sentence similarity calculation failed: {e}")
            # Fallback to first max_sentences
            return " ".join(sentences[:max_sentences])

    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting. Could be enhanced with nltk/spacy for better accuracy."""
        import re

        # Simple sentence splitting on common sentence endings
        sentences = re.split(r"[.!?]+", text)

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
