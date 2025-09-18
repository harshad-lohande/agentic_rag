#!/usr/bin/env python3
"""
Test script to validate the fixes for the three issues:
1. Semantic cache failure
2. SentenceTransformer model pre-loading
3. Aggressive compression (60%+ reduction)
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import json
from typing import List

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from langchain_core.documents import Document
    from agentic_rag.config import settings
    from agentic_rag.app.model_registry import model_registry
    from agentic_rag.app.fast_compression import fast_extractive_compressor, fast_compress_documents
    from agentic_rag.app.semantic_cache import SemanticCache
except ImportError as e:
    print(f"Import error: {e}")
    print("This test requires the agentic_rag package to be properly installed")
    sys.exit(1)


class TestSemanticCacheFix(unittest.TestCase):
    """Test the semantic cache metadata fix."""
    
    def setUp(self):
        self.cache = SemanticCache()
        
    def test_update_existing_cache_entry_with_none_metadata(self):
        """Test that updating an entry with None metadata doesn't crash."""
        # Mock the cache entry retrieval
        with patch.object(self.cache, '_get_cache_entry_by_id') as mock_get:
            with patch.object(self.cache, '_generate_cache_key') as mock_key:
                with patch.object(self.cache, 'redis_client') as mock_redis:
                    # Simulate existing entry with None metadata
                    mock_get.return_value = {
                        "cache_id": "test-id",
                        "answer": "old answer",
                        "access_count": 1,
                        "metadata": None  # This was causing the bug
                    }
                    mock_key.return_value = "test-key"
                    mock_redis.setex = AsyncMock(return_value=True)
                    
                    # This should not crash anymore
                    result = asyncio.run(self.cache._update_existing_cache_entry(
                        "test-id", 
                        "new answer", 
                        {"new_key": "new_value"}
                    ))
                    
                    self.assertTrue(result)
                    
    def test_update_existing_cache_entry_with_missing_metadata(self):
        """Test that updating an entry without metadata key doesn't crash."""
        with patch.object(self.cache, '_get_cache_entry_by_id') as mock_get:
            with patch.object(self.cache, '_generate_cache_key') as mock_key:
                with patch.object(self.cache, 'redis_client') as mock_redis:
                    # Simulate existing entry without metadata key
                    mock_get.return_value = {
                        "cache_id": "test-id",
                        "answer": "old answer", 
                        "access_count": 1
                        # No metadata key at all
                    }
                    mock_key.return_value = "test-key"
                    mock_redis.setex = AsyncMock(return_value=True)
                    
                    # This should not crash anymore
                    result = asyncio.run(self.cache._update_existing_cache_entry(
                        "test-id",
                        "new answer",
                        {"new_key": "new_value"}
                    ))
                    
                    self.assertTrue(result)


class TestModelRegistryFix(unittest.TestCase):
    """Test the model registry SentenceTransformer pre-loading."""
    
    def test_sentence_transformer_compression_getter(self):
        """Test that the new getter method exists."""
        # Test that the method exists
        self.assertTrue(hasattr(model_registry, 'get_sentence_transformer_compression'))
        
        # Test that it returns None when not initialized
        result = model_registry.get_sentence_transformer_compression()
        self.assertIsNone(result)
        
    def test_model_info_includes_sentence_transformer(self):
        """Test that model info includes the new SentenceTransformer."""
        info = model_registry.get_model_info()
        self.assertIn('sentence_transformer_compression', info)
        self.assertEqual(info['sentence_transformer_compression'], 'all-MiniLM-L6-v2')


class TestCompressionFix(unittest.TestCase):
    """Test the aggressive compression implementation."""
    
    def test_compression_config_exists(self):
        """Test that the new compression settings exist."""
        self.assertTrue(hasattr(settings, 'FAST_COMPRESSION_MAX_SENTENCES'))
        self.assertTrue(hasattr(settings, 'FAST_COMPRESSION_TARGET_RATIO'))
        self.assertEqual(settings.FAST_COMPRESSION_MAX_SENTENCES, 6)
        self.assertEqual(settings.FAST_COMPRESSION_TARGET_RATIO, 0.40)
        
    def test_additional_truncation_method_exists(self):
        """Test that the additional truncation method exists."""
        self.assertTrue(hasattr(fast_extractive_compressor, '_apply_additional_truncation'))
        
    def test_additional_truncation_reduces_content(self):
        """Test that additional truncation actually reduces content size."""
        # Create test documents
        long_text = "This is a sentence. " * 100  # 2000+ characters
        docs = [Document(
            page_content=long_text,
            metadata={"original_length": len(long_text)}
        )]
        
        # Apply additional truncation to 40% 
        result = fast_extractive_compressor._apply_additional_truncation(docs, 0.40)
        
        self.assertEqual(len(result), 1)
        compressed_length = len(result[0].page_content)
        original_length = len(long_text)
        
        # Should be significantly smaller
        ratio = compressed_length / original_length
        self.assertLess(ratio, 0.50)  # Should be less than 50% of original
        
    def test_fast_compression_uses_config_settings(self):
        """Test that fast_compress_documents uses the new configuration."""
        with patch.object(fast_extractive_compressor, 'compress_documents') as mock_compress:
            mock_compress.return_value = []
            
            test_docs = [Document(page_content="test content")]
            fast_compress_documents(test_docs, "test query")
            
            # Verify it was called with the configured max_sentences
            mock_compress.assert_called_once_with(
                test_docs, 
                "test query", 
                max_sentences=settings.FAST_COMPRESSION_MAX_SENTENCES
            )


class TestCompressionRatio(unittest.TestCase):
    """Test that compression actually achieves the target ratio."""
    
    def test_compression_achieves_target_ratio(self):
        """Test compression with realistic content."""
        # Create a realistic document with multiple sentences
        sentences = [
            "Machine learning is a subset of artificial intelligence.",
            "It focuses on algorithms that can learn from data.",
            "Deep learning uses neural networks with multiple layers.", 
            "Natural language processing deals with text understanding.",
            "Computer vision processes and analyzes visual information.",
            "Reinforcement learning learns through trial and error.",
            "Supervised learning uses labeled training data.",
            "Unsupervised learning finds patterns in unlabeled data.",
            "Feature engineering is crucial for model performance.",
            "Cross-validation helps assess model generalization.",
            "Overfitting occurs when models memorize training data.",
            "Regularization techniques prevent overfitting issues.",
            "Gradient descent optimizes model parameters iteratively.",
            "Ensemble methods combine multiple models together.",
            "Data preprocessing cleans and transforms raw data."
        ]
        
        long_text = " ".join(sentences)  # Should be substantial content
        original_length = len(long_text)
        
        print(f"Original text length: {original_length} characters")
        
        docs = [Document(page_content=long_text, metadata={})]
        
        # Mock the sentence transformer to avoid loading the actual model
        with patch('agentic_rag.app.fast_compression.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            with patch.object(fast_extractive_compressor, '_get_sentence_transformer') as mock_transformer:
                # Mock transformer and similarity calculation
                mock_model = Mock()
                mock_model.encode.return_value = Mock()
                mock_transformer.return_value = mock_model
                
                # Mock the sentence similarity to return the first 6 sentences
                with patch.object(fast_extractive_compressor, '_extract_relevant_sentences') as mock_extract:
                    # Return just the first N sentences based on max_sentences
                    def mock_extract_func(text, query_emb, transformer, max_sent):
                        sent_list = fast_extractive_compressor._split_into_sentences(text)
                        return " ".join(sent_list[:max_sent])
                    
                    mock_extract.side_effect = mock_extract_func
                    
                    # Compress the documents
                    compressed = fast_compress_documents(docs, "test query")
                    
                    self.assertEqual(len(compressed), 1)
                    compressed_length = len(compressed[0].page_content)
                    compression_ratio = compressed_length / original_length
                    
                    print(f"Compressed text length: {compressed_length} characters")
                    print(f"Compression ratio: {compression_ratio:.2%}")
                    
                    # Should achieve significant compression
                    self.assertLess(compression_ratio, 0.60, 
                                  f"Compression ratio {compression_ratio:.2%} should be less than 60%")


if __name__ == '__main__':
    print("Running tests for agentic RAG fixes...")
    print("=" * 50)
    
    # Run tests with verbose output
    unittest.main(verbosity=2)