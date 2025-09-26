"""
Unit tests for semantic cache functionality.
Tests cache hits/misses for similar, exact, and unrelated queries.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Tuple

from agentic_rag.app.semantic_cache import SemanticCache
from agentic_rag.config import settings


class TestSemanticCache:
    """Test semantic cache behavior with various query combinations."""

    @pytest.fixture
    def mock_semantic_cache(self):
        """Create a semantic cache instance with mocked dependencies."""
        cache = SemanticCache()
        
        # Mock Redis client
        cache.redis_client = AsyncMock()
        
        # Mock Weaviate client and vector store
        cache.weaviate_client = Mock()
        cache.cache_vector_store = Mock()
        
        # Mock embedding model
        cache.embedding_model = Mock()
        cache.embedding_model.embed_query.return_value = [0.1] * 768  # Mock embedding vector
        
        cache._initialized = True
        return cache

    @pytest.fixture
    def sample_cache_entries(self):
        """Sample cache entries for testing."""
        return {
            "nativepath_query": {
                "query": "What is Nativepath??",
                "answer": "NativePath is a company that offers health products, including MCT Oil Powder...",
                "cache_id": "cache_001",
                "created_at": "2025-01-24T01:00:00Z"
            },
            "mct_benefits_query": {
                "query": "What are the benefits of consuming MCTs daily?",
                "answer": "MCT (Medium Chain Triglycerides) benefits include improved energy metabolism...",
                "cache_id": "cache_002", 
                "created_at": "2025-01-24T01:01:00Z"
            }
        }

    @pytest.mark.asyncio
    async def test_exact_query_cache_hit(self, mock_semantic_cache, sample_cache_entries):
        """Test that exact query matches return cache hits."""
        cache = mock_semantic_cache
        entry = sample_cache_entries["nativepath_query"]
        
        # Mock exact match found in Redis
        cache.redis_client.get.return_value = entry["cache_id"]
        cache.redis_client.get.side_effect = None
        
        # Mock cache entry retrieval
        async def mock_get_cache_entry_by_id(cache_id):
            if cache_id == entry["cache_id"]:
                return entry
            return None
        cache._get_cache_entry_by_id = mock_get_cache_entry_by_id
        cache._update_cache_access = AsyncMock(return_value=entry)
        
        result = await cache.get_cached_answer(entry["query"])
        
        assert result is not None
        assert result["query"] == entry["query"]
        assert result["answer"] == entry["answer"]

    @pytest.mark.asyncio
    async def test_unrelated_queries_cache_miss(self, mock_semantic_cache, sample_cache_entries):
        """Test that completely unrelated queries should return cache miss."""
        cache = mock_semantic_cache
        
        # Mock no exact match
        cache.redis_client.get.return_value = None
        
        # Mock vector similarity search returns low similarity
        mock_doc = Mock()
        mock_doc.page_content = sample_cache_entries["nativepath_query"]["query"]
        mock_doc.metadata = {"cache_id": sample_cache_entries["nativepath_query"]["cache_id"]}
        
        # This is the critical test - unrelated queries should have LOW similarity, not 1.0
        cache._vector_similarity_search = AsyncMock(return_value=[(mock_doc, 0.3)])  # Low similarity
        
        # Mock other similarity methods
        cache._embedding_similarity = Mock(return_value=0.2)
        cache._ce_similarity = AsyncMock(return_value=0.1)
        cache._lexical_similarity = Mock(return_value=0.05)
        
        unrelated_query = sample_cache_entries["mct_benefits_query"]["query"]
        result = await cache.get_cached_answer(unrelated_query)
        
        # Should be cache miss for unrelated queries
        assert result is None

    @pytest.mark.asyncio
    async def test_similar_queries_cache_hit(self, mock_semantic_cache, sample_cache_entries):
        """Test that similar queries should return cache hits."""
        cache = mock_semantic_cache
        
        # Mock no exact match
        cache.redis_client.get.return_value = None
        
        # Mock vector similarity search returns high similarity for related queries
        original_query = sample_cache_entries["mct_benefits_query"]["query"]
        mock_doc = Mock()
        mock_doc.page_content = original_query
        mock_doc.metadata = {"cache_id": sample_cache_entries["mct_benefits_query"]["cache_id"]}
        
        # High similarity for related queries
        cache._vector_similarity_search = AsyncMock(return_value=[(mock_doc, 0.95)])
        
        # Mock cache entry retrieval
        async def mock_get_cache_entry_by_id(cache_id):
            if cache_id == sample_cache_entries["mct_benefits_query"]["cache_id"]:
                return sample_cache_entries["mct_benefits_query"]
            return None
        cache._get_cache_entry_by_id = mock_get_cache_entry_by_id
        cache._update_cache_access = AsyncMock(return_value=sample_cache_entries["mct_benefits_query"])
        
        # Test similar queries
        similar_queries = [
            "Explain the benefits of MCT",
            "What are the benefits of MCTs?", 
            "How consuming MCT can be beneficial to the health?",
            "What benefits can I expect if I consume MCT daily?"
        ]
        
        for similar_query in similar_queries:
            result = await cache.get_cached_answer(similar_query)
            assert result is not None, f"Similar query should get cache hit: {similar_query}"
            assert result["cache_id"] == sample_cache_entries["mct_benefits_query"]["cache_id"]

    def test_lexical_similarity(self, mock_semantic_cache):
        """Test lexical similarity calculation."""
        cache = mock_semantic_cache
        
        # Identical text should have similarity 1.0
        assert cache._lexical_similarity("hello world", "hello world") == 1.0
        
        # Completely different text should have similarity 0.0
        assert cache._lexical_similarity("hello world", "goodbye universe") == 0.0
        
        # Partial overlap should give intermediate similarity
        similarity = cache._lexical_similarity("hello world test", "hello world example")
        assert 0.0 < similarity < 1.0
        
        # The specific queries from the bug report should have very low lexical similarity
        query1 = "What is Nativepath??"
        query2 = "What are the benefits of consuming MCTs daily?"
        similarity = cache._lexical_similarity(query1, query2)
        assert similarity < 0.3  # Should be low for unrelated queries

    @pytest.mark.asyncio 
    async def test_bug_reproduction_different_queries(self, mock_semantic_cache, sample_cache_entries):
        """Reproduce the specific bug: unrelated queries getting cache hits."""
        cache = mock_semantic_cache
        
        # First, cache the Nativepath query
        nativepath_entry = sample_cache_entries["nativepath_query"]
        
        # Mock no exact match for the second query
        cache.redis_client.get.return_value = None
        
        # Mock vector similarity search - this is where the bug likely occurs
        # The bug shows similarity of 1.000 for completely unrelated queries
        mock_doc = Mock()
        mock_doc.page_content = nativepath_entry["query"]
        mock_doc.metadata = {"cache_id": nativepath_entry["cache_id"]}
        
        # Test what happens when vector search incorrectly returns high similarity
        cache._vector_similarity_search = AsyncMock(return_value=[(mock_doc, 1.0)])  # Incorrect high similarity
        
        # Mock other similarity methods to return low scores (as they should for unrelated queries)
        cache._embedding_similarity = Mock(return_value=0.2)  # Low embedding similarity
        cache._ce_similarity = AsyncMock(return_value=0.1)    # Low cross-encoder similarity  
        cache._lexical_similarity = Mock(return_value=0.05)   # Low lexical similarity
        
        # Mock cache entry retrieval
        async def mock_get_cache_entry_by_id(cache_id):
            if cache_id == nativepath_entry["cache_id"]:
                return nativepath_entry
            return None
        cache._get_cache_entry_by_id = mock_get_cache_entry_by_id
        cache._update_cache_access = AsyncMock(return_value=nativepath_entry)
        
        # This should NOT return a cache hit for unrelated query
        mct_query = sample_cache_entries["mct_benefits_query"]["query"]
        result = await cache.get_cached_answer(mct_query)
        
        # After fix, this should be None (cache miss) due to additional validation
        assert result is None, "Unrelated queries should not get cache hits even with high vector similarity"

    @pytest.mark.asyncio
    async def test_false_positive_prevention_high_vector_low_others(self, mock_semantic_cache):
        """Test that high vector similarity with low embedding/lexical similarity is rejected."""
        cache = mock_semantic_cache
        
        # Mock no exact match
        cache.redis_client.get.return_value = None
        
        # Mock a scenario where vector similarity is suspiciously high but other measures are low
        mock_doc = Mock()
        mock_doc.page_content = "What is artificial intelligence?"
        mock_doc.metadata = {"cache_id": "test_cache_id"}
        
        # High vector similarity (suspicious)
        cache._vector_similarity_search = AsyncMock(return_value=[(mock_doc, 0.99)])
        
        # But low embedding and lexical similarity (indicating false positive)
        cache._embedding_similarity = Mock(return_value=0.3)  # Low
        cache._lexical_similarity = Mock(return_value=0.05)   # Very low
        
        result = await cache.get_cached_answer("What are the benefits of chocolate?")
        
        # Should be rejected due to suspicious similarity pattern
        assert result is None


class TestSemanticCacheIntegration:
    """Integration tests for semantic cache with mock backends."""
    
    @pytest.mark.asyncio
    async def test_cache_workflow_integration(self):
        """Test complete cache workflow: store, retrieve, validate."""
        # This will be implemented after core fix
        pass