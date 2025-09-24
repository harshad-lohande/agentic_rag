"""
Integration tests for semantic cache with realistic query scenarios.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List, Tuple

from agentic_rag.app.semantic_cache import SemanticCache


class TestSemanticCacheIntegrationScenarios:
    """Integration tests with realistic query scenarios from the problem statement."""

    @pytest.fixture
    def real_scenario_cache(self):
        """Create semantic cache with realistic mocking for the bug scenario."""
        cache = SemanticCache()
        
        # Mock dependencies
        cache.redis_client = AsyncMock()
        cache.weaviate_client = Mock()
        cache.cache_vector_store = Mock()
        cache.embedding_model = Mock()
        cache._initialized = True
        
        return cache

    @pytest.mark.asyncio
    async def test_nativepath_vs_mct_benefits_queries_should_not_match(self, real_scenario_cache):
        """
        Test the exact scenario from the bug report:
        Query 1: "What is Nativepath??" (cached)
        Query 2: "What are the benefits of consuming MCTs daily?" (should NOT get cache hit)
        """
        cache = real_scenario_cache
        
        # Simulate the first query being cached
        nativepath_query = "What is Nativepath??"
        nativepath_entry = {
            "query": nativepath_query,
            "answer": "NativePath is a company that offers health products, including MCT Oil Powder...",
            "cache_id": "cache_nativepath_001",
            "created_at": "2025-01-24T01:00:00Z"
        }
        
        # Mock that the second query has no exact match
        cache.redis_client.get.return_value = None
        
        # Mock vector similarity search - Weaviate would return the cached Nativepath query
        mock_doc = Mock()
        mock_doc.page_content = nativepath_query
        mock_doc.metadata = {"cache_id": nativepath_entry["cache_id"]}
        
        # Simulate realistic vector similarity scores for unrelated queries
        # Even if there's some vector similarity (which shouldn't be 1.0 for unrelated queries),
        # our validation should catch it
        cache._vector_similarity_search = AsyncMock(return_value=[(mock_doc, 0.85)])  # Moderate similarity
        
        # Mock realistic similarity scores for unrelated queries
        cache._embedding_similarity = Mock(return_value=0.25)  # Low embedding similarity
        cache._ce_similarity = AsyncMock(return_value=0.15)    # Low cross-encoder similarity
        cache._lexical_similarity = Mock(return_value=0.08)    # Very low lexical similarity
        
        # Mock cache retrieval
        async def mock_get_cache_entry_by_id(cache_id):
            if cache_id == nativepath_entry["cache_id"]:
                return nativepath_entry
            return None
        cache._get_cache_entry_by_id = mock_get_cache_entry_by_id
        
        # Test the problematic second query
        mct_query = "What are the benefits of consuming MCTs daily?"
        result = await cache.get_cached_answer(mct_query)
        
        # Should be cache miss for unrelated queries
        assert result is None, f"Unrelated queries should not get cache hits: '{mct_query}' vs '{nativepath_query}'"

    @pytest.mark.asyncio
    async def test_similar_mct_queries_should_get_cache_hits(self, real_scenario_cache):
        """
        Test that legitimate similar queries get cache hits as specified in requirements.
        """
        cache = real_scenario_cache
        
        # Original cached query
        original_query = "What are the benefits of consuming MCT daily?"
        original_entry = {
            "query": original_query,
            "answer": "MCT (Medium Chain Triglycerides) benefits include improved energy metabolism, enhanced cognitive function...",
            "cache_id": "cache_mct_001",
            "created_at": "2025-01-24T01:00:00Z"
        }
        
        # Mock that similar queries have no exact match
        cache.redis_client.get.return_value = None
        
        # Mock vector similarity search returning the original query
        mock_doc = Mock()
        mock_doc.page_content = original_query
        mock_doc.metadata = {"cache_id": original_entry["cache_id"]}
        
        # Mock cache retrieval
        async def mock_get_cache_entry_by_id(cache_id):
            if cache_id == original_entry["cache_id"]:
                return original_entry
            return None
        cache._get_cache_entry_by_id = mock_get_cache_entry_by_id
        cache._update_cache_access = AsyncMock(return_value=original_entry)
        
        # Test cases: similar queries that should get cache hits
        similar_queries_and_expected_scores = [
            ("Explain the benefits of MCT", 0.94, 0.92, 0.88),                           # High similarity
            ("What are the benefits of MCTs?", 0.96, 0.94, 0.92),                      # Very high similarity  
            ("How consuming MCT can be beneficial to the health?", 0.93, 0.89, 0.85),   # Good similarity
            ("What benefits can I expect if I consume MCT daily?", 0.95, 0.91, 0.89),   # High similarity
        ]
        
        for similar_query, vec_sim, emb_sim, ce_sim in similar_queries_and_expected_scores:
            # Reset mocks for each query
            cache._vector_similarity_search = AsyncMock(return_value=[(mock_doc, vec_sim)])
            cache._embedding_similarity = Mock(return_value=emb_sim)
            cache._ce_similarity = AsyncMock(return_value=ce_sim)
            cache._lexical_similarity = Mock(return_value=0.6)  # Good lexical overlap
            
            result = await cache.get_cached_answer(similar_query)
            
            assert result is not None, f"Similar query should get cache hit: '{similar_query}'"
            assert result["cache_id"] == original_entry["cache_id"], f"Should return correct cached entry for: '{similar_query}'"
            assert result["answer"] == original_entry["answer"], f"Should return correct cached answer for: '{similar_query}'"

    @pytest.mark.asyncio
    async def test_exact_query_cache_hit_bypasses_similarity_checks(self, real_scenario_cache):
        """Test that exact query matches bypass similarity checks entirely."""
        cache = real_scenario_cache
        
        exact_query = "What is artificial intelligence?"
        exact_entry = {
            "query": exact_query,
            "answer": "Artificial intelligence (AI) is the simulation of human intelligence...",
            "cache_id": "cache_ai_001",
            "created_at": "2025-01-24T01:00:00Z"
        }
        
        # Mock exact match found in Redis
        cache.redis_client.get.return_value = exact_entry["cache_id"]
        
        # Mock cache entry retrieval
        async def mock_get_cache_entry_by_id(cache_id):
            if cache_id == exact_entry["cache_id"]:
                return exact_entry
            return None
        cache._get_cache_entry_by_id = mock_get_cache_entry_by_id
        cache._update_cache_access = AsyncMock(return_value=exact_entry)
        
        result = await cache.get_cached_answer(exact_query)
        
        assert result is not None
        assert result["query"] == exact_query
        assert result["answer"] == exact_entry["answer"]
        
        # Verify that vector similarity search was NOT called for exact matches
        cache._vector_similarity_search = AsyncMock()
        result = await cache.get_cached_answer(exact_query)
        cache._vector_similarity_search.assert_not_called()

    @pytest.mark.asyncio
    async def test_edge_case_suspiciously_high_similarity_rejected(self, real_scenario_cache):
        """Test edge case where vector similarity is suspiciously high but other metrics indicate false positive."""
        cache = real_scenario_cache
        
        # Mock no exact match
        cache.redis_client.get.return_value = None
        
        # Mock cached query
        mock_doc = Mock()
        mock_doc.page_content = "What is machine learning?"
        mock_doc.metadata = {"cache_id": "cache_ml_001"}
        
        # Mock suspiciously high vector similarity (could be due to a bug in vector search)
        cache._vector_similarity_search = AsyncMock(return_value=[(mock_doc, 0.995)])  # Nearly perfect
        
        # But other similarity measures indicate these are different queries
        cache._embedding_similarity = Mock(return_value=0.35)  # Low embedding similarity
        cache._lexical_similarity = Mock(return_value=0.02)    # Very low lexical similarity
        
        # Query that should NOT match
        different_query = "What are the health benefits of chocolate?"
        result = await cache.get_cached_answer(different_query)
        
        # Should be rejected due to suspicious similarity pattern
        assert result is None, "Suspiciously high vector similarity with low semantic overlap should be rejected"

    @pytest.mark.asyncio
    async def test_borderline_similarity_scores_with_multiple_validation(self, real_scenario_cache):
        """Test borderline similarity scores that require multiple validation layers."""
        cache = real_scenario_cache
        
        # Mock no exact match
        cache.redis_client.get.return_value = None
        
        # Mock cached query
        mock_doc = Mock()  
        mock_doc.page_content = "How to improve productivity at work?"
        mock_doc.metadata = {"cache_id": "cache_productivity_001"}
        
        cached_entry = {
            "query": "How to improve productivity at work?",
            "answer": "To improve productivity at work, consider...",
            "cache_id": "cache_productivity_001"
        }
        
        # Mock cache entry retrieval
        async def mock_get_cache_entry_by_id(cache_id):
            if cache_id == cached_entry["cache_id"]:
                return cached_entry
            return None
        cache._get_cache_entry_by_id = mock_get_cache_entry_by_id
        cache._update_cache_access = AsyncMock(return_value=cached_entry)
        
        # Test borderline case that should PASS (Rule 2: vector_min + cross-encoder + embedding)
        cache._vector_similarity_search = AsyncMock(return_value=[(mock_doc, 0.87)])  # Above vec_min (0.85)
        cache._embedding_similarity = Mock(return_value=0.90)     # Above emb_accept (0.88)
        cache._ce_similarity = AsyncMock(return_value=0.65)       # Above ce_accept (0.60)
        cache._lexical_similarity = Mock(return_value=0.25)       # Above lex_min (0.15)
        
        similar_query = "What are ways to boost workplace productivity?"
        result = await cache.get_cached_answer(similar_query)
        
        assert result is not None, "Queries meeting multi-metric thresholds should get cache hits"
        
        # Test borderline case that should FAIL (doesn't meet combined thresholds)
        cache._vector_similarity_search = AsyncMock(return_value=[(mock_doc, 0.87)])  # Above vec_min but...
        cache._embedding_similarity = Mock(return_value=0.85)     # Below emb_accept (0.88)
        cache._ce_similarity = AsyncMock(return_value=0.55)       # Below ce_accept (0.60)
        cache._lexical_similarity = Mock(return_value=0.10)       # Below lex_min (0.15)
        
        different_query = "How to cook pasta perfectly?"
        result = await cache.get_cached_answer(different_query)
        
        assert result is None, "Queries not meeting multi-metric thresholds should not get cache hits"