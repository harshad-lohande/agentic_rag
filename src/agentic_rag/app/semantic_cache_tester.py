"""
Semantic Cache Testing Framework

A comprehensive testing and simulation framework for the semantic cache implementation
that allows users to test and experiment with caching functionality without executing
the entire graph workflow or making LLM API calls.

Features:
- Direct cache entry creation and storage
- Similarity testing with multiple approaches (vector, cross-encoder, embedding, lexical)
- Cache retrieval and comparison
- Performance reporting and validation
- Full semantic cache functionality without LLM overhead
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel

from agentic_rag.app.semantic_cache import semantic_cache
from agentic_rag.logging_config import logger


# Pydantic models for API requests/responses
class CacheEntryRequest(BaseModel):
    """Request model for creating a cache entry."""
    query: str
    answer: str
    metadata: Optional[Dict[str, Any]] = None


class SimilarityTestRequest(BaseModel):
    """Request model for testing similarity between queries."""
    cached_query: str
    test_query: str


class SimilarityTestResponse(BaseModel):
    """Response model for similarity test results."""
    cached_query: str
    test_query: str
    vector_similarity: Optional[float]
    embedding_similarity: Optional[float]
    cross_encoder_similarity: Optional[float]
    lexical_similarity: float
    cache_hit_prediction: bool
    rule_triggered: Optional[str]
    execution_time_ms: float


class CacheTestResponse(BaseModel):
    """Response model for cache retrieval test."""
    query: str
    cache_hit: bool
    cached_entry: Optional[Dict[str, Any]]
    similarity_score: Optional[float]
    execution_time_ms: float


class SemanticCacheTester:
    """
    Testing framework for semantic cache functionality.
    
    Provides comprehensive testing capabilities for cache operations,
    similarity calculations, and performance validation without requiring
    full workflow execution or LLM API calls.
    """
    
    def __init__(self):
        self.cache = semantic_cache
    
    async def create_cache_entry(self, query: str, answer: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Directly create and store a cache entry, mimicking end-of-workflow caching.
        
        Args:
            query: The query to cache
            answer: The answer to cache
            metadata: Optional metadata to store with the entry
            
        Returns:
            Dictionary with operation result and details
        """
        start_time = time.time()
        
        try:
            # Ensure cache is initialized
            if not self.cache._initialized:
                await self.cache.initialize()
            
            # Create cache entry using the same method as workflow
            success = await self.cache.store_answer(query, answer, metadata)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Get the cache ID for the stored entry
            query_hash = self.cache._generate_query_hash(query)
            exact_match_key = f"exact_match:{query_hash}"
            
            cache_id = None
            try:
                if hasattr(self.cache.redis_client, 'get'):
                    cache_id = await self.cache.redis_client.get(exact_match_key)
                else:
                    cache_id = await asyncio.to_thread(self.cache.redis_client.get, exact_match_key)
            except Exception as e:
                logger.debug(f"Could not retrieve cache ID: {e}")
            
            return {
                "success": success,
                "query": query,
                "answer_length": len(answer),
                "cache_id": cache_id.decode() if cache_id and hasattr(cache_id, 'decode') else str(cache_id),
                "query_hash": query_hash,
                "metadata": metadata or {},
                "execution_time_ms": round(execution_time, 2),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Error creating cache entry: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "execution_time_ms": round(execution_time, 2),
                "timestamp": datetime.now().isoformat()
            }
    
    async def test_query_similarity(self, cached_query: str, test_query: str) -> SimilarityTestResponse:
        """
        Test similarity between two queries using all available similarity methods.
        
        Args:
            cached_query: The query that would be in cache
            test_query: The query to test against the cached query
            
        Returns:
            SimilarityTestResponse with detailed similarity metrics
        """
        start_time = time.time()
        
        try:
            # Ensure cache is initialized
            if not self.cache._initialized:
                await self.cache.initialize()
            
            # 1. Vector similarity (requires the cached query to be in vector store)
            vector_similarity = None
            try:
                # Perform vector similarity search
                similar_docs = await self.cache._vector_similarity_search(test_query, k=10)
                for doc, score in similar_docs:
                    if doc.page_content and doc.page_content.strip() == cached_query.strip():
                        vector_similarity = score
                        break
                
                # If not found in existing cache, we can't get vector similarity
                if vector_similarity is None:
                    logger.debug(f"Cached query '{cached_query}' not found in vector store for similarity test")
            except Exception as e:
                logger.debug(f"Vector similarity test failed: {e}")
            
            # 2. Embedding similarity
            embedding_similarity = None
            try:
                embedding_similarity = await asyncio.to_thread(
                    self.cache._embedding_similarity, test_query, cached_query
                )
            except Exception as e:
                logger.debug(f"Embedding similarity test failed: {e}")
            
            # 3. Cross-encoder similarity
            cross_encoder_similarity = None
            try:
                cross_encoder_similarity = await self.cache._ce_similarity(test_query, cached_query)
            except Exception as e:
                logger.debug(f"Cross-encoder similarity test failed: {e}")
            
            # 4. Lexical similarity (always available)
            lexical_similarity = self.cache._lexical_similarity(test_query, cached_query)
            
            # 5. Predict cache hit using the same rules as the actual cache
            cache_hit_prediction, rule_triggered = self._predict_cache_hit(
                vector_similarity, embedding_similarity, cross_encoder_similarity, lexical_similarity
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            return SimilarityTestResponse(
                cached_query=cached_query,
                test_query=test_query,
                vector_similarity=vector_similarity,
                embedding_similarity=embedding_similarity,
                cross_encoder_similarity=cross_encoder_similarity,
                lexical_similarity=lexical_similarity,
                cache_hit_prediction=cache_hit_prediction,
                rule_triggered=rule_triggered,
                execution_time_ms=round(execution_time, 2)
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Error testing query similarity: {e}", exc_info=True)
            
            return SimilarityTestResponse(
                cached_query=cached_query,
                test_query=test_query,
                vector_similarity=None,
                embedding_similarity=None,
                cross_encoder_similarity=None,
                lexical_similarity=0.0,
                cache_hit_prediction=False,
                rule_triggered=f"Error: {str(e)}",
                execution_time_ms=round(execution_time, 2)
            )
    
    def _predict_cache_hit(self, vector_sim: Optional[float], emb_sim: Optional[float],
                          ce_sim: Optional[float], lex_sim: float) -> Tuple[bool, str]:
        """
        Predict whether a query would get a cache hit using the same rules as the actual cache.
        
        Args:
            vector_sim: Vector similarity score (0-1, higher is better)
            emb_sim: Embedding similarity score (0-1, higher is better)
            ce_sim: Cross-encoder similarity score (0-1, higher is better)
            lex_sim: Lexical similarity score (0-1, higher is better)
            
        Returns:
            Tuple of (cache_hit_prediction, rule_triggered)
        """
        # Import settings to get current thresholds
        from agentic_rag.config import settings
        
        vec_accept = float(getattr(settings, "SEMANTIC_CACHE_VECTOR_ACCEPT", 0.92))
        vec_min = float(getattr(settings, "SEMANTIC_CACHE_VECTOR_MIN", 0.85))
        emb_min = float(getattr(settings, "SEMANTIC_CACHE_EMB_ACCEPT", 0.88))
        ce_min = float(getattr(settings, "SEMANTIC_CACHE_CE_ACCEPT", 0.60))
        lex_min = float(getattr(settings, "SEMANTIC_CACHE_LEXICAL_MIN", 0.15))
        
        if vector_sim is None:
            return False, "No vector similarity available"
        
        # Rule 1: very high vector similarity alone, with false positive detection
        if vector_sim >= vec_accept:
            # Additional validation for perfect or near-perfect scores
            if vector_sim >= 0.99:
                if emb_sim is not None and ((emb_sim < 0.7 and lex_sim < 0.1) or emb_sim < 0.4):
                    return False, f"Rule 1 rejected: Suspicious high vector similarity ({vector_sim:.3f}) with low semantic support"
            return True, f"Rule 1: High vector similarity ({vector_sim:.3f} >= {vec_accept})"
        
        # For other rules, we need embedding and cross-encoder scores
        if emb_sim is None or ce_sim is None:
            return False, "Insufficient similarity scores for multi-metric validation"
        
        # Rule 2: require BOTH cross-encoder and embedding support with vector above minimum
        if vector_sim >= vec_min and ce_sim >= ce_min and emb_sim >= emb_min:
            return True, f"Rule 2: Multi-metric validation (vec={vector_sim:.3f}, ce={ce_sim:.3f}, emb={emb_sim:.3f})"
        
        # Rule 3: tiny lexical support helps borderline cases
        if vector_sim >= vec_min and ce_sim >= ce_min and emb_sim >= (emb_min - 0.03) and lex_sim >= lex_min:
            return True, f"Rule 3: Lexical support (vec={vector_sim:.3f}, ce={ce_sim:.3f}, emb={emb_sim:.3f}, lex={lex_sim:.2f})"
        
        # Rule 4: more lenient rule for similar queries with good vector similarity and some semantic support
        if vector_sim >= (vec_min + 0.02) and (emb_sim >= (emb_min - 0.05) or ce_sim >= (ce_min + 0.05)):
            return True, f"Rule 4: Moderate vector + semantic support (vec={vector_sim:.3f}, ce={ce_sim:.3f}, emb={emb_sim:.3f})"
        
        return False, f"No rules triggered (vec={vector_sim:.3f}, ce={ce_sim:.3f}, emb={emb_sim:.3f}, lex={lex_sim:.2f})"
    
    async def test_cache_retrieval(self, query: str) -> CacheTestResponse:
        """
        Test cache retrieval for a given query using the same logic as the actual workflow.
        
        Args:
            query: The query to test for cache retrieval
            
        Returns:
            CacheTestResponse with retrieval results and performance metrics
        """
        start_time = time.time()
        
        try:
            # Ensure cache is initialized
            if not self.cache._initialized:
                await self.cache.initialize()
            
            # Use the same method as the actual workflow
            cached_result = await self.cache.get_cached_answer(query)
            
            execution_time = (time.time() - start_time) * 1000
            
            cache_hit = cached_result is not None
            similarity_score = cached_result.get("similarity") if cached_result else None
            
            return CacheTestResponse(
                query=query,
                cache_hit=cache_hit,
                cached_entry=cached_result,
                similarity_score=similarity_score,
                execution_time_ms=round(execution_time, 2)
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Error testing cache retrieval: {e}", exc_info=True)
            
            return CacheTestResponse(
                query=query,
                cache_hit=False,
                cached_entry=None,
                similarity_score=None,
                execution_time_ms=round(execution_time, 2)
            )
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics including custom testing metrics.
        
        Returns:
            Dictionary with detailed cache statistics
        """
        try:
            # Get base cache stats
            base_stats = await self.cache.get_cache_stats()
            
            # Add testing-specific information
            testing_info = {
                "testing_framework_version": "1.0.0",
                "available_similarity_methods": [
                    "vector_similarity",
                    "embedding_similarity", 
                    "cross_encoder_similarity",
                    "lexical_similarity"
                ],
                "cache_rules": [
                    "Rule 1: High vector similarity (≥0.92) with false positive detection",
                    "Rule 2: Multi-metric validation (vector + cross-encoder + embedding)",
                    "Rule 3: Lexical support for borderline cases",
                    "Rule 4: Lenient rule for similar queries (vector ≥0.87 + semantic support)"
                ],
                "cache_initialized": self.cache._initialized
            }
            
            # Combine stats
            return {
                **base_stats,
                "testing_framework": testing_info
            }
            
        except Exception as e:
            logger.error(f"Error getting cache statistics: {e}", exc_info=True)
            return {
                "error": str(e),
                "testing_framework": {
                    "testing_framework_version": "1.0.0",
                    "cache_initialized": False
                }
            }
    
    async def bulk_similarity_test(self, cached_query: str, test_queries: List[str]) -> List[SimilarityTestResponse]:
        """
        Test similarity between one cached query and multiple test queries.
        
        Args:
            cached_query: The query that would be in cache
            test_queries: List of queries to test against the cached query
            
        Returns:
            List of SimilarityTestResponse objects
        """
        results = []
        
        for test_query in test_queries:
            result = await self.test_query_similarity(cached_query, test_query)
            results.append(result)
        
        return results
    
    async def clear_cache_for_testing(self) -> Dict[str, Any]:
        """
        Clear the cache for testing purposes.
        
        Returns:
            Dictionary with operation result
        """
        try:
            success = await self.cache.clear_cache()
            return {
                "success": success,
                "message": "Cache cleared for testing" if success else "Failed to clear cache",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error clearing cache: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Global instance for use in API endpoints
semantic_cache_tester = SemanticCacheTester()