#!/usr/bin/env python3
"""
Test script to validate the semantic cache fix for similar queries.
"""

import asyncio
from unittest.mock import Mock, AsyncMock

# Mock the missing dependencies
import sys
from unittest.mock import MagicMock

# Mock langchain modules
sys.modules['langchain_core'] = MagicMock()
sys.modules['langchain_core.documents'] = MagicMock()
sys.modules['langchain_huggingface'] = MagicMock()
sys.modules['langchain_weaviate'] = MagicMock()
sys.modules['langchain_weaviate.vectorstores'] = MagicMock()
sys.modules['weaviate'] = MagicMock()
sys.modules['weaviate.collections'] = MagicMock()
sys.modules['weaviate.collections.classes'] = MagicMock()
sys.modules['weaviate.collections.classes.filters'] = MagicMock()
sys.modules['redis'] = MagicMock()
sys.modules['redis.asyncio'] = MagicMock()

from agentic_rag.app.semantic_cache import SemanticCache


def create_test_cache():
    """Create a semantic cache instance for testing."""
    cache = SemanticCache()
    
    # Mock dependencies
    cache.redis_client = AsyncMock()
    cache.weaviate_client = Mock()
    cache.cache_vector_store = Mock()
    cache.embedding_model = Mock()
    cache._initialized = True
    
    return cache


async def test_similar_queries_get_cache_hits():
    """Test that similar queries get cache hits with the updated thresholds."""
    print("🔍 Testing similar queries for cache hits...")
    
    cache = create_test_cache()
    
    # Original cached query
    original_query = "What are the benefits of MCTs"
    original_entry = {
        "query": original_query,
        "answer": "MCTs provide energy, cognitive benefits, and metabolic support...",
        "cache_id": "cache_mct_001"
    }
    
    # Mock no exact match
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
    
    # Test similar queries from the user's examples
    test_cases = [
        {
            "query": "What are the benefits of taking MCT",
            "vector_sim": 0.88,  # Above vec_min (0.85)
            "emb_sim": 0.85,     # Close to emb_min (0.88) 
            "ce_sim": 0.65,      # Above ce_min (0.60)
            "lex_sim": 0.65,     # Good lexical overlap
            "should_hit": True
        },
        {
            "query": "What benefits can I expect if I consume MCT daily?",
            "vector_sim": 0.87,  # Above vec_min (0.85)
            "emb_sim": 0.84,     # Slightly below emb_min but should trigger Rule 4
            "ce_sim": 0.70,      # Well above ce_min (0.60) 
            "lex_sim": 0.55,     # Moderate lexical overlap
            "should_hit": True
        },
        {
            "query": "What are the benefits of taking MCTs?",
            "vector_sim": 0.90,  # Above vec_min + 0.05 (0.90)
            "emb_sim": 0.85,     # Close to emb_min
            "ce_sim": 0.68,      # Above ce_min
            "lex_sim": 0.75,     # High lexical overlap
            "should_hit": True
        },
        {
            "query": "Tell me about chocolate benefits",  # Unrelated query
            "vector_sim": 0.45,  # Low vector similarity
            "emb_sim": 0.30,     # Low embedding similarity
            "ce_sim": 0.25,      # Low cross-encoder similarity
            "lex_sim": 0.10,     # Low lexical overlap
            "should_hit": False
        }
    ]
    
    results = []
    for test_case in test_cases:
        print(f"\n📝 Testing: '{test_case['query']}'")
        print(f"   Expected: {'HIT' if test_case['should_hit'] else 'MISS'}")
        
        # Mock similarity scores
        cache._vector_similarity_search = AsyncMock(return_value=[(mock_doc, test_case['vector_sim'])])
        cache._embedding_similarity = Mock(return_value=test_case['emb_sim'])
        cache._ce_similarity = AsyncMock(return_value=test_case['ce_sim'])
        cache._lexical_similarity = Mock(return_value=test_case['lex_sim'])
        
        result = await cache.get_cached_answer(test_case['query'])
        actual_hit = result is not None
        
        print(f"   Actual:   {'HIT' if actual_hit else 'MISS'}")
        print(f"   Scores:   vec={test_case['vector_sim']:.2f}, emb={test_case['emb_sim']:.2f}, ce={test_case['ce_sim']:.2f}, lex={test_case['lex_sim']:.2f}")
        
        success = actual_hit == test_case['should_hit']
        print(f"   Result:   {'✅ PASS' if success else '❌ FAIL'}")
        
        results.append({
            'query': test_case['query'],
            'expected': test_case['should_hit'],
            'actual': actual_hit,
            'success': success
        })
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['success'])
    
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        expected = "HIT" if result['expected'] else "MISS"
        actual = "HIT" if result['actual'] else "MISS"
        print(f"{status} {result['query'][:40]:<40} (expected {expected}, got {actual})")
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Similar queries should now get cache hits.")
    else:
        print("⚠️  Some tests failed. The fix may need further adjustments.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    asyncio.run(test_similar_queries_get_cache_hits())