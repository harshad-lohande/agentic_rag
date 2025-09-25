#!/usr/bin/env python3
"""
Test script to validate the cache retrieval endpoint fix.
"""

import asyncio
import json
from agentic_rag.app.semantic_cache_tester import SemanticCacheTester

async def test_retrieval_fix():
    """Test the cross-encoder similarity search and cache retrieval."""
    
    print("🧪 Testing semantic cache retrieval fix...")
    
    # Initialize the tester
    tester = SemanticCacheTester()
    
    try:
        # Clear cache first
        print("\n1. Clearing cache...")
        await tester.clear_cache_for_testing()
        
        # Create a test cache entry
        print("\n2. Creating test cache entry...")
        result = await tester.create_cache_entry(
            query="Explain the benefits of MCTs",
            answer="MCT benefits include enhanced metabolic function, improved cognitive vitality, and better heart health.",
            metadata={"category": "health"}
        )
        print(f"   ✅ Created cache entry: {result['cache_id']}")
        
        # Test similar query retrieval
        print("\n3. Testing cache retrieval for similar query...")
        similar_query = "What are the benefits of taking MCTs?"
        retrieval_result = await tester.test_cache_retrieval(similar_query)
        
        print(f"   Query: {similar_query}")
        print(f"   Cache hit: {retrieval_result.cache_hit}")
        print(f"   Similarity score: {retrieval_result.similarity_score}")
        print(f"   Execution time: {retrieval_result.execution_time_ms:.2f}ms")
        
        if retrieval_result.cache_hit:
            print("   ✅ Cache hit detected successfully!")
            cached_entry = retrieval_result.cached_entry
            print(f"   Original query: {cached_entry.get('query', 'N/A')}")
        else:
            print("   ❌ No cache hit - testing cross-encoder search...")
            
            # Test cross-encoder similarity directly
            print("\n4. Testing cross-encoder similarity...")
            similarity_result = await tester.test_query_similarity(
                cached_query="Explain the benefits of MCTs",
                test_query=similar_query
            )
            
            print(f"   Cross-encoder similarity: {similarity_result.cross_encoder_similarity}")
            print(f"   Lexical similarity: {similarity_result.lexical_similarity}")
            print(f"   Cache hit prediction: {similarity_result.cache_hit_prediction}")
            print(f"   Rule triggered: {similarity_result.rule_triggered}")
            
        # Test unrelated query
        print("\n5. Testing unrelated query (should be cache miss)...")
        unrelated_query = "What is the weather today?"
        unrelated_result = await tester.test_cache_retrieval(unrelated_query)
        
        print(f"   Query: {unrelated_query}")
        print(f"   Cache hit: {unrelated_result.cache_hit}")
        
        if not unrelated_result.cache_hit:
            print("   ✅ Correctly rejected unrelated query!")
        else:
            print("   ❌ False positive detected!")
            
        # Get cache stats
        print("\n6. Cache statistics:")
        stats = await tester.get_cache_statistics()
        print(f"   Total entries: {stats.get('total_entries', 0)}")
        print(f"   Framework version: {stats.get('testing_framework', {}).get('testing_framework_version', 'N/A')}")
        
        print("\n🎉 Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_retrieval_fix())