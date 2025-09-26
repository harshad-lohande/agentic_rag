#!/usr/bin/env python3
"""
Semantic Cache Testing Framework Demo

This script demonstrates how to use the semantic cache testing framework
to test and experiment with cache behavior without running the full workflow.

Usage:
    python semantic_cache_test_demo.py

Features demonstrated:
- Direct cache entry creation
- Similarity testing between queries
- Cache retrieval testing
- Bulk similarity testing
- Performance validation
"""

import asyncio
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from semantic_cache_tester import semantic_cache_tester


async def demo_cache_testing():
    """Demonstrate the semantic cache testing framework."""
    
    print("🧪 SEMANTIC CACHE TESTING FRAMEWORK DEMO")
    print("=" * 60)
    print()
    
    # Step 1: Get initial cache statistics
    print("📊 Step 1: Getting initial cache statistics...")
    stats = await semantic_cache_tester.get_cache_statistics()
    print(f"Cache initialized: {stats.get('testing_framework', {}).get('cache_initialized', False)}")
    print(f"Available similarity methods: {len(stats.get('testing_framework', {}).get('available_similarity_methods', []))}")
    print()
    
    # Step 2: Clear cache for clean testing
    print("🧹 Step 2: Clearing cache for clean testing...")
    clear_result = await semantic_cache_tester.clear_cache_for_testing()
    print(f"Cache cleared: {clear_result['success']}")
    print()
    
    # Step 3: Create test cache entries
    print("💾 Step 3: Creating test cache entries...")
    
    test_entries = [
        {
            "query": "What are the benefits of MCTs",
            "answer": "MCT (Medium Chain Triglycerides) benefits include improved energy metabolism, enhanced cognitive function, better fat burning, and support for heart health. They are easily digestible and provide quick energy for both body and brain.",
            "metadata": {"category": "health", "topic": "MCT"}
        },
        {
            "query": "What is artificial intelligence?",
            "answer": "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans. It encompasses machine learning, natural language processing, computer vision, and robotics.",
            "metadata": {"category": "technology", "topic": "AI"}
        },
        {
            "query": "How to cook pasta perfectly?",
            "answer": "To cook pasta perfectly: 1) Use plenty of salted boiling water, 2) Add pasta and stir immediately, 3) Cook according to package directions minus 1 minute, 4) Test for al dente texture, 5) Reserve pasta water before draining, 6) Combine with sauce immediately.",
            "metadata": {"category": "cooking", "topic": "pasta"}
        }
    ]
    
    created_entries = []
    for entry in test_entries:
        result = await semantic_cache_tester.create_cache_entry(
            query=entry["query"],
            answer=entry["answer"], 
            metadata=entry["metadata"]
        )
        created_entries.append(result)
        print(f"  ✅ Created: '{entry['query'][:40]}...' (Success: {result['success']})")
    
    print()
    
    # Step 4: Test similarity between queries
    print("🔍 Step 4: Testing query similarities...")
    
    similarity_tests = [
        {
            "name": "Similar MCT queries",
            "cached": "What are the benefits of MCTs",
            "tests": [
                "What are the benefits of taking MCT",
                "What benefits can I expect if I consume MCT daily?",
                "What are the benefits of taking MCTs?",
                "Explain the benefits of MCT"
            ]
        },
        {
            "name": "AI-related queries",
            "cached": "What is artificial intelligence?",
            "tests": [
                "What is AI?",
                "Explain artificial intelligence",
                "What are the benefits of chocolate?"  # Unrelated query
            ]
        }
    ]
    
    for test_group in similarity_tests:
        print(f"\n  📋 {test_group['name']}:")
        print(f"     Cached query: '{test_group['cached']}'")
        
        for test_query in test_group['tests']:
            result = await semantic_cache_tester.test_query_similarity(
                cached_query=test_group['cached'],
                test_query=test_query
            )
            
            hit_status = "🟢 HIT" if result.cache_hit_prediction else "🔴 MISS"
            print(f"     {hit_status} '{test_query}'")
            print(f"          Vector: {result.vector_similarity:.3f if result.vector_similarity else 'N/A'}")
            print(f"          Embedding: {result.embedding_similarity:.3f if result.embedding_similarity else 'N/A'}")
            print(f"          Cross-encoder: {result.cross_encoder_similarity:.3f if result.cross_encoder_similarity else 'N/A'}")
            print(f"          Lexical: {result.lexical_similarity:.3f}")
            print(f"          Rule: {result.rule_triggered}")
            print(f"          Time: {result.execution_time_ms:.1f}ms")
    
    print()
    
    # Step 5: Test actual cache retrieval
    print("🎯 Step 5: Testing actual cache retrieval...")
    
    retrieval_tests = [
        "What are the benefits of MCTs",  # Exact match
        "What are the benefits of taking MCT",  # Similar query
        "What benefits can I expect if I consume MCT daily?",  # Similar query
        "What are the health benefits of chocolate?",  # Unrelated query
    ]
    
    for query in retrieval_tests:
        result = await semantic_cache_tester.test_cache_retrieval(query)
        
        hit_status = "🟢 HIT" if result.cache_hit else "🔴 MISS"
        similarity = f" (sim: {result.similarity_score:.3f})" if result.similarity_score else ""
        
        print(f"  {hit_status} '{query}'{similarity}")
        print(f"       Time: {result.execution_time_ms:.1f}ms")
        
        if result.cache_hit and result.cached_entry:
            cached_query = result.cached_entry.get('query', 'Unknown')
            print(f"       Matched: '{cached_query}'")
    
    print()
    
    # Step 6: Bulk similarity testing
    print("📈 Step 6: Bulk similarity testing...")
    
    cached_query = "What are the benefits of MCTs"
    test_queries = [
        "What are the benefits of taking MCT",
        "What benefits can I expect if I consume MCT daily?",
        "What are the benefits of taking MCTs?",
        "Explain the benefits of MCT",
        "How does MCT help with weight loss?",
        "What are the side effects of MCT?",  # Related but different focus
        "What is the best chocolate brand?",  # Completely unrelated
    ]
    
    bulk_results = await semantic_cache_tester.bulk_similarity_test(cached_query, test_queries)
    
    print(f"  Testing {len(test_queries)} queries against: '{cached_query}'")
    print()
    
    hit_count = 0
    for result in bulk_results:
        hit_status = "🟢 HIT" if result.cache_hit_prediction else "🔴 MISS"
        if result.cache_hit_prediction:
            hit_count += 1
        
        print(f"  {hit_status} '{result.test_query}'")
        print(f"       Scores: V:{result.vector_similarity or 'N/A'} | "
              f"E:{result.embedding_similarity:.2f if result.embedding_similarity else 'N/A'} | "
              f"C:{result.cross_encoder_similarity:.2f if result.cross_encoder_similarity else 'N/A'} | "
              f"L:{result.lexical_similarity:.2f}")
        print(f"       Rule: {result.rule_triggered}")
    
    print()
    print(f"  📊 Summary: {hit_count}/{len(test_queries)} queries would get cache hits")
    
    # Step 7: Final cache statistics
    print()
    print("📊 Step 7: Final cache statistics...")
    final_stats = await semantic_cache_tester.get_cache_statistics()
    
    if 'total_entries' in final_stats:
        print(f"  Total cache entries: {final_stats['total_entries']}")
    if 'avg_access_count' in final_stats:
        print(f"  Average access count: {final_stats['avg_access_count']:.2f}")
    
    print()
    print("✅ Demo completed successfully!")
    print()
    print("🔧 API Endpoints Available:")
    print("  POST /cache/test/create-entry     - Create cache entries")
    print("  POST /cache/test/similarity       - Test query similarity")
    print("  POST /cache/test/retrieval        - Test cache retrieval")
    print("  POST /cache/test/bulk-similarity  - Bulk similarity testing")
    print("  GET  /cache/test/stats           - Get cache statistics")
    print("  POST /cache/test/clear           - Clear cache for testing")


async def demo_api_usage():
    """Demonstrate how to use the testing framework via API calls."""
    
    print("\n🌐 API USAGE EXAMPLES")
    print("=" * 60)
    print()
    
    print("To use the testing framework via API, start the server and use these endpoints:")
    print()
    
    print("1. Create a cache entry:")
    print("""
curl -X POST 'http://localhost:8000/cache/test/create-entry' \\
  -H 'Content-Type: application/json' \\
  -d '{
    "query": "What are the benefits of MCTs",
    "answer": "MCT benefits include...",
    "metadata": {"category": "health"}
  }'
""")
    
    print("2. Test similarity between queries:")
    print("""
curl -X POST 'http://localhost:8000/cache/test/similarity' \\
  -H 'Content-Type: application/json' \\
  -d '{
    "cached_query": "What are the benefits of MCTs",
    "test_query": "What are the benefits of taking MCT"
  }'
""")
    
    print("3. Test cache retrieval:")
    print("""
curl -X POST 'http://localhost:8000/cache/test/retrieval?query=What%20are%20the%20benefits%20of%20taking%20MCT'
""")
    
    print("4. Get cache statistics:")
    print("""
curl -X GET 'http://localhost:8000/cache/test/stats'
""")
    
    print("5. Clear cache for testing:")
    print("""
curl -X POST 'http://localhost:8000/cache/test/clear'
""")


def main():
    """Run the demo."""
    print("Starting Semantic Cache Testing Framework Demo...")
    print("Make sure you have set the required environment variables:")
    print("  HUGGINGFACEHUB_API_TOKEN, OPENAI_API_KEY, GOOGLE_API_KEY, LANGCHAIN_API_KEY")
    print()
    
    try:
        # Run the async demo
        asyncio.run(demo_cache_testing())
        
        # Show API usage examples
        asyncio.run(demo_api_usage())
        
    except KeyboardInterrupt:
        print("\n❌ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()