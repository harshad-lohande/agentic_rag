#!/usr/bin/env python3
"""
Debug script to understand what scores Weaviate actually returns.
"""

import asyncio
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agentic_rag.app.semantic_cache import semantic_cache


async def debug_weaviate_scores():
    """Debug what scores Weaviate actually returns."""
    
    print("🔍 DEBUGGING WEAVIATE SIMILARITY SCORES")
    print("=" * 60)
    
    try:
        # Initialize the cache
        if not semantic_cache._initialized:
            await semantic_cache.initialize()
        
        # Clear cache for clean testing
        await semantic_cache.clear_cache()
        print("✅ Cache cleared and initialized")
        
        # Create a test entry
        test_query = "How are MCTs different from LCTs?"
        test_answer = "MCT benefits include improved energy metabolism..."
        
        result = await semantic_cache.store_answer(test_query, test_answer, {"test": True})
        print(f"✅ Test entry created: {result}")
        
        # Now test various queries and see what raw scores Weaviate returns
        test_queries = [
            "How are MCTs different from LCTs?",  # Exact match
            "How MCTs differ from LCTs",  # Very similar
            "What is the difference between MCTs and LCTs?",  # Similar
            "How are MCTs different from LCT?",  # Very close
            "What are the benefits of chocolate?",  # Unrelated
        ]
        
        print("\n📊 RAW WEAVIATE SCORES:")
        print("-" * 60)
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            
            # Get raw results from vector store
            try:
                raw_results = await asyncio.to_thread(
                    semantic_cache.cache_vector_store.similarity_search_with_score,
                    query, k=1
                )
                
                if raw_results:
                    doc, raw_score = raw_results[0]
                    normalized_score = semantic_cache._normalize_similarity_score(raw_score)
                    
                    print(f"  Raw score from Weaviate: {raw_score}")
                    print(f"  Normalized score: {normalized_score}")
                    print(f"  Cached query: '{doc.page_content}'")
                    print(f"  Match: {doc.page_content.strip() == query.strip()}")
                else:
                    print("  No results returned")
                    
            except Exception as e:
                print(f"  Error: {e}")
        
        # Test the semantic cache search wrapper
        print("\n🔧 SEMANTIC CACHE WRAPPER RESULTS:")
        print("-" * 60)
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            
            try:
                results = await semantic_cache._vector_similarity_search(query, k=1)
                
                if results:
                    doc, score = results[0]
                    print(f"  Wrapper score: {score}")
                    print(f"  Cached query: '{doc.page_content}'")
                else:
                    print("  No results from wrapper")
                    
            except Exception as e:
                print(f"  Wrapper error: {e}")
        
        print("\n🧪 TESTING DIFFERENT SCORE MODES:")
        print("-" * 60)
        
        # Test different normalization approaches
        raw_score = 1.0  # What we're seeing from Weaviate
        
        print(f"Raw score: {raw_score}")
        
        # Current "distance" mode normalization
        distance_norm = 1.0 - (raw_score / 2.0)
        print(f"Current distance normalization (1 - raw/2): {distance_norm}")
        
        # Alternative distance normalization assuming range [0,1]
        alt_distance_norm = 1.0 - raw_score
        print(f"Alt distance normalization (1 - raw): {alt_distance_norm}")
        
        # Treat as similarity directly
        similarity_norm = raw_score
        print(f"Treat as similarity: {similarity_norm}")
        
        # Test if it's actually cosine similarity (range [-1,1] -> [0,1])
        cosine_norm = (raw_score + 1.0) / 2.0
        print(f"Cosine similarity normalization ((raw+1)/2): {cosine_norm}")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Make sure environment variables are set:")
    print("HUGGINGFACEHUB_API_TOKEN, OPENAI_API_KEY, GOOGLE_API_KEY, LANGCHAIN_API_KEY")
    print()
    
    asyncio.run(debug_weaviate_scores())