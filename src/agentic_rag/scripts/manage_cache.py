# src/agentic_rag/scripts/manage_cache.py

import argparse
import asyncio
import json
import time

from agentic_rag.app.semantic_cache import semantic_cache
from agentic_rag.logging_config import logger


async def show_cache_stats():
    """Display cache statistics."""
    print("\n" + "="*50)
    print("SEMANTIC CACHE STATISTICS")
    print("="*50)
    
    stats = semantic_cache.get_cache_stats()
    
    if not stats.get("enabled", False):
        print("❌ Semantic cache is disabled")
        return
    
    if "error" in stats:
        print(f"❌ Error getting cache stats: {stats['error']}")
        return
    
    print("✅ Cache Status: Enabled")
    print(f"📊 Total Entries: {stats.get('total_entries', 0)}")
    print(f"📏 Max Size: {stats.get('max_size', 0)}")
    print(f"⏰ TTL: {stats.get('ttl_seconds', 0)} seconds")
    print(f"🎯 Similarity Threshold: {stats.get('similarity_threshold', 0)}")
    print(f"📈 Avg Access Count: {stats.get('avg_access_count', 0)}")
    
    # Calculate fill percentage
    total = stats.get('total_entries', 0)
    max_size = stats.get('max_size', 1)
    fill_percentage = (total / max_size) * 100
    print(f"📊 Cache Fill: {fill_percentage:.1f}%")
    
    print("="*50)


async def clear_cache():
    """Clear all cache entries."""
    print("\n🗑️ Clearing semantic cache...")
    
    if not semantic_cache._is_cache_enabled():
        print("❌ Semantic cache is not enabled or not properly initialized")
        return False
    
    try:
        success = semantic_cache.clear_cache()
        if success:
            print("✅ Cache cleared successfully")
            return True
        else:
            print("❌ Failed to clear cache")
            return False
    except Exception as e:
        print(f"❌ Error clearing cache: {e}")
        return False


async def test_cache_functionality():
    """Test cache functionality with sample data."""
    print("\n🧪 Testing cache functionality...")
    
    if not semantic_cache._is_cache_enabled():
        print("❌ Semantic cache is not enabled or not properly initialized")
        return
    
    # Test queries
    test_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Explain neural networks"
    ]
    
    test_answers = [
        "Artificial intelligence is a field of computer science that focuses on creating intelligent machines.",
        "Machine learning is a subset of AI that enables computers to learn without being explicitly programmed.",
        "Neural networks are computing systems inspired by biological neural networks."
    ]
    
    # Test storing answers
    print("📝 Storing test answers in cache...")
    for query, answer in zip(test_queries, test_answers):
        try:
            success = await semantic_cache.store_answer(
                query, answer, {"test": True, "tokens": 100}
            )
            if success:
                print(f"  ✅ Stored: {query[:30]}...")
            else:
                print(f"  ❌ Failed to store: {query[:30]}...")
        except Exception as e:
            print(f"  ❌ Error storing '{query[:30]}...': {e}")
    
    # Test retrieving answers
    print("\n🔍 Testing cache retrieval...")
    for query in test_queries:
        try:
            cached_result = await semantic_cache.get_cached_answer(query)
            if cached_result:
                print(f"  ✅ Cache hit for: {query[:30]}...")
                print(f"     Answer: {cached_result['answer'][:50]}...")
            else:
                print(f"  ❌ Cache miss for: {query[:30]}...")
        except Exception as e:
            print(f"  ❌ Error retrieving '{query[:30]}...': {e}")
    
    # Test similar query retrieval
    print("\n🔄 Testing similar query retrieval...")
    similar_queries = [
        "What is AI?",  # Similar to "What is artificial intelligence?"
        "How does ML work?",  # Similar to "How does machine learning work?"
    ]
    
    for query in similar_queries:
        try:
            cached_result = await semantic_cache.get_cached_answer(query)
            if cached_result:
                print(f"  ✅ Similar query cache hit: {query}")
                print(f"     Original: {cached_result['query'][:50]}...")
            else:
                print(f"  ❌ No similar query found for: {query}")
        except Exception as e:
            print(f"  ❌ Error with similar query '{query}': {e}")


async def benchmark_cache_performance():
    """Benchmark cache performance."""
    print("\n⚡ Benchmarking cache performance...")
    
    if not semantic_cache._is_cache_enabled():
        print("❌ Semantic cache is not enabled")
        return
    
    import time
    
    test_query = "What is the meaning of life, universe, and everything?"
    test_answer = "42"
    
    # Benchmark storage
    start_time = time.time()
    store_success = await semantic_cache.store_answer(test_query, test_answer)
    store_time = (time.time() - start_time) * 1000  # Convert to ms
    
    if store_success:
        print(f"📝 Store time: {store_time:.2f} ms")
    else:
        print("❌ Failed to store test data")
        return
    
    # Benchmark retrieval
    retrieval_times = []
    for i in range(5):
        start_time = time.time()
        cached_result = await semantic_cache.get_cached_answer(test_query)
        retrieval_time = (time.time() - start_time) * 1000
        retrieval_times.append(retrieval_time)
        
        if cached_result:
            print(f"🔍 Retrieval {i+1}: {retrieval_time:.2f} ms")
        else:
            print(f"❌ Retrieval {i+1}: No result found")
    
    if retrieval_times:
        avg_retrieval = sum(retrieval_times) / len(retrieval_times)
        print(f"📊 Average retrieval time: {avg_retrieval:.2f} ms")
        
        # Calculate cache speedup (assuming normal RAG takes 5-10 seconds)
        normal_rag_time = 7000  # 7 seconds in ms
        speedup = normal_rag_time / avg_retrieval
        print(f"🚀 Cache speedup: {speedup:.1f}x faster than normal RAG")


async def export_cache_data(filename: str = None):
    """Export cache data to JSON file."""
    if not filename:
        filename = "cache_export.json"
    
    print(f"\n📤 Exporting cache data to {filename}...")
    
    if not semantic_cache._is_cache_enabled():
        print("❌ Semantic cache is not enabled")
        return False
    
    try:
        # Get cache statistics
        stats = semantic_cache.get_cache_stats()
        
        # Get all cache keys (limited export for demo)
        cache_keys = semantic_cache.redis_client.keys("cache_entry:*")
        
        export_data = {
            "export_timestamp": time.time(),
            "cache_stats": stats,
            "total_entries": len(cache_keys),
            "sample_entries": []
        }
        
        # Export first 10 entries as samples
        for key in cache_keys[:10]:
            try:
                data = semantic_cache.redis_client.get(key)
                if data:
                    entry = json.loads(data)
                    # Remove sensitive data and limit content
                    sample_entry = {
                        "query": entry.get("query", "")[:100] + "..." if len(entry.get("query", "")) > 100 else entry.get("query", ""),
                        "answer": entry.get("answer", "")[:200] + "..." if len(entry.get("answer", "")) > 200 else entry.get("answer", ""),
                        "created_at": entry.get("created_at"),
                        "access_count": entry.get("access_count", 0),
                        "metadata": entry.get("metadata", {})
                    }
                    export_data["sample_entries"].append(sample_entry)
            except Exception as e:
                logger.debug(f"Error processing cache entry {key}: {e}")
        
        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Cache data exported to {filename}")
        print(f"📊 Exported {len(export_data['sample_entries'])} sample entries")
        return True
        
    except Exception as e:
        print(f"❌ Error exporting cache data: {e}")
        return False


def main():
    """Main function for cache management CLI."""
    parser = argparse.ArgumentParser(description="Manage semantic cache")
    parser.add_argument(
        "action",
        choices=["stats", "clear", "test", "benchmark", "export"],
        help="Action to perform"
    )
    parser.add_argument(
        "--file",
        help="Output file for export action",
        default="cache_export.json"
    )
    
    args = parser.parse_args()
    
    if args.action == "stats":
        asyncio.run(show_cache_stats())
    elif args.action == "clear":
        confirm = input("⚠️  Are you sure you want to clear all cache entries? (y/N): ")
        if confirm.lower() == 'y':
            asyncio.run(clear_cache())
        else:
            print("❌ Cache clear cancelled")
    elif args.action == "test":
        asyncio.run(test_cache_functionality())
    elif args.action == "benchmark":
        asyncio.run(benchmark_cache_performance())
    elif args.action == "export":
        asyncio.run(export_cache_data(args.file))


if __name__ == "__main__":
    main()