# src/agentic_rag/scripts/manage_cache.py

import argparse
import asyncio
import json
import time

from agentic_rag.app.semantic_cache import get_semantic_cache
from agentic_rag.logging_config import logger


async def show_cache_stats():
    """Display cache statistics."""
    print("\n" + "="*50)
    print("SEMANTIC CACHE STATISTICS")
    print("="*50)
    
    cache = get_semantic_cache()
    stats = await cache.get_cache_stats()
    
    if not stats.get("enabled", False):
        print("❌ Semantic cache is disabled")
        return
    
    if "error" in stats:
        print(f"❌ Error getting cache stats: {stats['error']}")
        return
    
    print("✅ Cache Status: Enabled")
    print(f"📊 Total Entries: {stats.get('total_entries', 0)}")
    print(f"📏 Max Size: {stats.get('max_size', 0)}")
    print(f"📊 Cache Fill: {stats.get('fill_percentage', 0)}%")
    print(f"⏰ TTL: {stats.get('ttl_seconds', 0)} seconds")
    print(f"🎯 Similarity Threshold: {stats.get('similarity_threshold', 0)}")
    print(f"📈 Avg Access Count: {stats.get('avg_access_count', 0)}")
    print(f"🔄 Background GC: {'✅ Running' if stats.get('background_gc_enabled') else '❌ Stopped'}")
    print(f"⚡ Async Redis: {'✅ Enabled' if stats.get('async_redis') else '❌ Sync fallback'}")
    
    # Show Redis memory info if available
    redis_memory = stats.get('redis_memory', {})
    if redis_memory:
        print(f"💾 Redis Memory: {redis_memory.get('used_memory_human', 'unknown')}")
    
    print("="*50)


async def clear_cache():
    """Clear all cache entries."""
    print("\n🗑️ Clearing semantic cache...")
    
    cache = get_semantic_cache()
    if not await cache._is_cache_enabled():
        print("❌ Semantic cache is not enabled or not properly initialized")
        return False
    
    try:
        success = await cache.clear_cache()
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
    
    cache = get_semantic_cache()
    if not await cache._is_cache_enabled():
        print("❌ Semantic cache is not enabled or not properly initialized")
        return
    
    # Test queries
    test_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?", 
        "Explain neural networks",
        "What is AI?",  # Should be detected as duplicate of first query
    ]
    
    test_answers = [
        "Artificial intelligence is a field of computer science that focuses on creating intelligent machines.",
        "Machine learning is a subset of AI that enables computers to learn without being explicitly programmed.",
        "Neural networks are computing systems inspired by biological neural networks.",
        "AI is artificial intelligence, a technology for creating smart machines."
    ]
    
    # Test storing answers
    print("📝 Storing test answers in cache...")
    for i, (query, answer) in enumerate(zip(test_queries, test_answers)):
        try:
            success = await cache.store_answer(
                query, answer, {"test": True, "tokens": 100, "test_id": i}
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
            cached_result = await cache.get_cached_answer(query)
            if cached_result:
                print(f"  ✅ Cache hit for: {query[:30]}...")
                print(f"     Answer: {cached_result['answer'][:50]}...")
                print(f"     Access count: {cached_result.get('access_count', 0)}")
            else:
                print(f"  ❌ Cache miss for: {query[:30]}...")
        except Exception as e:
            print(f"  ❌ Error retrieving '{query[:30]}...': {e}")
    
    # Test similar query retrieval
    print("\n🔄 Testing similar query retrieval...")
    similar_queries = [
        "What is machine intelligence?",  # Similar to "What is artificial intelligence?"
        "How does deep learning work?",   # Similar to "How does machine learning work?"
    ]
    
    for query in similar_queries:
        try:
            cached_result = await cache.get_cached_answer(query)
            if cached_result:
                print(f"  ✅ Similar query cache hit: {query}")
                print(f"     Original: {cached_result['query'][:50]}...")
            else:
                print(f"  ❌ No similar query found for: {query}")
        except Exception as e:
            print(f"  ❌ Error with similar query '{query}': {e}")


async def health_check():
    """Perform health check on cache components."""
    print("\n🏥 Semantic Cache Health Check")
    print("="*40)
    
    cache = get_semantic_cache()
    health = await cache.health_check()
    
    print(f"📊 Cache Enabled: {'✅' if health.get('enabled') else '❌'}")
    print(f"🔧 Initialized: {'✅' if health.get('initialized') else '❌'}")
    print(f"📡 Redis Health: {'✅' if health.get('redis_healthy') else '❌'}")
    if not health.get('redis_healthy') and 'redis_error' in health:
        print(f"   Error: {health['redis_error']}")
    
    print(f"🔍 Weaviate Health: {'✅' if health.get('weaviate_healthy') else '❌'}")
    if not health.get('weaviate_healthy') and 'weaviate_error' in health:
        print(f"   Error: {health['weaviate_error']}")
    
    print(f"🔄 Background GC: {'✅' if health.get('background_gc_running') else '❌'}")
    
    print("="*40)


async def benchmark_cache_performance():
    """Benchmark cache performance."""
    print("\n⚡ Benchmarking cache performance...")
    
    cache = get_semantic_cache()
    if not await cache._is_cache_enabled():
        print("❌ Semantic cache is not enabled")
        return
    
    import time
    
    test_query = "What is the meaning of life, universe, and everything?"
    test_answer = "42, according to Douglas Adams' The Hitchhiker's Guide to the Galaxy."
    
    # Benchmark storage
    start_time = time.time()
    store_success = await cache.store_answer(test_query, test_answer)
    store_time = (time.time() - start_time) * 1000  # Convert to ms
    
    if store_success:
        print(f"📝 Store time: {store_time:.2f} ms")
    else:
        print("❌ Failed to store test data")
        return
    
    # Benchmark retrieval (exact match)
    print("\n🔍 Testing exact match retrieval...")
    retrieval_times = []
    for i in range(5):
        start_time = time.time()
        cached_result = await cache.get_cached_answer(test_query)
        retrieval_time = (time.time() - start_time) * 1000
        retrieval_times.append(retrieval_time)
        
        if cached_result:
            print(f"   Exact match {i+1}: {retrieval_time:.2f} ms")
        else:
            print(f"❌ Exact match {i+1}: No result found")
    
    # Benchmark semantic similarity retrieval
    print("\n🧠 Testing semantic similarity retrieval...")
    similar_query = "What's the answer to life, the universe and everything?"
    semantic_times = []
    for i in range(3):
        start_time = time.time()
        cached_result = await cache.get_cached_answer(similar_query)
        retrieval_time = (time.time() - start_time) * 1000
        semantic_times.append(retrieval_time)
        
        if cached_result:
            print(f"   Semantic match {i+1}: {retrieval_time:.2f} ms")
        else:
            print(f"❌ Semantic match {i+1}: No result found")
    
    # Calculate averages and speedup
    if retrieval_times:
        avg_exact = sum(retrieval_times) / len(retrieval_times)
        print(f"\n📊 Average exact match time: {avg_exact:.2f} ms")
        
        # Calculate cache speedup (assuming normal RAG takes 5-10 seconds)
        normal_rag_time = 7000  # 7 seconds in ms
        speedup = normal_rag_time / avg_exact
        print(f"🚀 Exact match speedup: {speedup:.1f}x faster than normal RAG")
    
    if semantic_times:
        avg_semantic = sum(semantic_times) / len(semantic_times)
        print(f"📊 Average semantic match time: {avg_semantic:.2f} ms")
        
        normal_rag_time = 7000  # 7 seconds in ms
        speedup = normal_rag_time / avg_semantic
        print(f"🚀 Semantic match speedup: {speedup:.1f}x faster than normal RAG")


async def export_cache_data(filename: str = None):
    """Export cache data to JSON file."""
    if not filename:
        filename = "cache_export.json"
    
    print(f"\n📤 Exporting cache data to {filename}...")
    
    cache = get_semantic_cache()
    if not await cache._is_cache_enabled():
        print("❌ Semantic cache is not enabled")
        return False
    
    try:
        # Get cache statistics
        stats = await cache.get_cache_stats()
        
        export_data = {
            "export_timestamp": time.time(),
            "cache_stats": stats,
            "sample_entries": []
        }
        
        # Export sample entries (last 10 from index)
        try:
            if hasattr(cache, 'redis_client'):
                index_key = "cache_index"
                if hasattr(cache.redis_client, 'zrange'):
                    # Async Redis
                    sample_cache_ids = await cache.redis_client.zrange(index_key, -10, -1)
                else:
                    # Sync Redis
                    sample_cache_ids = await asyncio.to_thread(cache.redis_client.zrange, index_key, -10, -1)
                
                for cache_id in sample_cache_ids:
                    try:
                        entry = await cache._get_cache_entry_by_id(cache_id)
                        if entry:
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
                        logger.debug(f"Error processing cache entry {cache_id}: {e}")
        except Exception as e:
            logger.debug(f"Error getting sample entries: {e}")
        
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
        choices=["stats", "clear", "test", "benchmark", "export", "health"],
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
    elif args.action == "health":
        asyncio.run(health_check())


if __name__ == "__main__":
    main()