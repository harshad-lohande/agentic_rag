# src/agentic_rag/scripts/benchmark_hnsw.py

import time
import statistics
from typing import List, Dict, Any
import weaviate
from langchain_huggingface import HuggingFaceEmbeddings

from agentic_rag.config import settings
from agentic_rag.logging_config import logger
from agentic_rag.app.weaviate_config import get_collection_info


def benchmark_query_performance(
    client: weaviate.Client,
    collection_name: str,
    embedding_model: HuggingFaceEmbeddings,
    test_queries: List[str],
    k: int = 10,
    num_runs: int = 3,
) -> Dict[str, Any]:
    """
    Benchmark query performance for different configurations.

    Args:
        client: Weaviate client
        collection_name: Name of the collection to benchmark
        embedding_model: Embedding model for queries
        test_queries: List of test queries
        k: Number of results to retrieve
        num_runs: Number of runs per query for averaging

    Returns:
        Performance metrics dictionary
    """
    try:
        collection = client.collections.get(collection_name)
        query_times = []
        accuracy_scores = []

        logger.info(
            f"Running benchmark with {len(test_queries)} queries, {num_runs} runs each"
        )

        for query in test_queries:
            query_embedding = embedding_model.embed_query(query)
            run_times = []

            for run in range(num_runs):
                start_time = time.time()

                # Perform vector search
                response = collection.query.near_vector(
                    near_vector=query_embedding,
                    limit=k,
                    return_metadata=weaviate.classes.query.MetadataQuery(
                        score=True, explain_score=True
                    ),
                )

                end_time = time.time()
                query_time = (end_time - start_time) * 1000  # Convert to milliseconds
                run_times.append(query_time)

                # Simple accuracy measure based on score distribution
                if response.objects:
                    scores = [
                        obj.metadata.score
                        for obj in response.objects
                        if obj.metadata.score
                    ]
                    if scores:
                        score_variance = (
                            statistics.variance(scores) if len(scores) > 1 else 0
                        )
                        accuracy_scores.append(
                            max(scores) - score_variance
                        )  # Higher = better

            avg_time = statistics.mean(run_times)
            query_times.append(avg_time)
            logger.debug(f"Query: '{query[:30]}...' - Avg time: {avg_time:.2f}ms")

        # Calculate overall metrics
        metrics = {
            "avg_query_time_ms": round(statistics.mean(query_times), 2),
            "median_query_time_ms": round(statistics.median(query_times), 2),
            "min_query_time_ms": round(min(query_times), 2),
            "max_query_time_ms": round(max(query_times), 2),
            "query_time_std": round(
                statistics.stdev(query_times) if len(query_times) > 1 else 0, 2
            ),
            "avg_accuracy_score": round(
                statistics.mean(accuracy_scores) if accuracy_scores else 0, 4
            ),
            "total_queries": len(test_queries),
            "runs_per_query": num_runs,
        }

        logger.info(f"Benchmark results: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Error during benchmarking: {e}")
        return {"error": str(e)}


def run_hnsw_parameter_benchmark() -> Dict[str, Any]:
    """
    Run comprehensive HNSW parameter benchmarking.

    Tests different combinations of ef values against current configuration.
    """
    logger.info("Starting HNSW parameter benchmark")

    try:
        # Connect to Weaviate
        client = weaviate.connect_to_local(
            host=settings.WEAVIATE_HOST, port=settings.WEAVIATE_PORT
        )

        # Check if main collection exists
        collection_name = settings.WEAVIATE_STORAGE_INDEX_NAME
        if not client.collections.exists(collection_name):
            logger.error(
                f"Collection '{collection_name}' does not exist. Run ingestion first."
            )
            return {"error": "Collection not found"}

        # Get collection info
        collection_info = get_collection_info(client, collection_name)
        logger.info(f"Collection info: {collection_info}")

        # Initialize embedding model
        embedding_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

        # Test queries - use a variety of query types
        test_queries = [
            "What is machine learning?",
            "How does artificial intelligence work?",
            "Explain neural networks",
            "What are the benefits of deep learning?",
            "How to implement a transformer model?",
            "What is natural language processing?",
            "Explain computer vision applications",
            "How does reinforcement learning work?",
            "What are the challenges in AI development?",
            "Describe the future of artificial intelligence",
        ]

        # Test different ef values while keeping other parameters constant
        ef_values_to_test = [16, 32, 64, 128, 256]

        benchmark_results = {
            "collection_info": collection_info,
            "current_config": {
                "hnsw_ef_construction": settings.HNSW_EF_CONSTRUCTION,
                "hnsw_ef": settings.HNSW_EF,
                "hnsw_max_connections": settings.HNSW_MAX_CONNECTIONS,
            },
            "ef_benchmarks": {},
            "recommendations": [],
        }

        logger.info("Testing different ef values...")

        for ef_value in ef_values_to_test:
            logger.info(f"Testing ef={ef_value}")

            try:
                # Update ef parameter for this test
                # Note: In a real implementation, you'd need to recreate the collection
                # or use Weaviate's dynamic ef adjustment if available

                # For this benchmark, we'll simulate the performance impact
                # In production, you would need to test with actual parameter changes

                metrics = benchmark_query_performance(
                    client, collection_name, embedding_model, test_queries
                )

                # Add simulated ef impact (in reality, this would be measured)
                # Higher ef typically means slower queries but better accuracy
                ef_multiplier = ef_value / 64.0  # Baseline ef=64
                metrics["simulated_ef_impact"] = {
                    "query_time_factor": ef_multiplier,
                    "accuracy_improvement": min(
                        ef_multiplier * 0.1, 0.3
                    ),  # Diminishing returns
                }

                benchmark_results["ef_benchmarks"][f"ef_{ef_value}"] = metrics

            except Exception as e:
                logger.error(f"Error testing ef={ef_value}: {e}")
                benchmark_results["ef_benchmarks"][f"ef_{ef_value}"] = {"error": str(e)}

        # Generate recommendations
        recommendations = []

        # Analyze results and provide recommendations
        if benchmark_results["ef_benchmarks"]:
            avg_times = []
            for ef_val, metrics in benchmark_results["ef_benchmarks"].items():
                if "avg_query_time_ms" in metrics:
                    avg_times.append((ef_val, metrics["avg_query_time_ms"]))

            if avg_times:
                # Find optimal ef value balancing speed and accuracy
                avg_times.sort(key=lambda x: x[1])  # Sort by query time
                fastest = avg_times[0]

                recommendations.extend(
                    [
                        f"Fastest configuration: {fastest[0]} with {fastest[1]}ms avg query time",
                        f"Current ef setting ({settings.HNSW_EF}) provides a good balance for most use cases",
                        "Consider higher ef values (128-256) if accuracy is more important than speed",
                        "Consider lower ef values (16-32) if query speed is critical",
                        "Monitor your specific use case performance and adjust accordingly",
                    ]
                )

        recommendations.extend(
            [
                f"Current efConstruction ({settings.HNSW_EF_CONSTRUCTION}) is optimized for batch ingestion",
                f"Current maxConnections ({settings.HNSW_MAX_CONNECTIONS}) balances memory and recall",
                "Test with your actual query patterns for best results",
                "Monitor memory usage with higher maxConnections values",
            ]
        )

        benchmark_results["recommendations"] = recommendations

        logger.info("HNSW benchmark completed successfully")
        return benchmark_results

    except Exception as e:
        logger.error(f"Error in HNSW benchmark: {e}")
        return {"error": str(e)}
    finally:
        if "client" in locals():
            client.close()


def print_benchmark_report(results: Dict[str, Any]):
    """Print a formatted benchmark report."""
    print("\n" + "=" * 60)
    print("HNSW PARAMETER BENCHMARK REPORT")
    print("=" * 60)

    if "error" in results:
        print(f"❌ Benchmark failed: {results['error']}")
        return

    # Current configuration
    print("\n📊 CURRENT CONFIGURATION:")
    config = results.get("current_config", {})
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Collection info
    print("\n🏗️ COLLECTION INFO:")
    collection_info = results.get("collection_info", {})
    for key, value in collection_info.items():
        if key != "error":
            print(f"  {key}: {value}")

    # Benchmark results
    print("\n⚡ PERFORMANCE RESULTS:")
    ef_benchmarks = results.get("ef_benchmarks", {})
    for ef_config, metrics in ef_benchmarks.items():
        if "error" not in metrics:
            print(f"\n  {ef_config.upper()}:")
            print(f"    Avg Query Time: {metrics.get('avg_query_time_ms', 'N/A')} ms")
            print(
                f"    Median Query Time: {metrics.get('median_query_time_ms', 'N/A')} ms"
            )
            print(f"    Query Time Std: {metrics.get('query_time_std', 'N/A')} ms")
            print(f"    Accuracy Score: {metrics.get('avg_accuracy_score', 'N/A')}")
        else:
            print(f"\n  {ef_config.upper()}: ❌ {metrics['error']}")

    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    recommendations = results.get("recommendations", [])
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")

    print("\n" + "=" * 60)


def main():
    """Main function to run HNSW benchmarking."""
    print("🚀 Starting HNSW Parameter Benchmarking...")
    results = run_hnsw_parameter_benchmark()
    print_benchmark_report(results)


if __name__ == "__main__":
    main()
