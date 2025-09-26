# Semantic Cache Testing Framework

A comprehensive testing and simulation framework for the semantic cache implementation that allows you to test and experiment with caching functionality without executing the entire graph workflow or making LLM API calls.

## Overview

This framework provides all the features of the `semantic_cache.py` implementation in a testing-focused interface that eliminates the overhead of LLM calls, query classification, summarization, answer generation, and grounding checks. It's designed for rapid experimentation and validation of semantic cache behavior.

## Features

### Core Functionality
- ✅ **Direct Cache Entry Creation**: Store query-answer pairs using the same `_create_new_cache_entry` logic as the workflow
- ✅ **Multi-Method Similarity Testing**: Test vector, embedding, cross-encoder, and lexical similarity between queries
- ✅ **Cache Retrieval Testing**: Test whether queries get cache hits using the same rules as the actual workflow
- ✅ **Performance Metrics**: Detailed timing and scoring information for all operations
- ✅ **Bulk Testing**: Test multiple queries against cached entries efficiently
- ✅ **Full Cache Features**: All semantic cache functionality (storage, retrieval, GC) works as in the actual workflow

### Similarity Methods Available (Cross-Encoder Based)
1. **Cross-Encoder Similarity**: Direct semantic similarity using cross-encoder models (most reliable and primary method)
2. **Lexical Similarity**: Token-based Jaccard similarity for baseline comparison and paraphrase detection

### Completely Removed Methods
- **Vector Similarity**: Completely removed due to persistent unreliable scoring issues (was returning 1.0 for both similar and unrelated queries)
- **Embedding Similarity**: Removed due to inconsistent results and performance issues

### 🎯 **Cross-Encoder Approach Benefits**
- **No false positives**: Eliminates vector similarity issues that were causing unrelated queries to get high similarity scores
- **Direct semantic understanding**: Cross-encoder models directly assess semantic similarity between query pairs
- **Consistent scoring**: Similar queries consistently get similar scores, unrelated queries get low scores
- **Better paraphrase detection**: Handles semantic variations and paraphrases more accurately
- **Transparent logic**: Easier to debug and understand why queries are considered similar or different

*Note: The semantic cache now uses a completely cross-encoder based approach for finding similar cached entries, eliminating all vector similarity dependencies that were causing scoring issues.*

## API Endpoints

### Create Cache Entry
```http
POST /cache/test/create-entry
Content-Type: application/json

{
  "query": "What are the benefits of MCTs",
  "answer": "MCT benefits include improved energy metabolism...",
  "metadata": {"category": "health", "topic": "MCT"}
}
```

### Test Query Similarity
```http
POST /cache/test/similarity
Content-Type: application/json

{
  "cached_query": "What are the benefits of MCTs",
  "test_query": "What are the benefits of taking MCT"
}
```

**Response:**
```json
{
  "cached_query": "What are the benefits of MCTs",
  "test_query": "What are the benefits of taking MCT",
  "vector_similarity": 0.887,
  "embedding_similarity": 0.856,
  "cross_encoder_similarity": 0.723,
  "lexical_similarity": 0.750,
  "cache_hit_prediction": true,
  "rule_triggered": "Rule 3: Lexical support (vec=0.887, ce=0.723, emb=0.856, lex=0.75)",
  "execution_time_ms": 45.2
}
```

### Test Cache Retrieval
```http
POST /cache/test/retrieval?query=What%20are%20the%20benefits%20of%20taking%20MCT
```

**Response:**
```json
{
  "query": "What are the benefits of taking MCT",
  "cache_hit": true,
  "cached_entry": {
    "cache_id": "uuid-here",
    "query": "What are the benefits of MCTs",
    "answer": "MCT benefits include...",
    "similarity": 0.887
  },
  "similarity_score": 0.887,
  "execution_time_ms": 23.1
}
```

### Bulk Similarity Testing
```http
POST /cache/test/bulk-similarity
Content-Type: application/json

{
  "cached_query": "What are the benefits of MCTs",
  "test_queries": [
    "What are the benefits of taking MCT",
    "What benefits can I expect if I consume MCT daily?",
    "What are the benefits of chocolate?"
  ]
}
```

### Get Cache Statistics
```http
GET /cache/test/stats
```

### Clear Cache for Testing
```http
POST /cache/test/clear
```

## Cache Hit Rules (Cross-Encoder Based)

The testing framework uses a cross-encoder based approach that completely eliminates vector similarity issues:

### Rule 1: High Cross-Encoder Similarity (≥0.85)
- Accepts queries with high cross-encoder similarity (most reliable semantic measure)
- Uses direct semantic understanding rather than vector similarity
- No additional validation required due to cross-encoder reliability

### Rule 2: Cross-Encoder with Lexical Support
- Requires cross-encoder ≥0.60 AND lexical ≥0.15
- Combines semantic understanding with lexical overlap for robust matching

### Rule 3: High Lexical Similarity (≥0.4)
- Accepts queries with very high lexical similarity (likely paraphrases)
- Useful for catching variations with same key terms

### 🎯 **Key Improvements**
- **Candidate Selection**: Uses cross-encoder similarity to find similar cached entries instead of unreliable vector search
- **Direct Scoring**: Cross-encoder scores are used directly without complex normalization
- **Consistent Results**: Eliminates false positives from vector similarity returning 1.0 for unrelated queries
- **Better Semantic Understanding**: Cross-encoder models provide more accurate semantic similarity assessment

**Note**: Vector similarity has been completely replaced with cross-encoder similarity throughout the entire cache system, including both candidate selection and similarity validation phases.

## Usage Examples

### Command Line Demo
```bash
# Run the comprehensive demo
python tests/semantic_cache_test_demo.py
```

### Programmatic Usage
```python
from agentic_rag.testing.semantic_cache_tester import semantic_cache_tester

# Create cache entry
result = await semantic_cache_tester.create_cache_entry(
    query="What are the benefits of MCTs",
    answer="MCT benefits include...",
    metadata={"category": "health"}
)

# Test similarity
similarity = await semantic_cache_tester.test_query_similarity(
    cached_query="What are the benefits of MCTs",
    test_query="What are the benefits of taking MCT"
)

# Test cache retrieval
retrieval = await semantic_cache_tester.test_cache_retrieval(
    "What are the benefits of taking MCT"
)
```

### API Usage with curl
```bash
# Create cache entry
curl -X POST 'http://localhost:8000/cache/test/create-entry' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What are the benefits of MCTs",
    "answer": "MCT benefits include improved energy metabolism...",
    "metadata": {"category": "health"}
  }'

# Test similarity
curl -X POST 'http://localhost:8000/cache/test/similarity' \
  -H 'Content-Type: application/json' \
  -d '{
    "cached_query": "What are the benefits of MCTs",
    "test_query": "What are the benefits of taking MCT"
  }'

# Test retrieval
curl -X POST 'http://localhost:8000/cache/test/retrieval?query=What%20are%20the%20benefits%20of%20taking%20MCT'
```

## Testing Scenarios

### 1. Validate Similar Queries Get Cache Hits
```python
# Cache the original query
await semantic_cache_tester.create_cache_entry(
    query="What are the benefits of MCTs",
    answer="MCT benefits include..."
)

# Test similar variations
similar_queries = [
    "What are the benefits of taking MCT",
    "What benefits can I expect if I consume MCT daily?",
    "What are the benefits of taking MCTs?",
    "Explain the benefits of MCT"
]

for query in similar_queries:
    result = await semantic_cache_tester.test_cache_retrieval(query)
    print(f"{query}: {'HIT' if result.cache_hit else 'MISS'}")
```

### 2. Validate False Positive Prevention
```python
# Test unrelated queries don't get cache hits
unrelated_queries = [
    "What are the health benefits of chocolate?",
    "How to cook pasta perfectly?",
    "What is artificial intelligence?"
]

for query in unrelated_queries:
    result = await semantic_cache_tester.test_cache_retrieval(query)
    print(f"{query}: {'HIT' if result.cache_hit else 'MISS'}")
```

### 3. Performance Validation
```python
# Test performance of similarity calculations
similarity = await semantic_cache_tester.test_query_similarity(
    cached_query="What are the benefits of MCTs",
    test_query="What are the benefits of taking MCT"
)

print(f"Execution time: {similarity.execution_time_ms}ms")
print(f"Vector similarity: {similarity.vector_similarity}")
print(f"Cache hit prediction: {similarity.cache_hit_prediction}")
```

## Configuration

The testing framework uses the same configuration as the main semantic cache:

```python
# Threshold settings (from agentic_rag.config)
SEMANTIC_CACHE_SIMILARITY_THRESHOLD: float = 0.95    # Main similarity threshold
SEMANTIC_CACHE_CE_ACCEPT: float = 0.60               # Cross-encoder acceptance
SEMANTIC_CACHE_LEXICAL_MIN: float = 0.15             # Minimum lexical support
```

## Troubleshooting

### Common Issues

1. **Cache not initialized**: The framework automatically initializes the cache on first use
2. **Vector similarity returns None**: Cached query may not exist in vector store
3. **Embedding/Cross-encoder similarity fails**: Check model availability in model registry
4. **Low performance**: Ensure models are pre-loaded in model registry

### Debugging

Enable debug logging to see detailed information:
```python
import logging
logging.getLogger('agentic_rag').setLevel(logging.DEBUG)
```

### Environment Variables Required
```bash
HUGGINGFACEHUB_API_TOKEN=your_token
OPENAI_API_KEY=your_key
GOOGLE_API_KEY=your_key
LANGCHAIN_API_KEY=your_key
```

## Benefits

### For Development
- **No LLM Costs**: Test cache behavior without API charges
- **Fast Iteration**: Rapid testing and validation of cache logic
- **Isolated Testing**: Test cache functionality without workflow dependencies
- **Performance Metrics**: Detailed timing and scoring information

### For Production
- **Validation**: Verify cache behavior before deployment
- **Tuning**: Experiment with threshold adjustments
- **Debugging**: Understand why queries do/don't get cache hits
- **Monitoring**: Validate cache performance in production

## Integration with Existing Workflow

The testing framework uses the same `semantic_cache` instance as the main workflow, so:

- ✅ Cache entries created via testing are available to the main workflow
- ✅ Cache entries created by the main workflow are available for testing
- ✅ All cache statistics and GC operations work across both systems
- ✅ Configuration changes affect both systems consistently

This makes it perfect for testing cache behavior in development and validating production cache performance.