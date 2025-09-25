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

### Similarity Methods Available
1. **Vector Similarity**: Uses the same vector store search as the workflow
2. **Embedding Similarity**: Cosine similarity using the configured embedding model
3. **Cross-Encoder Similarity**: Semantic similarity using cross-encoder models
4. **Lexical Similarity**: Token-based Jaccard similarity for baseline comparison

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

## Cache Hit Rules

The testing framework uses the same 4-rule system as the actual cache:

### Rule 1: High Vector Similarity (≥0.92)
- Accepts queries with high vector similarity
- Includes false positive detection for perfect scores (≥0.99)
- Rejects if embedding <0.7 AND lexical <0.1, OR embedding <0.4

### Rule 2: Multi-Metric Validation
- Requires vector ≥0.85 AND cross-encoder ≥0.60 AND embedding ≥0.88

### Rule 3: Lexical Support for Borderline Cases
- Requires vector ≥0.85 AND cross-encoder ≥0.60 AND embedding ≥0.85 AND lexical ≥0.15

### Rule 4: Lenient Rule for Similar Queries
- Requires vector ≥0.87 AND (embedding ≥0.83 OR cross-encoder ≥0.65)

## Usage Examples

### Command Line Demo
```bash
# Run the comprehensive demo
python semantic_cache_test_demo.py
```

### Programmatic Usage
```python
from agentic_rag.app.semantic_cache_tester import semantic_cache_tester

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
SEMANTIC_CACHE_VECTOR_ACCEPT = 0.92      # Rule 1 threshold
SEMANTIC_CACHE_VECTOR_MIN = 0.85         # Multi-metric minimum
SEMANTIC_CACHE_EMB_ACCEPT = 0.88         # Embedding threshold
SEMANTIC_CACHE_CE_ACCEPT = 0.60          # Cross-encoder threshold
SEMANTIC_CACHE_LEXICAL_MIN = 0.15        # Lexical threshold
SEMANTIC_CACHE_SCORE_MODE = "distance"   # Weaviate score interpretation
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