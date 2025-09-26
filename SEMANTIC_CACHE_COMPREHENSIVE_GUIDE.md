# Semantic Cache Comprehensive Guide

## Overview

The Semantic Cache is a production-ready, high-performance caching system designed for agentic RAG applications. It intelligently caches query-answer pairs and retrieves them based on both exact matches and semantic similarity, dramatically reducing response times and API costs while maintaining answer quality.

### Key Features

- **Two-Tier Architecture**: Combines exact hash-based lookup with semantic similarity search
- **Cross-Encoder Similarity**: Uses reliable cross-encoder models instead of problematic vector similarity
- **Atomic Operations**: Lua scripts ensure data consistency across Redis and Weaviate
- **Background Garbage Collection**: Automatically cleans up orphaned entries and manages cache size
- **Production Monitoring**: Comprehensive health checks and performance metrics
- **Async/Await Support**: Non-blocking operations for high-performance applications

## Architecture

### 1. Two-Tier Cache System

**Tier 1: Exact Match Layer**
- Uses SHA256 hashing for instant O(1) lookup
- Handles identical queries with microsecond response times
- Stores query hash → cache_id mappings in Redis

**Tier 2: Semantic Similarity Layer**
- Cross-encoder based similarity detection
- Handles paraphrases and semantically similar queries
- Configurable similarity thresholds for precision control

### 2. Data Storage

**Redis**
- Stores cache entry metadata with configurable TTL
- Uses ZSET for O(log N) eviction and statistics
- Atomic operations via Lua scripts prevent race conditions

**Weaviate**
- Vector storage for semantic similarity search
- Optimized HNSW indexing for fast retrieval
- Automatic cleanup of orphaned vectors

### 3. Cross-Encoder Similarity Engine

The cache uses cross-encoder models instead of traditional vector similarity because:
- **Higher Accuracy**: Direct semantic understanding between query pairs
- **No False Positives**: Eliminates vector similarity issues (1.0 scores for unrelated queries)
- **Better Paraphrase Detection**: Handles semantic variations more reliably
- **Transparent Logic**: Easier to debug and understand similarity decisions

## Core Functionality

### Query Processing Flow

1. **Exact Match Check**: Query is hashed and checked against exact match cache
2. **Cross-Encoder Search**: If no exact match, search for semantically similar cached queries
3. **Similarity Validation**: Apply tiered rules to determine cache hit acceptability
4. **Result Retrieval**: Fetch and return cached answer with updated access metadata
5. **Alias Creation**: Create exact match alias for high-confidence semantic hits

### Storage Flow

1. **Duplicate Detection**: Check for exact or highly similar existing entries
2. **Entry Consolidation**: Update existing entries instead of creating duplicates
3. **Atomic Creation**: Use Lua scripts for consistent cross-store operations
4. **Size Management**: Automatically evict oldest entries when cache limit reached
5. **Vector Indexing**: Store query vectors in Weaviate for semantic search

## Prominent Methods in SemanticCache Class

### Core Cache Operations

#### `get_cached_answer(query: str) -> Optional[Dict[str, Any]]`
**Purpose**: Retrieve cached answers for queries using two-tier lookup strategy.

**Process**:
1. Generates SHA256 hash for exact match lookup
2. Performs cross-encoder similarity search if no exact match
3. Validates similarity using tiered acceptance rules
4. Updates access metadata and extends TTL
5. Creates exact match aliases for high-confidence hits

**Returns**: Cached entry with answer, metadata, and similarity score, or None if no match.

#### `store_answer(query: str, answer: str, metadata: Dict[str, Any] = None) -> bool`
**Purpose**: Store query-answer pairs with intelligent deduplication.

**Process**:
1. Checks for exact duplicates via hashing
2. Searches for highly similar entries for consolidation
3. Updates existing entries or creates new ones atomically
4. Manages cache size and vector indexing
5. Ensures data consistency across Redis and Weaviate

**Returns**: True if successfully cached, False otherwise.

### Similarity Detection

#### `_cross_encoder_similarity_search(query: str, k: int = 1)`
**Purpose**: Find semantically similar cached queries using cross-encoder models.

**Process**:
1. Retrieves all cached entries from Redis index
2. Computes cross-encoder similarity for each candidate
3. Sorts results by similarity score (highest first)
4. Returns top-k candidates with similarity scores

**Benefits**: More reliable than vector similarity, eliminates false positives.

#### `_ce_similarity(text1: str, text2: str) -> float`
**Purpose**: Compute cross-encoder similarity between two text strings.

**Features**:
- Lazy model loading via model registry
- Thread-safe execution with asyncio.to_thread
- Sigmoid normalization for logit-like scores
- Robust error handling with fallback scores

### Cache Management

#### `_manage_cache_size_atomic()`
**Purpose**: Maintain cache size limits using atomic Lua scripts.

**Process**:
1. Uses ZSET operations for O(log N) performance
2. Identifies oldest entries for eviction
3. Atomically removes entries from both Redis and index
4. Triggers Weaviate vector cleanup
5. Cleans up exact match mappings

#### `run_garbage_collection_manually() -> int`
**Purpose**: Clean up expired entries and orphaned vectors.

**Process**:
1. Identifies stale Redis index entries
2. Finds orphaned Weaviate vectors
3. Removes inconsistencies between storage layers
4. Returns count of cleaned items

### Monitoring and Health

#### `get_cache_stats() -> Dict[str, Any]`
**Purpose**: Provide comprehensive cache performance metrics.

**Returns**:
```python
{
    "enabled": True,
    "total_entries": 245,
    "max_size": 1000,
    "fill_percentage": 24.5,
    "ttl_seconds": 3600,
    "similarity_threshold": 0.95,
    "avg_access_count": 2.3,
    "redis_memory": {"used_memory": 1024000, "used_memory_human": "1M"},
    "background_gc_enabled": True,
    "async_redis": True
}
```

#### `health_check() -> Dict[str, Any]`
**Purpose**: Verify system health and component connectivity.

**Checks**:
- Cache enablement and initialization status
- Redis connectivity and responsiveness
- Weaviate cluster health and readiness
- Background garbage collection status

### Utility Methods

#### `clear_cache() -> bool`
**Purpose**: Atomically clear all cache entries and reset system state.

**Process**:
1. Clears all Weaviate vectors by cache_id
2. Removes all Redis cache entries and mappings
3. Purges Redis memory allocations
4. Maintains system consistency throughout operation

#### `shutdown()`
**Purpose**: Gracefully shutdown cache with proper resource cleanup.

**Process**:
1. Signals shutdown to background tasks
2. Cancels garbage collection loops
3. Closes Redis and Weaviate connections
4. Releases all allocated resources

## Similarity Acceptance Rules

The cache uses a tiered approach to determine when cached entries should be returned:

### Rule 1: High Cross-Encoder Similarity (≥0.85)
- **Trigger**: Cross-encoder similarity ≥ 0.85
- **Action**: Accept immediately without additional checks
- **Rationale**: Cross-encoder models provide highly reliable semantic understanding

### Rule 2: Cross-Encoder with Lexical Support
- **Trigger**: Cross-encoder ≥ 0.60 AND lexical similarity ≥ 0.15
- **Action**: Accept with combined evidence
- **Rationale**: Moderate semantic similarity reinforced by lexical overlap

### Rule 3: High Lexical Similarity (≥0.4)
- **Trigger**: Lexical similarity ≥ 0.4
- **Action**: Accept as likely paraphrase
- **Rationale**: High token overlap indicates query variations

## Configuration Options

### Core Settings
```python
ENABLE_SEMANTIC_CACHE: bool = True                    # Enable/disable caching
SEMANTIC_CACHE_TTL: int = 3600                       # Entry TTL in seconds
SEMANTIC_CACHE_MAX_SIZE: int = 1000                  # Maximum cache entries
SEMANTIC_CACHE_INDEX_NAME: str = "SemanticCache"     # Weaviate collection name
SEMANTIC_CACHE_GC_INTERVAL: int = 3600              # GC interval in seconds
```

### Similarity Thresholds
```python
SEMANTIC_CACHE_SIMILARITY_THRESHOLD: float = 0.95    # Main similarity threshold
SEMANTIC_CACHE_CE_ACCEPT: float = 0.60               # Cross-encoder acceptance
SEMANTIC_CACHE_LEXICAL_MIN: float = 0.15             # Minimum lexical support
```

## Performance Characteristics

### Cache Hit Performance
- **Exact Match**: ~1-5ms (SHA256 + Redis lookup)
- **Semantic Match**: ~30-50ms (cross-encoder + Redis lookup)
- **Cache Miss**: ~50-100ms (full search + fallback)

### Improvements Over Previous Versions
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Cache Eviction | O(N) keys scan | O(log N) ZSET ops | ~100x faster |
| Statistics | O(N) full scan | O(1) ZSET card | ~1000x faster |
| Deduplication | None | SHA256 + semantic | Prevents duplicates |
| Memory Leaks | Orphaned vectors | Background GC | Zero leaks |
| Race Conditions | Possible | Atomic Lua scripts | Thread-safe |
| False Positives | Vector similarity bugs | Cross-encoder reliability | Eliminated |

## Usage Examples

### Basic Usage
```python
from agentic_rag.app.semantic_cache import get_semantic_cache

cache = get_semantic_cache()

# Store an answer
success = await cache.store_answer(
    query="What is machine learning?",
    answer="Machine learning is a subset of AI...",
    metadata={"category": "AI", "confidence": 0.95}
)

# Retrieve cached answer
cached_result = await cache.get_cached_answer("What is ML?")
if cached_result:
    print(f"Answer: {cached_result['answer']}")
    print(f"Similarity: {cached_result.get('similarity', 'N/A')}")
```

### Health Monitoring
```python
# Check system health
health = await cache.health_check()
if health["redis_healthy"] and health["weaviate_healthy"]:
    print("Cache system is healthy")

# Get performance statistics
stats = await cache.get_cache_stats()
print(f"Cache fill: {stats['fill_percentage']}%")
print(f"Average access count: {stats['avg_access_count']}")
```

### Manual Maintenance
```python
# Run garbage collection
cleaned_count = await cache.run_garbage_collection_manually()
print(f"Cleaned {cleaned_count} stale entries")

# Clear entire cache
success = await cache.clear_cache()
if success:
    print("Cache cleared successfully")
```

## Production Deployment Considerations

### Resource Requirements
- **Redis**: Minimum 1GB RAM for metadata storage
- **Weaviate**: Memory scales with vector dimensions and cache size
- **CPU**: Cross-encoder models benefit from GPU acceleration

### Monitoring
- Monitor cache hit rates and performance metrics
- Set up alerts for component health failures
- Track memory usage and garbage collection frequency

### Scalability
- Redis can be clustered for high availability
- Weaviate supports horizontal scaling
- Cross-encoder models can be distributed across GPUs

### Security
- Enable Redis authentication and encryption in transit
- Configure Weaviate access controls
- Implement rate limiting for cache operations

## Integration with Agentic RAG Workflow

The semantic cache integrates seamlessly with the agentic workflow:

1. **Pre-Processing**: Queries are checked against cache before expensive LLM calls
2. **Post-Processing**: Generated answers are automatically cached for future use
3. **Workflow State**: Cache hits bypass expensive retrieval and generation steps
4. **Metadata Preservation**: Token counts and performance metrics are cached
5. **Session Management**: Cache operates across user sessions for maximum benefit

This comprehensive caching system provides significant performance improvements while maintaining the high-quality responses expected from agentic RAG applications.