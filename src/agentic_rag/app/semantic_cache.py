# src/agentic_rag/app/semantic_cache.py

import hashlib
import json
import asyncio
import time
import uuid
import inspect
import threading
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore

try:
    import redis.asyncio as aioredis
    AIOREDIS_AVAILABLE = True
except ImportError:
    AIOREDIS_AVAILABLE = False
    import redis

import weaviate

from agentic_rag.config import settings
from agentic_rag.logging_config import logger
from agentic_rag.app.model_registry import model_registry


class SemanticCache:
    """
    Production-ready semantic caching system with atomic operations and consistency guarantees.
    
    Architecture:
    1. Redis with ZSET indexing for O(log N) eviction and metadata storage
    2. Weaviate for semantic similarity search of queries  
    3. Atomic operations via Lua scripts for consistency
    4. Background garbage collection for orphaned vector cleanup
    5. Query deduplication via SHA256 hashing
    """
    
    def __init__(self):
        self.redis_client = None
        self.weaviate_client = None
        self.cache_vector_store = None
        self.embedding_model = None
        self._initialized = False
        self._init_lock = threading.Lock()
        self._gc_task = None
        self._shutdown_event = asyncio.Event() if asyncio.iscoroutinefunction(lambda: None) else None
    
    async def _initialize_clients(self):
        """Lazy initialization of Redis and Weaviate clients for caching."""
        if self._initialized:
            return True
            
        with self._init_lock:
            if self._initialized:  # Double-check locking
                return True
                
            try:
                # Initialize async Redis client
                if AIOREDIS_AVAILABLE:
                    self.redis_client = aioredis.from_url(
                        f"redis://{settings.REDIS_HOST}:6379",
                        decode_responses=True,
                        max_connections=20,
                        retry_on_timeout=True
                    )
                    await self.redis_client.ping()
                else:
                    # Fallback to sync Redis for backward compatibility
                    self.redis_client = redis.Redis(
                        host=settings.REDIS_HOST, 
                        port=6379, 
                        decode_responses=True,
                        max_connections=20
                    )
                    self.redis_client.ping()
                
                logger.info(f"Connected to Redis for semantic caching (async: {AIOREDIS_AVAILABLE})")
                
                # Initialize Weaviate client in thread pool for async compatibility
                await asyncio.to_thread(self._init_weaviate_and_embeddings)
                
                # Start background garbage collection
                await self._start_background_gc()
                
                self._initialized = True
                logger.info("Semantic cache initialized successfully with optimizations")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize semantic cache: {e}")
                await self._cleanup_failed_init()
                return False
    
    async def _vector_similarity_search(self, query: str, k: int = 1, score_threshold: float | None = None):
        """
        Robust wrapper around the underlying vector store search API.
        Returns list of tuples: [(doc, score), ...]. Score may be None if unavailable.

        Tries, in order:
         - cache_vector_store.similarity_search_with_score(...) with best-effort kwargs
         - cache_vector_store.similarity_search(...) (returns docs only)
         - cache_vector_store.hybrid(...) or other common variants (without unsupported kwargs)
        """
        if not self.cache_vector_store:
            return []

        def call_sync(fn, *a, **kw):
            try:
                return fn(*a, **kw)
            except TypeError:
                # try calling without kwargs if signature mismatch
                try:
                    return fn(*a)
                except Exception:
                    raise

        try:
            # Preferred method: similarity_search_with_score
            if hasattr(self.cache_vector_store, "similarity_search_with_score"):
                fn = getattr(self.cache_vector_store, "similarity_search_with_score")
                sig = inspect.signature(fn)
                call_kwargs = {}
                if "k" in sig.parameters:
                    call_kwargs["k"] = k
                elif "top_k" in sig.parameters:
                    call_kwargs["top_k"] = k
                if "score_threshold" in sig.parameters and score_threshold is not None:
                    call_kwargs["score_threshold"] = score_threshold
                # run in thread if sync
                res = await asyncio.to_thread(call_sync, fn, query, **call_kwargs)
                # Expecting [(doc, score), ...] or [doc,...]
                if not res:
                    return []
                # Normalize: if entries are docs only, assign None as score
                normalized = []
                for item in res:
                    if isinstance(item, tuple) and len(item) >= 2:
                        normalized.append((item[0], item[1]))
                    else:
                        normalized.append((item, None))
                return normalized

            # Fallback: similarity_search (docs only)
            if hasattr(self.cache_vector_store, "similarity_search"):
                fn = getattr(self.cache_vector_store, "similarity_search")
                sig = inspect.signature(fn)
                call_kwargs = {}
                if "k" in sig.parameters:
                    call_kwargs["k"] = k
                elif "top_k" in sig.parameters:
                    call_kwargs["top_k"] = k
                res = await asyncio.to_thread(call_sync, fn, query, **call_kwargs)
                if not res:
                    return []
                return [(doc, None) for doc in res]

            # Fallback: hybrid (weaviate client variants)
            if hasattr(self.cache_vector_store, "hybrid"):
                fn = getattr(self.cache_vector_store, "hybrid")
                sig = inspect.signature(fn)
                call_kwargs = {}
                if "top_k" in sig.parameters:
                    call_kwargs["top_k"] = k
                res = await asyncio.to_thread(call_sync, fn, query, **call_kwargs)
                if not res:
                    return []
                # try to normalize possible return shapes
                normalized = []
                for item in res:
                    if isinstance(item, tuple) and len(item) >= 2:
                        normalized.append((item[0], item[1]))
                    else:
                        normalized.append((item, None))
                return normalized

        except Exception as e:
            logger.debug("Vector similarity search failed: %s", e)
            return []

        return []

    def _init_weaviate_and_embeddings(self):
        """Initialize Weaviate and embeddings in sync context."""
        # Initialize Weaviate client
        self.weaviate_client = weaviate.connect_to_local(
            host=settings.WEAVIATE_HOST, 
            port=settings.WEAVIATE_PORT
        )
        
        # Use pre-loaded embedding model from registry for performance optimization
        self.embedding_model = model_registry.get_embedding_model()
        if self.embedding_model is None:
            # Fallback to on-demand loading if registry not initialized
            logger.warning("Model registry not initialized for semantic cache, loading embedding model on-demand")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL
            )
        else:
            logger.debug("Semantic cache using pre-loaded embedding model from registry")
        
        # Create cache vector store
        self.cache_vector_store = WeaviateVectorStore(
            client=self.weaviate_client,
            index_name=settings.SEMANTIC_CACHE_INDEX_NAME,
            text_key="query_text",
            embedding=self.embedding_model,
        )

    async def _cleanup_failed_init(self):
        """Clean up resources after failed initialization."""
        try:
            if self.redis_client and AIOREDIS_AVAILABLE:
                await self.redis_client.close()
            if self.weaviate_client:
                await asyncio.to_thread(self.weaviate_client.close)
        except Exception as e:
            logger.debug(f"Error during cleanup: {e}")
        finally:
            self.redis_client = None
            self.weaviate_client = None
            self.cache_vector_store = None
            self.embedding_model = None
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent hashing."""
        return query.strip().lower()
    
    def _generate_query_hash(self, query: str) -> str:
        """Generate deterministic hash for query deduplication."""
        normalized_query = self._normalize_query(query)
        return hashlib.sha256(normalized_query.encode()).hexdigest()
    
    def _generate_cache_key(self, cache_id: str) -> str:
        """Generate Redis key for cache entry."""
        return f"cache_entry:{cache_id}"
    
    async def _is_cache_enabled(self) -> bool:
        """Check if semantic caching is enabled and properly initialized."""
        if not settings.ENABLE_SEMANTIC_CACHE:
            return False
            
        if not self._initialized:
            await self._initialize_clients()
            
        return (
            self._initialized and 
            self.redis_client is not None and 
            self.cache_vector_store is not None
        )
    
    @property
    def _lua_scripts(self):
        """Lua scripts for atomic Redis operations."""
        return {
            # Atomic cache entry addition with ZSET indexing
            'add_cache_entry': """
                local cache_key = KEYS[1]
                local index_key = KEYS[2] 
                local cache_id = ARGV[1]
                local entry_data = ARGV[2]
                local ttl = tonumber(ARGV[3])
                local created_ts = tonumber(ARGV[4])
                
                -- Set cache entry with TTL
                redis.call('SETEX', cache_key, ttl, entry_data)
                
                -- Add to sorted set index
                redis.call('ZADD', index_key, created_ts, cache_id)
                
                return 1
            """,
            
            # Atomic cache trimming with cross-store cleanup
            'trim_cache': """
                local index_key = KEYS[1]
                local max_size = tonumber(ARGV[1])
                
                -- Get current size
                local current_size = redis.call('ZCARD', index_key)
                if current_size <= max_size then
                    return {}
                end
                
                -- Get oldest entries to remove
                local to_remove = current_size - max_size
                local oldest_entries = redis.call('ZRANGE', index_key, 0, to_remove - 1)
                
                if #oldest_entries == 0 then
                    return {}
                end
                
                -- Remove from index and get cache keys to delete
                local cache_keys_to_delete = {}
                for i, cache_id in ipairs(oldest_entries) do
                    redis.call('ZREM', index_key, cache_id)
                    table.insert(cache_keys_to_delete, 'cache_entry:' .. cache_id)
                end
                
                -- Delete cache entries
                if #cache_keys_to_delete > 0 then
                    redis.call('DEL', unpack(cache_keys_to_delete))
                end
                
                return oldest_entries
            """,
            
            # Get cache stats efficiently 
            'get_cache_stats': """
                local index_key = KEYS[1]
                local entry_pattern = ARGV[1]
                
                local total_entries = redis.call('ZCARD', index_key)
                local sample_cache_ids = redis.call('ZRANGE', index_key, -10, -1)
                
                return {total_entries, sample_cache_ids}
            """
        }

    async def _execute_lua_script(self, script_name: str, keys: List[str], args: List[str]) -> Any:
        """Execute a Lua script atomically."""
        try:
            script = self._lua_scripts[script_name]
            if AIOREDIS_AVAILABLE:
                return await self.redis_client.eval(script, len(keys), *keys, *args)
            else:
                # Use sync Redis in thread pool
                return await asyncio.to_thread(
                    self.redis_client.eval, script, len(keys), *keys, *args
                )
        except Exception as e:
            logger.error(f"Error executing Lua script {script_name}: {e}")
            raise
    
    async def get_cached_answer(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached answer for semantically similar queries.
        
        Args:
            query: The user query to search for
            
        Returns:
            Cached answer data if found, None otherwise
        """
        if not await self._is_cache_enabled():
            return None
        
        try:
            # First check for exact query match using hash
            query_hash = self._generate_query_hash(query)
            exact_match_key = f"exact_match:{query_hash}"
            
            # Check Redis for exact match cache_id
            cache_id = None
            if AIOREDIS_AVAILABLE:
                cache_id = await self.redis_client.get(exact_match_key)
            else:
                cache_id = await asyncio.to_thread(self.redis_client.get, exact_match_key)
            
            if cache_id:
                # Exact match found, retrieve from cache
                cached_result = await self._get_cache_entry_by_id(cache_id)
                if cached_result:
                    logger.info(f"✅ Exact cache hit for query: {query[:50]}...")
                    return await self._update_cache_access(cache_id, cached_result)
            
            # No exact match, perform semantic search
            similar_docs = await self._vector_similarity_search(
                query, k=1, score_threshold=settings.SEMANTIC_CACHE_SIMILARITY_THRESHOLD
            )

            
            if not similar_docs:
                logger.debug(f"No cached answer found for query: {query[:50]}...")
                return None
            
            doc, similarity_score = similar_docs[0]
            
            # Validate similarity score orientation and threshold
            # Note: Different vector stores may return distance vs similarity
            # For most similarity metrics, higher is better; for distance, lower is better
            # We assume similarity_search_with_score returns similarity (0-1, higher is better)
            if similarity_score < settings.SEMANTIC_CACHE_SIMILARITY_THRESHOLD:
                logger.debug(f"Similarity {similarity_score:.3f} below threshold {settings.SEMANTIC_CACHE_SIMILARITY_THRESHOLD}")
                return None
            
            # Get cache entry ID from document metadata
            cache_id = doc.metadata.get("cache_id")
            if not cache_id:
                logger.warning("Found similar query but no cache_id in metadata")
                return None
            
            # Retrieve full cache entry from Redis
            cached_result = await self._get_cache_entry_by_id(cache_id)
            
            if not cached_result:
                logger.debug(f"Cache entry {cache_id} expired or not found in Redis")
                # Clean up orphaned vector entry
                doc_id = doc.metadata.get("doc_id")
                if doc_id:
                    await self._cleanup_orphaned_vector_entry(doc_id)
                return None
            
            logger.info(f"✅ Semantic cache hit for query: {query[:50]}... (similarity: {similarity_score:.3f})")
            return await self._update_cache_access(cache_id, cached_result)
            
        except Exception as e:
            logger.error(f"Error retrieving cached answer: {e}")
            return None

    async def _get_cache_entry_by_id(self, cache_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cache entry by ID from Redis."""
        try:
            cache_key = self._generate_cache_key(cache_id)
            
            if AIOREDIS_AVAILABLE:
                cached_data = await self.redis_client.get(cache_key)
            else:
                cached_data = await asyncio.to_thread(self.redis_client.get, cache_key)
            
            if cached_data:
                return json.loads(cached_data)
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving cache entry {cache_id}: {e}")
            return None

    async def _update_cache_access(self, cache_id: str, cache_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Update cache entry access metadata and extend TTL."""
        try:
            # Update access metadata
            cache_entry["last_accessed"] = datetime.now().isoformat()
            cache_entry["access_count"] = cache_entry.get("access_count", 0) + 1
            
            # Update in Redis with extended TTL
            cache_key = self._generate_cache_key(cache_id)
            
            if AIOREDIS_AVAILABLE:
                await self.redis_client.setex(
                    cache_key, 
                    settings.SEMANTIC_CACHE_TTL, 
                    json.dumps(cache_entry)
                )
            else:
                await asyncio.to_thread(
                    self.redis_client.setex,
                    cache_key, 
                    settings.SEMANTIC_CACHE_TTL, 
                    json.dumps(cache_entry)
                )
            
            return cache_entry
            
        except Exception as e:
            logger.error(f"Error updating cache access for {cache_id}: {e}")
            return cache_entry
    
    async def store_answer(self, query: str, answer: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Store a query-answer pair in the semantic cache with deduplication.
        
        Args:
            query: The user query
            answer: The generated answer
            metadata: Additional metadata (tokens, generation time, etc.)
            
        Returns:
            True if successfully cached, False otherwise
        """
        if not await self._is_cache_enabled():
            return False
        
        try:
            # Check for exact query match first (deduplication)
            query_hash = self._generate_query_hash(query)
            exact_match_key = f"exact_match:{query_hash}"
            
            # Check if exact match already exists
            existing_cache_id = None
            if AIOREDIS_AVAILABLE:
                existing_cache_id = await self.redis_client.get(exact_match_key)
            else:
                existing_cache_id = await asyncio.to_thread(self.redis_client.get, exact_match_key)
            
            if existing_cache_id:
                # Update existing entry instead of creating duplicate
                logger.info(f"Updating existing cache entry for duplicate query: {query[:50]}...")
                return await self._update_existing_cache_entry(existing_cache_id, answer, metadata)
            
            # Check for high semantic similarity to avoid near-duplicates
            similar_entries = await self._find_highly_similar_entries(query)
            if similar_entries:
                # Update most similar entry instead of creating new one
                cache_id, similarity = similar_entries[0]
                logger.info(f"Updating similar cache entry (sim: {similarity:.3f}) for query: {query[:50]}...")
                return await self._update_existing_cache_entry(cache_id, answer, metadata)
            
            # Generate unique IDs
            cache_id = str(uuid.uuid4())
            doc_id = str(uuid.uuid4())
            created_ts = int(time.time())
            
            # Prepare cache entry
            cache_entry = {
                "cache_id": cache_id,
                "doc_id": doc_id,
                "query": query,
                "query_hash": query_hash,
                "answer": answer,
                "created_at": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat(),
                "access_count": 0,
                "metadata": metadata or {}
            }
            
            # Store in Redis using atomic Lua script
            cache_key = self._generate_cache_key(cache_id)
            index_key = "cache_index"
            
            await self._execute_lua_script(
                'add_cache_entry',
                keys=[cache_key, index_key],
                args=[cache_id, json.dumps(cache_entry), str(settings.SEMANTIC_CACHE_TTL), str(created_ts)]
            )
            
            # Store exact match mapping
            if AIOREDIS_AVAILABLE:
                await self.redis_client.setex(exact_match_key, settings.SEMANTIC_CACHE_TTL, cache_id)
            else:
                await asyncio.to_thread(
                    self.redis_client.setex, exact_match_key, settings.SEMANTIC_CACHE_TTL, cache_id
                )
            
            # Store query vector in Weaviate for similarity search
            cache_doc = Document(
                page_content=query,
                metadata={
                    "cache_id": cache_id,
                    "doc_id": doc_id,
                    "query_hash": query_hash,
                    "created_at": cache_entry["created_at"],
                    "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer
                }
            )
            
            await asyncio.to_thread(self.cache_vector_store.add_documents, [cache_doc])
            
            # Manage cache size atomically
            await self._manage_cache_size_atomic()
            
            logger.info(f"Cached new answer for query: {query[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error storing answer in cache: {e}")
            return False

    async def _find_highly_similar_entries(self, query: str, similarity_threshold: float = 0.98) -> List[Tuple[str, float]]:
        """Find highly similar entries to avoid near-duplicates."""
        try:
            similar_docs = await self._vector_similarity_search(query, k=3, score_threshold=similarity_threshold)
            
            similar_entries = []
            for doc, score in similar_docs:
                cache_id = doc.metadata.get("cache_id")
                if cache_id and score >= similarity_threshold:
                    similar_entries.append((cache_id, score))
            
            return similar_entries
            
        except Exception as e:
            logger.debug(f"Error finding similar entries: {e}")
            return []

    async def _update_existing_cache_entry(self, cache_id: str, new_answer: str, new_metadata: Dict[str, Any] = None) -> bool:
        """Update an existing cache entry with new answer/metadata."""
        try:
            # Get existing entry
            existing_entry = await self._get_cache_entry_by_id(cache_id)
            if not existing_entry:
                return False
            
            # Update with new data
            existing_entry["answer"] = new_answer
            existing_entry["last_accessed"] = datetime.now().isoformat()
            existing_entry["access_count"] = existing_entry.get("access_count", 0) + 1
            
            if new_metadata:
                existing_entry["metadata"].update(new_metadata)
            
            # Store updated entry
            cache_key = self._generate_cache_key(cache_id)
            
            if AIOREDIS_AVAILABLE:
                await self.redis_client.setex(
                    cache_key, 
                    settings.SEMANTIC_CACHE_TTL, 
                    json.dumps(existing_entry)
                )
            else:
                await asyncio.to_thread(
                    self.redis_client.setex,
                    cache_key, 
                    settings.SEMANTIC_CACHE_TTL, 
                    json.dumps(existing_entry)
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating existing cache entry {cache_id}: {e}")
            return False
    
    async def _manage_cache_size_atomic(self):
        """Atomically manage cache size using Lua script for O(log N) performance."""
        try:
            index_key = "cache_index"
            evicted_cache_ids = await self._execute_lua_script(
                'trim_cache',
                keys=[index_key],
                args=[str(settings.SEMANTIC_CACHE_MAX_SIZE)]
            )
            
            if evicted_cache_ids:
                # Clean up corresponding Weaviate vectors
                await self._cleanup_weaviate_vectors(evicted_cache_ids)
                
                # Clean up exact match mappings
                await self._cleanup_exact_match_mappings(evicted_cache_ids)
                
                logger.info(f"Evicted {len(evicted_cache_ids)} cache entries to maintain size limit")
                
        except Exception as e:
            logger.error(f"Error managing cache size: {e}")

    async def _cleanup_weaviate_vectors(self, cache_ids: List[str]):
        """Clean up Weaviate vectors for evicted cache entries."""
        try:
            if not cache_ids:
                return
                
            # Delete vectors by cache_id filter
            # Note: This is a simplified approach. In production, you might want to
            # batch deletions or use more efficient Weaviate deletion methods
            for cache_id in cache_ids[:10]:  # Limit batch size
                try:
                    await asyncio.to_thread(
                        self._delete_weaviate_vector_by_cache_id, cache_id
                    )
                except Exception as e:
                    logger.debug(f"Error deleting vector for cache_id {cache_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up Weaviate vectors: {e}")

    def _delete_weaviate_vector_by_cache_id(self, cache_id: str):
        """Delete a specific vector from Weaviate by cache_id (sync method for thread pool)."""
        try:
            # This is a simplified deletion. In production, you'd use proper Weaviate deletion APIs
            # based on your Weaviate client version and schema
            collection = self.weaviate_client.collections.get(settings.SEMANTIC_CACHE_INDEX_NAME)
            
            # Delete by metadata filter
            collection.data.delete_many(
                where={"path": ["cache_id"], "operator": "Equal", "valueString": cache_id}
            )
        except Exception as e:
            logger.debug(f"Error deleting Weaviate vector for {cache_id}: {e}")

    async def _cleanup_exact_match_mappings(self, cache_ids: List[str]):
        """Clean up exact match hash mappings for evicted entries."""
        try:
            # Get the query hashes for these cache_ids to clean up exact match mappings
            # This is a simplified approach - in production you might store reverse mappings
            
            for cache_id in cache_ids:
                try:
                    # We can't easily reverse-lookup query_hash from cache_id after deletion,
                    # so we'll rely on Redis TTL for cleanup of exact_match keys
                    # Alternatively, you could store a reverse mapping cache_id -> query_hash
                    pass
                except Exception as e:
                    logger.debug(f"Error cleaning exact match mapping for {cache_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up exact match mappings: {e}")

    async def _start_background_gc(self):
        """Start background garbage collection task."""
        try:
            if hasattr(asyncio, 'create_task'):
                self._gc_task = asyncio.create_task(self._background_gc_loop())
            logger.debug("Started background garbage collection task")
        except Exception as e:
            logger.error(f"Error starting background GC: {e}")

    async def _background_gc_loop(self):
        """Background garbage collection loop."""
        gc_interval = getattr(settings, 'SEMANTIC_CACHE_GC_INTERVAL', 3600)  # 1 hour default
        
        while self._initialized and not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(gc_interval)
                await self._run_garbage_collection()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background GC loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _run_garbage_collection(self):
        """Run garbage collection to clean up orphaned vectors."""
        try:
            logger.info("Running garbage collection for semantic cache")
            
            # Get all cache_ids from Redis index
            index_key = "cache_index"
            
            if AIOREDIS_AVAILABLE:
                redis_cache_ids = set(await self.redis_client.zrange(index_key, 0, -1))
            else:
                redis_cache_ids = set(await asyncio.to_thread(self.redis_client.zrange, index_key, 0, -1))
            
            # Find orphaned vectors in Weaviate
            orphaned_vectors = await asyncio.to_thread(self._find_orphaned_vectors, redis_cache_ids)
            
            if orphaned_vectors:
                logger.info(f"Found {len(orphaned_vectors)} orphaned vectors, cleaning up...")
                for cache_id in orphaned_vectors:
                    await asyncio.to_thread(self._delete_weaviate_vector_by_cache_id, cache_id)
            
            logger.info("Garbage collection completed")
            
        except Exception as e:
            logger.error(f"Error during garbage collection: {e}")

    def _find_orphaned_vectors(self, redis_cache_ids: set) -> List[str]:
        """Find vectors in Weaviate that don't have corresponding Redis entries (sync method)."""
        try:
            # Get all cache_ids from Weaviate
            collection = self.weaviate_client.collections.get(settings.SEMANTIC_CACHE_INDEX_NAME)
            
            # This is a simplified approach. In production, you'd use proper pagination
            # and more efficient querying methods based on your Weaviate setup
            weaviate_docs = collection.query.fetch_objects(
                limit=1000,  # Adjust based on your cache size
                return_metadata=["cache_id"]
            )
            
            weaviate_cache_ids = set()
            for doc in weaviate_docs.objects:
                cache_id = doc.metadata.get("cache_id")
                if cache_id:
                    weaviate_cache_ids.add(cache_id)
            
            # Find orphaned cache_ids (in Weaviate but not in Redis)
            orphaned = weaviate_cache_ids - redis_cache_ids
            return list(orphaned)
            
        except Exception as e:
            logger.error(f"Error finding orphaned vectors: {e}")
            return []

    async def _cleanup_orphaned_vector_entry(self, doc_id: str):
        """Clean up orphaned vector entries that no longer have Redis counterparts."""
        try:
            if doc_id:
                await asyncio.to_thread(self._delete_weaviate_vector_by_doc_id, doc_id)
                logger.debug(f"Cleaned up orphaned vector entry: {doc_id}")
        except Exception as e:
            logger.error(f"Error cleaning up orphaned vector entry {doc_id}: {e}")

    def _delete_weaviate_vector_by_doc_id(self, doc_id: str):
        """Delete a Weaviate vector by doc_id (sync method for thread pool)."""
        try:
            collection = self.weaviate_client.collections.get(settings.SEMANTIC_CACHE_INDEX_NAME)
            collection.data.delete_many(
                where={"path": ["doc_id"], "operator": "Equal", "valueString": doc_id}
            )
        except Exception as e:
            logger.debug(f"Error deleting Weaviate vector by doc_id {doc_id}: {e}")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics efficiently using ZSET operations."""
        if not await self._is_cache_enabled():
            return {"enabled": False}
        
        try:
            # Use efficient Lua script for stats
            index_key = "cache_index"
            stats_result = await self._execute_lua_script(
                'get_cache_stats',
                keys=[index_key],
                args=["cache_entry:*"]
            )
            
            total_entries, sample_cache_ids = stats_result
            
            # Calculate access count from sample
            total_access_count = 0
            sample_size = len(sample_cache_ids)
            
            if sample_size > 0:
                for cache_id in sample_cache_ids:
                    try:
                        cache_entry = await self._get_cache_entry_by_id(cache_id)
                        if cache_entry:
                            total_access_count += cache_entry.get("access_count", 0)
                    except Exception as e:
                        logger.debug(f"Error sampling cache entry {cache_id}: {e}")
            
            avg_access_count = total_access_count / sample_size if sample_size > 0 else 0
            
            # Calculate additional stats
            fill_percentage = (total_entries / settings.SEMANTIC_CACHE_MAX_SIZE) * 100
            
            # Get Redis memory usage if available
            redis_memory_info = {}
            try:
                if AIOREDIS_AVAILABLE:
                    info = await self.redis_client.info("memory")
                else:
                    info = await asyncio.to_thread(self.redis_client.info, "memory")
                redis_memory_info = {
                    "used_memory": info.get("used_memory", 0),
                    "used_memory_human": info.get("used_memory_human", "unknown")
                }
            except Exception as e:
                logger.debug(f"Could not get Redis memory info: {e}")
            
            return {
                "enabled": True,
                "total_entries": total_entries,
                "max_size": settings.SEMANTIC_CACHE_MAX_SIZE,
                "fill_percentage": round(fill_percentage, 1),
                "ttl_seconds": settings.SEMANTIC_CACHE_TTL,
                "similarity_threshold": settings.SEMANTIC_CACHE_SIMILARITY_THRESHOLD,
                "avg_access_count": round(avg_access_count, 2),
                "sample_size": sample_size,
                "redis_memory": redis_memory_info,
                "background_gc_enabled": self._gc_task is not None and not self._gc_task.done(),
                "async_redis": AIOREDIS_AVAILABLE
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"enabled": True, "error": str(e)}

    async def clear_cache(self) -> bool:
        """Clear all cache entries atomically."""
        if not await self._is_cache_enabled():
            return False
        
        try:
            # Get all cache entries from index
            index_key = "cache_index"
            
            if AIOREDIS_AVAILABLE:
                all_cache_ids = await self.redis_client.zrange(index_key, 0, -1)
            else:
                all_cache_ids = await asyncio.to_thread(self.redis_client.zrange, index_key, 0, -1)
            
            if not all_cache_ids:
                logger.info("Cache is already empty")
                return True
            
            # Clear Redis entries
            cache_keys = [self._generate_cache_key(cache_id) for cache_id in all_cache_ids]
            exact_match_keys = await self._get_exact_match_keys()
            
            all_keys_to_delete = cache_keys + exact_match_keys + [index_key]
            
            if AIOREDIS_AVAILABLE:
                if all_keys_to_delete:
                    await self.redis_client.delete(*all_keys_to_delete)
            else:
                if all_keys_to_delete:
                    await asyncio.to_thread(self.redis_client.delete, *all_keys_to_delete)
            
            # Clear Weaviate collection
            await asyncio.to_thread(self._clear_weaviate_collection)
            
            logger.info(f"✅ Cleared {len(all_cache_ids)} cache entries")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    async def _get_exact_match_keys(self) -> List[str]:
        """Get all exact match keys for cleanup."""
        try:
            pattern = "exact_match:*"
            if AIOREDIS_AVAILABLE:
                return await self.redis_client.keys(pattern)
            else:
                return await asyncio.to_thread(self.redis_client.keys, pattern)
        except Exception as e:
            logger.debug(f"Error getting exact match keys: {e}")
            return []

    def _clear_weaviate_collection(self):
        """Clear the Weaviate cache collection (sync method for thread pool)."""
        try:
            collection = self.weaviate_client.collections.get(settings.SEMANTIC_CACHE_INDEX_NAME)
            
            # Delete all objects in the collection
            collection.data.delete_many(where={})
            
            logger.info("Cleared Weaviate cache collection")
            
        except Exception as e:
            logger.warning(f"Error clearing Weaviate collection: {e}")

    async def shutdown(self):
        """Gracefully shutdown the semantic cache."""
        try:
            logger.info("Shutting down semantic cache...")
            
            # Signal shutdown to background tasks
            if self._shutdown_event:
                self._shutdown_event.set()
            
            # Cancel background GC task
            if self._gc_task and not self._gc_task.done():
                self._gc_task.cancel()
                try:
                    await self._gc_task
                except asyncio.CancelledError:
                    pass
            
            # Close Redis connection
            if self.redis_client and AIOREDIS_AVAILABLE:
                await self.redis_client.close()
            
            # Close Weaviate connection
            if self.weaviate_client:
                await asyncio.to_thread(self.weaviate_client.close)
            
            self._initialized = False
            logger.info("Semantic cache shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during cache shutdown: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on cache components."""
        health = {
            "enabled": settings.ENABLE_SEMANTIC_CACHE,
            "initialized": self._initialized,
            "redis_healthy": False,
            "weaviate_healthy": False,
            "background_gc_running": False
        }
        
        if not await self._is_cache_enabled():
            return health
        
        # Check Redis health
        try:
            if AIOREDIS_AVAILABLE:
                await self.redis_client.ping()
            else:
                await asyncio.to_thread(self.redis_client.ping)
            health["redis_healthy"] = True
        except Exception as e:
            health["redis_error"] = str(e)
        
        # Check Weaviate health
        try:
            await asyncio.to_thread(self.weaviate_client.is_ready)
            health["weaviate_healthy"] = True
        except Exception as e:
            health["weaviate_error"] = str(e)
        
        # Check background GC
        health["background_gc_running"] = (
            self._gc_task is not None and not self._gc_task.done()
        )
        
        return health


# Global semantic cache instance with lazy initialization
_semantic_cache_instance = None
_cache_lock = threading.Lock()

def get_semantic_cache() -> SemanticCache:
    """Get the global semantic cache instance with thread-safe lazy initialization."""
    global _semantic_cache_instance
    
    if _semantic_cache_instance is None:
        with _cache_lock:
            if _semantic_cache_instance is None:
                _semantic_cache_instance = SemanticCache()
    
    return _semantic_cache_instance

# Maintain backward compatibility
semantic_cache = get_semantic_cache()