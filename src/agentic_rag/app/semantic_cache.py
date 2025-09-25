# src/agentic_rag/app/semantic_cache.py

import hashlib
import json
import asyncio
import time
import uuid
import math
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
from weaviate.collections.classes.filters import Filter

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

    @staticmethod
    def _lexical_similarity(text1: str, text2: str) -> float:
        """Simple token Jaccard overlap for guard-rail decisions."""
        t1 = set((text1 or "").lower().split())
        t2 = set((text2 or "").lower().split())
        if not t1 or not t2:
            return 0.0
        inter = len(t1 & t2)
        union = len(t1 | t2)
        return inter / union if union else 0.0
    
    async def _ce_similarity(self, text1: str, text2: str) -> float:
        """
        Cross-encoder pairwise similarity in [0,1]. Loaded lazily via model_registry.
        Runs in a thread to avoid blocking the event loop.
        """
        try:
            ce = model_registry.get_cross_encoder_guard()
            if ce is None:
                ce = await model_registry.ensure_cross_encoder_guard()
            if ce is None:
                return -1.0

            def _predict():
                # sentence-transformers CrossEncoder expects list of (a,b) pairs
                scores = ce.predict([(text1 or "", text2 or "")])
                s = float(scores[0])
                # normalize to [0,1] if raw logit-like
                if s < 0.0 or s > 1.0:
                    try:
                        return 1.0 / (1.0 + math.exp(-s))
                    except OverflowError:
                        return 0.0 if s < 0 else 1.0
                return s

            return await asyncio.to_thread(_predict)
        except Exception:
            return -1.0
    
    def _embedding_similarity(self, text1: str, text2: str) -> float:
        """
        DEPRECATED: Cosine similarity of query embeddings in [0,1].
        This method has been deprecated due to unreliable results.
        The semantic cache now uses only cross-encoder and lexical similarity.
        """
        logger.warning("_embedding_similarity is deprecated and should not be used")
        return 0.0

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
    
    def _normalize_similarity_score(self, score: float | None) -> float | None:
        """
        Normalize backend score to similarity in [0,1].
        
        IMPORTANT: There appears to be an issue with the Weaviate langchain integration
        where it may return similarity scores instead of distance scores, despite the
        documentation stating it returns cosine distances.
        
        Based on user testing:
        - Exact matches return raw score 1.0 but should be distance 0.0
        - This suggests the scores are actually similarities, not distances
        """
        if score is None:
            return None
        mode = getattr(settings, "SEMANTIC_CACHE_SCORE_MODE", "similarity").lower()
        try:
            s = float(score)
        except Exception:
            return None
        
        logger.debug(f"Normalizing score: raw={s}, mode={mode}")
        
        if mode == "distance":
            # INVESTIGATION FINDINGS: The Weaviate langchain integration appears to return
            # similarity scores (where 1.0 = perfect match) despite claiming to return
            # distance scores. This is evidenced by:
            # 1. Exact matches returning raw score 1.0 (should be distance 0.0)
            # 2. The similarity correlates with embedding similarity, not distance
            
            # Detect if we're actually getting similarity scores disguised as distances
            if s >= 0.95:  # Very high "distance" suggests it's actually similarity
                logger.info(f"High raw score {s} detected - treating as similarity score instead of distance")
                return max(0.0, min(1.0, s))  # Treat as similarity directly
            elif s <= 0.1:  # Very low score - likely actual distance  
                return 1.0 - s  # Convert distance to similarity
            else:
                # Ambiguous range - use original distance formula but log warning
                logger.warning(f"Ambiguous score {s} - using distance interpretation")
                s_clamped = max(0.0, min(2.0, s))
                return 1.0 - (s_clamped / 2.0)
        else:
            # Similarity mode - clamp to [0,1]
            return max(0.0, min(1.0, s))

    async def _vector_similarity_search(self, query: str, k: int = 1):
        """
        Robust wrapper around the underlying vector store search API.
        Returns list of (doc, similarity_in_[0,1]) where higher is better.
        """
        if not self.cache_vector_store:
            return []

        try:
            if hasattr(self.cache_vector_store, "similarity_search_with_score"):
                results = await asyncio.to_thread(
                    self.cache_vector_store.similarity_search_with_score, query, k=k
                )
                if not results:
                    return []
                
                logger.debug(f"Raw vector search results for query '{query[:50]}...': {results}")
                
                # DEBUG: Compare vectors for identical queries
                if results and len(results) > 0:
                    doc, raw_score = results[0]
                    if doc.page_content.strip() == query.strip():
                        # Identical queries - investigate why distance is not 0.0
                        logger.warning(f"IDENTICAL QUERY VECTOR DEBUG: '{query}' vs '{doc.page_content}' raw_score={raw_score}")
                        
                        # Get the embedding that would be generated for this query
                        try:
                            query_embedding = self.embedding_model.embed_query(query)
                            logger.debug(f"Query embedding dimensions: {len(query_embedding) if query_embedding else None}")
                            logger.debug(f"Query embedding preview: {query_embedding[:5] if query_embedding else None}")
                        except Exception as e:
                            logger.error(f"Failed to generate embedding for debugging: {e}")
                
                normalized = []
                for item in results:
                    if isinstance(item, tuple) and len(item) >= 2:
                        doc, raw = item[0], item[1]
                        sim = self._normalize_similarity_score(raw)
                        logger.info(f"Vector search: query='{query[:30]}...' cached='{doc.page_content[:30]}...' raw_score={raw} normalized={sim}")
                        normalized.append((doc, sim))
                    else:
                        normalized.append((item, None))
                return normalized

            if hasattr(self.cache_vector_store, "similarity_search"):
                logger.debug("Falling back to similarity_search (scores will be unavailable)")
                results = await asyncio.to_thread(self.cache_vector_store.similarity_search, query, k=k)
                if not results:
                    return []
                return [(doc, None) for doc in results]

        except Exception as e:
            logger.error(f"Vector similarity search failed: {e}", exc_info=True)
            return []

        logger.warning("No suitable vector search method found on the cache_vector_store.")
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
        Retrieve a cached answer for a given query. It first checks for an
        exact match via hashing for maximum speed. If no exact match is found,
        it performs a semantic search for the most similar query.
        """
        if not await self._is_cache_enabled():
            return None
        
        try:
            # 1. Check for an exact query match first.
            query_hash = self._generate_query_hash(query)
            exact_match_key = f"exact_match:{query_hash}"
            
            cache_id = None
            if AIOREDIS_AVAILABLE:
                cache_id = await self.redis_client.get(exact_match_key)
            else:
                cache_id = await asyncio.to_thread(self.redis_client.get, exact_match_key)
            
            if cache_id:
                cached_result = await self._get_cache_entry_by_id(cache_id)
                if cached_result:
                    logger.info(f"Exact cache hit for query: {query[:50]}...")
                    return await self._update_cache_access(cache_id, cached_result)
            
            # 2. No exact match, so perform a semantic search.
            # We fetch the single best candidate and then verify its score.
            similar_docs = await self._vector_similarity_search(query, k=1)
            
            if not similar_docs:
                logger.debug(f"No similar documents found in cache for query: {query[:50]}...")
                return None
            
            doc, similarity_score = similar_docs[0]

            # 3. Validate the result using stricter, tiered rules
            if similarity_score is None:
                logger.warning("Similarity search returned a document but no score. Cannot validate cache hit.")
                return None

            cached_query = doc.page_content or ""

            # Use configured thresholds for similarity validation
            # SIMPLIFIED APPROACH: Use only cross-encoder and lexical similarity
            # Vector similarity and embedding similarity have proven unreliable
            ce_min = float(getattr(settings, "SEMANTIC_CACHE_CE_ACCEPT", 0.60))
            lex_min = float(getattr(settings, "SEMANTIC_CACHE_LEXICAL_MIN", 0.15))

            accept = False
            reason = ""

            # Compute semantic similarities
            ce_sim = await self._ce_similarity(query, cached_query)
            lex = self._lexical_similarity(query, cached_query)

            # Simplified rules using only reliable similarity measures
            # Rule 1: High cross-encoder similarity (most reliable semantic measure)
            if ce_sim >= 0.85:
                accept = True
                reason = f"high cross-encoder similarity (ce={ce_sim:.3f})"
            # Rule 2: Good cross-encoder with lexical support
            elif ce_sim >= ce_min and lex >= lex_min:
                accept = True
                reason = f"cross-encoder & lexical support (ce={ce_sim:.3f}, lex={lex:.2f})"
            # Rule 3: Very high lexical similarity (likely paraphrases)
            elif lex >= 0.4:
                accept = True
                reason = f"high lexical similarity (lex={lex:.2f})"

            if not accept:
                logger.info(
                    f"Cache miss. ce={ce_sim:.3f}, lex={lex:.2f} did not meet acceptance rules"
                )
                return None

            cache_id = str(doc.metadata.get("cache_id"))
            if not cache_id:
                logger.warning("Found similar document but it is missing a cache_id in its metadata.")
                return None

            # 4. Fetch entry and alias on true hit
            cached_result = await self._get_cache_entry_by_id(cache_id)
            if not cached_result:
                logger.warning(f"Cache entry {cache_id} found in vector index but not in Redis. Cleaning up orphan.")
                doc_id = doc.metadata.get("doc_id")
                if doc_id:
                    await self._cleanup_orphaned_vector_entry(doc_id)
                return None

            logger.info(f"Semantic cache hit ({reason}) for query: {query[:50]}... (sim: {similarity_score:.3f})")
            cached_result["similarity"] = similarity_score
            updated = await self._update_cache_access(cache_id, cached_result)

            # Alias only on very high-confidence hits to avoid poisoning
            try:
                if reason.startswith("vector>=") and similarity_score >= (vec_accept + 0.01):
                    qh = self._generate_query_hash(query)
                    exact_key = f"exact_match:{qh}"
                    ttl = int(getattr(settings, "SEMANTIC_CACHE_TTL", 3600))
                    if AIOREDIS_AVAILABLE:
                        await self.redis_client.setex(exact_key, ttl, cache_id)
                    else:
                        await asyncio.to_thread(self.redis_client.setex, exact_key, ttl, cache_id)
            except Exception as e:
                logger.debug(f"Failed to set exact-match alias for semantic hit: {e}")

            return updated
            
        except Exception as e:
            logger.error(f"Error retrieving cached answer: {e}", exc_info=True)
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
        Store a query-answer pair in the semantic cache. It first checks for
        exact duplicates via hashing. If none are found, it checks for highly
        similar entries to update instead of creating a new one.
        
        Args:
            query: The user query.
            answer: The generated answer.
            metadata: Additional metadata to store with the entry.
            
        Returns:
            True if successfully cached, False otherwise.
        """
        if not await self._is_cache_enabled():
            return False
        
        try:
            ttl_seconds = int(getattr(settings, "SEMANTIC_CACHE_TTL", 3600))

            # 1. Check for an exact query match first for deduplication
            query_hash = self._generate_query_hash(query)
            exact_match_key = f"exact_match:{query_hash}"
            
            existing_cache_id = None
            if AIOREDIS_AVAILABLE:
                existing_cache_id = await self.redis_client.get(exact_match_key)
            else:
                existing_cache_id = await asyncio.to_thread(self.redis_client.get, exact_match_key)
            
            if existing_cache_id:
                logger.info(f"Updating existing entry due to exact query match: {query[:50]}...")
                return await self._update_existing_cache_entry(existing_cache_id, query, answer, metadata)

            # 2. Check for a high semantic similarity match to update an existing entry
            # This makes the cache smarter by consolidating very similar questions.
            similar_entries = await self._find_highly_similar_entries(query)
            if similar_entries:
                cache_id, similarity = similar_entries[0]
                logger.info(f"Updating similar cache entry (sim: {similarity:.3f}) for query: {query[:50]}...")
                return await self._update_existing_cache_entry(cache_id, query, answer, metadata)
            
            # 3. If no similar entry is found, create a new one
            return await self._create_new_cache_entry(query, answer, metadata)
            
        except Exception as e:
            logger.error(f"Error storing answer in cache: {e}", exc_info=True)
            return False

    async def _find_highly_similar_entries(self, query: str) -> List[Tuple[str, float]]:
        """
        For the store path, be conservative to avoid wrong merges.
        """
        try:
            similar_docs = await self._vector_similarity_search(query, k=1)
            out: List[Tuple[str, float]] = []
            if not similar_docs:
                return out

            doc, score = similar_docs[0]
            cache_id = doc.metadata.get("cache_id")
            if score is None or not cache_id:
                return out

            vec_accept = float(getattr(settings, "SEMANTIC_CACHE_VECTOR_ACCEPT", 0.97))
            vec_min = float(getattr(settings, "SEMANTIC_CACHE_VECTOR_MIN", 0.90))
            ce_min = float(getattr(settings, "SEMANTIC_CACHE_CE_ACCEPT", 0.85))
            emb_min = float(getattr(settings, "SEMANTIC_CACHE_EMB_ACCEPT", 0.88))

            accept = False
            if score >= vec_accept:
                accept = True
            else:
                ce_sim = await self._ce_similarity(query, doc.page_content or "")
                emb_sim = await asyncio.to_thread(self._embedding_similarity, query, doc.page_content or "")
                # For merges, demand BOTH ce and emb support in addition to vector >= vec_min
                if score >= vec_min and ce_sim >= ce_min and emb_sim >= emb_min:
                    accept = True

            if accept:
                out.append((str(cache_id), float(score)))
            return out
        except Exception as e:
            logger.debug(f"Error finding similar entries: {e}")
            return []

    async def _update_existing_cache_entry(self, cache_id: str, query: str, new_answer: str, new_metadata: Dict[str, Any] = None) -> bool:
        """
        Update an existing cache entry. If the Redis entry is missing (orphaned vector),
        upsert a fresh entry for the current query and clean up the orphaned vector.
        """
        try:
            ttl_seconds = int(getattr(settings, "SEMANTIC_CACHE_TTL", 3600))
            existing_entry = await self._get_cache_entry_by_id(cache_id)

            if not existing_entry:
                # Orphaned vector: remove and create a fresh entry
                logger.info(f"Orphaned vector detected for cache_id={cache_id} during update; re-inserting fresh entry.")
                try:
                    await self._cleanup_weaviate_vectors([cache_id])
                except Exception:
                    logger.debug("Best-effort cleanup of orphaned vector failed for cache_id=%s", cache_id)
                return await self._create_new_cache_entry(query, new_answer, new_metadata)

            # Normalize metadata container
            if not isinstance(existing_entry.get("metadata"), dict):
                existing_entry["metadata"] = {}

            # Apply updates
            existing_entry["answer"] = new_answer
            existing_entry["last_accessed"] = datetime.now().isoformat()
            existing_entry["access_count"] = existing_entry.get("access_count", 0) + 1
            if new_metadata:
                try:
                    existing_entry["metadata"].update(new_metadata)
                except Exception:
                    existing_entry["metadata"] = new_metadata

            # Write back to Redis and refresh ZSET position
            cache_key = self._generate_cache_key(cache_id)
            payload = json.dumps(existing_entry)

            if AIOREDIS_AVAILABLE:
                await self.redis_client.setex(cache_key, ttl_seconds, payload)
                # refresh ZSET score to keep hot entries
                try:
                    await self.redis_client.zadd("cache_index", {"%s" % cache_id: int(time.time())})
                except TypeError:
                    # some clients expect (key, score, member)
                    await self.redis_client.zadd("cache_index", int(time.time()), cache_id)
            else:
                await asyncio.to_thread(self.redis_client.setex, cache_key, ttl_seconds, payload)
                try:
                    await asyncio.to_thread(self.redis_client.zadd, "cache_index", {cache_id: int(time.time())})
                except TypeError:
                    await asyncio.to_thread(self.redis_client.zadd, "cache_index", int(time.time()), cache_id)

            # Ensure exact-match mapping is set for this query
            qhash = self._generate_query_hash(query)
            exact_key = f"exact_match:{qhash}"
            if AIOREDIS_AVAILABLE:
                await self.redis_client.setex(exact_key, ttl_seconds, cache_id)
            else:
                await asyncio.to_thread(self.redis_client.setex, exact_key, ttl_seconds, cache_id)

            return True

        except Exception as e:
            logger.error(f"Error updating existing cache entry {cache_id}: {e}")
            return False

    async def _create_new_cache_entry(self, query: str, answer: str, metadata: Dict[str, Any] | None) -> bool:
        """
        Create a fresh cache entry and vector doc (used by upsert fallback when update fails).
        """
        try:
            ttl_seconds = int(getattr(settings, "SEMANTIC_CACHE_TTL", 3600))
            cache_id = str(uuid.uuid4())
            doc_id = str(uuid.uuid4())
            created_ts = int(time.time())
            query_hash = self._generate_query_hash(query)

            entry = {
                "cache_id": cache_id,
                "doc_id": doc_id,
                "query": query,
                "query_hash": query_hash,
                "answer": answer,
                "created_at": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat(),
                "access_count": 0,
                "metadata": metadata or {},
            }

            cache_key = self._generate_cache_key(cache_id)
            await self._execute_lua_script(
                "add_cache_entry",
                keys=[cache_key, "cache_index"],
                args=[cache_id, json.dumps(entry), str(ttl_seconds), str(created_ts)],
            )

            # exact match mapping
            if AIOREDIS_AVAILABLE:
                await self.redis_client.setex(f"exact_match:{query_hash}", ttl_seconds, cache_id)
            else:
                await asyncio.to_thread(self.redis_client.setex, f"exact_match:{query_hash}", ttl_seconds, cache_id)

            # add vector
            cache_doc = Document(
                page_content=query,
                metadata={"cache_id": cache_id, "doc_id": doc_id, "query_hash": query_hash, "created_at": entry["created_at"],
                          "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer}
            )
            await asyncio.to_thread(self.cache_vector_store.add_documents, [cache_doc])

            # enforce size
            await self._manage_cache_size_atomic()
            logger.info("Upserted new cache entry for query: %s", query[:50])
            return True
        except Exception as e:
            logger.error("Error creating new cache entry (upsert): %s", e)
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
                where=Filter.by_property("cache_id").equal(cache_id)
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
        
        while self._initialized and not (self._shutdown_event and self._shutdown_event.is_set()):
            try:
                await asyncio.sleep(gc_interval)
                await self._run_garbage_collection()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background GC loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _run_garbage_collection(self) -> int:
        """
        Run garbage collection to clean up expired Redis index entries and orphaned vectors.
        Returns the total number of cleaned items (Redis index + Weaviate orphans).
        """
        try:
            logger.info("Running garbage collection for semantic cache")
            
            # Part 1: Clean up stale entries from the Redis cache_index
            index_key = "cache_index"
            if AIOREDIS_AVAILABLE:
                all_indexed_ids = await self.redis_client.zrange(index_key, 0, -1)
            else:
                all_indexed_ids = await asyncio.to_thread(self.redis_client.zrange, index_key, 0, -1)
            
            stale_index_ids = []
            active_redis_ids = set()

            if all_indexed_ids:
                # Use a pipeline for efficient checking
                if AIOREDIS_AVAILABLE:
                    pipe = self.redis_client.pipeline()
                    for cache_id in all_indexed_ids:
                        pipe.exists(self._generate_cache_key(cache_id))
                    exists_results = await pipe.execute()
                else:
                    # Sync pipeline execution in thread
                    def sync_pipeline_exists(keys):
                        pipe = self.redis_client.pipeline()
                        for key in keys:
                            pipe.exists(key)
                        return pipe.execute()
                    
                    batch_keys = [self._generate_cache_key(id) for id in all_indexed_ids]
                    exists_results = await asyncio.to_thread(sync_pipeline_exists, batch_keys)

                for i, exists in enumerate(exists_results):
                    if exists:
                        active_redis_ids.add(all_indexed_ids[i])
                    else:
                        stale_index_ids.append(all_indexed_ids[i])
            
            # Remove all stale entries from the index at once
            if stale_index_ids:
                logger.info(f"Found {len(stale_index_ids)} stale entries in Redis index, cleaning up...")
                if AIOREDIS_AVAILABLE:
                    await self.redis_client.zrem(index_key, *stale_index_ids)
                else:
                    await asyncio.to_thread(self.redis_client.zrem, index_key, *stale_index_ids)
            
            # Part 2: Clean up orphaned vectors in Weaviate
            orphaned_vectors = await asyncio.to_thread(self._find_orphaned_vectors, active_redis_ids)
            
            cleaned_count = len(stale_index_ids) + len(orphaned_vectors)

            if orphaned_vectors:
                logger.info(f"Found {len(orphaned_vectors)} orphaned vectors, cleaning up...")
                for cache_id in orphaned_vectors:
                    await asyncio.to_thread(self._delete_weaviate_vector_by_cache_id, cache_id)
            
            if cleaned_count > 0:
                logger.info(f"Garbage collection completed. Cleaned {cleaned_count} total items.")
            else:
                logger.info("Garbage collection completed, no stale items or orphans found.")
                
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error during garbage collection: {e}")
            return 0

    async def run_garbage_collection_manually(self) -> int:
        """Manually triggers a single run of the garbage collection process."""
        if not await self._is_cache_enabled():
            logger.error("Cannot run GC: Cache is not enabled or initialized.")
            return 0
        return await self._run_garbage_collection()

    def _find_orphaned_vectors(self, redis_cache_ids: set) -> List[str]:
        """
        Find vectors in Weaviate that don't have corresponding Redis entries.
        This is a sync method for the thread pool and iterates through all objects.
        """
        try:
            collection = self.weaviate_client.collections.get(settings.SEMANTIC_CACHE_INDEX_NAME)
            weaviate_cache_ids = set()
            
            # FIX: Iterate through all objects in the collection, not just the first 1000
            for item in collection.iterator(include_vector=False, return_properties=["cache_id"]):
                cache_id = item.properties.get("cache_id")
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
                where=Filter.by_property("doc_id").equal(doc_id)
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
            index_key = "cache_index"
            if AIOREDIS_AVAILABLE:
                all_cache_ids = await self.redis_client.zrange(index_key, 0, -1)
            else:
                all_cache_ids = await asyncio.to_thread(self.redis_client.zrange, index_key, 0, -1)
            
            # Always attempt to clear Weaviate (handles orphan vectors too)
            await asyncio.to_thread(self._clear_weaviate_collection)

            # If nothing in Redis index, still return success after Weaviate clear
            if not all_cache_ids:
                logger.info("Cache index empty; Weaviate collection cleared (if any).")
                # Best-effort memory purge to release allocator memory
                try:
                    if AIOREDIS_AVAILABLE:
                        await self.redis_client.execute_command("MEMORY", "PURGE")
                    else:
                        await asyncio.to_thread(self.redis_client.execute_command, "MEMORY", "PURGE")
                except Exception as e:
                    logger.debug(f"Redis MEMORY PURGE not available: {e}")
                return True

            # Clear Redis entries
            cache_keys = [self._generate_cache_key(cache_id) for cache_id in all_cache_ids]
            exact_match_keys = await self._get_exact_match_keys()
            all_keys_to_delete = cache_keys + exact_match_keys + [index_key]

            if all_keys_to_delete:
                # Prefer DEL; UNLINK if you want async-freeing (DEL + PURGE reclaims faster)
                if AIOREDIS_AVAILABLE:
                    await self.redis_client.delete(*all_keys_to_delete)
                else:
                    await asyncio.to_thread(self.redis_client.delete, *all_keys_to_delete)

            # Best-effort memory purge (may not immediately reflect in used_memory_human)
            try:
                if AIOREDIS_AVAILABLE:
                    await self.redis_client.execute_command("MEMORY", "PURGE")
                else:
                    await asyncio.to_thread(self.redis_client.execute_command, "MEMORY", "PURGE")
            except Exception as e:
                logger.debug(f"Redis MEMORY PURGE not available: {e}")

            logger.info(f"✅ Cleared {len(all_cache_ids)} cache entries (Redis + Weaviate)")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    async def _get_exact_match_keys(self) -> List[str]:
        """Get all exact match keys for cleanup."""
        try:
            pattern = "exact_match:*"
            if AIOREDIS_AVAILABLE:
                # Use scan for production-safe key iteration
                keys = []
                async for key in self.redis_client.scan_iter(pattern):
                    keys.append(key)
                return keys
            else:
                # Sync scan
                return await asyncio.to_thread(
                    lambda: list(self.redis_client.scan_iter(pattern))
                )
        except Exception as e:
            logger.debug(f"Error getting exact match keys: {e}")
            return []

    def _clear_weaviate_collection(self):
        """
        Clear the Weaviate cache collection by deleting all objects that have cache_id.
        Robust against filter quirks by iterating and deleting per cache_id.
        """
        try:
            collection = self.weaviate_client.collections.get(settings.SEMANTIC_CACHE_INDEX_NAME)

            # Collect all cache_ids first
            cache_ids: list[str] = []
            for item in collection.iterator(include_vector=False, return_properties=["cache_id"]):
                cid = item.properties.get("cache_id")
                if cid:
                    cache_ids.append(str(cid))

            if not cache_ids:
                logger.info("Weaviate cache collection already empty.")
                return

            # Delete per cache_id (robust; small N is typical for cache)
            deleted, failed = 0, 0
            for cid in cache_ids:
                try:
                    res = collection.data.delete_many(where=Filter.by_property("cache_id").equal(cid))
                    # Some client versions return dict-like results; be tolerant
                    ok = getattr(res, "successful", None)
                    if ok is None:
                        deleted += 1  # assume success if no error thrown
                    else:
                        deleted += int(ok)
                        failed += int(getattr(res, "failed", 0) or 0)
                except Exception as e:
                    failed += 1
                    logger.debug(f"Error deleting Weaviate vector for cache_id={cid}: {e}")

            logger.info(f"Cleared Weaviate cache collection. Results: {deleted} successful, {failed} failed.")
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