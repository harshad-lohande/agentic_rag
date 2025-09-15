# src/agentic_rag/app/semantic_cache.py

import hashlib
import json
import redis
import weaviate
import uuid
from typing import Optional, Dict, Any
from datetime import datetime
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore

from agentic_rag.config import settings
from agentic_rag.logging_config import logger


class SemanticCache:
    """
    Semantic caching system that stores query-answer pairs based on semantic similarity.
    
    Uses a hybrid approach:
    1. Redis for fast metadata storage and TTL management
    2. Weaviate for semantic similarity search of queries
    """
    
    def __init__(self):
        self.redis_client = None
        self.weaviate_client = None
        self.cache_vector_store = None
        self.embedding_model = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize Redis and Weaviate clients for caching."""
        try:
            # Initialize Redis client
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST, 
                port=6379, 
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Connected to Redis for semantic caching")
            
            # Initialize Weaviate client for cache vector store
            self.weaviate_client = weaviate.connect_to_local(
                host=settings.WEAVIATE_HOST, 
                port=settings.WEAVIATE_PORT
            )
            
            # Initialize embedding model
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL
            )
            
            # Create cache vector store
            self.cache_vector_store = WeaviateVectorStore(
                client=self.weaviate_client,
                index_name=settings.SEMANTIC_CACHE_INDEX_NAME,
                text_key="query_text",
                embedding=self.embedding_model,
            )
            
            logger.info("Semantic cache initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize semantic cache: {e}")
            self.redis_client = None
            self.weaviate_client = None
            self.cache_vector_store = None
    
    def _generate_cache_key(self, query: str) -> str:
        """Generate a unique cache key for a query."""
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        return f"cache:{query_hash}"
    
    def _is_cache_enabled(self) -> bool:
        """Check if semantic caching is enabled and properly initialized."""
        return (
            settings.ENABLE_SEMANTIC_CACHE and 
            self.redis_client is not None and 
            self.cache_vector_store is not None
        )
    
    async def get_cached_answer(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached answer for semantically similar queries.
        
        Args:
            query: The user query to search for
            
        Returns:
            Cached answer data if found, None otherwise
        """
        if not self._is_cache_enabled():
            return None
        
        try:
            # Search for semantically similar queries
            similar_docs = self.cache_vector_store.similarity_search_with_score(
                query, 
                k=1,
                score_threshold=settings.SEMANTIC_CACHE_SIMILARITY_THRESHOLD
            )
            
            if not similar_docs:
                logger.debug(f"No cached answer found for query: {query[:50]}...")
                return None
            
            doc, similarity_score = similar_docs[0]
            
            # Check if similarity meets threshold
            if similarity_score < settings.SEMANTIC_CACHE_SIMILARITY_THRESHOLD:
                logger.debug(f"Similarity {similarity_score} below threshold {settings.SEMANTIC_CACHE_SIMILARITY_THRESHOLD}")
                return None
            
            # Get cache entry ID from document metadata
            cache_id = doc.metadata.get("cache_id")
            if not cache_id:
                logger.warning("Found similar query but no cache_id in metadata")
                return None
            
            # Retrieve full cache entry from Redis
            cache_key = f"cache_entry:{cache_id}"
            cached_data = self.redis_client.get(cache_key)
            
            if not cached_data:
                logger.debug(f"Cache entry {cache_id} expired or not found in Redis")
                # Clean up orphaned vector entry
                await self._cleanup_orphaned_vector_entry(doc.metadata.get("doc_id"))
                return None
            
            cache_entry = json.loads(cached_data)
            
            # Update access metadata
            cache_entry["last_accessed"] = datetime.now().isoformat()
            cache_entry["access_count"] = cache_entry.get("access_count", 0) + 1
            
            # Update TTL
            self.redis_client.setex(
                cache_key, 
                settings.SEMANTIC_CACHE_TTL, 
                json.dumps(cache_entry)
            )
            
            logger.info(f"Cache hit for query: {query[:50]}... (similarity: {similarity_score:.3f})")
            return cache_entry
            
        except Exception as e:
            logger.error(f"Error retrieving cached answer: {e}")
            return None
    
    async def store_answer(self, query: str, answer: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Store a query-answer pair in the semantic cache.
        
        Args:
            query: The user query
            answer: The generated answer
            metadata: Additional metadata (tokens, generation time, etc.)
            
        Returns:
            True if successfully cached, False otherwise
        """
        if not self._is_cache_enabled():
            return False
        
        try:
            # Generate unique cache ID
            cache_id = str(uuid.uuid4())
            doc_id = str(uuid.uuid4())
            
            # Prepare cache entry
            cache_entry = {
                "cache_id": cache_id,
                "query": query,
                "answer": answer,
                "created_at": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat(),
                "access_count": 0,
                "metadata": metadata or {}
            }
            
            # Store in Redis with TTL
            cache_key = f"cache_entry:{cache_id}"
            self.redis_client.setex(
                cache_key, 
                settings.SEMANTIC_CACHE_TTL, 
                json.dumps(cache_entry)
            )
            
            # Store query vector in Weaviate for similarity search
            cache_doc = Document(
                page_content=query,
                metadata={
                    "cache_id": cache_id,
                    "doc_id": doc_id,
                    "created_at": cache_entry["created_at"],
                    "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer
                }
            )
            
            self.cache_vector_store.add_documents([cache_doc])
            
            # Manage cache size
            await self._manage_cache_size()
            
            logger.info(f"Cached answer for query: {query[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error storing answer in cache: {e}")
            return False
    
    async def _manage_cache_size(self):
        """Manage cache size by removing oldest entries if needed."""
        try:
            # Count current cache entries
            cache_pattern = "cache_entry:*"
            cache_keys = self.redis_client.keys(cache_pattern)
            
            if len(cache_keys) <= settings.SEMANTIC_CACHE_MAX_SIZE:
                return
            
            # Get entries with creation timestamps
            entries_with_timestamps = []
            for key in cache_keys:
                data = self.redis_client.get(key)
                if data:
                    entry = json.loads(data)
                    created_at = entry.get("created_at")
                    if created_at:
                        entries_with_timestamps.append((key, created_at, entry.get("cache_id")))
            
            # Sort by creation time (oldest first)
            entries_with_timestamps.sort(key=lambda x: x[1])
            
            # Remove oldest entries
            entries_to_remove = len(cache_keys) - settings.SEMANTIC_CACHE_MAX_SIZE
            for i in range(entries_to_remove):
                cache_key, _, cache_id = entries_with_timestamps[i]
                
                # Remove from Redis
                self.redis_client.delete(cache_key)
                
                # Remove from Weaviate (this is more complex and might be done periodically)
                logger.debug(f"Removed cache entry: {cache_id}")
                
        except Exception as e:
            logger.error(f"Error managing cache size: {e}")
    
    async def _cleanup_orphaned_vector_entry(self, doc_id: str):
        """Clean up orphaned vector entries that no longer have Redis counterparts."""
        try:
            if doc_id:
                # This would require custom Weaviate deletion logic
                # For now, just log it for periodic cleanup
                logger.debug(f"Orphaned vector entry detected: {doc_id}")
        except Exception as e:
            logger.error(f"Error cleaning up orphaned vector entry: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._is_cache_enabled():
            return {"enabled": False}
        
        try:
            cache_keys = self.redis_client.keys("cache_entry:*")
            total_entries = len(cache_keys)
            
            # Sample some entries for stats
            sample_size = min(10, total_entries)
            total_access_count = 0
            
            if sample_size > 0:
                sample_keys = cache_keys[:sample_size]
                for key in sample_keys:
                    data = self.redis_client.get(key)
                    if data:
                        entry = json.loads(data)
                        total_access_count += entry.get("access_count", 0)
            
            avg_access_count = total_access_count / sample_size if sample_size > 0 else 0
            
            return {
                "enabled": True,
                "total_entries": total_entries,
                "max_size": settings.SEMANTIC_CACHE_MAX_SIZE,
                "ttl_seconds": settings.SEMANTIC_CACHE_TTL,
                "similarity_threshold": settings.SEMANTIC_CACHE_SIMILARITY_THRESHOLD,
                "avg_access_count": round(avg_access_count, 2)
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"enabled": True, "error": str(e)}
    
    def clear_cache(self) -> bool:
        """Clear all cache entries."""
        if not self._is_cache_enabled():
            return False
        
        try:
            # Clear Redis entries
            cache_keys = self.redis_client.keys("cache_entry:*")
            if cache_keys:
                self.redis_client.delete(*cache_keys)
            
            # Clear Weaviate collection (would require recreating the collection)
            logger.warning("Weaviate cache collection not cleared - requires manual intervention")
            
            logger.info(f"Cleared {len(cache_keys)} cache entries from Redis")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False


# Global semantic cache instance
semantic_cache = SemanticCache()