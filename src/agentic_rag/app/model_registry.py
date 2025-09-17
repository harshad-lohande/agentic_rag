# src/agentic_rag/app/model_registry.py

import threading
import asyncio
from typing import Optional, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from agentic_rag.config import settings
from agentic_rag.logging_config import logger


class ModelRegistry:
    """
    Singleton registry for pre-loaded ML models to eliminate repetitive loading.
    
    This addresses the critical performance bottleneck where SentenceTransformer 
    and CrossEncoder models were being loaded multiple times per request,
    adding 80-90 seconds of unnecessary overhead.
    """
    
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_models'):
            self._models: Dict[str, Any] = {}
            self._init_lock = threading.Lock()
    
    async def initialize_models(self):
        """
        Pre-load all ML models at application startup.
        This runs once and eliminates per-request model loading overhead.
        """
        if self._initialized:
            return
            
        with self._init_lock:
            if self._initialized:  # Double-check locking
                return
                
            logger.info("🚀 Pre-loading ML models for performance optimization...")
            start_time = asyncio.get_event_loop().time()
            
            try:
                # Pre-load embedding model (used by semantic cache, retrieval, compression)
                await self._load_embedding_model()
                
                # Pre-load cross-encoder models (used by reranking)
                await self._load_cross_encoders()
                
                # Pre-load any additional models that were causing bottlenecks
                await self._load_additional_models()
                
                elapsed = asyncio.get_event_loop().time() - start_time
                logger.info(f"✅ All models pre-loaded successfully in {elapsed:.2f}s")
                logger.info(f"📊 Models cached: {list(self._models.keys())}")
                
                self._initialized = True
                
            except Exception as e:
                logger.error(f"❌ Failed to pre-load models: {e}")
                raise
    
    async def _load_embedding_model(self):
        """Load the SentenceTransformer embedding model."""
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        
        # Load in thread pool to avoid blocking event loop
        embedding_model = await asyncio.to_thread(
            HuggingFaceEmbeddings,
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},  # Ensure CPU usage for consistency
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self._models['embedding_model'] = embedding_model
        logger.info(f"✅ Embedding model loaded: {settings.EMBEDDING_MODEL}")
    
    async def _load_cross_encoders(self):
        """Load CrossEncoder models for reranking."""
        # Load small cross-encoder
        if hasattr(settings, 'CROSS_ENCODER_MODEL_SMALL'):
            logger.info(f"Loading small cross-encoder: {settings.CROSS_ENCODER_MODEL_SMALL}")
            small_cross_encoder = await asyncio.to_thread(
                HuggingFaceCrossEncoder,
                model_name=settings.CROSS_ENCODER_MODEL_SMALL
            )
            self._models['cross_encoder_small'] = small_cross_encoder
            logger.info(f"✅ Small cross-encoder loaded: {settings.CROSS_ENCODER_MODEL_SMALL}")
        
        # Load large cross-encoder
        if hasattr(settings, 'CROSS_ENCODER_MODEL_LARGE'):
            logger.info(f"Loading large cross-encoder: {settings.CROSS_ENCODER_MODEL_LARGE}")
            large_cross_encoder = await asyncio.to_thread(
                HuggingFaceCrossEncoder,
                model_name=settings.CROSS_ENCODER_MODEL_LARGE
            )
            self._models['cross_encoder_large'] = large_cross_encoder
            logger.info(f"✅ Large cross-encoder loaded: {settings.CROSS_ENCODER_MODEL_LARGE}")
    
    async def _load_additional_models(self):
        """Load any additional models that may be causing performance issues."""
        # This method can be extended to pre-load other models as needed
        pass
    
    def get_embedding_model(self) -> Optional[HuggingFaceEmbeddings]:
        """Get the pre-loaded embedding model."""
        if not self._initialized:
            logger.warning("Model registry not initialized. Models may be loaded on-demand.")
            return None
        return self._models.get('embedding_model')
    
    def get_cross_encoder_small(self) -> Optional[HuggingFaceCrossEncoder]:
        """Get the pre-loaded small cross-encoder model."""
        if not self._initialized:
            logger.warning("Model registry not initialized. Models may be loaded on-demand.")
            return None
        return self._models.get('cross_encoder_small')
    
    def get_cross_encoder_large(self) -> Optional[HuggingFaceCrossEncoder]:
        """Get the pre-loaded large cross-encoder model."""
        if not self._initialized:
            logger.warning("Model registry not initialized. Models may be loaded on-demand.")
            return None
        return self._models.get('cross_encoder_large')
    
    def is_initialized(self) -> bool:
        """Check if all models have been pre-loaded."""
        return self._initialized
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about loaded models."""
        return {
            'embedding_model': settings.EMBEDDING_MODEL,
            'cross_encoder_small': getattr(settings, 'CROSS_ENCODER_MODEL_SMALL', 'Not configured'),
            'cross_encoder_large': getattr(settings, 'CROSS_ENCODER_MODEL_LARGE', 'Not configured'),
            'initialized': str(self._initialized),
            'loaded_models': list(self._models.keys())
        }


# Global singleton instance
model_registry = ModelRegistry()