# src/agentic_rag/config.py

import logging
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

from agentic_rag.aws import get_secrets

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    # Allow loading from a .env file for local development
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- RAG Application Settings ---
    APP_ENDPOINT_API_KEY: str = ""

    # --- Retrieval -> Rerank -> Compression knobs ---
    RETRIEVAL_CANDIDATES_K: int = 10  # initial recall before rerank/compress
    RERANK_TOP_K: int = 3  # keep top-N after rerank

    # --- Contextual compression ---
    ENABLE_CONTEXTUAL_COMPRESSION: bool = True
    COMPRESSION_MAX_TOKENS: int = 200
    COMPRESSION_OVERLAP_TOKENS: int = 30
    COMPRESSION_REDUNDANCY_SIM: float = 0.95
    # Dedicated fast compression SentenceTransformer (singleton)
    FAST_COMPRESSION_MODEL: str = "all-MiniLM-L6-v2"

    # --- Performance optimization settings ---
    ENABLE_FAST_COMPRESSION: bool = (
        True  # Use fast extractive compression instead of LLM-based
    )
    ENABLE_MODEL_PRELOADING: bool = True  # Pre-load models at startup
    ENABLE_OPTIMIZED_WORKFLOW: bool = (
        True  # Use streamlined workflow without correction loops
    )

    # --- Compression LLM (open-source) ---
    COMPRESSION_LLM_PROVIDER: Literal["hf_endpoint", "ollama", "openai", "google"] = (
        "ollama"
    )
    HF_COMPRESSION_MODEL: str = "mistralai/Mistral-7B-Instruct-v0.3"
    HUGGINGFACEHUB_API_TOKEN: str = ""

    # --- Optional Ollama alternative (if you run an Ollama server) ---
    OLLAMA_HOST: str = "http://localhost:11434"
    COMPRESSION_LLM_MODEL: str = "llama3.1:8b"  # for provider="ollama"

    # --- Chunking Settings ---
    CHUNKING_STRATEGY: Literal["recursive", "semantic"] = "semantic"
    SEMANTIC_BREAKPOINT_TYPE: Literal["percentile", "standard_deviation"] = "percentile"
    SEMANTIC_BREAKPOINT_AMOUNT: float = 90.0  # 95.0 for percentile, 1.0 for std dev
    SEMANTIC_BUFFER_SIZE: int = 20
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # --- Redis Setting ---
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    # --- Provider Settings ---
    LLM_PROVIDER: Literal["openai", "google"] = "openai"

    # --- Ingestion settings from Phase 1 ---
    WEAVIATE_HOST: str = "localhost"
    WEAVIATE_PORT: int = 8080
    # EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    # EMBEDDING_MODEL: str = "mixedbread-ai/mxbai-embed-large-v1"
    EMBEDDING_MODEL: str = "intfloat/e5-large-v2"
    INDEX_NAME: str = "AgenticRAG"
    DATA_TO_INDEX: str = "data"

    # --- Settings from Phase 1 ---
    OPENAI_MAIN_MODEL_NAME: str = "gpt-4.1-mini"
    OPENAI_FAST_MODEL_NAME: str = "gpt-4.1-nano"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_API_KEY: str = ""

    # --- Google Settings ---
    GOOGLE_MAIN_MODEL_NAME: str = "gemini-2.5-flash"
    GOOGLE_FAST_MODEL_NAME: str = "gemini-2.5-flash-lite"
    GOOGLE_EMBEDDING_MODEL: str = "models/text-embedding-004"
    GOOGLE_API_KEY: str = ""

    LANGCHAIN_TRACING_V2: str = "true"
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_PROJECT: str = "Agentic RAG System"

    # --- HuggingFace Cross-Encoder Settings ---
    CROSS_ENCODER_MODEL_SMALL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    CROSS_ENCODER_MODEL_LARGE: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    CROSS_ENCODER_MODEL_ROBERTA: str = "cross-encoder/stsb-roberta-large"

    # --- HNSW Index Configuration ---
    # efConstruction: Higher values create more accurate index (build-time)
    # Recommended range: 64-512, higher for better accuracy/slower builds
    HNSW_EF_CONSTRUCTION: int = 256

    # ef: Query-time search parameter (runtime)
    # Recommended range: 16-256, higher for better accuracy/slower queries
    HNSW_EF: int = 64

    # maxConnections: Max connections per node in HNSW graph
    # Recommended range: 16-64, higher for better recall/more memory
    HNSW_MAX_CONNECTIONS: int = 32

    # --- Semantic Caching Configuration ---
    ENABLE_SEMANTIC_CACHE: bool = True
    SEMANTIC_CACHE_SIMILARITY_THRESHOLD: float = 0.85
    SEMANTIC_CACHE_TTL: int = 3600  # Cache TTL in seconds
    SEMANTIC_CACHE_MAX_SIZE: int = 1000  # Maximum number of cached queries
    SEMANTIC_CACHE_INDEX_NAME: str = "SemanticCache"
    SEMANTIC_CACHE_GC_INTERVAL: int = 3600  # Garbage collection interval in seconds

    # Robust semantic hit acceptance (tunable)
    SEMANTIC_CACHE_CE_ACCEPT: float = 0.60  # cross-encoder accept
    SEMANTIC_CE_SIM_HIGH: float = 0.90  # high cross-encoder support when needed
    SEMANTIC_CACHE_LEXICAL_MIN: float = 0.15  # tiny lexical support when needed
    SEMANTIC_CACHE_LEXICAL_HIGH: float = 0.4  # high lexical support when needed
    SEMANTIC_CACHE_LEXICAL_MODERATE: float = (
        0.30  # lexical support for moderate similarity
    )

    # --- Settings for cloud environment ---
    APP_ENVIRONMENT: Literal["development", "production"] = "development"
    AWS_SECRET_NAME: str = "agentic-rag/api_keys"
    AWS_REGION: str = "us-east-1" # Or your preferred region


def load_settings() -> Settings:
    """
    Loads the application settings.

    In a 'production' environment, it fetches secrets from AWS Secrets Manager.
    Otherwise, it falls back to loading from environment variables / .env file.
    """
    # First, load the base settings which will read APP_ENVIRONMENT from the env
    base_settings = Settings()

    if base_settings.APP_ENVIRONMENT == "production":
        logger.info("Production environment detected. Loading secrets from AWS Secrets Manager.")
        try:
            # Fetch the secrets dictionary from AWS
            aws_secrets = get_secrets(
                secret_name=base_settings.AWS_SECRET_NAME,
                region_name=base_settings.AWS_REGION
            )
            # Create a new Settings instance, overriding values with those from AWS
            # Any setting not in AWS will keep its default or value from the environment
            return Settings(**aws_secrets)
        except ValueError as e:
            logger.critical(f"Failed to load settings from AWS: {e}")
            # In a real production system, you might want the app to fail to start
            # if secrets can't be loaded.
            raise
    else:
        logger.info("Development environment detected. Loading settings from .env file.")
        # In a development environment, return the settings loaded from the .env file
        return base_settings

# Create a single, globally accessible settings object
settings = load_settings()
