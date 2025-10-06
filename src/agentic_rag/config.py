# src/agentic_rag/config.py

import logging
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

# Import from both cloud modules
from agentic_rag.aws import get_secrets as get_aws_secrets
from agentic_rag.gcp import get_secrets as get_gcp_secrets

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    # Allow loading from a .env file for local development
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- RAG Application Settings ---
    APP_ENDPOINT_API_KEY: str = ""
    BACKEND_URL: str = "http://localhost:8000/query"
    USE_RECENT_HISTORY_IN_REWRITE: bool = True
    APP_ENVIRONMENT: Literal["development", "production"] = "production"
    CLOUD_PROVIDER: Literal["aws", "gcp", "none"] = "aws"

    # --- Retrieval -> Rerank -> Compression knobs ---
    RETRIEVAL_CANDIDATES_K: int = 10  # initial recall before rerank/compress
    RERANK_TOP_K: int = 3  # keep top-N after rerank
    RETRIEVER_BALANCE: float = 0.5  # balance between vector and keyword search

    # --- Contextual compression settings ---
    ENABLE_CONTEXTUAL_COMPRESSION: bool = True
    COMPRESSION_MAX_TOKENS: int = 200
    COMPRESSION_OVERLAP_TOKENS: int = 30
    COMPRESSION_REDUNDANCY_SIM: float = 0.95
    COMPRESSION_LLM_MODEL: str = "llama3.1:8b"  # for provider="ollama"
    COMPRESSION_MODEL_TEMP: float = 0.2
    COMPRESSION_MODEL_CONTEXT_SIZE: int = 4096
    COMPRESSION_LLM_PROVIDER: Literal["hf_endpoint", "ollama", "openai", "google"] = (
        "ollama"
    )
    HF_COMPRESSION_MODEL: str = "mistralai/Mistral-7B-Instruct-v0.3"

    # --- Fast compression settings ---
    ENABLE_FAST_COMPRESSION: bool = (
        True  # Use fast extractive compression instead of LLM-based
    )
    ENABLE_MODEL_PRELOADING: bool = True  # Pre-load models at startup
    ENABLE_OPTIMIZED_WORKFLOW: bool = (
        True  # Use streamlined workflow without correction loops
    )
    # Dedicated fast compression SentenceTransformer (singleton)
    FAST_COMPRESSION_MODEL: str = "all-MiniLM-L6-v2"
    FAST_COMPRESSION_MAX_SENTENCES: int = 5

    # --- Huggingface Settings ---
    HUGGINGFACEHUB_API_TOKEN: str = ""

    # --- Optional Ollama alternative (if you run an Ollama server) ---
    OLLAMA_HOST: str = "http://localhost:11434"

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
    REDIS_MAX_CONNECTIONS: int = 20

    # --- Provider Settings ---
    LLM_PROVIDER: Literal["openai", "google"] = "openai"

    # --- Ingestion Settings ---
    WEAVIATE_HOST: str = "localhost"
    WEAVIATE_PORT: int = 8080
    # EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    # EMBEDDING_MODEL: str = "mixedbread-ai/mxbai-embed-large-v1"
    EMBEDDING_MODEL: str = "intfloat/e5-large-v2"
    WEAVIATE_STORAGE_INDEX_NAME: str = "AgenticRAG"
    # Path to the data to index (local directory or S3/GCS bucket)
    DATA_TO_INDEX: str = "s3://agentic-rag-ingestion-data-051025/data/"

    # --- OpenAI Settings ---
    OPENAI_MAIN_MODEL_NAME: str = "gpt-4.1-mini"
    OPENAI_FAST_MODEL_NAME: str = "gpt-4.1-nano"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_API_KEY: str = ""

    # --- Google Settings ---
    GOOGLE_MAIN_MODEL_NAME: str = "gemini-2.5-flash"
    GOOGLE_FAST_MODEL_NAME: str = "gemini-2.5-flash-lite"
    GOOGLE_EMBEDDING_MODEL: str = "models/text-embedding-004"
    GOOGLE_API_KEY: str = ""

    # Model temperature for LLM providers (other than compression)
    MODEL_TEMP: float = 0.7

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
    SEMANTIC_CACHE_LEXICAL_HIGH: float = 0.7  # high lexical support when needed
    SEMANTIC_CACHE_LEXICAL_MODERATE: float = (
        0.30  # lexical support for moderate similarity
    )

    # Redis keys and patterns
    REDIS_ZSET_KEY: str = "cache_index"
    REDIS_ENTRY_PREFIX: str = "cache_entry:"
    REDIS_ENTRY_PREFIX_PATTERN: str = "cache_entry:*"
    REDIS_EXACT_MATCH_PREFIX: str = "exact_match:"
    REDIS_SCAN_PATTERN: str = "exact_match:*"

    # --- AWS Specific Settings ---
    AWS_SECRET_NAME: str = "agentic-rag/api_keys"
    AWS_REGION: str = "us-east-1"  # Or your preferred region

    # --- GCP Specific Settings ---
    GCP_PROJECT_ID: str = "agentic-rag-system-473914"
    GCP_SECRET_ID: str = "api-keys"


def load_settings() -> Settings:
    """
    Loads settings, fetching from a cloud provider in a production environment.
    """
    base_settings = Settings()

    if base_settings.APP_ENVIRONMENT == "production":
        cloud_provider = base_settings.CLOUD_PROVIDER
        logger.info(
            f"Production environment detected. Using cloud provider: {cloud_provider}"
        )

        secrets = {}
        try:
            if cloud_provider == "aws":
                secrets = get_aws_secrets(
                    secret_name=base_settings.AWS_SECRET_NAME,
                    region_name=base_settings.AWS_REGION,
                )
            elif cloud_provider == "gcp":
                secrets = get_gcp_secrets(
                    project_id=base_settings.GCP_PROJECT_ID,
                    secret_id=base_settings.GCP_SECRET_ID,
                )
            else:
                logger.warning(
                    "Running in production mode without a specified cloud provider. Using default settings."
                )
                return base_settings

            return Settings(**secrets)

        except ValueError as e:
            logger.critical(f"Failed to load settings from {cloud_provider}: {e}")
            raise
    else:
        logger.info(
            "Development environment detected. Loading settings from .env file."
        )
        return base_settings


# Create a single, globally accessible settings object
settings = load_settings()
