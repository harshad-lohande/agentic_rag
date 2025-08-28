# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Provider Settings ---
    LLM_PROVIDER: Literal["openai", "google"] = "openai"

    # --- Ingestion settings from Phase 1 ---
    WEAVIATE_HOST: str = "localhost"
    WEAVIATE_PORT: int = 8080
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    INDEX_NAME: str = "AgenticRAG"
    DATA_TO_INDEX: str = "data"

    # --- Settings from Phase 1 ---
    OPENAI_MAIN_MODEL_NAME: str = "gpt-4.1-mini"
    OPENAI_FAST_MODEL_NAME: str = "gpt-4.1-nano"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_API_KEY: str

    # --- Google Settings ---
    GOOGLE_MAIN_MODEL_NAME: str = "gemini-2.5-flash"
    GOOGLE_FAST_MODEL_NAME: str = "gemini-2.5-flash-lite"
    GOOGLE_EMBEDDING_MODEL: str = "models/text-embedding-004"
    GOOGLE_API_KEY: str

    LANGCHAIN_TRACING_V2: str = "true"
    LANGCHAIN_API_KEY: str
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_PROJECT: str = "Agentic RAG System"

    # --- HuggingFace Cross-Encoder Settings ---
    CROSS_ENCODER_MODEL_SMALL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    CROSS_ENCODER_MODEL_LARGE: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    CROSS_ENCODER_MODEL_ROBERTA: str = "cross-encoder/stsb-roberta-large"

settings = Settings()