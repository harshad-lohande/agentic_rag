# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Ingestion settings from Phase 1 ---
    WEAVIATE_HOST: str = "localhost"
    WEAVIATE_PORT: int = 8080
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    INDEX_NAME: str = "AgenticRAG"
    DATA_TO_INDEX: str = "data"

    # --- Settings from Phase 1 ---
    OPENAI_MODEL_NAME: str = "gpt-4.1-nano"
    OPENAI_API_KEY: str

    LANGCHAIN_TRACING_V2: str = "true"
    LANGCHAIN_API_KEY: str
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_PROJECT: str = "Agentic RAG System"

settings = Settings()