from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings. Override with env vars prefixed SCOPEFLOW_ (or a .env file)."""

    model_config = SettingsConfigDict(env_prefix="SCOPEFLOW_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://scopeflow:scopeflow@localhost:5433/scopeflow"
    ollama_url: str = "http://localhost:11434"
    default_model: str = "scopeflow-q4"
    embedding_model: str = "nomic-embed-text"
    qdrant_url: str = "http://localhost:6333"
    letta_url: str = "http://localhost:8283"


@lru_cache
def get_settings() -> Settings:
    return Settings()
