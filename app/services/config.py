"""Central application settings.

All configuration lives here, loaded from environment variables (and a local
.env file for development). Every other module imports `get_settings()`
rather than reading `os.environ` directly, so there is exactly one place
that defines what's configurable and what the defaults are.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Required — no default. Missing this causes startup to fail immediately
    # (a clear config error) instead of failing confusingly on first request.
    openrouter_api_key: str

    # LLM (via OpenRouter's OpenAI-compatible API)
    llm_model: str = "openai/gpt-4o-mini"

    # Embeddings (local, no API key needed)
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # Vector store
    chroma_persist_dir: str = "./data/chroma_db"
    collection_name: str = "mahima_knowledge_base"

    # Retrieval
    top_k: int = 8

    # Ingestion source
    data_dir: str = "./data"


@lru_cache
def get_settings() -> Settings:
    """Cached so Settings() is only constructed once per process."""
    return Settings()
