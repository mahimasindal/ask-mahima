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

    # Embeddings (local, no API key needed). Currently unused — the app uses
    # ChromaDB's built-in ONNX embedding function (see
    # app/services/embeddings.py), which isn't configurable by model name.
    # Kept as a field so an EMBEDDING_MODEL_NAME env var set on Render (or
    # elsewhere) doesn't break settings loading.
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # Vector store
    chroma_persist_dir: str = "./data/chroma_db"
    collection_name: str = "mahima_knowledge_base"

    # Retrieval
    # top_k is the FINAL number of chunks sent to the LLM as context — after
    # reranking, if enabled. It is not how many the vector store is queried
    # for; see retrieval_candidates below.
    # Lowered 8 -> 3 once reranking (below) brought Recall@1 to 1.000 across
    # the eval dataset — see README "Iteration 4". Verified via
    # evaluation/: top_k=2 was rejected because it silently drops a
    # genuinely relevant chunk in 3 of 10 eval queries; top_k=3 does not,
    # in any of them.
    top_k: int = 3

    # Reranking — a second, LLM-judged relevance pass over retrieval
    # candidates (see app/services/reranker.py for why this exists).
    use_reranker: bool = True
    # How many chunks the vector store is asked for before reranking narrows
    # them down to top_k. Must be >= top_k for reranking to have anything
    # extra to choose from; raise it to give the reranker a wider net.
    retrieval_candidates: int = 10

    # Ingestion source
    data_dir: str = "./data"


@lru_cache
def get_settings() -> Settings:
    """Cached so Settings() is only constructed once per process."""
    return Settings()
