"""FastAPI dependency providers.

Each provider is cached with lru_cache so the embedding model, ChromaDB
client, and OpenAI client are constructed exactly once at process start
(the first time they're needed) rather than on every request — loading the
embedding model per request would be both slow and wasteful.
"""

from functools import lru_cache

from app.services.config import get_settings
from app.services.llm import LLMClient
from app.services.rag import RagService
from app.services.retrieval import ChromaVectorStore, Retriever, VectorStore


@lru_cache
def get_vector_store() -> VectorStore:
    return ChromaVectorStore(get_settings())


@lru_cache
def get_retriever() -> Retriever:
    return Retriever(get_vector_store(), top_k=get_settings().top_k)


@lru_cache
def get_llm_client() -> LLMClient:
    return LLMClient(get_settings())


@lru_cache
def get_rag_service() -> RagService:
    return RagService(get_retriever(), get_llm_client())
