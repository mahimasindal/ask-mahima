"""Shared embedding function.

Both ingestion (scripts/ingest.py) and retrieval (app/services/retrieval.py)
must build their ChromaDB collection through this exact same function. If
they ever diverged — different model, different config — stored vectors and
query vectors would live in different embedding spaces. ChromaDB would not
raise an error; search would just silently return bad results. Centralizing
the function here makes that divergence impossible.
"""

from chromadb.utils import embedding_functions

from app.services.config import Settings


def get_embedding_function(
    settings: Settings,
) -> embedding_functions.SentenceTransformerEmbeddingFunction:
    """Local sentence-transformers embedding function — no external API/key needed."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model_name
    )
