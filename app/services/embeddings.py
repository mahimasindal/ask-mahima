"""Shared embedding function.

Both ingestion (scripts/ingest.py) and retrieval (app/services/retrieval.py)
must build their ChromaDB collection through this exact same function. If
they ever diverged — different model, different config — stored vectors and
query vectors would live in different embedding spaces. ChromaDB would not
raise an error; search would just silently return bad results. Centralizing
the function here makes that divergence impossible.

Uses ChromaDB's built-in DefaultEmbeddingFunction (all-MiniLM-L6-v2 via a
quantized ONNX Runtime model) rather than `sentence-transformers`. Same base
model, but `sentence-transformers` pulls in `torch`, and loading a torch
model at request time was blowing past Render's free-tier 512MB RAM limit
and getting the instance OOM-killed — that's what "no chat responses" turned
out to be. ONNX Runtime gets the same embeddings for a fraction of the
memory and with no torch dependency at all.
"""

from chromadb.utils import embedding_functions

from app.services.config import Settings


def get_embedding_function(
    settings: Settings,
) -> embedding_functions.DefaultEmbeddingFunction:
    """Local, lightweight embedding function — no external API/key needed."""
    return embedding_functions.DefaultEmbeddingFunction()
