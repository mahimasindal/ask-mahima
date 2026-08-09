"""Lightweight retrieval tests.

These assume `python scripts/ingest.py` has already been run against the
data/*.md files, so the local ChromaDB collection is populated. This is a
deliberate V0 simplification — no fixture re-ingests a throwaway DB per test
run, since that's more machinery than a personal-project V0 needs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.config import get_settings
from app.services.retrieval import ChromaVectorStore, Retriever


def _make_retriever() -> Retriever:
    settings = get_settings()
    store = ChromaVectorStore(settings)
    return Retriever(store, top_k=settings.top_k)


def test_retrieve_returns_chunks_with_expected_source():
    retriever = _make_retriever()
    chunks = retriever.retrieve("What is Mahima's career and current role?")
    assert len(chunks) > 0
    sources = {chunk.source for chunk in chunks}
    assert "career.md" in sources


def test_retrieve_returns_top_k_or_fewer():
    settings = get_settings()
    retriever = _make_retriever()
    chunks = retriever.retrieve("Tell me about Mahima")
    assert len(chunks) <= settings.top_k


def test_retrieve_empty_query_returns_no_chunks():
    retriever = _make_retriever()
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []
