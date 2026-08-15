"""Unit tests for app/services/reranker.py — parsing logic and fallback
behavior, using a stub LLMClient so no real API calls happen."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.retrieval import RetrievedChunk
from app.services.llm import LLMServiceError
from app.services.reranker import LLMReranker, NoOpReranker, _parse_ranking


def _chunk(chunk_id: str, source: str) -> RetrievedChunk:
    return RetrievedChunk(text=f"text for {chunk_id}", source=source, document_type="doc", chunk_id=chunk_id, distance=1.0)


class StubLLMClient:
    """Minimal stand-in for LLMClient — returns a canned response or raises."""

    def __init__(self, response: str = "", error: bool = False):
        self._response = response
        self._error = error

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self._error:
            raise LLMServiceError("stub failure")
        return self._response


def test_parse_ranking_valid():
    assert _parse_ranking("[3, 1, 2]", n=3) == [2, 0, 1]


def test_parse_ranking_extracts_json_from_surrounding_text():
    assert _parse_ranking("Here is the ranking: [2, 1] — done.", n=2) == [1, 0]


def test_parse_ranking_rejects_wrong_count():
    assert _parse_ranking("[1, 2]", n=3) is None


def test_parse_ranking_rejects_duplicates():
    assert _parse_ranking("[1, 1, 2]", n=3) is None


def test_parse_ranking_rejects_out_of_range():
    assert _parse_ranking("[1, 2, 5]", n=3) is None


def test_parse_ranking_rejects_malformed_json():
    assert _parse_ranking("not a list at all", n=3) is None


def test_noop_reranker_truncates_preserving_order():
    chunks = [_chunk("a", "a.md"), _chunk("b", "b.md"), _chunk("c", "c.md")]
    result = NoOpReranker().rerank("query", chunks, top_k=2)
    assert [c.chunk_id for c in result] == ["a", "b"]


def test_llm_reranker_reorders_by_parsed_ranking():
    chunks = [_chunk("a", "a.md"), _chunk("b", "b.md"), _chunk("c", "c.md")]
    reranker = LLMReranker(StubLLMClient(response="[3, 1, 2]"))
    result = reranker.rerank("query", chunks, top_k=2)
    assert [c.chunk_id for c in result] == ["c", "a"]


def test_llm_reranker_falls_back_on_malformed_response():
    chunks = [_chunk("a", "a.md"), _chunk("b", "b.md")]
    reranker = LLMReranker(StubLLMClient(response="not json"))
    result = reranker.rerank("query", chunks, top_k=2)
    assert [c.chunk_id for c in result] == ["a", "b"]


def test_llm_reranker_falls_back_on_llm_error():
    chunks = [_chunk("a", "a.md"), _chunk("b", "b.md")]
    reranker = LLMReranker(StubLLMClient(error=True))
    result = reranker.rerank("query", chunks, top_k=2)
    assert [c.chunk_id for c in result] == ["a", "b"]


def test_llm_reranker_skips_llm_call_for_single_chunk():
    chunks = [_chunk("a", "a.md")]
    reranker = LLMReranker(StubLLMClient(error=True))  # would raise if called
    result = reranker.rerank("query", chunks, top_k=5)
    assert [c.chunk_id for c in result] == ["a"]
