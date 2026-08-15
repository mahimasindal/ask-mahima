"""Reranking — a second, LLM-judged relevance pass over retrieval
candidates, run after ChromaVectorStore.query() and before chunks are sent
to the LLM as context.

Why this exists: the retrieval eval harness (evaluation/) showed a
persistent chunk (links.md's contact URLs) winning rank #1 by raw
embedding similarity for most queries regardless of topic — a measured
weakness of all-MiniLM-L6-v2 over short, low-information text (chunk
length correlated with similarity-to-arbitrary-query at Pearson r=-0.616;
see README's "Known Limitations"). Fixing that by reordering or rewording
the data hit a ceiling. A reranker re-scores the *same* candidates using
actual relevance judgment instead of embedding distance, so a smaller
final top_k can be trusted to still contain the right context — which is
the whole point: it's what makes shrinking top_k for token savings safe.

`Reranker` is the interface the rest of the app depends on (mirrors the
VectorStore/ChromaVectorStore split in app/services/retrieval.py), so the
implementation can be swapped or disabled — see NoOpReranker — without
touching RagService.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Optional

from app.models.retrieval import RetrievedChunk
from app.services.llm import LLMClient, LLMServiceError

RERANK_SYSTEM_PROMPT = (
    "You are a relevance-ranking assistant. You will be given a user "
    "question and a numbered list of candidate text passages. Return ONLY "
    "a JSON array of the passage numbers, ordered from most to least "
    "relevant to answering the question. Include every passage number "
    "exactly once. Do not include any other text, explanation, or "
    "formatting — just the JSON array, e.g. [3, 1, 4, 2]."
)

# Enough for the LLM to judge relevance without paying to re-read every
# full chunk twice (once here, once at generation time).
PREVIEW_CHARS = 300


class Reranker(ABC):
    """Reorders (and truncates to top_k) retrieval candidates by relevance
    to the query."""

    @abstractmethod
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]: ...


class NoOpReranker(Reranker):
    """Pass-through: keeps the vector store's original similarity order and
    just truncates to top_k. Used when Settings.use_reranker=False, and
    also what LLMReranker itself falls back to if the LLM call fails or
    its output can't be parsed — reranking is an enhancement, never a hard
    dependency for answering a question."""

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return chunks[:top_k]


def _build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    lines = [f"Question: {query}", "", "Candidate passages:"]
    for i, chunk in enumerate(chunks, start=1):
        preview = chunk.text.replace("\n", " ").strip()
        if len(preview) > PREVIEW_CHARS:
            preview = preview[:PREVIEW_CHARS] + "..."
        lines.append(f"[{i}] (source: {chunk.source}) {preview}")
    return "\n".join(lines)


def _parse_ranking(raw: str, n: int) -> Optional[list[int]]:
    """Parses the LLM's response into a 0-indexed ranking over n candidates.

    Returns None (rather than raising) on any malformed output — wrong
    count, duplicate/out-of-range indices, unparseable JSON — so the caller
    falls back to the original order instead of failing the whole request
    over a formatting slip.
    """
    match = re.search(r"\[[\d,\s]+\]", raw)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or len(parsed) != n:
        return None

    indices: list[int] = []
    seen: set[int] = set()
    for item in parsed:
        if not isinstance(item, int) or not (1 <= item <= n) or item in seen:
            return None
        seen.add(item)
        indices.append(item - 1)
    return indices


class LLMReranker(Reranker):
    """Reranks candidates with one extra LLM call that judges relevance
    directly, instead of relying on embedding distance.

    Chosen over a cross-encoder model specifically to avoid pulling in
    torch/sentence-transformers — app/services/embeddings.py already
    avoided that dependency once, to stay under Render free tier's 512MB
    RAM limit, and a cross-encoder would reopen that constraint. Trade-off:
    one extra LLM round-trip (latency + a small amount of token cost) per
    /chat request, in exchange for a context that's cheaper and more
    reliable to build (see Settings.use_reranker to disable).
    """

    def __init__(self, llm_client: LLMClient):
        self._llm_client = llm_client

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if len(chunks) <= 1:
            return chunks[:top_k]

        prompt = _build_prompt(query, chunks)
        try:
            raw = self._llm_client.generate(RERANK_SYSTEM_PROMPT, prompt)
        except LLMServiceError:
            return chunks[:top_k]

        order = _parse_ranking(raw, len(chunks))
        if order is None:
            return chunks[:top_k]

        return [chunks[i] for i in order][:top_k]
