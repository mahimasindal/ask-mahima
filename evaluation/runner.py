"""Evaluation runner — executes an EvalDataset against the retrieval
pipeline (optionally including reranking) and produces a typed EvalReport.

Builds its own Retriever/ChromaVectorStore/Reranker instances rather than
reusing app.api.deps' cached singletons — exactly like
scripts/check_retrieval.py and tests/test_retrieval.py already do. This
keeps evaluation a read-only consumer of the same persisted collection the
production app queries, without ever sharing state with (or risking
mutating) the request-serving path. Nothing here calls
VectorStore.add()/reset(), and nothing here touches app/api, app/main.py,
or the deps.py dependency graph.

Note: with reranking on (the default, matching Settings.use_reranker),
running this makes one real LLM call per query via LLMReranker — unlike
pure vector-similarity evaluation, this is not free. Pass
use_reranker=False for a zero-cost, LLM-free baseline check.
"""

from __future__ import annotations

from typing import Optional

from app.models.retrieval import RetrievedChunk
from app.services.config import get_settings
from app.services.llm import LLMClient
from app.services.reranker import LLMReranker, NoOpReranker, Reranker
from app.services.retrieval import ChromaVectorStore, Retriever

from evaluation.metrics import recall_at_k
from evaluation.results import EvalExampleResult, EvalReport
from evaluation.schema import EvalDataset, EvalExample

DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5, 8)


def _ids_for_recall(chunks: list[RetrievedChunk], use_chunk_ids: bool) -> list[str]:
    return [chunk.chunk_id for chunk in chunks] if use_chunk_ids else [chunk.source for chunk in chunks]


class EvaluationRunner:
    """Runs eval datasets against a Retriever (+ optional Reranker)
    configured for evaluation."""

    def __init__(
        self,
        retriever: Retriever,
        k_values: tuple[int, ...] = DEFAULT_K_VALUES,
        reranker: Optional[Reranker] = None,
    ):
        self._retriever = retriever
        self._k_values = k_values
        self._reranker = reranker

    @classmethod
    def from_settings(
        cls,
        k_values: tuple[int, ...] = DEFAULT_K_VALUES,
        use_reranker: Optional[bool] = None,
    ) -> "EvaluationRunner":
        """Builds standalone Retriever/Reranker instances against the same
        persisted collection and settings the production app reads from —
        separate objects from what app/api/deps.py hands to requests via
        lru_cache, constructed fresh per run and never registered as a
        request dependency.

        use_reranker defaults to Settings.use_reranker (matching production
        behavior); pass explicitly to override for a given eval run.
        """
        settings = get_settings()
        rerank = settings.use_reranker if use_reranker is None else use_reranker

        store = ChromaVectorStore(settings)
        candidate_pool = max(settings.retrieval_candidates, max(k_values)) if rerank else max(k_values)
        retriever = Retriever(store, top_k=candidate_pool)

        reranker = LLMReranker(LLMClient(settings)) if rerank else NoOpReranker()
        return cls(retriever, k_values, reranker=reranker)

    def run(self, dataset: EvalDataset) -> EvalReport:
        results = [self._run_example(example) for example in dataset.examples]
        mean_recall = {
            k: (sum(r.recall_at_k[k] for r in results) / len(results)) if results else 0.0
            for k in self._k_values
        }
        return EvalReport(
            dataset_name=dataset.name,
            k_values=list(self._k_values),
            results=results,
            mean_recall_at_k=mean_recall,
        )

    def _run_example(self, example: EvalExample) -> EvalExampleResult:
        candidates = self._retriever.retrieve(example.query)

        # Ask the reranker for a *full* reordering (top_k = however many
        # candidates there are), not just its top pick — so Recall@K can be
        # computed at every K by slicing one ranked list locally, instead of
        # paying for a separate LLM call per K value.
        chunks = (
            self._reranker.rerank(example.query, candidates, top_k=len(candidates))
            if self._reranker is not None
            else candidates
        )

        use_chunk_ids = bool(example.relevant_chunk_ids)
        relevant_ids = example.relevant_chunk_ids if use_chunk_ids else example.relevant_sources
        retrieved_ids = _ids_for_recall(chunks, use_chunk_ids)

        recall_scores = {k: recall_at_k(retrieved_ids, relevant_ids, k) for k in self._k_values}

        return EvalExampleResult(
            query=example.query,
            relevant_sources=example.relevant_sources,
            retrieved=chunks,
            recall_at_k=recall_scores,
        )
