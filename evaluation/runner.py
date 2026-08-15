"""Evaluation runner — executes an EvalDataset against the retrieval
pipeline and produces a typed EvalReport.

Builds its own Retriever/ChromaVectorStore instance rather than reusing
app.api.deps' cached singletons — exactly like scripts/check_retrieval.py
and tests/test_retrieval.py already do. This keeps evaluation a read-only
consumer of the same persisted collection the production app queries,
without ever sharing state with (or risking mutating) the request-serving
path. Nothing here calls VectorStore.add()/reset(), and nothing here
touches app/api, app/main.py, or the deps.py dependency graph.
"""

from __future__ import annotations

from app.models.retrieval import RetrievedChunk
from app.services.config import get_settings
from app.services.retrieval import ChromaVectorStore, Retriever

from evaluation.metrics import recall_at_k
from evaluation.results import EvalExampleResult, EvalReport
from evaluation.schema import EvalDataset, EvalExample

DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5, 8)


def _ids_for_recall(chunks: list[RetrievedChunk], use_chunk_ids: bool) -> list[str]:
    return [chunk.chunk_id for chunk in chunks] if use_chunk_ids else [chunk.source for chunk in chunks]


class EvaluationRunner:
    """Runs eval datasets against a Retriever configured for evaluation."""

    def __init__(self, retriever: Retriever, k_values: tuple[int, ...] = DEFAULT_K_VALUES):
        self._retriever = retriever
        self._k_values = k_values

    @classmethod
    def from_settings(cls, k_values: tuple[int, ...] = DEFAULT_K_VALUES) -> "EvaluationRunner":
        """Builds a standalone Retriever against the same persisted collection
        the production app reads from, sized to fetch enough chunks for the
        largest K being evaluated.

        This is a separate Retriever instance from the one app/api/deps.py
        hands to requests via lru_cache — same underlying ChromaDB
        collection, but its own object, constructed fresh per run and never
        registered as a request dependency.
        """
        settings = get_settings()
        store = ChromaVectorStore(settings)
        retriever = Retriever(store, top_k=max(k_values))
        return cls(retriever, k_values)

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
        chunks = self._retriever.retrieve(example.query)

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
