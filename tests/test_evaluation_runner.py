"""Tests for evaluation/runner.py's aggregate-exclusion behavior: examples
with no ground truth (e.g. category='no_answer') must not silently inflate
Mean Recall@K. Uses a stub retriever and NoOpReranker — no ChromaDB, no LLM
calls, fully deterministic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.retrieval import RetrievedChunk
from app.services.reranker import NoOpReranker
from evaluation.runner import EvaluationRunner
from evaluation.schema import EvalDataset, EvalExample


class StubRetriever:
    """Returns a fixed set of chunks regardless of query — one from
    career.md, one from links.md, so recall depends only on which source is
    labeled relevant."""

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(text="career text", source="career.md", document_type="career", chunk_id="career_0", distance=0.5),
            RetrievedChunk(text="links text", source="links.md", document_type="links", chunk_id="links_0", distance=0.9),
        ]


def test_no_ground_truth_example_excluded_from_mean_recall():
    dataset = EvalDataset(
        name="test",
        examples=[
            EvalExample(query="answerable question", relevant_sources=["links.md"], category="simple_factual"),
            EvalExample(query="unanswerable question", relevant_sources=[], category="no_answer"),
        ],
    )
    runner = EvaluationRunner(StubRetriever(), k_values=(1, 2), reranker=NoOpReranker())
    report = runner.run(dataset)

    assert report.scored_example_count == 1
    assert len(report.results) == 2

    # links.md is at rank 2 in the stub's fixed order, so recall@1=0 recall@2=1
    # for the answerable example — mean should equal exactly that example's
    # scores, not be diluted/inflated by the no_answer example's trivial 1.0s.
    assert report.mean_recall_at_k[1] == 0.0
    assert report.mean_recall_at_k[2] == 1.0

    scored_flags = {r.query: r.has_ground_truth for r in report.results}
    assert scored_flags["answerable question"] is True
    assert scored_flags["unanswerable question"] is False


def test_all_no_ground_truth_gives_empty_mean_without_crashing():
    dataset = EvalDataset(
        name="test",
        examples=[EvalExample(query="unanswerable", relevant_sources=[], category="no_answer")],
    )
    runner = EvaluationRunner(StubRetriever(), k_values=(1, 3), reranker=NoOpReranker())
    report = runner.run(dataset)

    assert report.scored_example_count == 0
    assert report.mean_recall_at_k == {1: 0.0, 3: 0.0}
