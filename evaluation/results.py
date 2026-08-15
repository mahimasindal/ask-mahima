"""Result models produced by running an EvalDataset through the retriever.

Kept separate from schema.py (the hand-authored input format) because these
are generated output, not something a human writes by hand.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from app.models.retrieval import RetrievedChunk


class EvalExampleResult(BaseModel):
    query: str
    relevant_sources: list[str]
    category: Optional[str] = None
    retrieved: list[RetrievedChunk]
    recall_at_k: dict[int, float]  # k -> recall, one entry per evaluated K
    # False for examples with no ground truth (relevant_sources == [] and
    # relevant_chunk_ids == [], e.g. a 'no_answer' example) — recall_at_k is
    # still computed for these (metrics.recall_at_k trivially returns 1.0
    # when there's nothing to find), but that 1.0 does NOT mean "the system
    # correctly declined to answer." It means "there was no ground truth to
    # miss." See report.py for how this is surfaced, and EvalReport below
    # for why these are excluded from the aggregate.
    has_ground_truth: bool = True


class EvalReport(BaseModel):
    dataset_name: str
    k_values: list[int]
    results: list[EvalExampleResult]
    # Computed only over results where has_ground_truth is True — mixing in
    # no-ground-truth examples' trivial 1.0 scores would silently inflate
    # this number without it meaning anything.
    mean_recall_at_k: dict[int, float]
    scored_example_count: int  # len(results) with has_ground_truth == True
