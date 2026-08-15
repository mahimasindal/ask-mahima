"""Result models produced by running an EvalDataset through the retriever.

Kept separate from schema.py (the hand-authored input format) because these
are generated output, not something a human writes by hand.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.models.retrieval import RetrievedChunk


class EvalExampleResult(BaseModel):
    query: str
    relevant_sources: list[str]
    retrieved: list[RetrievedChunk]
    recall_at_k: dict[int, float]  # k -> recall, one entry per evaluated K


class EvalReport(BaseModel):
    dataset_name: str
    k_values: list[int]
    results: list[EvalExampleResult]
    mean_recall_at_k: dict[int, float]
