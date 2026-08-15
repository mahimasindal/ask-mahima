"""Evaluation dataset schema.

Defines the on-disk format for hand-labeled retrieval eval examples
(evaluation/data/*.json) and the typed models the rest of the evaluation
package works with. Deliberately separate from app/models/ — these are
eval-only shapes and never part of the production API contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, Field


class EvalExample(BaseModel):
    """One hand-labeled query with its known-relevant source documents.

    Ground truth is at the *source file* level (e.g. "career.md") by
    default, rather than chunk_id, because chunk ids/boundaries shift
    whenever ingest.py's chunking logic or the underlying markdown changes —
    labeling at the file level keeps a hand-written dataset stable across
    re-chunking. If a specific example needs finer-grained precision, set
    relevant_chunk_ids too; metrics/runner will use those instead of
    relevant_sources whenever they're non-empty.
    """

    query: str = Field(..., min_length=1)
    relevant_sources: list[str] = Field(
        default_factory=list,
        description=(
            "Filenames that should be retrieved for this query, e.g. ['career.md']. "
            "Leave empty for a genuinely unanswerable question — see the 'no_answer' "
            "category and the caveat in evaluation/report.py about what an empty "
            "list does (and doesn't) mean for Recall@K."
        ),
    )
    relevant_chunk_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Optional finer-grained ground truth, e.g. ['career_1']. "
            "Leave empty to evaluate at source-file granularity."
        ),
    )
    category: Optional[str] = Field(
        default=None,
        description=(
            "Free-text tag for grouping examples in reports, e.g. 'simple_factual', "
            "'multi_document', 'ambiguous', 'requires_context', 'no_answer', "
            "'conflicting_or_old_info'. Not validated against a fixed set — purely "
            "descriptive, so new categories don't require a schema change."
        ),
    )
    notes: Optional[str] = Field(
        default=None, description="Why this example matters, for humans reading the dataset."
    )


class EvalDataset(BaseModel):
    """A named collection of EvalExamples, loaded from / saved to JSON."""

    name: str
    examples: list[EvalExample] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EvalDataset":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(raw)

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")
