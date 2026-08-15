"""Unit tests for evaluation/metrics.py — pure functions, no vector store or
network access needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import recall_at_k


def test_recall_at_k_all_relevant_found():
    retrieved = ["career.md", "career.md", "projects.md"]
    assert recall_at_k(retrieved, ["career.md"], k=3) == 1.0


def test_recall_at_k_partial_match():
    retrieved = ["career.md", "interests.md"]
    relevant = ["career.md", "writing.md"]
    assert recall_at_k(retrieved, relevant, k=2) == 0.5


def test_recall_at_k_respects_k_truncation():
    retrieved = ["interests.md", "projects.md", "career.md"]
    # "career.md" only shows up at rank 3, so it's excluded at k=2.
    assert recall_at_k(retrieved, ["career.md"], k=2) == 0.0
    assert recall_at_k(retrieved, ["career.md"], k=3) == 1.0


def test_recall_at_k_empty_relevant_is_perfect_by_convention():
    assert recall_at_k(["career.md"], [], k=5) == 1.0


def test_recall_at_k_no_matches():
    assert recall_at_k(["links.md"], ["career.md"], k=5) == 0.0
