"""Retrieval quality metrics.

Pure functions with no I/O and no dependency on the vector store — they
operate on plain id lists (source filenames or chunk ids) so they're
trivially unit-testable and reusable if another metric (precision@K, MRR)
is ever added alongside Recall@K.
"""

from __future__ import annotations


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of relevant_ids found anywhere among the top-k retrieved_ids.

    `retrieved_ids` is assumed to already be ranked best-first (as returned
    by Retriever.retrieve); only the first `k` are considered. Duplicate ids
    in `retrieved_ids` (e.g. two chunks from the same source file) are
    harmless since membership, not count, is what's checked.

    Returns 1.0 when relevant_ids is empty — there's nothing to miss, so
    treating it as a perfect score avoids a divide-by-zero and avoids a
    misleading 0.0 silently dragging down an average.
    """
    if not relevant_ids:
        return 1.0
    top_k = set(retrieved_ids[:k])
    hits = sum(1 for relevant_id in relevant_ids if relevant_id in top_k)
    return hits / len(relevant_ids)
