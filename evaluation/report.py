"""Human-readable Markdown report for an EvalReport.

Shows, per query: the retrieved chunks (source, chunk id, distance, text
preview) and Recall@K, plus a summary table across the whole dataset. Plain
string building — no template engine, since nothing else in this project
pulls one in and the format is simple enough not to need one.

Note on "distance" vs "similarity score": RetrievedChunk.distance is
ChromaDB's raw distance metric (lower = more similar — see
app/models/retrieval.py), not a normalized cosine similarity. This report
shows that raw distance directly rather than inventing a `1 - distance`
"similarity score", since Chroma's default metric isn't guaranteed to be
cosine and that conversion could misrepresent it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from app.models.retrieval import RetrievedChunk
from evaluation.results import EvalExampleResult, EvalReport


def _format_chunk_row(rank: int, chunk: RetrievedChunk) -> str:
    preview = chunk.text.replace("\n", " ").replace("|", "\\|").strip()
    if len(preview) > 140:
        preview = preview[:140] + "..."
    return f"| {rank} | {chunk.source} | {chunk.chunk_id} | {chunk.distance:.4f} | {preview} |"


def _format_example(result: EvalExampleResult) -> str:
    recall_str = ", ".join(f"R@{k}={v:.2f}" for k, v in sorted(result.recall_at_k.items()))
    lines = [
        f"### {result.query}",
        "",
        f"**Relevant sources:** {', '.join(result.relevant_sources)}",
        f"**Recall@K:** {recall_str}",
        "",
    ]
    if not result.retrieved:
        lines.append("_No chunks retrieved._")
        lines.append("")
        return "\n".join(lines)

    lines.append("| Rank | Source | Chunk ID | Distance (lower = more similar) | Preview |")
    lines.append("|---|---|---|---|---|")
    for i, chunk in enumerate(result.retrieved, start=1):
        lines.append(_format_chunk_row(i, chunk))
    lines.append("")
    return "\n".join(lines)


def render_markdown(report: EvalReport) -> str:
    lines = [
        f"# Retrieval Evaluation Report — {report.dataset_name}",
        "",
        f"Examples evaluated: {len(report.results)}",
        "",
        "## Summary",
        "",
        "| K | Mean Recall@K |",
        "|---|---|",
    ]
    for k in report.k_values:
        lines.append(f"| {k} | {report.mean_recall_at_k[k]:.3f} |")
    lines.append("")
    lines.append("## Per-query results")
    lines.append("")
    for result in report.results:
        lines.append(_format_example(result))
    return "\n".join(lines)


def save_markdown(report: EvalReport, path: Union[str, Path]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(report), encoding="utf-8")
    return out_path
