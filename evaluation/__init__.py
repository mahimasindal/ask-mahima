"""Retrieval evaluation module — offline quality checks for the RAG pipeline.

This package is a read-only *consumer* of app.services.retrieval /
app.services.config. Nothing in app/ imports from evaluation/, and nothing
here writes to the vector store or touches app/api, app/main.py, or the
deps.py dependency graph used to serve requests. It builds its own
Retriever/ChromaVectorStore instance against the same persisted collection
(see runner.py), exactly like scripts/check_retrieval.py and
tests/test_retrieval.py already do — so running an evaluation can never
change production behavior.

Layout:
    schema.py   — EvalExample / EvalDataset: the hand-labeled input format.
    metrics.py  — pure scoring functions (Recall@K), no I/O.
    runner.py   — EvaluationRunner: executes a dataset against retrieval.
    results.py  — EvalExampleResult / EvalReport: typed run output.
    report.py   — renders an EvalReport to Markdown.
    data/       — sample eval datasets (JSON).
    reports/    — generated report output (gitignored).

Entry point: scripts/evaluate_retrieval.py
"""
