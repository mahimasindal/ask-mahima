"""Retrieval evaluation CLI — runs a hand-labeled eval dataset through the
retrieval pipeline and writes a Recall@K report with retrieved chunks and
distances.

This is read-only: it queries the existing ChromaDB collection through its
own Retriever instance (see evaluation/runner.py) and never writes to it or
touches the production app/api serving path.

Run:
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py evaluation/data/eval_dataset.json
    python scripts/evaluate_retrieval.py --k 1 3 5 --out evaluation/reports/latest.md
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.report import render_markdown, save_markdown
from evaluation.runner import DEFAULT_K_VALUES, EvaluationRunner
from evaluation.schema import EvalDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "data" / "eval_dataset.json"
DEFAULT_REPORT = PROJECT_ROOT / "evaluation" / "reports" / "latest.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality (Recall@K).")
    parser.add_argument(
        "dataset", nargs="?", default=str(DEFAULT_DATASET), help="Path to an eval dataset JSON file."
    )
    parser.add_argument(
        "--k", type=int, nargs="+", default=list(DEFAULT_K_VALUES), help="K values to compute Recall@K for."
    )
    parser.add_argument("--out", default=str(DEFAULT_REPORT), help="Where to write the Markdown report.")
    parser.add_argument("--quiet", action="store_true", help="Don't also print the report to stdout.")
    args = parser.parse_args()

    dataset = EvalDataset.load(args.dataset)
    if not dataset.examples:
        print(f"{args.dataset} has no examples — nothing to evaluate.")
        return

    runner = EvaluationRunner.from_settings(k_values=tuple(sorted(set(args.k))))
    report = runner.run(dataset)

    out_path = save_markdown(report, args.out)
    print(f"Wrote report to {out_path}")

    if not args.quiet:
        print()
        print(render_markdown(report))


if __name__ == "__main__":
    main()
