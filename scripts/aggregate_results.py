"""Aggregate run, score, trace, and attribution JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mas_contribution_bench.utils.io import ensure_dir, iter_jsonl, write_jsonl  # noqa: E402


def collect_jsonl(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not root.exists():
        return records
    for path in sorted(root.rglob("*.jsonl")):
        for record in iter_jsonl(path):
            record = dict(record)
            record["_source_file"] = str(path.relative_to(PROJECT_ROOT))
            records.append(record)
    return records


def summarize_scores(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any], list[float]] = defaultdict(list)
    for record in scores:
        score = record.get("score")
        if score is None:
            continue
        grouped[(record.get("dataset"), record.get("architecture_id"))].append(float(score))
    rows = []
    for (dataset, architecture_id), values in sorted(grouped.items()):
        rows.append(
            {
                "dataset": dataset,
                "architecture_id": architecture_id,
                "n": len(values),
                "mean_score": sum(values) / len(values),
                "min_score": min(values),
                "max_score": max(values),
            }
        )
    return rows


def summarize_attribution(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any, Any, Any], list[float]] = defaultdict(list)
    for record in records:
        score = record.get("score")
        if score is None:
            continue
        key = (
            record.get("dataset"),
            record.get("architecture_id"),
            record.get("method"),
            record.get("role"),
        )
        grouped[key].append(float(score))
    rows = []
    for (dataset, architecture_id, method, role), values in sorted(grouped.items()):
        rows.append(
            {
                "dataset": dataset,
                "architecture_id": architecture_id,
                "method": method,
                "role": role,
                "n": len(values),
                "mean_contribution": sum(values) / len(values),
                "min_contribution": min(values),
                "max_contribution": max(values),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate MASContributionBench result files.")
    parser.add_argument("--runs-dir", default="data/runs", help="Directory containing run JSONL files.")
    parser.add_argument("--scores-dir", default="data/results/scores", help="Directory containing score JSONL files.")
    parser.add_argument("--attribution-dir", default="data/results/attribution", help="Directory containing attribution JSONL files.")
    parser.add_argument("--output-dir", default="outputs/results", help="Output directory for aggregated JSONL/JSON.")
    args = parser.parse_args()

    runs = collect_jsonl(PROJECT_ROOT / args.runs_dir)
    scores = collect_jsonl(PROJECT_ROOT / args.scores_dir)
    attribution = collect_jsonl(PROJECT_ROOT / args.attribution_dir)

    output_dir = ensure_dir(PROJECT_ROOT / args.output_dir)
    write_jsonl(output_dir / "all_runs.jsonl", runs)
    write_jsonl(output_dir / "all_scores.jsonl", scores)
    write_jsonl(output_dir / "all_attribution.jsonl", attribution)
    score_summary = summarize_scores(scores)
    attribution_summary = summarize_attribution(attribution)
    write_jsonl(output_dir / "score_summary.jsonl", score_summary)
    write_jsonl(output_dir / "attribution_summary.jsonl", attribution_summary)

    summary = {
        "runs": len(runs),
        "scores": len(scores),
        "attribution": len(attribution),
        "score_summary_rows": len(score_summary),
        "attribution_summary_rows": len(attribution_summary),
        "run_status": dict(Counter(record.get("status") for record in runs)),
        "output_dir": str(output_dir),
    }
    (output_dir / "aggregate_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
