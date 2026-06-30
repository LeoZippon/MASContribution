"""Prepare MASContributionBench processed task files.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --required-only
    python scripts/prepare_data.py --dataset humaneval --dataset mbpp
    python scripts/prepare_data.py --dataset mbpp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mas_contribution_bench.data.converters import (  # noqa: E402
    convert_humaneval,
    convert_marble,
    convert_mbpp,
    convert_multiagentbench,
    convert_swebench_lite,
    convert_teambench,
)


CONVERTERS = {
    "humaneval": (convert_humaneval, "humaneval", "humaneval_tasks.jsonl"),
    "mbpp": (convert_mbpp, "mbpp", "mbpp_tasks.jsonl"),
    "teambench": (convert_teambench, "teambench", "teambench_tasks.jsonl"),
    "multiagentbench": (
        convert_multiagentbench,
        "multiagentbench",
        "multiagentbench_tasks.jsonl",
    ),
    "marble": (convert_marble, "marble", "marble_tasks.jsonl"),
    "swebench_lite": (convert_swebench_lite, "swebench_lite", "swebench_lite_tasks.jsonl"),
}


FIRST_STAGE_DATASETS = ["humaneval", "mbpp"]
ALL_DATASETS = list(CONVERTERS.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw datasets into unified TaskRecord JSONL.")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=ALL_DATASETS,
        help="Dataset to convert. Can be passed multiple times. Defaults to all datasets.",
    )
    parser.add_argument(
        "--required-only",
        action="store_true",
        help="Only convert first-stage required datasets: HumanEval and MBPP.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=ROOT / "data" / "raw",
        help="Raw data root directory.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=ROOT / "data" / "processed",
        help="Processed data root directory.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failed dataset conversion.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.required_only:
        datasets = FIRST_STAGE_DATASETS
    elif args.dataset:
        datasets = args.dataset
    else:
        datasets = ALL_DATASETS

    tasks_dir = args.processed_root / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, object]] = {}
    for name in datasets:
        converter, raw_name, out_name = CONVERTERS[name]
        raw_dir = args.raw_root / raw_name
        out_file = tasks_dir / out_name

        if not raw_dir.exists():
            summary[name] = {
                "status": "skipped",
                "reason": f"missing raw directory: {raw_dir}",
                "output": str(out_file),
            }
            continue

        try:
            count = converter(raw_dir, out_file)
            summary[name] = {
                "status": "ok",
                "records": count,
                "output": str(out_file),
            }
        except Exception as exc:
            summary[name] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "output": str(out_file),
            }
            if args.fail_fast:
                break

    summary_path = args.processed_root / "prepare_data_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSummary written to: {summary_path}")

    failed = [name for name, item in summary.items() if item["status"] == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
