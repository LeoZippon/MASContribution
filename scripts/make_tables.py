"""Create CSV tables from aggregated MASContributionBench results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mas_contribution_bench.utils.io import ensure_dir, read_jsonl  # noqa: E402


def write_csv(records: list[dict], path: Path) -> None:
    import pandas as pd

    ensure_dir(path.parent)
    pd.DataFrame.from_records(records).to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper-ready CSV tables.")
    parser.add_argument("--results-dir", default="outputs/results", help="Aggregated results directory.")
    parser.add_argument("--output-dir", default="outputs/tables", help="Directory for CSV tables.")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / args.results_dir
    output_dir = ensure_dir(PROJECT_ROOT / args.output_dir)
    table_specs = {
        "table_full_system_scores.csv": results_dir / "score_summary.jsonl",
        "table_agent_contribution.csv": results_dir / "attribution_summary.jsonl",
        "table_all_scores.csv": results_dir / "all_scores.jsonl",
        "table_all_attribution.csv": results_dir / "all_attribution.jsonl",
    }
    written = []
    for filename, source in table_specs.items():
        if not source.exists():
            continue
        records = read_jsonl(source)
        write_csv(records, output_dir / filename)
        written.append(str(output_dir / filename))
    print({"written": written, "output_dir": str(output_dir)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
