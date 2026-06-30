"""Create simple diagnostic figures from aggregated results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mas_contribution_bench.utils.io import ensure_dir, read_jsonl  # noqa: E402


def plot_score_summary(records: list[dict], path: Path) -> bool:
    if not records:
        return False
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ModuleNotFoundError as exc:
        print(f"Skip figures: missing optional dependency {exc.name}. Install matplotlib to enable plots.")
        return False

    df = pd.DataFrame.from_records(records)
    if df.empty or "mean_score" not in df:
        return False
    labels = (df["dataset"].astype(str) + "\n" + df["architecture_id"].astype(str)).tolist()
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 4.5))
    ax.bar(labels, df["mean_score"])
    ax.set_ylabel("Mean score")
    ax.set_title("Full-system score summary")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def plot_attribution_summary(records: list[dict], path: Path) -> bool:
    if not records:
        return False
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ModuleNotFoundError as exc:
        print(f"Skip figures: missing optional dependency {exc.name}. Install matplotlib to enable plots.")
        return False

    df = pd.DataFrame.from_records(records)
    if df.empty or "mean_contribution" not in df:
        return False
    grouped = df.groupby("role", as_index=False)["mean_contribution"].mean().sort_values("mean_contribution", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(grouped["role"].astype(str), grouped["mean_contribution"])
    ax.set_ylabel("Mean contribution")
    ax.set_title("Agent contribution by role")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Build diagnostic figures.")
    parser.add_argument("--results-dir", default="outputs/results", help="Aggregated results directory.")
    parser.add_argument("--output-dir", default="outputs/figures", help="Directory for figures.")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / args.results_dir
    output_dir = ensure_dir(PROJECT_ROOT / args.output_dir)
    written = []
    if plot_score_summary(read_jsonl(results_dir / "score_summary.jsonl"), output_dir / "figure_full_system_scores.png"):
        written.append(str(output_dir / "figure_full_system_scores.png"))
    if plot_attribution_summary(read_jsonl(results_dir / "attribution_summary.jsonl"), output_dir / "figure_agent_contribution.png"):
        written.append(str(output_dir / "figure_agent_contribution.png"))
    print({"written": written, "output_dir": str(output_dir)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
