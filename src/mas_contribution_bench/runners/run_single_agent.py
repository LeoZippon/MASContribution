"""Run baseline experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mas_contribution_bench.runners.run_full_system import run_full_system


def run_single_agent_baseline(config_path: str | Path, max_tasks: int | None = None) -> dict[str, Any]:
    return run_full_system(config_path, max_tasks=max_tasks)
