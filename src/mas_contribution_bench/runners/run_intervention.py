"""Intervention runner placeholders with schema-compatible outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mas_contribution_bench.runners.run_attribution import run_loo_attribution


def run_intervention(config_path: str | Path, max_tasks: int | None = None) -> dict[str, Any]:
    return run_loo_attribution(config_path, max_tasks=max_tasks)
