"""Run full-system MAS experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mas_contribution_bench.runners.common import (
    load_experiment,
    run_mas_once,
    select_architectures,
    select_tasks,
    write_run_bundle,
)


def run_full_system(config_path: str | Path, max_tasks: int | None = None) -> dict[str, Any]:
    experiment = load_experiment(config_path)
    tasks = select_tasks(experiment)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    architectures = select_architectures(experiment)
    seeds = experiment.raw.get("seeds", [0])
    runs, traces, evaluations = [], [], []
    for seed in seeds:
        for architecture_id in architectures:
            if architecture_id not in experiment.benchmark.architectures:
                continue
            for task in tasks:
                run, trace, evaluation = run_mas_once(experiment, task, architecture_id, int(seed))
                runs.append(run)
                traces.extend(trace)
                evaluations.append(evaluation)
    outputs = experiment.raw.get("outputs", {})
    return write_run_bundle(
        experiment,
        runs,
        traces,
        evaluations,
        run_file=f"{outputs.get('run_dir', 'data/runs/full_system')}/runs.jsonl",
        trace_file=f"{outputs.get('trace_dir', 'data/traces')}/{experiment.experiment_id}_traces.jsonl",
        evaluation_file=outputs.get("score_file", f"data/results/scores/{experiment.experiment_id}.jsonl"),
    )
