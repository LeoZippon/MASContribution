"""Run attribution experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mas_contribution_bench.data.schemas import (
    AttributionMethod,
    AttributionRecord,
    CoalitionInfo,
    RemovalProtocol,
)
from mas_contribution_bench.runners.common import (
    load_experiment,
    run_mas_once,
    select_architectures,
    select_tasks,
)
from mas_contribution_bench.utils.io import stable_id, write_jsonl


def run_loo_attribution(config_path: str | Path, max_tasks: int | None = None) -> dict[str, Any]:
    experiment = load_experiment(config_path)
    tasks = select_tasks(experiment)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    architectures = select_architectures(experiment)
    seeds = experiment.raw.get("seeds", [0])
    attribution_cfg = experiment.raw.get("attribution", {})
    protocol = attribution_cfg.get("primary_removal_protocol") or attribution_cfg.get("removal_protocol", "null_agent_replacement")
    records: list[AttributionRecord] = []
    for seed in seeds:
        for architecture_id in architectures:
            architecture = experiment.benchmark.architectures.get(architecture_id)
            if architecture is None:
                continue
            for task in tasks:
                full_run, _, full_eval = run_mas_once(experiment, task, architecture_id, int(seed))
                full_score = float(full_eval.score or 0.0)
                for agent in architecture.roles:
                    ablated_run, _, ablated_eval = run_mas_once(
                        experiment,
                        task,
                        architecture_id,
                        int(seed),
                        removed_agents={agent},
                        removal_protocol=protocol,
                    )
                    ablated_score = float(ablated_eval.score or 0.0)
                    records.append(
                        AttributionRecord(
                            attribution_id=stable_id(experiment.experiment_id, task["task_id"], architecture_id, seed, agent, "loo"),
                            experiment_id=experiment.experiment_id,
                            task_id=task["task_id"],
                            dataset=task["dataset"],
                            architecture_id=architecture_id,
                            agent_id=agent,
                            role=agent,
                            method=AttributionMethod.LOO,
                            score=full_score - ablated_score,
                            baseline_score=0.0,
                            coalition=CoalitionInfo(
                                active_agents=[r for r in architecture.roles if r != agent],
                                removed_agents=[agent],
                            ),
                            removal_protocol=RemovalProtocol(protocol),
                            full_team_score=full_score,
                            ablated_score=ablated_score,
                            sampling_seed=int(seed),
                            metadata={"full_run_id": full_run.run_id, "ablated_run_id": ablated_run.run_id},
                        )
                    )
    out = experiment.raw.get("outputs", {}).get("attribution_file", f"data/results/attribution/{experiment.experiment_id}.jsonl")
    out_path = experiment.benchmark.project_root / out
    write_jsonl(out_path, records)
    return {"records": len(records), "attribution_file": str(out_path)}


def run_attribution(config_path: str | Path, max_tasks: int | None = None) -> dict[str, Any]:
    return run_loo_attribution(config_path, max_tasks=max_tasks)
