"""Shared runner primitives."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any

from mas_contribution_bench.agents import build_agents
from mas_contribution_bench.config import ExperimentSpec, load_experiment_spec
from mas_contribution_bench.data.loaders import load_jsonl_tasks
from mas_contribution_bench.data.schemas import (
    CoalitionInfo,
    RemovalInfo,
    RemovalProtocol,
    RunRecord,
    RunStatus,
)
from mas_contribution_bench.evaluation import evaluate_task_output
from mas_contribution_bench.graphs import MASGraphBuilder
from mas_contribution_bench.tracing import build_trace_records, trace_cost
from mas_contribution_bench.utils.io import append_jsonl, ensure_dir, stable_id, write_jsonl
from mas_contribution_bench.utils.seeds import set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def should_execute_code(experiment: ExperimentSpec) -> bool:
    env_value = os.getenv("MAS_EXECUTE_CODE")
    if env_value is not None:
        return env_value.lower() in {"1", "true", "yes", "y"}
    evaluation_cfg = experiment.raw.get("evaluation") or {}
    return bool(evaluation_cfg.get("execute_code", False))


def sandbox_backend(experiment: ExperimentSpec) -> str:
    return (
        os.getenv("MAS_SANDBOX_BACKEND")
        or (experiment.raw.get("evaluation") or {}).get("sandbox_backend")
        or "auto"
    )


def load_experiment(config_path: str | Path, project_root: str | Path = PROJECT_ROOT) -> ExperimentSpec:
    return load_experiment_spec(config_path, project_root)


def select_tasks(experiment: ExperimentSpec, seed: int | None = None) -> list[dict[str, Any]]:
    tasks = []
    root = experiment.benchmark.project_root
    for dataset_spec in experiment.raw.get("datasets", []):
        task_file = root / dataset_spec["task_file"]
        loaded = load_jsonl_tasks(task_file)
        split = dataset_spec.get("split")
        if split:
            loaded = [task for task in loaded if task.split == split]
        max_tasks = dataset_spec.get("max_tasks")
        if max_tasks is not None:
            loaded = loaded[: int(max_tasks)]
        tasks.extend(task.model_dump(mode="json") for task in loaded)
    return tasks


def select_architectures(experiment: ExperimentSpec) -> list[str]:
    raw = experiment.raw
    if isinstance(raw.get("architectures"), dict):
        return list(raw["architectures"].get("include", []))
    if raw.get("base_architectures"):
        return list(raw["base_architectures"])
    return []


def run_mas_once(
    experiment: ExperimentSpec,
    task: dict[str, Any],
    architecture_id: str,
    seed: int,
    removed_agents: set[str] | None = None,
    removal_protocol: str = "none",
) -> tuple[RunRecord, list[Any], Any]:
    set_seed(seed)
    architecture = experiment.benchmark.architectures[architecture_id]
    active_roles = [role for role in architecture.roles if role not in (removed_agents or set())]
    agents = build_agents(experiment.benchmark.agents, architecture.roles, model_overrides=experiment.raw.get("model", {}))
    null_replacement = removal_protocol == "null_agent_replacement"
    graph = MASGraphBuilder(
        architecture=architecture,
        agents=agents,
        removed_agents=removed_agents or set(),
        null_replacement=null_replacement,
    ).build()
    run_id = stable_id(experiment.experiment_id, task["task_id"], architecture_id, seed, sorted(removed_agents or []))
    started = datetime.now(timezone.utc)
    result = graph.invoke({"task": task, "messages": []})
    trace_records = build_trace_records(run_id, task["task_id"], result.state)
    cost = trace_cost(trace_records)
    evaluation = evaluate_task_output(
        run_id=run_id,
        task=task,
        architecture_id=architecture_id,
        final_answer=result.final_answer,
        cost=cost,
        execute_code=should_execute_code(experiment),
        sandbox_backend=sandbox_backend(experiment),
    )
    ended = datetime.now(timezone.utc)
    run = RunRecord(
        run_id=run_id,
        experiment_id=experiment.experiment_id,
        task_id=task["task_id"],
        dataset=task["dataset"],
        architecture_id=architecture_id,
        seed=seed,
        coalition=CoalitionInfo(active_agents=active_roles, removed_agents=sorted(removed_agents or [])),
        removal=RemovalInfo(protocol=RemovalProtocol(removal_protocol), removed_agents=sorted(removed_agents or [])),
        config_hash=experiment.config_hash,
        started_at=started,
        ended_at=ended,
        status=RunStatus.SUCCEEDED,
        cost=cost,
        failure_type=evaluation.failure_type,
        metadata={"dry_run": not should_execute_code(experiment)},
    )
    return run, trace_records, evaluation


def write_run_bundle(
    experiment: ExperimentSpec,
    runs: list[RunRecord],
    traces: list[Any],
    evaluations: list[Any],
    run_file: str | Path,
    trace_file: str | Path,
    evaluation_file: str | Path,
) -> dict[str, Any]:
    root = experiment.benchmark.project_root
    run_path = root / run_file
    trace_path = root / trace_file
    eval_path = root / evaluation_file
    write_jsonl(run_path, runs)
    write_jsonl(trace_path, traces)
    write_jsonl(eval_path, evaluations)
    return {
        "runs": len(runs),
        "traces": len(traces),
        "evaluations": len(evaluations),
        "run_file": str(run_path),
        "trace_file": str(trace_path),
        "evaluation_file": str(eval_path),
    }
