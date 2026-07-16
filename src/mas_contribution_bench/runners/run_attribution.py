"""Run attribution experiments with checkpointed outputs."""

from __future__ import annotations

import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from mas_contribution_bench.data.schemas import (
    AttributionMethod,
    AttributionRecord,
    CoalitionInfo,
    RemovalProtocol,
)
from mas_contribution_bench.runners.common import (
    backup_existing_file,
    completed_run_ids,
    load_experiment,
    print_progress,
    run_mas_once,
    select_architectures,
    select_tasks,
)
from mas_contribution_bench.utils.io import append_jsonl, iter_jsonl, stable_id


def _use_checkpointing() -> bool:
    return os.getenv("MAS_DISABLE_CHECKPOINT", "").lower() not in {"1", "true", "yes", "y"}


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _standard_error(values: list[float]) -> float | None:
    if len(values) <= 1:
        return None
    avg = _mean(values)
    variance = sum((x - avg) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance) / math.sqrt(len(values))


def _full_system_score_index(project_root: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Index corrected exp01 full-system scores by task, architecture, and seed."""
    runs_by_id: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(project_root / "data/runs/full_system/runs.jsonl"):
        if row.get("experiment_id") == "exp01_full_system":
            runs_by_id[str(row.get("run_id"))] = row

    index: dict[tuple[str, str, int], dict[str, Any]] = {}
    for score in iter_jsonl(project_root / "data/results/scores/full_system_scores.jsonl"):
        run = runs_by_id.get(str(score.get("run_id")))
        if not run:
            continue
        key = (str(score.get("task_id")), str(score.get("architecture_id")), int(run.get("seed", 0)))
        index[key] = {
            "score": _as_float(score.get("score")),
            "run_id": score.get("run_id"),
            "passed": score.get("passed"),
            "failure_type": score.get("failure_type"),
        }
    return index


def _completed_attribution_ids(path: Path) -> set[str]:
    return {str(row.get("attribution_id")) for row in iter_jsonl(path) if row.get("attribution_id")}


def _coalition_key(
    experiment_id: str,
    task_id: str,
    architecture_id: str,
    seed: int,
    active_agents: set[str] | list[str] | tuple[str, ...],
) -> str:
    return stable_id(
        experiment_id,
        task_id,
        architecture_id,
        seed,
        "coalition",
        sorted(active_agents),
    )


def _load_coalition_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        key = row.get("coalition_id")
        if key:
            cache[str(key)] = row
    return cache


def _cfg_int(cfg: dict[str, Any], *paths: str, default: int) -> int:
    for path in paths:
        current: Any = cfg
        ok = True
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                ok = False
                break
            current = current[part]
        if ok and current is not None:
            try:
                return int(current)
            except (TypeError, ValueError):
                pass
    return default


def _normalize_method(method: str) -> str:
    if method == "sampled_shapley":
        return "shapley_sampled"
    if method == "sampled_banzhaf":
        return "banzhaf_sampled"
    return method


def run_loo_attribution(config_path: str | Path, max_tasks: int | None = None) -> dict[str, Any]:
    experiment = load_experiment(config_path)
    tasks = select_tasks(experiment)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    architectures = select_architectures(experiment)
    seeds = [int(seed) for seed in experiment.raw.get("seeds", [0])]
    attribution_cfg = experiment.raw.get("attribution", {})
    protocol = attribution_cfg.get("primary_removal_protocol") or attribution_cfg.get(
        "removal_protocol", "null_agent_replacement"
    )
    outputs = experiment.raw.get("outputs", {})

    root = experiment.benchmark.project_root
    run_path = root / f"{outputs.get('run_dir', 'data/runs/loo')}/runs.jsonl"
    trace_path = root / f"{outputs.get('trace_dir', 'data/traces/loo')}/{experiment.experiment_id}_traces.jsonl"
    attribution_path = root / outputs.get("attribution_file", f"data/results/attribution/{experiment.experiment_id}.jsonl")
    evaluation_path = root / outputs.get("score_file", f"data/results/scores/{experiment.experiment_id}_scores.jsonl")

    fresh = os.getenv("MAS_FRESH_RUN", "").lower() in {"1", "true", "yes", "y"}
    if _use_checkpointing() and fresh:
        for file_path in (run_path, trace_path, attribution_path, evaluation_path):
            backup = backup_existing_file(file_path)
            if backup:
                print_progress(f"[backup] {file_path} -> {backup}")
            file_path.unlink(missing_ok=True)

    full_scores = _full_system_score_index(root)
    done_runs = completed_run_ids(run_path) if _use_checkpointing() else set()
    done_attr = _completed_attribution_ids(attribution_path) if _use_checkpointing() else set()

    architecture_roles = [
        (architecture_id, experiment.benchmark.architectures[architecture_id].roles)
        for architecture_id in architectures
        if architecture_id in experiment.benchmark.architectures
    ]
    total = len(tasks) * len(seeds) * sum(len(roles) for _, roles in architecture_roles)
    completed = len(done_attr)
    written_runs = 0
    written_traces = 0
    written_evaluations = 0
    written_attribution = 0
    print_progress(f"[start] experiment={experiment.experiment_id} total_attributions={total} already_done={completed}")

    for task_index, task in enumerate(tasks, start=1):
        for architecture_id, roles in architecture_roles:
            architecture = experiment.benchmark.architectures[architecture_id]
            for seed in seeds:
                full_key = (str(task["task_id"]), architecture_id, int(seed))
                full_info = full_scores.get(full_key)
                if full_info is None:
                    print_progress(f"[missing_full] task={task['task_id']} arch={architecture_id} seed={seed}; running full once")
                    full_run, _, full_eval = run_mas_once(experiment, task, architecture_id, int(seed))
                    full_info = {"score": _as_float(full_eval.score), "run_id": full_run.run_id}
                full_score = _as_float(full_info.get("score"))

                for agent in roles:
                    attribution_id = stable_id(experiment.experiment_id, task["task_id"], architecture_id, seed, agent, "loo")
                    label = f"task={task['task_id']} arch={architecture_id} seed={seed} remove={agent}"
                    if attribution_id in done_attr:
                        print_progress(f"[skip] {completed}/{total} {label} attribution_id={attribution_id}")
                        continue

                    ablated_run_id = stable_id(
                        experiment.experiment_id,
                        task["task_id"],
                        architecture_id,
                        seed,
                        sorted([agent]),
                    )
                    print_progress(f"[run] {completed + 1}/{total} task_index={task_index}/{len(tasks)} {label}")
                    if ablated_run_id in done_runs:
                        print_progress(f"[warn] ablated run already exists without attribution; rerunning {ablated_run_id}")

                    ablated_run, traces, ablated_eval = run_mas_once(
                        experiment,
                        task,
                        architecture_id,
                        int(seed),
                        removed_agents={agent},
                        removal_protocol=protocol,
                    )
                    ablated_score = _as_float(ablated_eval.score)

                    record = AttributionRecord(
                        attribution_id=attribution_id,
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
                            active_agents=[role for role in architecture.roles if role != agent],
                            removed_agents=[agent],
                        ),
                        removal_protocol=RemovalProtocol(protocol),
                        full_team_score=full_score,
                        ablated_score=ablated_score,
                        sampling_seed=int(seed),
                        metadata={
                            "full_run_id": full_info.get("run_id"),
                            "ablated_run_id": ablated_run.run_id,
                            "full_passed": full_info.get("passed"),
                            "full_failure_type": full_info.get("failure_type"),
                        },
                    )

                    append_jsonl(run_path, [ablated_run])
                    append_jsonl(trace_path, traces)
                    append_jsonl(evaluation_path, [ablated_eval])
                    append_jsonl(attribution_path, [record])

                    done_runs.add(ablated_run.run_id)
                    done_attr.add(record.attribution_id)
                    completed += 1
                    written_runs += 1
                    written_traces += len(traces)
                    written_evaluations += 1
                    written_attribution += 1
                    print_progress(
                        f"[done] {completed}/{total} attribution_id={record.attribution_id} "
                        f"full={full_score} ablated={ablated_score} contribution={record.score} "
                        f"failure={ablated_eval.failure_type} traces={len(traces)}"
                    )

    summary = {
        "records": completed,
        "new_records": written_attribution,
        "runs": written_runs,
        "traces": written_traces,
        "evaluations": written_evaluations,
        "attribution_file": str(attribution_path),
        "run_file": str(run_path),
        "trace_file": str(trace_path),
        "evaluation_file": str(evaluation_path),
        "checkpointing": _use_checkpointing(),
    }
    print_progress(f"[complete] {summary}")
    return summary


def run_coalition_attribution(config_path: str | Path, max_tasks: int | None = None) -> dict[str, Any]:
    """Run sampled Shapley/Banzhaf attribution with null-agent coalition replacement.

    A coalition is evaluated by keeping active_agents real and replacing all
    other agents with null agents through run_mas_once(..., removed_agents=...).
    """

    experiment = load_experiment(config_path)
    tasks = select_tasks(experiment)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    architectures = select_architectures(experiment)
    seeds = [int(seed) for seed in experiment.raw.get("seeds", [0])]
    attribution_cfg = experiment.raw.get("attribution", {})
    methods = [_normalize_method(str(m)) for m in attribution_cfg.get("methods", [])]
    if not methods:
        method = attribution_cfg.get("method", "shapley_sampled")
        methods = [_normalize_method(str(method))]

    methods = [m for m in methods if m in {"shapley_sampled", "banzhaf_sampled"}]
    if not methods:
        methods = ["shapley_sampled"]

    protocol = attribution_cfg.get("primary_removal_protocol") or attribution_cfg.get(
        "removal_protocol", "null_agent_replacement"
    )
    protocol = str(protocol or "null_agent_replacement")

    shapley_samples = _cfg_int(
        attribution_cfg,
        "shapley.num_permutations",
        "shapley.num_samples",
        "num_samples",
        default=8,
    )
    banzhaf_samples = _cfg_int(
        attribution_cfg,
        "banzhaf.num_coalitions",
        "banzhaf.num_samples",
        "num_samples",
        default=8,
    )

    outputs = experiment.raw.get("outputs", {})
    root = experiment.benchmark.project_root

    run_path = root / f"{outputs.get('run_dir', 'data/runs/shapley')}/runs.jsonl"
    trace_path = root / f"{outputs.get('trace_dir', 'data/traces/shapley')}/{experiment.experiment_id}_traces.jsonl"
    attribution_path = root / outputs.get(
        "attribution_file",
        f"data/results/attribution/{experiment.experiment_id}.jsonl",
    )
    coalition_path = root / outputs.get(
        "coalition_file",
        f"data/results/attribution/{experiment.experiment_id}_coalitions.jsonl",
    )
    evaluation_path = root / outputs.get(
        "score_file",
        f"data/results/scores/{experiment.experiment_id}_scores.jsonl",
    )

    fresh = os.getenv("MAS_FRESH_RUN", "").lower() in {"1", "true", "yes", "y"}
    if _use_checkpointing() and fresh:
        for file_path in (run_path, trace_path, attribution_path, coalition_path, evaluation_path):
            backup = backup_existing_file(file_path)
            if backup:
                print_progress(f"[backup] {file_path} -> {backup}")
            file_path.unlink(missing_ok=True)

    full_scores = _full_system_score_index(root)
    done_attr = _completed_attribution_ids(attribution_path) if _use_checkpointing() else set()
    coalition_cache = _load_coalition_cache(coalition_path) if _use_checkpointing() else {}

    architecture_roles = [
        (architecture_id, experiment.benchmark.architectures[architecture_id].roles)
        for architecture_id in architectures
        if architecture_id in experiment.benchmark.architectures
    ]

    total = len(tasks) * len(seeds) * sum(len(roles) for _, roles in architecture_roles) * len(methods)
    completed = len(done_attr)
    written_runs = 0
    written_traces = 0
    written_evaluations = 0
    written_attribution = 0
    written_coalitions = 0

    print_progress(
        f"[start] experiment={experiment.experiment_id} methods={methods} "
        f"total_attributions={total} already_done={completed}"
    )

    def evaluate_coalition(
        task: dict[str, Any],
        architecture_id: str,
        seed: int,
        roles: list[str],
        active_agents: set[str],
    ) -> dict[str, Any]:
        active_agents = set(active_agents)
        removed_agents = set(roles) - active_agents
        coalition_id = _coalition_key(
            experiment.experiment_id,
            str(task["task_id"]),
            architecture_id,
            seed,
            active_agents,
        )

        cached = coalition_cache.get(coalition_id)
        if cached is not None:
            return cached

        full_key = (str(task["task_id"]), architecture_id, int(seed))
        if len(active_agents) == len(roles) and full_key in full_scores:
            full_info = full_scores[full_key]
            row = {
                "coalition_id": coalition_id,
                "experiment_id": experiment.experiment_id,
                "task_id": task["task_id"],
                "dataset": task["dataset"],
                "architecture_id": architecture_id,
                "sampling_seed": int(seed),
                "active_agents": sorted(active_agents),
                "removed_agents": sorted(removed_agents),
                "score": _as_float(full_info.get("score")),
                "run_id": full_info.get("run_id"),
                "passed": full_info.get("passed"),
                "failure_type": full_info.get("failure_type"),
                "source": "exp01_full_system_cache",
            }
            coalition_cache[coalition_id] = row
            append_jsonl(coalition_path, [row])
            return row

        print_progress(
            f"[coalition] task={task['task_id']} arch={architecture_id} seed={seed} "
            f"active={sorted(active_agents)} removed={sorted(removed_agents)}"
        )
        run, traces, evaluation = run_mas_once(
            experiment,
            task,
            architecture_id,
            int(seed),
            removed_agents=removed_agents,
            removal_protocol=protocol,
        )
        score = _as_float(evaluation.score)
        row = {
            "coalition_id": coalition_id,
            "experiment_id": experiment.experiment_id,
            "task_id": task["task_id"],
            "dataset": task["dataset"],
            "architecture_id": architecture_id,
            "sampling_seed": int(seed),
            "active_agents": sorted(active_agents),
            "removed_agents": sorted(removed_agents),
            "score": score,
            "run_id": run.run_id,
            "passed": getattr(evaluation, "passed", None),
            "failure_type": getattr(evaluation, "failure_type", None),
            "source": "coalition_run",
        }
        coalition_cache[coalition_id] = row

        append_jsonl(run_path, [run])
        append_jsonl(trace_path, traces)
        append_jsonl(evaluation_path, [evaluation])
        append_jsonl(coalition_path, [row])

        nonlocal written_runs, written_traces, written_evaluations, written_coalitions
        written_runs += 1
        written_traces += len(traces)
        written_evaluations += 1
        written_coalitions += 1
        return row

    for task_index, task in enumerate(tasks, start=1):
        for architecture_id, roles in architecture_roles:
            roles = list(roles)
            all_agents = set(roles)

            for seed in seeds:
                full_info = evaluate_coalition(task, architecture_id, int(seed), roles, all_agents)
                full_score = _as_float(full_info.get("score"))

                for method in methods:
                    rng = random.Random(
                        stable_id(
                            experiment.experiment_id,
                            task["task_id"],
                            architecture_id,
                            seed,
                            method,
                        )
                    )

                    marginals_by_agent: dict[str, list[float]] = defaultdict(list)
                    example_coalition_by_agent: dict[str, list[str]] = {}
                    example_permutation_by_agent: dict[str, list[str]] = {}

                    if method == "shapley_sampled":
                        for sample_index in range(shapley_samples):
                            permutation = roles[:]
                            rng.shuffle(permutation)

                            active: set[str] = set()
                            prev_score = _as_float(
                                evaluate_coalition(task, architecture_id, int(seed), roles, active).get("score")
                            )

                            for agent in permutation:
                                before = set(active)
                                active.add(agent)
                                current_score = _as_float(
                                    evaluate_coalition(task, architecture_id, int(seed), roles, active).get("score")
                                )
                                marginal = current_score - prev_score
                                marginals_by_agent[agent].append(marginal)
                                example_coalition_by_agent.setdefault(agent, sorted(before))
                                example_permutation_by_agent.setdefault(agent, permutation[:])
                                prev_score = current_score

                            print_progress(
                                f"[sample] method={method} task={task['task_id']} arch={architecture_id} "
                                f"seed={seed} sample={sample_index + 1}/{shapley_samples}"
                            )

                    elif method == "banzhaf_sampled":
                        for sample_index in range(banzhaf_samples):
                            for agent in roles:
                                others = [role for role in roles if role != agent]
                                subset = {role for role in others if rng.random() < 0.5}
                                with_agent = set(subset)
                                with_agent.add(agent)

                                without_score = _as_float(
                                    evaluate_coalition(task, architecture_id, int(seed), roles, subset).get("score")
                                )
                                with_score = _as_float(
                                    evaluate_coalition(task, architecture_id, int(seed), roles, with_agent).get("score")
                                )
                                marginal = with_score - without_score
                                marginals_by_agent[agent].append(marginal)
                                example_coalition_by_agent.setdefault(agent, sorted(subset))

                            print_progress(
                                f"[sample] method={method} task={task['task_id']} arch={architecture_id} "
                                f"seed={seed} sample={sample_index + 1}/{banzhaf_samples}"
                            )

                    for agent in roles:
                        sample_count = len(marginals_by_agent.get(agent, []))
                        attribution_id = stable_id(
                            experiment.experiment_id,
                            task["task_id"],
                            architecture_id,
                            seed,
                            method,
                            agent,
                        )
                        label = (
                            f"method={method} task={task['task_id']} arch={architecture_id} "
                            f"seed={seed} agent={agent}"
                        )
                        if attribution_id in done_attr:
                            print_progress(f"[skip] {completed}/{total} {label} attribution_id={attribution_id}")
                            continue

                        values = marginals_by_agent.get(agent, [])
                        score = _mean(values)
                        stderr = _standard_error(values)

                        active_example = example_coalition_by_agent.get(agent, [])
                        removed_example = [role for role in roles if role not in set(active_example)]

                        record = AttributionRecord(
                            attribution_id=attribution_id,
                            experiment_id=experiment.experiment_id,
                            task_id=task["task_id"],
                            dataset=task["dataset"],
                            architecture_id=architecture_id,
                            agent_id=agent,
                            role=agent,
                            method=AttributionMethod(method),
                            utility_type=attribution_cfg.get("utility", "task"),
                            score=score,
                            baseline_score=0.0,
                            coalition=CoalitionInfo(
                                active_agents=active_example,
                                removed_agents=removed_example,
                            ),
                            removal_protocol=RemovalProtocol(protocol),
                            full_team_score=full_score,
                            ablated_score=None,
                            sampling_seed=int(seed),
                            num_samples=sample_count,
                            permutation_order=example_permutation_by_agent.get(agent),
                            standard_error=stderr,
                            metadata={
                                "method": method,
                                "marginal_values": values,
                                "coalition_cache_size": len(coalition_cache),
                                "shapley_samples": shapley_samples if method == "shapley_sampled" else None,
                                "banzhaf_samples": banzhaf_samples if method == "banzhaf_sampled" else None,
                            },
                        )

                        append_jsonl(attribution_path, [record])
                        done_attr.add(record.attribution_id)
                        completed += 1
                        written_attribution += 1

                        print_progress(
                            f"[done] {completed}/{total} method={method} attribution_id={record.attribution_id} "
                            f"agent={agent} score={score} samples={sample_count} stderr={stderr}"
                        )

    summary = {
        "records": completed,
        "new_records": written_attribution,
        "runs": written_runs,
        "traces": written_traces,
        "evaluations": written_evaluations,
        "coalitions": written_coalitions,
        "attribution_file": str(attribution_path),
        "coalition_file": str(coalition_path),
        "run_file": str(run_path),
        "trace_file": str(trace_path),
        "evaluation_file": str(evaluation_path),
        "checkpointing": _use_checkpointing(),
        "methods": methods,
        "shapley_samples": shapley_samples,
        "banzhaf_samples": banzhaf_samples,
    }
    print_progress(f"[complete] {summary}")
    return summary


def run_attribution(config_path: str | Path, max_tasks: int | None = None) -> dict[str, Any]:
    experiment = load_experiment(config_path)
    attribution_cfg = experiment.raw.get("attribution", {})
    method = _normalize_method(str(attribution_cfg.get("method", "")))
    methods = [_normalize_method(str(m)) for m in attribution_cfg.get("methods", [])]

    if (
        experiment.experiment_id == "exp03_loo_attribution"
        or method == "loo"
        or "loo" in methods
    ):
        return run_loo_attribution(config_path, max_tasks=max_tasks)

    if (
        method in {"shapley_sampled", "banzhaf_sampled"}
        or any(m in {"shapley_sampled", "banzhaf_sampled"} for m in methods)
        or "shapley" in experiment.experiment_id
        or "banzhaf" in experiment.experiment_id
    ):
        return run_coalition_attribution(config_path, max_tasks=max_tasks)

    return run_loo_attribution(config_path, max_tasks=max_tasks)