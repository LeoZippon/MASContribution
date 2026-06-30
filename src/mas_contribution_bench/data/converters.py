"""Dataset converters for MASContributionBench.

Each converter reads one raw dataset and writes JSONL rows that validate
against TaskRecord. The converters deliberately keep raw provenance so every
processed task can be traced back to the original dataset file.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    import pandas as pd
except ImportError:  # pragma: no cover - handled at runtime with a clear error.
    pd = None

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from .schemas import (
    ContributionMetadata,
    DatasetName,
    EvaluationConfig,
    EvaluatorType,
    MASMetadata,
    SourceInfo,
    TaskDifficulty,
    TaskRecord,
    TaskType,
)


CONVERSION_VERSION = "v1"


def _require_pandas() -> None:
    if pd is None:
        raise ImportError(
            "pandas is required to read parquet files. Install project dependencies with "
            "`pip install -r requirements.txt`."
        )


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _write_jsonl(path: Path, rows: Iterable[TaskRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            payload = row.model_dump(mode="json")
            f.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")
            count += 1
    return count


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _safe_read_text(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _stringify_tests(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value)


def _first_present(record: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        return value
    return None


def _count_constraints(text: str | None) -> int | None:
    if not text:
        return None
    markers = ["must", "should", "required", "requirement", "only", "after", "before", "需要", "必须"]
    lowered = text.lower()
    return sum(lowered.count(marker) for marker in markers)


def _code_task_metadata(
    *, requires_research: bool = False, estimated_agents: list[str] | None = None
) -> MASMetadata:
    return MASMetadata(
        requires_planning=True,
        requires_coding=True,
        requires_verification=True,
        requires_research=requires_research,
        requires_tool_use=True,
        estimated_agents=estimated_agents or ["planner", "coder", "verifier"],
    )


def _default_contribution_metadata(
    *,
    roles: list[str] | None = None,
    architectures: list[str] | None = None,
    intervention_tags: list[str] | None = None,
) -> ContributionMetadata:
    return ContributionMetadata(
        eligible_roles=roles or ["planner", "coder", "verifier", "critic"],
        default_architectures=architectures or ["A1_pev", "A2_chain", "A3_dag", "A5_star", "A6_debate"],
        permission_requirements=["read_task", "write_solution", "run_tests"],
        intervention_tags=intervention_tags or ["topology", "role", "permission"],
    )


def convert_humaneval(raw_dir: Path, out_file: Path) -> int:
    """Convert OpenAI HumanEval parquet into TaskRecord JSONL."""

    _require_pandas()
    parquet = _first_existing(raw_dir.rglob("*.parquet"))
    if parquet is None:
        raise FileNotFoundError(f"No HumanEval parquet file found under {raw_dir}")

    df = pd.read_parquet(parquet)

    def rows() -> Iterable[TaskRecord]:
        for idx, record in enumerate(df.to_dict("records")):
            raw_id = str(record.get("task_id") or f"HumanEval/{idx}")
            prompt = str(record.get("prompt") or "")
            tests = record.get("test")
            yield TaskRecord(
                task_id=f"humaneval/{raw_id.split('/')[-1]}",
                dataset=DatasetName.HUMANEVAL,
                split="test",
                task_type=TaskType.CODE_GENERATION,
                prompt=prompt,
                output_format="python_function",
                entry_point=record.get("entry_point"),
                reference_solution=record.get("canonical_solution"),
                tests=_stringify_tests(tests),
                evaluation=EvaluationConfig(
                    evaluator_type=EvaluatorType.UNIT_TEST,
                    metric="pass_at_1",
                    timeout_seconds=10,
                    sandbox="python",
                ),
                mas_metadata=_code_task_metadata(),
                difficulty=TaskDifficulty(
                    source="not_computed",
                    level="unknown",
                    input_length=len(prompt),
                    constraint_count=_count_constraints(prompt),
                    requires_cross_file_edit=False,
                ),
                contribution_metadata=_default_contribution_metadata(),
                source=SourceInfo(
                    raw_dataset="HumanEval",
                    raw_task_id=raw_id,
                    raw_file_path=str(parquet.relative_to(raw_dir.parents[1])),
                    conversion_version=CONVERSION_VERSION,
                ),
                metadata={"raw_columns": sorted(record.keys())},
            )

    return _write_jsonl(out_file, rows())


def convert_mbpp(raw_dir: Path, out_file: Path, subset: str = "sanitized") -> int:
    """Convert MBPP parquet files into TaskRecord JSONL.

    By default this uses the sanitized split, because it is the cleaner version
    commonly used for code-generation evaluation.
    """

    _require_pandas()
    subset_dir = raw_dir / subset
    if not subset_dir.exists():
        subset_dir = raw_dir / "full"
    if not subset_dir.exists():
        raise FileNotFoundError(f"No MBPP subset directory found under {raw_dir}")

    parquet_files = sorted(subset_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No MBPP parquet files found under {subset_dir}")

    def iter_records() -> Iterable[tuple[str, dict[str, Any], Path]]:
        for parquet in parquet_files:
            split = parquet.name.split("-")[0]
            df = pd.read_parquet(parquet)
            for record in df.to_dict("records"):
                yield split, record, parquet

    def rows() -> Iterable[TaskRecord]:
        for idx, (split, record, parquet) in enumerate(iter_records()):
            raw_task_id = record.get("task_id", idx)
            prompt = str(record.get("prompt") or record.get("text") or "")
            code = record.get("code")
            tests = _first_present(
                record,
                ["test_list", "tests", "challenge_test_list", "test"],
            )
            test_imports = record.get("test_imports")
            test_setup = record.get("test_setup_code")
            tests_text = _stringify_tests(tests)
            if test_imports is not None:
                tests_text = _stringify_tests(test_imports) + "\n" + (tests_text or "")
            if test_setup is not None:
                tests_text = str(test_setup) + "\n" + (tests_text or "")

            yield TaskRecord(
                task_id=f"mbpp/{subset}/{split}/{raw_task_id}",
                dataset=DatasetName.MBPP,
                split=split,
                task_type=TaskType.CODE_GENERATION,
                prompt=prompt,
                output_format="python_function",
                reference_solution=code,
                tests=tests_text,
                evaluation=EvaluationConfig(
                    evaluator_type=EvaluatorType.UNIT_TEST,
                    metric="pass_at_1",
                    timeout_seconds=10,
                    sandbox="python",
                ),
                mas_metadata=_code_task_metadata(),
                difficulty=TaskDifficulty(
                    source="not_computed",
                    level="unknown",
                    input_length=len(prompt),
                    constraint_count=_count_constraints(prompt),
                    requires_cross_file_edit=False,
                ),
                contribution_metadata=_default_contribution_metadata(),
                source=SourceInfo(
                    raw_dataset=f"MBPP/{subset}",
                    raw_task_id=str(raw_task_id),
                    raw_file_path=str(parquet.relative_to(raw_dir.parents[1])),
                    conversion_version=CONVERSION_VERSION,
                ),
                metadata={"raw_columns": sorted(record.keys())},
            )

    return _write_jsonl(out_file, rows())


def convert_teambench(raw_dir: Path, out_file: Path) -> int:
    """Convert TeamBench shared metadata and task folders into TaskRecord JSONL."""

    dataset_file = raw_dir / "shared" / "teambench_dataset.json"
    if not dataset_file.exists():
        raise FileNotFoundError(f"Missing TeamBench dataset file: {dataset_file}")

    data = json.loads(dataset_file.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        records = list(data.values())
    else:
        records = list(data)

    tasks_dir = raw_dir / "tasks"
    raw_id_counts = Counter(str(record.get("task_id") or record.get("id")) for record in records)
    raw_id_seen: defaultdict[str, int] = defaultdict(int)

    def rows() -> Iterable[TaskRecord]:
        for record in records:
            raw_task_id = str(record.get("task_id") or record.get("id"))
            occurrence_index = raw_id_seen[raw_task_id]
            raw_id_seen[raw_task_id] += 1

            actual_task_id = raw_task_id
            if raw_id_counts[raw_task_id] > 1:
                seeded_task_id = f"{raw_task_id}_seed{occurrence_index}"
                if (tasks_dir / seeded_task_id).exists():
                    actual_task_id = seeded_task_id

            task_dir = tasks_dir / actual_task_id
            title = record.get("title") or raw_task_id
            category = record.get("category")
            difficulty = str(record.get("difficulty") or "unknown").lower()
            if difficulty not in {"unknown", "easy", "medium", "hard", "expert"}:
                difficulty = "unknown"

            brief = _safe_read_text(task_dir / "brief.md")
            spec = _safe_read_text(task_dir / "spec.md")
            task_yaml = None
            if yaml is not None and (task_dir / "task.yaml").exists():
                task_yaml = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8")) or {}

            prompt_parts = [f"Task: {title}"]
            if category:
                prompt_parts.append(f"Category: {category}")
            if brief:
                prompt_parts.append("Brief:\n" + brief)
            if spec:
                prompt_parts.append("Spec:\n" + spec)
            if not brief and not spec and task_yaml:
                prompt_parts.append(json.dumps(task_yaml, ensure_ascii=False, indent=2))
            prompt = "\n\n".join(prompt_parts)

            grade_path = task_dir / "grade.sh"
            setup_path = task_dir / "setup.sh"
            workspace_path = task_dir / "workspace"

            yield TaskRecord(
                task_id=f"teambench/{actual_task_id}",
                dataset=DatasetName.TEAMBENCH,
                split=str(record.get("split") or "test"),
                task_type=TaskType.SOFTWARE_ENGINEERING,
                prompt=prompt,
                context=spec,
                input_format="task_brief_and_workspace",
                output_format="workspace_patch_or_final_answer",
                tests=_safe_read_text(grade_path),
                evaluation=EvaluationConfig(
                    evaluator_type=EvaluatorType.OFFICIAL_GRADER if grade_path.exists() else EvaluatorType.CUSTOM,
                    metric="official_score",
                    timeout_seconds=120,
                    sandbox="shell",
                    script_path=str(grade_path.relative_to(raw_dir.parents[1])) if grade_path.exists() else None,
                    extra={
                        "setup_script": str(setup_path.relative_to(raw_dir.parents[1])) if setup_path.exists() else None,
                        "workspace_path": str(workspace_path.relative_to(raw_dir.parents[1])) if workspace_path.exists() else None,
                    },
                ),
                mas_metadata=MASMetadata(
                    requires_planning=True,
                    requires_coding=True,
                    requires_verification=True,
                    requires_research=False,
                    requires_tool_use=True,
                    estimated_agents=["planner", "coder", "verifier"],
                ),
                difficulty=TaskDifficulty(
                    source="teambench_metadata",
                    level=difficulty,  # type: ignore[arg-type]
                    input_length=len(prompt),
                    constraint_count=_count_constraints(prompt),
                    requires_cross_file_edit=workspace_path.exists(),
                ),
                contribution_metadata=_default_contribution_metadata(
                    roles=["planner", "coder", "verifier", "critic", "supervisor"],
                    architectures=["A1_pev", "A2_chain", "A3_dag", "A5_star"],
                    intervention_tags=["role", "permission", "topology", "information"],
                ),
                source=SourceInfo(
                    raw_dataset="TeamBench",
                    raw_task_id=actual_task_id,
                    raw_file_path=str(dataset_file.relative_to(raw_dir.parents[1])),
                    conversion_version=CONVERSION_VERSION,
                    extra={
                        "base_task_id": raw_task_id,
                        "duplicate_occurrence_index": occurrence_index
                        if raw_id_counts[raw_task_id] > 1
                        else None,
                    },
                ),
                metadata={
                    "title": title,
                    "category": category,
                    "base_task_id": raw_task_id,
                    "actual_task_id": actual_task_id,
                    "task_dir_exists": task_dir.exists(),
                    "has_generator": record.get("has_generator"),
                    "has_brief": brief is not None,
                    "has_spec": spec is not None,
                    "has_grade": grade_path.exists(),
                    "raw": record,
                },
            )

    return _write_jsonl(out_file, rows())


def convert_multiagentbench(raw_dir: Path, out_file: Path) -> int:
    """Convert MultiAgentBench JSONL scenarios into TaskRecord JSONL."""

    jsonl_files = sorted(raw_dir.glob("*/*_main.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No MultiAgentBench *_main.jsonl files found under {raw_dir}")

    def rows() -> Iterable[TaskRecord]:
        for jsonl_file in jsonl_files:
            scenario = jsonl_file.parent.name
            for record in _read_jsonl(jsonl_file):
                raw_task_id = str(record.get("task_id"))
                task = record.get("task") or {}
                task_content = task.get("content") if isinstance(task, dict) else str(task)
                output_format = task.get("output_format") if isinstance(task, dict) else None
                agents = record.get("agents") or []
                relationships = record.get("relationships") or []
                metrics = record.get("metrics") or {}

                role_names = []
                for agent in agents:
                    if isinstance(agent, dict):
                        role_names.append(str(agent.get("type") or agent.get("agent_id") or "agent"))

                task_type = {
                    "coding": TaskType.CODE_GENERATION,
                    "database": TaskType.DATABASE,
                    "research": TaskType.RESEARCH,
                    "bargaining": TaskType.BARGAINING,
                }.get(scenario, TaskType.GENERAL_MAS)

                yield TaskRecord(
                    task_id=f"multiagentbench/{scenario}/{raw_task_id}",
                    dataset=DatasetName.MULTIAGENTBENCH,
                    split="test",
                    task_type=task_type,
                    prompt=str(task_content or ""),
                    context=json.dumps(
                        {
                            "agents": agents,
                            "relationships": relationships,
                            "environment": record.get("environment"),
                            "memory": record.get("memory"),
                            "engine_planner": record.get("engine_planner"),
                        },
                        ensure_ascii=False,
                    ),
                    input_format="multi_agent_scenario",
                    output_format=output_format,
                    evaluation=EvaluationConfig(
                        evaluator_type=EvaluatorType.ENVIRONMENT_REWARD,
                        metric="scenario_score",
                        timeout_seconds=None,
                        sandbox=str((record.get("environment") or {}).get("type") or ""),
                        extra={"metrics": metrics},
                    ),
                    mas_metadata=MASMetadata(
                        requires_planning=True,
                        requires_coding=scenario in {"coding", "database"},
                        requires_verification=True,
                        requires_research=scenario == "research",
                        requires_tool_use=scenario in {"coding", "database", "research"},
                        estimated_agents=[str(a.get("agent_id")) for a in agents if isinstance(a, dict) and a.get("agent_id")],
                        extra={"coordinate_mode": record.get("coordinate_mode")},
                    ),
                    difficulty=TaskDifficulty(
                        source="not_computed",
                        level="unknown",
                        input_length=len(str(task_content or "")),
                        constraint_count=_count_constraints(str(task_content or "")),
                        requires_cross_file_edit=scenario == "coding",
                    ),
                    contribution_metadata=_default_contribution_metadata(
                        roles=role_names or ["planner", "coder", "verifier", "researcher"],
                        architectures=["A5_star", "A6_debate", "A7_graph"],
                        intervention_tags=["topology", "role", "communication", "permission"],
                    ),
                    source=SourceInfo(
                        raw_dataset=f"MultiAgentBench/{scenario}",
                        raw_task_id=raw_task_id,
                        raw_file_path=str(jsonl_file.relative_to(raw_dir.parents[1])),
                        conversion_version=CONVERSION_VERSION,
                    ),
                    metadata={
                        "scenario": scenario,
                        "relationships": relationships,
                        "raw_keys": sorted(record.keys()),
                    },
                )

    return _write_jsonl(out_file, rows())


def convert_marble(raw_dir: Path, out_file: Path) -> int:
    """Create lightweight TaskRecord rows from MARBLE configs.

    MARBLE is primarily a framework with environments/configs rather than a
    single flat task dataset. This converter records runnable config-like
    entries so the benchmark can track MARBLE as an external-validation source.
    MultiAgentBench tasks embedded in MARBLE should be converted with
    convert_multiagentbench.
    """

    config_files = sorted((raw_dir / "marble" / "configs").glob("*.yaml"))
    if yaml is None:
        config_files = []
    if not config_files:
        # Keep the output file valid but empty if there are no parseable configs.
        return _write_jsonl(out_file, [])

    def rows() -> Iterable[TaskRecord]:
        for config_file in config_files:
            config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
            prompt = json.dumps(config, ensure_ascii=False, indent=2)
            env = config.get("environment") if isinstance(config, dict) else None
            env_name = env.get("name") if isinstance(env, dict) else config_file.stem
            yield TaskRecord(
                task_id=f"marble/{config_file.stem}",
                dataset=DatasetName.MARBLE,
                split="test",
                task_type=TaskType.GENERAL_MAS,
                prompt=prompt,
                context=prompt,
                input_format="marble_config",
                output_format="environment_result",
                evaluation=EvaluationConfig(
                    evaluator_type=EvaluatorType.ENVIRONMENT_REWARD,
                    metric="environment_score",
                    extra={"environment": env},
                ),
                mas_metadata=MASMetadata(
                    requires_planning=True,
                    requires_coding=False,
                    requires_verification=True,
                    requires_research=False,
                    requires_tool_use=True,
                    estimated_agents=[],
                ),
                difficulty=TaskDifficulty(
                    source="not_computed",
                    level="unknown",
                    input_length=len(prompt),
                    constraint_count=_count_constraints(prompt),
                ),
                contribution_metadata=_default_contribution_metadata(
                    roles=["planner", "researcher", "coder", "verifier", "supervisor"],
                    architectures=["A5_star", "A7_graph"],
                    intervention_tags=["topology", "communication", "permission"],
                ),
                source=SourceInfo(
                    raw_dataset="MARBLE",
                    raw_task_id=config_file.stem,
                    raw_file_path=str(config_file.relative_to(raw_dir.parents[1])),
                    conversion_version=CONVERSION_VERSION,
                ),
                metadata={"environment_name": env_name, "raw_config": config},
            )

    return _write_jsonl(out_file, rows())


def convert_swebench_lite(raw_dir: Path, out_file: Path) -> int:
    """Convert SWE-bench Lite parquet files into TaskRecord JSONL."""

    _require_pandas()
    parquet_files = sorted((raw_dir / "data").glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No SWE-bench Lite parquet files found under {raw_dir / 'data'}")

    def rows() -> Iterable[TaskRecord]:
        for parquet in parquet_files:
            split = parquet.name.split("-")[0]
            df = pd.read_parquet(parquet)
            for idx, record in enumerate(df.to_dict("records")):
                instance_id = str(record.get("instance_id") or record.get("task_id") or idx)
                problem = str(record.get("problem_statement") or record.get("prompt") or "")
                repo = record.get("repo")
                base_commit = record.get("base_commit")
                tests = record.get("test_patch") or record.get("FAIL_TO_PASS")
                yield TaskRecord(
                    task_id=f"swebench_lite/{split}/{instance_id}",
                    dataset=DatasetName.SWEBENCH_LITE,
                    split=split,
                    task_type=TaskType.SOFTWARE_ENGINEERING,
                    prompt=problem,
                    context=json.dumps(
                        {
                            "repo": repo,
                            "base_commit": base_commit,
                            "hints_text": record.get("hints_text"),
                            "created_at": record.get("created_at"),
                        },
                        ensure_ascii=False,
                        default=_json_default,
                    ),
                    input_format="github_issue",
                    output_format="patch",
                    reference_solution=record.get("patch"),
                    tests=_stringify_tests(tests),
                    evaluation=EvaluationConfig(
                        evaluator_type=EvaluatorType.OFFICIAL_GRADER,
                        metric="resolved",
                        timeout_seconds=1800,
                        sandbox="swebench_harness",
                        extra={"repo": repo, "base_commit": base_commit},
                    ),
                    mas_metadata=MASMetadata(
                        requires_planning=True,
                        requires_coding=True,
                        requires_verification=True,
                        requires_research=True,
                        requires_tool_use=True,
                        estimated_agents=["planner", "researcher", "coder", "verifier"],
                    ),
                    difficulty=TaskDifficulty(
                        source="dataset_type",
                        level="hard",
                        input_length=len(problem),
                        constraint_count=_count_constraints(problem),
                        requires_cross_file_edit=True,
                    ),
                    contribution_metadata=_default_contribution_metadata(
                        roles=["planner", "researcher", "coder", "verifier", "critic"],
                        architectures=["A2_chain", "A3_dag", "A4_metagpt_lite", "A7_graph"],
                        intervention_tags=["role", "permission", "information", "topology"],
                    ),
                    source=SourceInfo(
                        raw_dataset="SWE-bench_Lite",
                        raw_task_id=instance_id,
                        raw_file_path=str(parquet.relative_to(raw_dir.parents[1])),
                        conversion_version=CONVERSION_VERSION,
                    ),
                    metadata={"raw_columns": sorted(record.keys())},
                )

    return _write_jsonl(out_file, rows())


def convert_all(raw_root: Path, processed_root: Path, include_optional: bool = True) -> dict[str, int | str]:
    """Run all available converters and return dataset -> row count/status."""

    tasks_dir = processed_root / "tasks"
    results: dict[str, int | str] = {}

    jobs = [
        ("humaneval", convert_humaneval, raw_root / "humaneval", tasks_dir / "humaneval_tasks.jsonl"),
        ("mbpp", convert_mbpp, raw_root / "mbpp", tasks_dir / "mbpp_tasks.jsonl"),
    ]
    if include_optional:
        jobs.extend(
            [
                ("teambench", convert_teambench, raw_root / "teambench", tasks_dir / "teambench_tasks.jsonl"),
                (
                    "multiagentbench",
                    convert_multiagentbench,
                    raw_root / "multiagentbench",
                    tasks_dir / "multiagentbench_tasks.jsonl",
                ),
                ("marble", convert_marble, raw_root / "marble", tasks_dir / "marble_tasks.jsonl"),
                (
                    "swebench_lite",
                    convert_swebench_lite,
                    raw_root / "swebench_lite",
                    tasks_dir / "swebench_lite_tasks.jsonl",
                ),
            ]
        )

    for name, func, raw_dir, out_file in jobs:
        if not raw_dir.exists():
            results[name] = f"skipped: missing {raw_dir}"
            continue
        try:
            results[name] = func(raw_dir, out_file)
        except Exception as exc:  # Keep other datasets convertible.
            results[name] = f"failed: {type(exc).__name__}: {exc}"

    return results
