"""MBPP evaluator adapter."""

from __future__ import annotations

import ast
import re
from typing import Any

from mas_contribution_bench.evaluation.humaneval_eval import extract_python_code
from mas_contribution_bench.evaluation.sandbox import SandboxConfig, SandboxResult, run_python_script


IMPORT_RE = re.compile(r"^(?:from\s+\S+\s+import\s+.+|import\s+.+)$")
ASSERT_RE = re.compile(r"^assert\s+.+")


def _as_statement_list(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        literal = ast.literal_eval(text)
    except Exception:
        literal = None
    if isinstance(literal, (list, tuple)):
        return [str(item).strip() for item in literal if str(item).strip()]
    return [line.strip().rstrip(",") for line in text.splitlines() if line.strip()]


def parse_mbpp_tests(tests_blob: str | None) -> tuple[list[str], list[str]]:
    """Return import/setup statements and assert statements for an MBPP task."""
    imports_or_setup: list[str] = []
    asserts: list[str] = []
    for statement in _as_statement_list(tests_blob):
        if statement == "[]":
            continue
        if IMPORT_RE.match(statement):
            imports_or_setup.append(statement)
        elif ASSERT_RE.match(statement):
            asserts.append(statement)
        elif "assert " in statement:
            # Fallback for legacy processed rows that accidentally packed many
            # asserts into one string. This is intentionally conservative.
            for part in re.split(r"(?=\bassert\s+)", statement):
                part = part.strip().strip(",")
                if ASSERT_RE.match(part):
                    asserts.append(part)
        else:
            imports_or_setup.append(statement)
    return imports_or_setup, asserts


def _ensure_entry_point_present(code: str, entry_point: str | None) -> None:
    if entry_point and not re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", code, flags=re.MULTILINE):
        raise ValueError(f"Prediction does not define required MBPP entry point: {entry_point}")


def build_mbpp_program(task: dict[str, Any], prediction: str | None) -> str:
    code = extract_python_code(prediction)
    if not code:
        raise ValueError("Empty prediction")
    imports_or_setup, asserts = parse_mbpp_tests(task.get("tests"))
    if not asserts:
        raise ValueError("MBPP task has no assert tests")
    _ensure_entry_point_present(code, task.get("entry_point"))
    return "\n".join([*imports_or_setup, "", code, "", *asserts, ""])


def evaluate_mbpp(
    task: dict[str, Any],
    prediction: str | None,
    *,
    timeout_seconds: float | None = None,
    sandbox_backend: str = "auto",
) -> SandboxResult:
    program = build_mbpp_program(task, prediction)
    config = SandboxConfig(
        backend=sandbox_backend,
        timeout_seconds=float(timeout_seconds or (task.get("evaluation") or {}).get("timeout_seconds") or 10.0),
    )
    return run_python_script(program, config=config)
