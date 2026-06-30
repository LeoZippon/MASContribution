"""MBPP evaluator adapter."""

from __future__ import annotations

import ast
import re
from typing import Any

from mas_contribution_bench.evaluation.humaneval_eval import extract_python_code
from mas_contribution_bench.evaluation.sandbox import SandboxConfig, SandboxResult, run_python_script


IMPORT_RE = re.compile(r"^(?:from\s+\S+\s+import\s+.+|import\s+.+)$")


def _literal_list_from_line(line: str) -> list[str] | None:
    try:
        value = ast.literal_eval(line.strip())
    except Exception:
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    return None


def _clean_statement(text: str) -> str:
    item = text.strip()
    while item and item[0] in "[,'\" ":
        item = item[1:].strip()
    while item and item[-1] in "],'\" ":
        item = item[:-1].strip()
    return item


def _split_asserts(text: str) -> list[str]:
    starts = [match.start() for match in re.finditer(r"\bassert\s+", text)]
    statements: list[str] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        statement = _clean_statement(text[start:end])
        if statement.startswith("assert "):
            statements.append(statement)
    return statements


def parse_mbpp_tests(tests_blob: str | None) -> tuple[list[str], list[str]]:
    if not tests_blob:
        return [], []
    imports: list[str] = []
    asserts: list[str] = []
    lines = [line.strip() for line in tests_blob.splitlines() if line.strip()]
    for line in lines:
        literal = _literal_list_from_line(line)
        if literal is not None:
            for item in literal:
                item = item.strip()
                if IMPORT_RE.match(item):
                    imports.append(item)
                elif item.startswith("assert "):
                    asserts.append(item)
            continue
        asserts.extend(_split_asserts(line))
        if IMPORT_RE.match(line):
            imports.append(line)
    if not asserts:
        asserts.extend(_split_asserts(tests_blob))
    return imports, asserts


def build_mbpp_program(task: dict[str, Any], prediction: str | None) -> str:
    code = extract_python_code(prediction)
    if not code:
        raise ValueError("Empty prediction")
    imports, asserts = parse_mbpp_tests(task.get("tests"))
    if not asserts:
        raise ValueError("MBPP task has no assert tests")
    return "\n".join(
        [
            *imports,
            "",
            code,
            "",
            *asserts,
            "",
        ]
    )


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
