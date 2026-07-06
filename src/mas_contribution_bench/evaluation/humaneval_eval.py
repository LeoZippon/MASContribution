"""HumanEval evaluator adapter."""

from __future__ import annotations

from typing import Any

from mas_contribution_bench.evaluation.sandbox import SandboxConfig, SandboxResult, run_python_script


def extract_python_code(text: str | None) -> str:
    """Extract executable Python code from model output.

    Agents often return JSON such as {"artifact": "...code..."}, sometimes
    wrapped in ```json fences. Evaluators should execute the artifact, not the
    surrounding explanation or serialization wrapper.
    """
    if not text:
        return ""
    stripped = text.strip()

    if "```" in stripped:
        parts = stripped.split("```")
        fenced_blocks: list[str] = []
        for part in parts[1::2]:
            candidate = part.strip()
            if not candidate:
                continue
            if candidate.startswith("python"):
                return extract_python_code(candidate[len("python") :].strip())
            if candidate.startswith("json"):
                fenced_blocks.append(candidate[len("json") :].strip())
            else:
                fenced_blocks.append(candidate)
        for block in fenced_blocks:
            extracted = extract_python_code(block)
            if extracted:
                return extracted

    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            import json

            payload = json.loads(stripped)
            for key in ("artifact", "code", "final_answer", "answer", "solution"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return extract_python_code(value)
        except Exception:
            pass

    return stripped.rstrip()


def build_humaneval_program(task: dict[str, Any], prediction: str | None) -> str:
    prompt = task.get("prompt") or ""
    tests = task.get("tests") or ""
    entry_point = task.get("entry_point")
    code = extract_python_code(prediction)
    if not code:
        raise ValueError("Empty prediction")
    if entry_point and f"def {entry_point}" in code:
        candidate_code = code
    else:
        candidate_code = prompt + code
    if not entry_point:
        raise ValueError("HumanEval task is missing entry_point")
    return "\n".join(
        [
            candidate_code,
            "",
            tests,
            "",
            f"check({entry_point})",
            "",
        ]
    )


def evaluate_humaneval(
    task: dict[str, Any],
    prediction: str | None,
    *,
    timeout_seconds: float | None = None,
    sandbox_backend: str = "auto",
) -> SandboxResult:
    program = build_humaneval_program(task, prediction)
    config = SandboxConfig(
        backend=sandbox_backend,
        timeout_seconds=float(timeout_seconds or (task.get("evaluation") or {}).get("timeout_seconds") or 10.0),
    )
    return run_python_script(program, config=config)
