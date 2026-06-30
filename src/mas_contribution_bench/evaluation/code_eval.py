"""Code-task evaluators for HumanEval and MBPP."""

from __future__ import annotations

from typing import Any

from mas_contribution_bench.data.schemas import EvaluationRecord, FailureType
from mas_contribution_bench.evaluation.humaneval_eval import evaluate_humaneval
from mas_contribution_bench.evaluation.mbpp_eval import evaluate_mbpp
from mas_contribution_bench.evaluation.sandbox import SandboxResult


def score_text_answer(final_answer: str | None, reference: str | None = None) -> tuple[float, bool, FailureType]:
    if not final_answer or not final_answer.strip():
        return 0.0, False, FailureType.EMPTY_OUTPUT
    if reference and final_answer.strip() == reference.strip():
        return 1.0, True, FailureType.NONE
    return 0.0, False, FailureType.UNKNOWN


def failure_type_from_sandbox(result: SandboxResult) -> FailureType:
    if result.passed:
        return FailureType.NONE
    if result.timed_out:
        return FailureType.TIMEOUT
    stderr = result.stderr or ""
    if "SyntaxError" in stderr or "IndentationError" in stderr:
        return FailureType.SYNTAX_ERROR
    if "AssertionError" in stderr:
        return FailureType.TEST_FAILURE
    if "Traceback" in stderr:
        return FailureType.RUNTIME_ERROR
    return FailureType.TEST_FAILURE


def evaluate_code_prediction(
    task: dict[str, Any],
    prediction: str | None,
    *,
    sandbox_backend: str = "auto",
) -> tuple[float, bool, FailureType, dict[str, Any]]:
    dataset = str(task.get("dataset", "")).lower()
    timeout = (task.get("evaluation") or {}).get("timeout_seconds")
    try:
        if dataset == "humaneval":
            result = evaluate_humaneval(
                task,
                prediction,
                timeout_seconds=timeout,
                sandbox_backend=sandbox_backend,
            )
        elif dataset == "mbpp":
            result = evaluate_mbpp(
                task,
                prediction,
                timeout_seconds=timeout,
                sandbox_backend=sandbox_backend,
            )
        else:
            score, passed, failure_type = score_text_answer(prediction, task.get("reference_solution"))
            return score, passed, failure_type, {"evaluator": "text_fallback"}
    except ValueError as exc:
        return 0.0, False, FailureType.INVALID_FORMAT, {"error": str(exc)}
    except Exception as exc:
        return 0.0, False, FailureType.EVALUATOR_ERROR, {"error": repr(exc)}
    failure_type = failure_type_from_sandbox(result)
    return (
        1.0 if result.passed else 0.0,
        result.passed,
        failure_type,
        {
            "backend": result.backend,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
            "duration_seconds": result.duration_seconds,
            "metadata": result.metadata,
        },
    )


def evaluate_task_output(
    run_id: str,
    task: dict[str, Any],
    architecture_id: str,
    final_answer: str | None,
    cost=None,
    execute_code: bool = False,
    sandbox_backend: str = "auto",
) -> EvaluationRecord:
    if execute_code:
        score, passed, failure_type, raw_output = evaluate_code_prediction(
            task,
            final_answer,
            sandbox_backend=sandbox_backend,
        )
    else:
        score, passed, failure_type = score_text_answer(final_answer, task.get("reference_solution"))
        raw_output = {"safe_mode": True, "code_execution": False}
    return EvaluationRecord(
        run_id=run_id,
        task_id=task["task_id"],
        dataset=task["dataset"],
        architecture_id=architecture_id,
        final_answer=final_answer,
        score=score,
        metric=(task.get("evaluation") or {}).get("metric", "pass_at_1"),
        passed=passed,
        failure_type=failure_type,
        evaluator=task["evaluation"],
        cost=cost,
        raw_evaluator_output=raw_output,
    )
