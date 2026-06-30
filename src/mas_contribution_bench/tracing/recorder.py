"""Trace construction from runner state."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from mas_contribution_bench.data.schemas import CostInfo, TraceEventType, TraceRecord


def build_trace_records(run_id: str, task_id: str, state: dict[str, Any]) -> list[TraceRecord]:
    records: list[TraceRecord] = []
    index = 0
    for role, output in (state.get("agent_outputs") or {}).items():
        records.append(
            TraceRecord(
                run_id=run_id,
                task_id=task_id,
                event_id=f"{run_id}:event:{index}",
                event_index=index,
                event_type=TraceEventType.MESSAGE,
                sender=role,
                receiver="next",
                role=role,
                timestamp=datetime.now(timezone.utc),
                input_tokens=int(output.get("input_tokens", 0)),
                output_tokens=int(output.get("output_tokens", 0)),
                content=output.get("content", ""),
                metadata=output.get("metadata", {}),
            )
        )
        index += 1
    records.append(
        TraceRecord(
            run_id=run_id,
            task_id=task_id,
            event_id=f"{run_id}:event:{index}",
            event_index=index,
            event_type=TraceEventType.FINAL_ANSWER,
            sender="system",
            receiver="evaluator",
            timestamp=datetime.now(timezone.utc),
            content=state.get("final_answer", ""),
        )
    )
    return records


def trace_cost(records: list[TraceRecord]) -> CostInfo:
    input_tokens = sum(record.input_tokens for record in records)
    output_tokens = sum(record.output_tokens for record in records)
    tool_calls = sum(1 for record in records if record.tool_call is not None)
    return CostInfo(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        tool_calls=tool_calls,
    )
