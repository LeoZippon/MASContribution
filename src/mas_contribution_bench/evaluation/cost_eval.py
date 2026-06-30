"""Cost utility helpers."""

from __future__ import annotations

from mas_contribution_bench.data.schemas import CostInfo


def net_score(task_score: float | None, cost: CostInfo, token_penalty: float = 0.0) -> float:
    base = float(task_score or 0.0)
    return base - token_penalty * float(cost.total_tokens)


def aggregate_cost(costs: list[CostInfo]) -> CostInfo:
    return CostInfo(
        input_tokens=sum(c.input_tokens for c in costs),
        output_tokens=sum(c.output_tokens for c in costs),
        total_tokens=sum(c.total_tokens for c in costs),
        tool_calls=sum(c.tool_calls for c in costs),
        estimated_cost_usd=sum(c.estimated_cost_usd or 0.0 for c in costs) or None,
    )
