"""Factory functions for role agents."""

from __future__ import annotations

from typing import Any

from mas_contribution_bench.agents.base import BaseAgent, ModelClient
from mas_contribution_bench.config.loaders import LoadedAgentSpec


def build_agent(
    spec: LoadedAgentSpec,
    model_client: ModelClient | None = None,
    model_overrides: dict[str, Any] | None = None,
) -> BaseAgent:
    model_kwargs = {
        "model": spec.model,
        "temperature": spec.temperature,
        "max_tokens": spec.max_tokens,
    }
    if model_overrides:
        model_kwargs.update(model_overrides)
    return BaseAgent(
        agent_id=spec.role,
        role=spec.role,
        prompt=spec.prompt,
        permissions=spec.permissions,
        model_client=model_client,
        model_kwargs=model_kwargs,
    )


def build_agents(
    specs: dict[str, LoadedAgentSpec],
    roles: list[str],
    model_client: ModelClient | None = None,
    model_overrides: dict[str, Any] | None = None,
) -> dict[str, BaseAgent]:
    missing = [role for role in roles if role not in specs]
    if missing:
        raise KeyError(f"Unknown agent roles: {missing}")
    return {
        role: build_agent(specs[role], model_client=model_client, model_overrides=model_overrides)
        for role in roles
    }
