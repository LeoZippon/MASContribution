"""Graph execution layer.

The class exposes a LangGraph-compatible boundary but also provides a pure
Python fallback used by tests and dry-runs. This keeps the benchmark runnable
before a real LLM backend is configured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mas_contribution_bench.agents.base import BaseAgent
from mas_contribution_bench.graphs.architectures import execution_order


@dataclass
class MASExecutionResult:
    state: dict[str, Any]
    final_answer: str


class MASGraphBuilder:
    def __init__(
        self,
        architecture,
        agents: dict[str, BaseAgent],
        removed_agents: set[str] | None = None,
        null_replacement: bool = False,
    ):
        self.architecture = architecture
        self.agents = agents
        self.removed_agents = removed_agents or set()
        self.null_replacement = null_replacement

    def build(self):
        return self

    def invoke(self, state: dict[str, Any]) -> MASExecutionResult:
        state = dict(state)
        state.setdefault("messages", [])
        state.setdefault("agent_outputs", {})
        order = execution_order(self.architecture)
        for role in order:
            if role not in self.agents:
                continue
            if role in self.removed_agents and not self.null_replacement:
                continue
            if role in self.removed_agents and self.null_replacement:
                content = '{"summary": "null agent replacement", "artifact": "", "evidence": [], "confidence": "low", "failure_modes": []}'
                output = {
                    "agent_id": role,
                    "role": role,
                    "content": content,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "tool_calls": 0,
                    "metadata": {"null_agent": True},
                }
            else:
                agent_output = self.agents[role].invoke(state)
                output = {
                    "agent_id": agent_output.agent_id,
                    "role": agent_output.role,
                    "content": agent_output.content,
                    "input_tokens": agent_output.input_tokens,
                    "output_tokens": agent_output.output_tokens,
                    "tool_calls": agent_output.tool_calls,
                    "metadata": agent_output.metadata or {},
                }
            state["agent_outputs"][role] = output
            state["messages"].append({"sender": role, "receiver": "next", "content": output["content"]})
        final_role = self._final_role(order)
        final_answer = ""
        if final_role and final_role in state["agent_outputs"]:
            final_answer = state["agent_outputs"][final_role]["content"]
        elif state["agent_outputs"]:
            final_answer = list(state["agent_outputs"].values())[-1]["content"]
        state["final_answer"] = final_answer
        return MASExecutionResult(state=state, final_answer=final_answer)

    def _final_role(self, order: list[str]) -> str | None:
        for preferred in ("finalizer", "aggregator", "supervisor", "verifier", "coder", "executor"):
            if preferred in order:
                return preferred
        return order[-1] if order else None
