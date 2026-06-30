"""Agent abstractions used by the benchmark runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class ModelClient(Protocol):
    """Minimal model interface.

    A real implementation can wrap LangChain/OpenAI/etc. The benchmark core only
    requires a deterministic `complete` method returning text.
    """

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        ...


@dataclass
class AgentOutput:
    agent_id: str
    role: str
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    metadata: dict[str, Any] | None = None


class DryRunModelClient:
    """Offline model client for smoke tests and framework validation."""

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        role = kwargs.get("role", "agent")
        task_id = kwargs.get("task_id", "unknown_task")
        last = messages[-1]["content"] if messages else ""
        return (
            "{\n"
            f'  "summary": "Dry-run {role} response for {task_id}.",\n'
            f'  "artifact": {last[:500]!r},\n'
            '  "evidence": [],\n'
            '  "confidence": "low",\n'
            '  "failure_modes": []\n'
            "}"
        )


class BaseAgent:
    def __init__(
        self,
        agent_id: str,
        role: str,
        prompt: str,
        permissions: dict[str, bool],
        model_client: ModelClient | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ):
        self.agent_id = agent_id
        self.role = role
        self.prompt = prompt
        self.permissions = permissions
        self.model_client = model_client or DryRunModelClient()
        self.model_kwargs = model_kwargs or {}

    def build_messages(self, state: dict[str, Any]) -> list[dict[str, str]]:
        task = state.get("task", {})
        history = state.get("messages", [])
        history_text = "\n".join(
            f"{item.get('sender')}: {item.get('content')}" for item in history[-8:]
        )
        user_content = (
            f"Task ID: {task.get('task_id')}\n"
            f"Dataset: {task.get('dataset')}\n"
            f"Prompt:\n{task.get('prompt', '')}\n\n"
            f"Recent collaboration history:\n{history_text}"
        )
        return [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": user_content},
        ]

    def invoke(self, state: dict[str, Any]) -> AgentOutput:
        messages = self.build_messages(state)
        content = self.model_client.complete(
            messages,
            role=self.role,
            agent_id=self.agent_id,
            task_id=(state.get("task") or {}).get("task_id"),
            **self.model_kwargs,
        )
        input_tokens = sum(len(m["content"].split()) for m in messages)
        output_tokens = len(content.split())
        return AgentOutput(
            agent_id=self.agent_id,
            role=self.role,
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata={"permissions": self.permissions},
        )
