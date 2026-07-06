"""Agent abstractions used by the benchmark runner."""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any, Protocol

import requests


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


class DeepSeekModelClient:
    """DeepSeek chat-completions client.

    The API key is read from DEEPSEEK_API_KEY at call time. Keep the key in
    environment variables or a local .env file that is never committed.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout_seconds: float = 120.0,
        max_retries: int | None = None,
        retry_backoff_seconds: float | None = None,
    ):
        self.api_key = api_key
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        self.default_model = default_model or os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"
        self.timeout_seconds = timeout_seconds
        self.max_retries = int(os.getenv("DEEPSEEK_MAX_RETRIES", str(max_retries if max_retries is not None else 3)))
        self.retry_backoff_seconds = float(os.getenv("DEEPSEEK_RETRY_BACKOFF", str(retry_backoff_seconds if retry_backoff_seconds is not None else 2.0)))

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        api_key = self.api_key or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set. Export it before using MAS_MODEL_BACKEND=deepseek.")

        model = kwargs.get("model") or self.default_model
        if isinstance(model, str) and model.startswith("${"):
            model = self.default_model
        temperature = kwargs.get("temperature", 0.2)
        max_tokens = kwargs.get("max_tokens", 2048)

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                if response.status_code >= 500 and attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * (2 ** attempt))
                    continue
                if response.status_code >= 400:
                    raise RuntimeError(f"DeepSeek API error {response.status_code}: {response.text[:1000]}")
                data = response.json()
                break
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise RuntimeError(f"DeepSeek API request failed after {self.max_retries + 1} attempts: {exc}") from exc
                time.sleep(self.retry_backoff_seconds * (2 ** attempt))
        else:
            raise RuntimeError(f"DeepSeek API request failed: {last_error}")
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected DeepSeek response: {data}") from exc


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
        task_details = [
            f"Task ID: {task.get('task_id')}",
            f"Dataset: {task.get('dataset')}",
            f"Prompt:\n{task.get('prompt', '')}",
        ]
        if task.get("entry_point"):
            task_details.append(f"Required entry point: {task.get('entry_point')}")
        if task.get("tests"):
            task_details.append(f"Visible tests/assertions:\n{task.get('tests')}")
        user_content = (
            "\n\n".join(task_details)
            + f"\n\nRecent collaboration history:\n{history_text}"
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
