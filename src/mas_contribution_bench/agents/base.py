"""Agent abstractions used by the benchmark runner."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, ClassVar, Protocol

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

    _shared_cache_indexes: ClassVar[dict[str, dict[str, dict[str, Any]]]] = {}

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
        self.last_usage: dict[str, Any] = {}
        self.last_cache_metadata: dict[str, Any] = {}

    def _cache_enabled(self) -> bool:
        return os.getenv("MAS_LLM_CACHE_ENABLED", "1").lower() not in {"0", "false", "no", "n"}

    def _cache_path(self, model: str) -> Path:
        configured = os.getenv("MAS_LLM_CACHE_FILE")
        if configured:
            return Path(configured)
        cache_dir = Path(os.getenv("MAS_LLM_CACHE_DIR", "data/cache/llm_calls"))
        safe_model = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in model)
        return cache_dir / f"deepseek_{safe_model}.jsonl"

    def _cache_key(self, payload: dict[str, Any]) -> str:
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _load_cache_index(self, path: Path) -> dict[str, dict[str, Any]]:
        cache_id = str(path)
        if cache_id in self._shared_cache_indexes:
            return self._shared_cache_indexes[cache_id]
        index: dict[str, dict[str, Any]] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = row.get("cache_key")
                    if key:
                        index[str(key)] = row
        self._shared_cache_indexes[cache_id] = index
        return index

    def _append_cache_row(self, path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.last_usage = {}
        self.last_cache_metadata = {}
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
        cache_path = self._cache_path(str(model))
        cache_key = self._cache_key(payload)
        if self._cache_enabled():
            cache_index = self._load_cache_index(cache_path)
            cached = cache_index.get(cache_key)
            if cached is not None:
                self.last_usage = dict(cached.get("usage") or {})
                self.last_cache_metadata = {
                    "local_cache_hit": True,
                    "local_cache_key": cache_key,
                    "local_cache_path": str(cache_path),
                    "provider_cache_hit_tokens": self.last_usage.get("prompt_cache_hit_tokens", 0),
                    "provider_cache_miss_tokens": self.last_usage.get("prompt_cache_miss_tokens", 0),
                }
                return str(cached.get("content", ""))

        api_key = self.api_key or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set. Export it before using MAS_MODEL_BACKEND=deepseek.")

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
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected DeepSeek response: {data}") from exc
        self.last_usage = dict(data.get("usage") or {})
        self.last_cache_metadata = {
            "local_cache_hit": False,
            "local_cache_key": cache_key,
            "local_cache_path": str(cache_path),
            "provider_cache_hit_tokens": self.last_usage.get("prompt_cache_hit_tokens", 0),
            "provider_cache_miss_tokens": self.last_usage.get("prompt_cache_miss_tokens", 0),
        }
        if self._cache_enabled():
            row = {
                "cache_key": cache_key,
                "created_at": int(time.time()),
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": messages,
                "content": content,
                "usage": self.last_usage,
            }
            self._append_cache_row(cache_path, row)
            self._load_cache_index(cache_path)[cache_key] = row
        return content


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
        permission_lines = [
            f"- {name}: {str(value).lower()}"
            for name, value in sorted(self.permissions.items())
        ]
        task_details.append(
            "Current role permissions:\n"
            + "\n".join(permission_lines)
            + "\nOnly perform actions and claim authority that are enabled above."
        )
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
        usage = dict(getattr(self.model_client, "last_usage", {}) or {})
        cache_metadata = dict(getattr(self.model_client, "last_cache_metadata", {}) or {})
        input_tokens = sum(len(m["content"].split()) for m in messages)
        output_tokens = len(content.split())
        if usage:
            input_tokens = int(usage.get("prompt_tokens") or input_tokens)
            output_tokens = int(usage.get("completion_tokens") or output_tokens)
        return AgentOutput(
            agent_id=self.agent_id,
            role=self.role,
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata={
                "permissions": self.permissions,
                "model_usage": usage,
                "cache": cache_metadata,
            },
        )
