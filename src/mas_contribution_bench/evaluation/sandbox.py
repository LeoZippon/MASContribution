"""Sandboxed execution for generated Python code.

The evaluator runs untrusted model-generated code. Prefer the Docker backend
for formal experiments. The subprocess backend is provided only for local smoke
tests on trusted code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any


@dataclass
class SandboxConfig:
    backend: str = "auto"
    timeout_seconds: float = 10.0
    docker_image: str = "python:3.10-slim"
    cpus: str = "1"
    memory: str = "512m"
    pids_limit: int = 128
    network_disabled: bool = True
    keep_tmpdir: bool = False
    extra_docker_args: list[str] = field(default_factory=list)


@dataclass
class SandboxResult:
    passed: bool
    timed_out: bool
    exit_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    backend: str
    tmpdir: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def docker_available() -> bool:
    return shutil.which("docker") is not None


def _resolve_backend(config: SandboxConfig) -> str:
    backend = config.backend or "auto"
    if backend == "auto":
        return "docker" if docker_available() else "subprocess"
    if backend not in {"docker", "subprocess"}:
        raise ValueError(f"Unsupported sandbox backend: {backend}")
    return backend


def _docker_command(tmpdir: Path, script_name: str, config: SandboxConfig) -> list[str]:
    cmd = ["docker", "run", "--rm"]
    if config.network_disabled:
        cmd += ["--network", "none"]
    cmd += [
        "--cpus",
        str(config.cpus),
        "--memory",
        str(config.memory),
        "--pids-limit",
        str(config.pids_limit),
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,size=64m",
    ]
    cmd += config.extra_docker_args
    cmd += [
        "-v",
        f"{tmpdir}:/work:ro",
        "-w",
        "/work",
        config.docker_image,
        "python",
        script_name,
    ]
    return cmd


def _subprocess_command(script_name: str) -> list[str]:
    return ["python", script_name]


def run_python_script(
    script: str,
    *,
    config: SandboxConfig | None = None,
    filename: str = "evaluate_one.py",
) -> SandboxResult:
    config = config or SandboxConfig(
        backend=os.getenv("MAS_SANDBOX_BACKEND", "auto"),
        timeout_seconds=float(os.getenv("MAS_EVAL_TIMEOUT", "10")),
        docker_image=os.getenv("MAS_SANDBOX_IMAGE", "python:3.10-slim"),
    )
    backend = _resolve_backend(config)
    started = time.monotonic()
    tmp_obj = tempfile.TemporaryDirectory(prefix="mas_eval_")
    tmpdir = Path(tmp_obj.name)
    script_path = tmpdir / filename
    script_path.write_text(script, encoding="utf-8")
    if backend == "docker":
        cmd = _docker_command(tmpdir, filename, config)
        cwd = None
    else:
        cmd = _subprocess_command(filename)
        cwd = tmpdir
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            timeout=config.timeout_seconds,
            capture_output=True,
            text=True,
            check=False,
        )
        result = SandboxResult(
            passed=proc.returncode == 0,
            timed_out=False,
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration_seconds=time.monotonic() - started,
            backend=backend,
            tmpdir=str(tmpdir) if config.keep_tmpdir else None,
            metadata={"command": cmd},
        )
    except subprocess.TimeoutExpired as exc:
        result = SandboxResult(
            passed=False,
            timed_out=True,
            exit_code=None,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            duration_seconds=time.monotonic() - started,
            backend=backend,
            tmpdir=str(tmpdir) if config.keep_tmpdir else None,
            metadata={"command": cmd},
        )
    finally:
        if config.keep_tmpdir:
            tmp_obj._finalizer.detach()  # type: ignore[attr-defined]
        else:
            tmp_obj.cleanup()
    return result
