"""Subprocess plumbing shared by the real adapters."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path


def which(binary: str) -> str | None:
    return shutil.which(binary)


def child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for a spawned agent.

    ``RMC_CHILD=1`` is the recursion guard: RMC's hooks see it and do nothing,
    so a compression run cannot trigger further compression runs.
    """
    env = dict(os.environ)
    env["RMC_CHILD"] = "1"
    env["RMC_DISABLE"] = "1"
    env.pop("RMC_HOME", None)  # child must not inherit a redirected store
    if extra:
        env.update(extra)
    return env


def run_cmd(
    argv: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 180,
    stdin: str | None = None,
    env_extra: dict[str, str] | None = None,
) -> tuple[int, str, str, float]:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=child_env(env_extra),
        )
        return proc.returncode, proc.stdout or "", proc.stderr or "", time.monotonic() - started
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s", time.monotonic() - started
    except FileNotFoundError as exc:
        return 127, "", str(exc), time.monotonic() - started
    except Exception as exc:  # pragma: no cover - defensive
        return 1, "", f"{type(exc).__name__}: {exc}", time.monotonic() - started
