"""Claude Code backend: ``claude -p --output-format json``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import AgentResult, Session, extract_json, schema_instruction
from ._proc import run_cmd, which

# Tools a meta-call (compress / diagnose / judge) has no business touching.
# These calls are pure text transforms; anything that mutates the user's repo
# during one is a bug, so we take the belt-and-braces route of denying them.
_META_DENY = [
    "Bash",
    "Edit",
    "Write",
    "NotebookEdit",
    "WebFetch",
    "WebSearch",
    "Task",
    "Agent",
]


class ClaudeCodeAdapter:
    name = "claude"

    def __init__(self, *, model: str | None = None, binary: str = "claude") -> None:
        self.model = model
        self.binary = binary

    def available(self) -> bool:
        return which(self.binary) is not None

    def run(
        self,
        prompt: str,
        *,
        system: str | None = None,
        cwd: Path | None = None,
        schema: dict[str, Any] | None = None,
        tools: bool = False,
        timeout: int = 180,
        session: Session | None = None,
        allowed_tools: list[str] | None = None,
    ) -> AgentResult:
        if not self.available():
            return AgentResult(ok=False, error=f"{self.binary} not on PATH", backend=self.name)

        full_prompt = prompt + (schema_instruction(schema) if schema else "")

        argv = [self.binary, "-p", "--output-format", "json"]
        if session is None:
            # Nothing to reuse, so leave nothing behind.
            argv.append("--no-session-persistence")
        elif session.resume:
            # Branch a throwaway child: the question is answered against the
            # stored prefix without being appended to it, so the next call sees
            # the same conversation and the same cache entry.
            argv += ["--resume", session.id, "--fork-session"]
        else:
            argv += ["--session-id", session.id]
        if self.model:
            argv += ["--model", self.model]
        if system:
            argv += ["--append-system-prompt", system]
        if allowed_tools:
            # An explicit allowlist, for a call that must search but must not
            # change anything — selection is the case. Nothing outside the list
            # is reachable, so this is stricter than the deny list below rather
            # than a relaxation of it: in headless mode a tool that is not
            # allowed is refused, not prompted for.
            argv += ["--allowedTools", *allowed_tools]
        elif tools:
            # Replay needs to actually do the work; accept edits without prompting
            # but stay inside the sandbox directory we were handed.
            argv += ["--permission-mode", "acceptEdits"]
        else:
            argv += ["--disallowedTools", *_META_DENY]

        code, out, err, dur = run_cmd(
            argv, cwd=cwd, timeout=timeout, stdin=full_prompt
        )
        if code != 0 and not out.strip():
            return AgentResult(
                ok=False,
                error=(err or f"exit {code}").strip()[:2000],
                duration_s=dur,
                backend=self.name,
                raw=out,
            )

        text, tin, tout, cached, created = _parse_envelope(out)
        data = extract_json(text) if schema else None
        if schema and data is None:
            return AgentResult(
                ok=False,
                text=text,
                error="model did not return parseable JSON",
                tokens_in=tin,
                tokens_out=tout,
                cached_in=cached,
                created_in=created,
                duration_s=dur,
                backend=self.name,
                raw=out[:4000],
            )
        return AgentResult(
            ok=True,
            text=text,
            data=data,
            tokens_in=tin,
            tokens_out=tout,
            cached_in=cached,
            created_in=created,
            duration_s=dur,
            backend=self.name,
            raw=out[:4000],
        )


def _parse_envelope(stdout: str) -> tuple[str, int, int, int, int]:
    """Pull the assistant text and token usage out of ``--output-format json``.

    Falls back to treating stdout as plain text, so a change to the envelope
    shape degrades to "still works, no token accounting" rather than breaking.
    """
    stdout = (stdout or "").strip()
    if not stdout:
        return "", 0, 0, 0, 0
    try:
        payload = json.loads(stdout)
    except Exception:
        return stdout, 0, 0, 0, 0

    if isinstance(payload, list):  # stream-json transcript
        text_parts, usage = [], {}
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            if entry.get("type") == "result" and isinstance(entry.get("result"), str):
                text_parts = [entry["result"]]
                usage = entry.get("usage") or usage
            elif entry.get("usage"):
                usage = entry["usage"]
        return (
            "\n".join(text_parts).strip(),
            _tok(usage, "input"),
            _tok(usage, "output"),
            _cached(usage, "cache_read_input_tokens"),
            _cached(usage, "cache_creation_input_tokens"),
        )

    if not isinstance(payload, dict):
        return stdout, 0, 0, 0, 0

    text = payload.get("result")
    if not isinstance(text, str):
        content = payload.get("content")
        if isinstance(content, list):
            text = "".join(
                c.get("text", "") for c in content if isinstance(c, dict)
            )
        else:
            text = stdout
    usage = payload.get("usage") or {}
    return (
        (text or "").strip(),
        _tok(usage, "input"),
        _tok(usage, "output"),
        _cached(usage, "cache_read_input_tokens"),
        _cached(usage, "cache_creation_input_tokens"),
    )


def _cached(usage: Any, key: str) -> int:
    """One of the provider's cache counters, or zero if it did not report it."""
    if not isinstance(usage, dict):
        return 0
    value = usage.get(key)
    return value if isinstance(value, int) else 0


def _tok(usage: Any, side: str) -> int:
    if not isinstance(usage, dict):
        return 0
    for key in (f"{side}_tokens", f"{side}Tokens", side):
        if isinstance(usage.get(key), int):
            base = usage[key]
            break
    else:
        base = 0
    if side == "input":
        for key in ("cache_read_input_tokens", "cache_creation_input_tokens"):
            if isinstance(usage.get(key), int):
                base += usage[key]
    return base
