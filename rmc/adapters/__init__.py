"""Execution backends. One interface, three implementations.

RMC needs to spawn *fresh* agent processes for two jobs:

1. meta-calls (compress, diagnose, judge) — no tools, structured output;
2. replay (re-run a recorded task against a candidate lesson) — tools on.

Every spawned process gets ``RMC_CHILD=1`` in its environment. RMC's own hooks
check that variable and no-op, which is the only thing preventing a compression
run from recursively triggering compression runs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class AgentResult:
    ok: bool
    text: str = ""
    data: dict[str, Any] | None = None
    error: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    # Input tokens the provider served from its prompt cache. Reported
    # separately from tokens_in because it is the only observable that says
    # whether a warm-prefix strategy is actually working — cache TTLs are not
    # published per-request, so warmth has to be measured after the fact rather
    # than predicted.
    cached_in: int = 0
    # Input tokens the provider wrote *into* the cache on this call. On a
    # seeding call this is how much of our own prefix it stored, which is the
    # amount a later fork has to read back for the trick to have worked.
    created_in: int = 0
    duration_s: float = 0.0
    backend: str = ""
    raw: str = ""

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out


@dataclass
class Session:
    """A conversation to start, or to branch a one-off question from.

    Repeated routing calls differ only in the question: the candidate list in
    front of it is the same apex layer every time. Sending it fresh on every
    prompt pays full price for text that never changed, which is precisely what
    a provider prompt cache exists to avoid — but only if the prefix arrives as
    the same conversation rather than as a new one.

    `start` seeds that conversation once. `resume` branches a throwaway child
    from it, so the shared prefix stays exactly where it was and the transcript
    never grows with questions nobody will ask again.
    """

    id: str
    resume: bool = False


class Adapter(Protocol):
    name: str

    def run(
        self,
        prompt: str,
        *,
        system: str | None = None,
        cwd: Path | None = None,
        schema: dict[str, Any] | None = None,
        tools: bool = False,
        timeout: int = 180,
        session: "Session | None" = None,
        allowed_tools: list[str] | None = None,
    ) -> AgentResult: ...

    def available(self) -> bool: ...


# --------------------------------------------------------------------------- #
# JSON extraction — shared by adapters whose backend has no native schema mode
# --------------------------------------------------------------------------- #

_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort recovery of a JSON object from a model's prose.

    Tries, in order: the whole string, fenced blocks (last first, since models
    tend to restate the final answer), then balanced-brace scanning.
    """
    if not text:
        return None
    candidates: list[str] = [text.strip()]
    candidates.extend(reversed(_FENCE_RE.findall(text)))
    for blob in candidates:
        blob = blob.strip()
        if not blob:
            continue
        try:
            parsed = json.loads(blob)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    # balanced scan
    depth, start, in_str, esc = 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    parsed = json.loads(text[start : i + 1])
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    start = -1
    return None


def schema_instruction(schema: dict[str, Any]) -> str:
    """Prompt fragment used by backends without native structured output."""
    return (
        "\n\n## Output contract\n"
        "Respond with a single JSON object and nothing else — no prose before or "
        "after, no markdown fence. It must validate against this JSON Schema:\n\n"
        f"{json.dumps(schema, indent=2)}\n"
    )


def get_adapter(name: str, *, model: str | None = None, **kwargs: Any) -> Adapter:
    from .claude_code import ClaudeCodeAdapter
    from .codex import CodexAdapter
    from .mock import MockAdapter

    key = (name or "claude").strip().lower()
    if key in ("claude", "claude-code", "cc"):
        return ClaudeCodeAdapter(model=model, **kwargs)
    if key in ("codex", "openai"):
        return CodexAdapter(model=model, **kwargs)
    if key == "mock":
        return MockAdapter(**kwargs)
    raise ValueError(f"unknown agent backend: {name!r} (want claude | codex | mock)")


def available_backends() -> list[str]:
    from .claude_code import ClaudeCodeAdapter
    from .codex import CodexAdapter

    out = []
    if ClaudeCodeAdapter().available():
        out.append("claude")
    if CodexAdapter().available():
        out.append("codex")
    out.append("mock")
    return out


__all__ = [
    "Adapter",
    "AgentResult",
    "extract_json",
    "schema_instruction",
    "get_adapter",
    "available_backends",
]
