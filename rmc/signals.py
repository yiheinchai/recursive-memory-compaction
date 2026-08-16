"""Reading a transcript into facts. Nothing here decides what those facts mean.

This module used to classify sessions with regex phrase banks and hand-tuned
weights — `-0.65` if the user said something matching a "correction" pattern,
`+0.6` for an "approval" pattern. That approach cannot work: whether "actually,
let's use the other one" is a correction or a change of mind is a reading of
intent, and a pattern list only matches the surface forms someone thought of in
advance. Worse, it looks like a judgement while being a lookup table.

So the split is now strict:

* **here** — parsing. Turn a transcript into structured facts: who said what,
  which tool ran with which input, what came back, what the host itself marked
  as a refusal or a meta turn. All of this is reading a file format.
* **`judge.assess`** — meaning. Did it go well, was the agent corrected, what
  did it work out by trial.

The one thing this module still decides is *whether a session is worth judging
at all* (`worth_assessing`), which is a structural question about size, not a
semantic one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Hosts inject synthetic turns that wear the user role but were never typed by a
# human: slash-command envelopes, harness reminders, command stdout. These are
# matched by tag name, not by meaning — this is format parsing, not classification.
_SYNTHETIC_BLOCK = re.compile(
    r"<(command-name|command-message|command-args|local-command-stdout|"
    r"system-reminder|cross-session-message|task-notification)>.*?"
    r"</\1>",
    re.DOTALL | re.IGNORECASE,
)
_SYNTHETIC_OPEN = re.compile(
    r"<(command-name|command-message|command-args|local-command-stdout|"
    r"system-reminder)>.*",
    re.DOTALL | re.IGNORECASE,
)


def strip_synthetic(text: str) -> str:
    """Remove host-injected envelopes, leaving only what a human actually typed."""
    if not text:
        return ""
    cleaned = _SYNTHETIC_BLOCK.sub(" ", text)
    cleaned = _SYNTHETIC_OPEN.sub(" ", cleaned)  # unclosed/truncated envelopes
    return cleaned.strip()


@dataclass
class ToolEvent:
    """One tool call and what came back.

    Kept in order and paired by id. ``ok`` is set only from what the host
    actually reported — an `is_error` flag or a non-zero exit code. When the host
    says nothing, ``ok`` stays ``None`` rather than being guessed from the text,
    and the model decides from the transcript instead.
    """

    name: str = ""
    detail: str = ""  # command line, path, or another short digest of the input
    output: str = ""
    ok: bool | None = None
    uid: str = ""

    def render(self, *, limit: int = 400) -> str:
        mark = {True: "ok", False: "FAILED", None: "?"}[self.ok]
        out = " ".join((self.output or "").split())[:limit]
        return f"  [{self.name} {mark}] {self.detail[:200]}\n      -> {out}"


@dataclass
class SessionFacts:
    """Flattened, backend-agnostic view of a transcript. Facts only."""

    user_messages: list[str] = field(default_factory=list)
    assistant_messages: list[str] = field(default_factory=list)
    tool_outputs: list[str] = field(default_factory=list)
    tool_events: list[ToolEvent] = field(default_factory=list)
    tool_calls: int = 0
    first_prompt: str = ""
    last_assistant: str = ""
    denied: bool = False  # the host reported a refused or interrupted tool call

    @property
    def follow_ups(self) -> list[str]:
        return self.user_messages[1:]

    @property
    def failures(self) -> list[ToolEvent]:
        return [e for e in self.tool_events if e.ok is False]


# --------------------------------------------------------------------------- #
# transcript parsing
# --------------------------------------------------------------------------- #


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("content"), (str, list)):
                    parts.append(_text_of(block["content"]))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def _input_digest(payload: Any, limit: int = 220) -> str:
    """A short, human-readable stand-in for a tool's input."""
    if not isinstance(payload, dict):
        return str(payload or "")[:limit]
    for key in ("command", "file_path", "path", "pattern", "query", "url", "prompt"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())[:limit]
    try:
        return json.dumps(payload)[:limit]
    except Exception:
        return str(payload)[:limit]


def parse_transcript(path: Path | str, *, max_lines: int = 20000) -> SessionFacts:
    """Read a JSONL transcript. Tolerant by design: unknown shapes are skipped.

    Handles Claude Code's transcript format and Codex's rollout JSONL, which
    differ in nesting but agree on role-tagged messages.
    """
    facts = SessionFacts()
    path = Path(path)
    if not path.exists():
        return facts
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return facts

    for line in lines[-max_lines:]:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        _absorb(row, facts)

    if facts.user_messages:
        facts.first_prompt = facts.user_messages[0]
    if facts.assistant_messages:
        facts.last_assistant = facts.assistant_messages[-1]
    return facts


def _absorb(row: dict[str, Any], facts: SessionFacts) -> None:
    kind = row.get("type") or row.get("role") or ""
    message = row.get("message") if isinstance(row.get("message"), dict) else row
    role = message.get("role") or kind
    content = message.get("content")

    if role == "user":
        text = _text_of(content)

        # The host's own metadata, not guesswork: Claude Code marks harness turns
        # `isMeta`, tool results `toolUseResult`, and refusals `toolDenialKind`.
        if row.get("toolDenialKind") or row.get("interruptedMessageId"):
            facts.denied = True
            return
        if row.get("isMeta"):
            return
        if row.get("toolUseResult") is not None:
            facts.tool_outputs.append(text)
            _attach_result(facts, content, text, row.get("toolUseResult"))
            return
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        ):
            facts.tool_outputs.append(text)
            _attach_result(facts, content, text, None)
            return

        human = strip_synthetic(text)
        # Some hosts re-inject a standing instruction on every turn; counting it
        # repeatedly would let one phrase dominate the record shown to the model.
        if human and human not in facts.user_messages:
            facts.user_messages.append(human)
        return

    if role == "assistant":
        text = _text_of(content)
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    facts.tool_calls += 1
                    facts.tool_events.append(
                        ToolEvent(
                            name=str(block.get("name") or ""),
                            detail=_input_digest(block.get("input")),
                            uid=str(block.get("id") or ""),
                        )
                    )
        if text.strip():
            facts.assistant_messages.append(text)
        return

    if kind in ("tool_result", "function_call_output", "item.completed"):
        text = _text_of(content) or _text_of(row.get("output"))
        if text:
            facts.tool_outputs.append(text)


def _attach_result(facts: SessionFacts, content: Any, text: str, raw: Any) -> None:
    """Pair a tool result back to its call, and record what the host said about it."""
    uid = ""
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("tool_use_id"):
                uid = str(block["tool_use_id"])
                break
    event = None
    if uid:
        event = next((e for e in reversed(facts.tool_events) if e.uid == uid), None)
    if event is None:
        event = next((e for e in reversed(facts.tool_events) if e.ok is None), None)
    if event is None:
        return

    event.output = text[:2000]
    # Only the host's own error signal counts. Left as None when absent — an
    # unknown is honest, and the model reads the output anyway.
    #
    # Presence matters, not truthiness: `is_error: false` is an explicit
    # statement that the call succeeded, and treating it as "no signal" throws
    # away exactly the successes that make an error-then-fix pair legible.
    if isinstance(raw, dict):
        for key in ("is_error", "isError"):
            if key in raw and isinstance(raw[key], bool):
                event.ok = not raw[key]
                return
        code = raw.get("exit_code", raw.get("exitCode"))
        if isinstance(code, int):
            event.ok = code == 0
            return
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and "is_error" in block:
                event.ok = not block["is_error"]
                return


# --------------------------------------------------------------------------- #
# structural gate and digest
# --------------------------------------------------------------------------- #


def worth_assessing(facts: SessionFacts, *, min_tool_calls: int = 8) -> bool:
    """Is there enough here to be worth a judgement?

    Structural, not semantic: a session with two tool calls and no follow-up has
    nothing to teach regardless of what it says, and asking would be waste.
    """
    if facts.tool_calls >= min_tool_calls:
        return True
    if len(facts.user_messages) > 1:
        return True  # the human said something after the first prompt
    return facts.denied


def digest(facts: SessionFacts, *, limit: int = 11000) -> str:
    """The session rendered for judgement.

    Interleaves the human's turns with the tool trail, because both questions
    the model is asked — was the agent corrected, what did it discover — depend
    on the *order* of what happened.
    """
    parts: list[str] = []
    if facts.first_prompt:
        parts.append(f"[the request]\n{facts.first_prompt[:1500]}")

    for msg in facts.follow_ups[:8]:
        parts.append(f"[the human then said]\n{msg[:800]}")

    # The agent's own reasoning, not just what it ran. Attribution asks whether
    # a lesson bore on the work, and for a lesson that shapes *how you think* —
    # a principle, a constraint, a way of deciding — the only trace is here. A
    # digest of commands and outcomes shows a lesson that named a flag and is
    # blind to one that changed an approach, which credits the cheap kind of
    # knowledge and silently starves the expensive kind.
    for msg in facts.assistant_messages[1:-1][:6]:
        text = " ".join(msg.split())
        if len(text) > 80:  # skip one-line acknowledgements
            parts.append(f"[the agent reasoned]\n{text[:700]}")

    if facts.denied:
        parts.append("[the human refused or interrupted a tool call]")

    if facts.tool_events:
        parts.append("[what the agent ran, in order]")
        events = facts.tool_events
        # Keep the ends: the opening moves and the resolution are where the
        # useful signal is. The middle of a long grind rarely adds anything.
        shown = events if len(events) <= 40 else events[:20] + events[-20:]
        if len(events) > 40:
            parts.append(f"  ... {len(events) - 40} further calls omitted ...")
        parts.extend(e.render() for e in shown)

    if facts.last_assistant:
        parts.append(f"[the agent's closing message]\n{facts.last_assistant[:1500]}")

    return "\n\n".join(parts)[:limit]


def summarise_work(facts: SessionFacts, *, limit: int = 1500) -> str:
    """A compact record of what the agent ended up doing, for replay comparison."""
    text = facts.last_assistant.strip()
    if not text:
        text = "\n".join(facts.assistant_messages[-2:]).strip()
    return text[:limit]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                yield json.loads(line)
            except Exception:
                continue
