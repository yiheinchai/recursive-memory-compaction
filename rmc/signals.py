"""Reading outcomes out of a real session.

In a scripted harness you write an oracle by hand. Running ambiently inside
someone's repo you do not get that, so the outcome has to be inferred from what
the human did next.

The signals, strongest first:

* the user explicitly corrected the agent            -> failure   (high conf)
* the user denied a tool call                        -> failure   (medium)
* a test command exited non-zero and was never fixed -> failure   (medium)
* the user said some version of "yes, that"          -> success   (high)
* tests passed / a commit landed after the work      -> success   (medium)
* the session just ended with real work done         -> success   (low)

None of these is trustworthy alone, which is why they are combined into a
confidence and why anything below `signals.min_confidence` is recorded as
`unknown` and excluded from both stats and the replay corpus. The corpus is the
thing that must stay clean: it is what every future compression is judged
against.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# phrase banks
# --------------------------------------------------------------------------- #

_CORRECTION = re.compile(
    r"(?i)\b("
    r"no,?\s+(?:don'?t|do not|that'?s|you|i|it|use|stop)"
    r"|that'?s (?:wrong|not right|not what|incorrect)"
    r"|not what i (?:asked|wanted|said|meant)"
    r"|i (?:said|told you|asked for|meant)\b"
    r"|you (?:were|are) (?:wrong|supposed to)"
    r"|revert|undo (?:that|this)|roll ?back"
    r"|stop (?:doing|using)"
    r"|why did you|don'?t do that"
    r"|actually,? (?:no|it|we|you|i)"
    r"|wrong (?:approach|file|answer|assumption)"
    r"|doesn'?t work|didn'?t work|still (?:broken|failing|fails)"
    r"|try again"
    r")\b"
)

_APPROVAL = re.compile(
    r"(?i)\b("
    r"perfect|exactly|lgtm|looks good|ship it|that'?s (?:it|right|correct)"
    r"|nice work|great work|works now|that works|thanks[,! ]*(?:that|it)?"
    r"|yes,? (?:that|exactly|please)"
    r")\b"
)

_TEST_PASS = re.compile(
    r"(?i)("
    r"\b\d+ passed|\ball tests? passed|\btests? pass\b|\bbuild succeeded|✓ \d+"
    r"|\b0 failed|\bno errors found|\bcompiled successfully"
    # unittest prints a bare "OK" on its own line after "Ran N tests"
    r"|^OK$|^OK \(|\bRan \d+ tests? in [\d.]+s\s*\n+OK"
    r")",
    re.MULTILINE,
)

_TEST_FAIL = re.compile(
    r"(?i)\b("
    r"\d+ failed|test failures?|assertionerror|traceback \(most recent"
    r"|build failed|compilation (?:error|failed)|error TS\d+|panicked at"
    r")\b"
)

_DENIED = re.compile(
    r"(?i)(user (?:denied|rejected|doesn'?t want)|tool use was rejected"
    r"|permission denied by user|request interrupted by user)"
)

# Hosts inject synthetic turns that wear the user role but were never typed by a
# human: slash-command envelopes, harness reminders, command stdout. Scoring
# them as intent is how a `/goal` payload gets read as the user approving the
# work. Strip them before anything looks for corrections or approvals.
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
class SessionFacts:
    """Flattened, backend-agnostic view of a transcript."""

    user_messages: list[str] = field(default_factory=list)
    assistant_messages: list[str] = field(default_factory=list)
    tool_outputs: list[str] = field(default_factory=list)
    tool_calls: int = 0
    first_prompt: str = ""
    last_assistant: str = ""
    denied: bool = False

    @property
    def follow_ups(self) -> list[str]:
        """User turns after the first — where corrections live."""
        return self.user_messages[1:]


@dataclass
class Outcome:
    label: str = "unknown"  # success | failure | unknown
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "confidence": self.confidence, "evidence": self.evidence}


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

        # Prefer the host's own metadata over guessing from the text. Claude
        # Code marks harness-injected turns with `isMeta`, tool results with
        # `toolUseResult`, and refusals with `toolDenialKind`. Without this, a
        # slash-command payload gets scored as the user approving the work.
        if row.get("toolDenialKind") or row.get("interruptedMessageId"):
            facts.denied = True
            return
        if row.get("isMeta"):
            return
        if row.get("toolUseResult") is not None:
            facts.tool_outputs.append(text)
            return

        # Tool results also arrive wearing the user role on hosts that do not
        # set `toolUseResult`; keep them out of the human-intent channel or
        # every tool output reads as a correction.
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        ):
            facts.tool_outputs.append(text)
            if _DENIED.search(text):
                facts.denied = True
            return
        if _DENIED.search(text):
            facts.denied = True
        human = strip_synthetic(text)
        # Some hosts re-inject a standing instruction on every turn. Counting it
        # once per turn would let a single phrase dominate the whole score.
        if human and human not in facts.user_messages:
            facts.user_messages.append(human)
        return

    if role == "assistant":
        text = _text_of(content)
        if isinstance(content, list):
            calls = sum(1 for b in content if isinstance(b, dict) and b.get("type") == "tool_use")
            facts.tool_calls += calls
        if text.strip():
            facts.assistant_messages.append(text)
        return

    if kind in ("tool_result", "function_call_output", "item.completed"):
        text = _text_of(content) or _text_of(row.get("output"))
        if text:
            facts.tool_outputs.append(text)
        if _DENIED.search(text):
            facts.denied = True


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def _last_pos(pattern: re.Pattern[str], text: str) -> int:
    """Index of the last match, or -1. Used to decide red-then-green ordering."""
    last = -1
    for match in pattern.finditer(text):
        last = match.start()
    return last


def classify(facts: SessionFacts, *, min_tool_calls: int = 8) -> Outcome:
    """Combine signals into a single labelled outcome with a confidence."""
    evidence: list[str] = []
    score = 0.0  # positive -> success, negative -> failure

    corrections = [m for m in facts.follow_ups if _CORRECTION.search(m)]
    if corrections:
        score -= 0.65
        evidence.append(f"user correction ×{len(corrections)}: {corrections[0][:120]!r}")

    approvals = [m for m in facts.follow_ups if _APPROVAL.search(m)]
    if approvals:
        score += 0.6
        evidence.append(f"user approval: {approvals[0][:120]!r}")

    if facts.denied:
        score -= 0.35
        evidence.append("a tool call was denied")

    # Red-then-green is the normal shape of successful work, so what matters is
    # which signal came *last*, not which appeared at all.
    tail = "\n".join(facts.tool_outputs[-12:])
    last_pass = _last_pos(_TEST_PASS, tail)
    last_fail = _last_pos(_TEST_FAIL, tail)
    if last_pass > last_fail:
        score += 0.35
        evidence.append("tests passing at the end of the session")
    elif last_fail > last_pass:
        score -= 0.3
        evidence.append("tests failing at the end of the session")

    if facts.tool_calls >= min_tool_calls and not corrections:
        score += 0.25
        evidence.append(f"{facts.tool_calls} tool calls, no correction")

    if facts.tool_calls < min_tool_calls and not corrections and not approvals:
        evidence.append("session too small to judge")
        return Outcome("unknown", 0.0, evidence)

    if score >= 0.3:
        return Outcome("success", min(1.0, score), evidence)
    if score <= -0.3:
        return Outcome("failure", min(1.0, abs(score)), evidence)
    return Outcome("unknown", abs(score), evidence)


def summarise_work(facts: SessionFacts, *, limit: int = 1500) -> str:
    """A compact record of what the agent ended up doing, for replay comparison."""
    text = facts.last_assistant.strip()
    if not text:
        text = "\n".join(facts.assistant_messages[-2:]).strip()
    return text[:limit]


def correction_text(facts: SessionFacts) -> str:
    """The steering the human supplied — the raw material for a new lesson."""
    return "\n\n".join(m for m in facts.follow_ups if _CORRECTION.search(m))[:2000]


def excerpt(facts: SessionFacts, *, limit: int = 6000) -> str:
    """A readable slice of the session for the reflector."""
    parts: list[str] = []
    for i, msg in enumerate(facts.user_messages[:6]):
        parts.append(f"[user] {msg[:900]}")
        if i < len(facts.assistant_messages):
            parts.append(f"[assistant] {facts.assistant_messages[i][:900]}")
    if facts.last_assistant:
        parts.append(f"[assistant final] {facts.last_assistant[:1200]}")
    out = "\n\n".join(parts)
    return out[:limit]


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
