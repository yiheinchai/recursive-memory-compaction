"""In-process backend for tests, ablations and offline development.

Beyond canned responses, this adapter can simulate a *knowledge world*: each
task declares the facts required to solve it, and the mock agent succeeds iff
every required fact appears in the lesson text it was given.

That turns the entire RMC control flow into something deterministically
testable. A compression that drops ``backoff-constants`` really does fail
replay; a descent that restores the delta holding that fact really does rescue
it. No tokens, no flakiness, and ablations (delta-patch vs stepwise) become
measurable rather than asserted.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from . import AgentResult

FACT_RE = re.compile(r"@([a-z0-9][a-z0-9_-]*)")


class MockWorld:
    """Maps task id -> set of fact tokens needed to solve it."""

    def __init__(self, requirements: dict[str, set[str]] | None = None) -> None:
        self.requirements = requirements or {}

    def required(self, task_id: str) -> set[str]:
        return set(self.requirements.get(task_id, set()))

    @staticmethod
    def facts_in(text: str) -> set[str]:
        return set(FACT_RE.findall(text or ""))

    def solves(self, task_id: str, context: str) -> tuple[bool, set[str]]:
        need = self.required(task_id)
        have = self.facts_in(context)
        missing = need - have
        return (not missing), missing


class MockAdapter:
    """Scriptable backend.

    ``router`` wins if provided; otherwise ``responses`` are popped in order;
    otherwise the built-in behaviours (compress / diagnose / judge / solve) run
    against ``world``.
    """

    name = "mock"

    def __init__(
        self,
        *,
        responses: list[Any] | None = None,
        router: Callable[[str, dict | None], Any] | None = None,
        world: MockWorld | None = None,
        model: str | None = None,
        diagnosis_kind: str = "edge-case",
    ) -> None:
        self.responses = list(responses or [])
        self.router = router
        self.world = world or MockWorld()
        self.model = model
        # Matches the kind `_compress` assigns to what it drops, so the
        # categorical join in the selector is exercised rather than bypassed.
        self.diagnosis_kind = diagnosis_kind
        self.calls: list[dict[str, Any]] = []

    def available(self) -> bool:
        return True

    def run(
        self,
        prompt: str,
        *,
        system: str | None = None,
        cwd: Path | None = None,
        schema: dict[str, Any] | None = None,
        tools: bool = False,
        timeout: int = 180,
        session: Any = None,
        allowed_tools: list[str] | None = None,
    ) -> AgentResult:
        self.calls.append(
            {
                "prompt": prompt,
                "schema": schema,
                "tools": tools,
                "session": session,
                "allowed_tools": allowed_tools,
            }
        )

        if self.router is not None:
            payload = self.router(prompt, schema)
        elif self.responses:
            payload = self.responses.pop(0)
        else:
            payload = self._builtin(prompt, schema)

        if isinstance(payload, AgentResult):
            return payload
        if isinstance(payload, dict):
            return AgentResult(
                ok=True, text=json.dumps(payload), data=payload, backend=self.name
            )
        return AgentResult(ok=True, text=str(payload), backend=self.name)

    # ------------------------------------------------------------- behaviours
    def _builtin(self, prompt: str, schema: dict | None) -> Any:
        kind = _classify(prompt)
        handler = {
            "compress": self._compress,
            "diagnose": self._diagnose,
            "judge": self._judge,
            "relevance": self._relevance,
            "related": self._relevance,
            "repair": self._repair,
            "assess": self._assess,
        }.get(kind)
        return handler(prompt) if handler else self._solve(prompt)

    # -------------------------------------------------- judgement stand-ins
    #
    # These simulate a model's opinion using the @fact world. They are crude on
    # purpose: their job is to make the *control flow* deterministic, not to be
    # good judges. Tests that care about a specific verdict pass a `router`.

    def _relevance(self, prompt: str) -> dict[str, Any]:
        """Relevant iff the candidate shares a @fact with the question."""
        question = _section(prompt, "WORK") or _section(prompt, "NEW LESSON")
        wanted = self.facts(question)
        picks = []
        for ident, text in _candidates(prompt):
            shared = bool(self.facts(text) & wanted)
            picks.append(
                {
                    "id": ident,
                    "verdict": "relevant" if shared else "unrelated",
                    "descend": False,
                    "why": "shares a fact" if shared else "no shared fact",
                }
            )
        return {"picks": picks}

    def _repair(self, prompt: str) -> dict[str, Any]:
        """Useful iff the option supplies a @fact the failure says is missing."""
        missing = self.facts(_section(prompt, "FAILURE"))
        ranked = []
        for key, text in _keyed_options(prompt):
            supplies = self.facts(text) & missing
            ranked.append({"key": key, "usefulness": 1.0 if supplies else 0.0})
        return {"ranked": ranked}

    def _assess(self, prompt: str) -> dict[str, Any]:
        digest = _section(prompt, "SESSION")
        corrected = "[the human then said]" in digest
        failed = digest.count("FAILED")
        return {
            "outcome": "success",
            "confidence": 0.9,
            "corrected": corrected,
            "correction": _after(digest, "[the human then said]"),
            "evidence": [f"{failed} failed tool calls"],
            "discoveries": [] if not failed else [{"what_failed": "x", "what_worked": "y"}],
            "summary": "mock summary",
        }

    def _compress(self, prompt: str) -> dict[str, Any]:
        """Drop the last fact-bearing *block*, and honestly declare it as a delta.

        Blocks, not lines: a real compressor removes a whole idea, and a lesson
        written with wrapped bullets would otherwise lose a sentence fragment
        while the concept it belongs to stays behind.

        This is the well-behaved compressor. ``MockAdapter(router=...)`` is how
        tests simulate a badly-behaved one that under-reports its manifest.
        """
        body = _section(prompt, "LESSON")
        blocks = [b for b in re.split(r"\n\s*\n", body) if b.strip()]
        if len(blocks) < 2:  # unwrapped lesson: fall back to line granularity
            blocks = [ln for ln in body.splitlines() if ln.strip()]

        preserve = set(FACT_RE.findall(_section(prompt, "PRESERVE")))
        droppable = [
            b for b in blocks if FACT_RE.search(b) and not (self.facts(b) & preserve)
        ]

        if not droppable:
            kept, dropped = blocks, []
        else:
            victim = droppable[-1]
            kept = [b for b in blocks if b != victim]
            dropped = [victim]

        return {
            "body": "\n\n".join(b.strip() for b in kept).strip() or "(empty)",
            "dropped": [
                {
                    "claim": " ".join(b.split()).lstrip("- ").strip(),
                    "kind": "edge-case",
                    "holder": None,
                }
                for b in dropped
            ],
            "rationale": "mock: removed the trailing fact-bearing block",
        }

    def _diagnose(self, prompt: str) -> dict[str, Any]:
        complaint = _section(prompt, "MISSING")
        # Prefer explicit @fact tokens, but fall back to the raw complaint text
        # so that a verifier which reports in prose still yields something the
        # selector can match lexically — exactly as a real diagnoser would.
        # Keep the @ prefix: the failure text is what `_repair` later matches
        # against, and stripping it there makes the two halves of the simulation
        # unable to talk to each other.
        missing = [f"@{f}" for f in sorted(self.facts(complaint))] or [complaint.strip()]
        return {
            "category": self.diagnosis_kind,
            "missing": [m for m in missing if m] or ["unspecified detail"],
            "wrong_step": "mock diagnosis",
            "confidence": 0.9,
        }

    def _judge(self, prompt: str) -> dict[str, Any]:
        task_id = _field(prompt, "TASK_ID") or "unknown"
        context = _section(prompt, "CONTEXT")
        ok, missing = self.world.solves(task_id, context)
        return {
            "pass": ok,
            "reason": "all required facts present" if ok else f"missing: {sorted(missing)}",
            "missing": sorted(missing),
        }

    def _solve(self, prompt: str) -> dict[str, Any]:
        task_id = _field(prompt, "TASK_ID") or "unknown"
        ok, missing = self.world.solves(task_id, prompt)
        return {
            "pass": ok,
            "output": f"solved {task_id}" if ok else f"failed {task_id}",
            "missing": sorted(missing),
        }

    @staticmethod
    def facts(text: str) -> set[str]:
        return set(FACT_RE.findall(text or ""))


# --------------------------------------------------------------------------- #
# prompt introspection helpers
# --------------------------------------------------------------------------- #


def _classify(prompt: str) -> str:
    head = (prompt or "")[:4000].lower()
    for kind in ("compress", "diagnose", "judge", "relevance", "related", "repair", "assess"):
        if f"rmc:{kind}" in head:
            return kind
    return "solve"


def _candidates(prompt: str) -> list[tuple[str, str]]:
    """Parse the `[id: n_x] ...` blocks a relevance/related prompt renders."""
    body = _section(prompt, "LESSONS") or _section(prompt, "REMEMBERED")
    out: list[tuple[str, str]] = []
    for chunk in re.split(r"\n\s*\n(?=\[id: )", body):
        match = re.match(r"\[id: ([^\]]+)\]", chunk.strip())
        if match:
            out.append((match.group(1).strip(), chunk))
    return out


def _keyed_options(prompt: str) -> list[tuple[str, str]]:
    """Parse the `[key: ...]` blocks a repair prompt renders."""
    body = _section(prompt, "REMOVED DETAILS")
    out: list[tuple[str, str]] = []
    for chunk in re.split(r"\n\s*\n(?=\[key: )", body):
        match = re.match(r"\[key: ([^\]]+)\]", chunk.strip())
        if match:
            out.append((match.group(1).strip(), chunk))
    return out


def _after(text: str, marker: str, limit: int = 400) -> str:
    idx = text.find(marker)
    return text[idx + len(marker) : idx + len(marker) + limit].strip() if idx >= 0 else ""


def _section(prompt: str, name: str) -> str:
    """Read a ``<<<NAME ... NAME>>>`` block. RMC prompts delimit inputs this way."""
    match = re.search(
        rf"<<<{re.escape(name)}\s*\n(.*?)\n{re.escape(name)}>>>", prompt or "", re.DOTALL
    )
    return match.group(1) if match else ""


def _field(prompt: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}:\s*(.+)$", prompt or "", re.MULTILINE)
    return match.group(1).strip() if match else ""
