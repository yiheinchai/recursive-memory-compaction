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
    ) -> None:
        self.responses = list(responses or [])
        self.router = router
        self.world = world or MockWorld()
        self.model = model
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
    ) -> AgentResult:
        self.calls.append({"prompt": prompt, "schema": schema, "tools": tools})

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
        if kind == "compress":
            return self._compress(prompt)
        if kind == "diagnose":
            return self._diagnose(prompt)
        if kind == "judge":
            return self._judge(prompt)
        return self._solve(prompt)

    def _compress(self, prompt: str) -> dict[str, Any]:
        """Drop the last fact-bearing line, and honestly declare it as a delta.

        A well-behaved compressor. ``MockAdapter(router=...)`` is how tests
        simulate a badly-behaved one that under-reports its manifest.
        """
        body = _section(prompt, "LESSON")
        lines = [ln for ln in body.splitlines() if ln.strip()]
        fact_lines = [ln for ln in lines if FACT_RE.search(ln)]
        preserve = set(FACT_RE.findall(_section(prompt, "PRESERVE")))

        droppable = [ln for ln in fact_lines if not (self.facts(ln) & preserve)]
        if not droppable:
            kept, dropped_lines = lines, []
        else:
            victim = droppable[-1]
            kept = [ln for ln in lines if ln != victim]
            dropped_lines = [victim]

        return {
            "body": "\n".join(kept).strip() or "(empty)",
            "dropped": [
                {"claim": ln.strip(), "kind": "parameter", "holder": None}
                for ln in dropped_lines
            ],
            "rationale": "mock: removed the trailing fact line",
        }

    def _diagnose(self, prompt: str) -> dict[str, Any]:
        missing = sorted(self.facts(_section(prompt, "MISSING")))
        return {
            "category": "parameter",
            "missing": missing or ["unspecified detail"],
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
    if "rmc:compress" in head:
        return "compress"
    if "rmc:diagnose" in head:
        return "diagnose"
    if "rmc:judge" in head:
        return "judge"
    return "solve"


def _section(prompt: str, name: str) -> str:
    """Read a ``<<<NAME ... NAME>>>`` block. RMC prompts delimit inputs this way."""
    match = re.search(
        rf"<<<{re.escape(name)}\s*\n(.*?)\n{re.escape(name)}>>>", prompt or "", re.DOTALL
    )
    return match.group(1) if match else ""


def _field(prompt: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}:\s*(.+)$", prompt or "", re.MULTILINE)
    return match.group(1).strip() if match else ""
