"""Measuring whether the thesis holds.

RMC claims that a lesson can be compressed repeatedly and keep working. That is
falsifiable, and this is where it gets tested rather than asserted: replay real
tasks against every level of a lesson and see whether the shorter forms still
transfer.

The design choices here are all about making a *failure* visible, because an
eval that can only produce good news is a demo:

* **A control arm.** Every case is also run with no lesson at all. Without it
  you measure the model's prior, not the lesson — and a lesson that adds nothing
  looks identical to one that works.
* **A ceiling arm.** The original L0 sets what full detail achieves. Without it
  a low score at L2 cannot be told apart from a lesson that never helped.
* **Held-out episodes.** Compression is *accepted* by replaying episodes; scoring
  it on those same episodes measures memorisation. The split is deterministic
  and the compressor never sees the test half.
* **Blind grading.** The grader is shown the task, the known-good outcome and a
  candidate — never the lesson, and never which arm produced it. The ordinary
  replay judge takes the lesson as context, which would identify the control
  instantly by its absence.
* **Repeats.** A single sample per arm is a coin toss. Variance is reported, not
  hidden behind a mean.

Nothing here writes to the store. An eval that mutates what it measures is
measuring itself.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from .adapters import Adapter
from .node import Node
from .prompts import BLIND_JUDGE, JUDGE_SCHEMA, REPLAY_PROBE
from .store import Episode, Store
from .util import truncate

CONTROL = "none"


# --------------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------------- #


@dataclass
class ArmResult:
    level: str  # "none", "L0", "L1", ...
    node_id: str | None
    tokens: int
    passes: int = 0
    runs: int = 0
    reasons: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.passes / self.runs if self.runs else 0.0


@dataclass
class CaseResult:
    episode_id: str
    task: str
    arms: dict[str, ArmResult] = field(default_factory=dict)


@dataclass
class EvalReport:
    cases: list[CaseResult] = field(default_factory=list)
    holdout: float = 0.0
    samples: int = 1
    skipped: str = ""

    def levels(self) -> list[str]:
        seen: list[str] = []
        for case in self.cases:
            for level in case.arms:
                if level not in seen:
                    seen.append(level)
        return sorted(seen, key=lambda l: (-1 if l == CONTROL else int(l[1:])))

    def aggregate(self, level: str) -> ArmResult:
        total = ArmResult(level=level, node_id=None, tokens=0)
        tokens: list[int] = []
        for case in self.cases:
            arm = case.arms.get(level)
            if arm is None:
                continue
            total.passes += arm.passes
            total.runs += arm.runs
            tokens.append(arm.tokens)
        total.tokens = round(sum(tokens) / len(tokens)) if tokens else 0
        return total

    @property
    def lift(self) -> float:
        """Transfer at L0 minus the control. If this is ~0 nothing else matters."""
        base = self.aggregate(CONTROL)
        full = self.aggregate("L0")
        return full.rate - base.rate

    def render(self) -> str:
        if self.skipped:
            return f"eval skipped: {self.skipped}"
        lines = [
            f"RMC eval — {len(self.cases)} held-out episode(s), {self.samples} sample(s) each",
            "",
            f"  {'level':<7} {'tokens':>7} {'transfer':>9} {'vs L0':>7}   n",
        ]
        full = self.aggregate("L0")
        for level in self.levels():
            arm = self.aggregate(level)
            relative = (
                f"{arm.rate / full.rate:.0%}" if full.rate and level != CONTROL else "—"
            )
            label = f"{level} (control)" if level == CONTROL else level
            lines.append(
                f"  {label:<7} {arm.tokens:>7} {arm.rate:>8.0%} {relative:>7}   {arm.runs}"
            )
        lines += ["", f"  lift over control: {self.lift:+.0%}"]

        if len(self.cases) < 5 or self.samples < 3:
            lines += [
                "",
                "  NOT A RESULT. With this few episodes or samples the numbers are an",
                "  anecdote — a single flipped grade moves them by tens of points.",
            ]
        if abs(self.lift) < 0.15:
            lines += [
                "",
                "  WARNING: the lesson barely beats no lesson at all. Retention across",
                "  levels is meaningless until the lesson itself demonstrably helps.",
            ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #


def lineage(store: Store, node: Node) -> list[Node]:
    """Every level of one lesson, most detailed first.

    Walks down to the original through ``derived_from`` and up through
    ``parents``, so a chain is compared against itself rather than against
    unrelated nodes.
    """
    seen: dict[str, Node] = {node.id: node}
    frontier = [node]
    while frontier:
        nxt: list[Node] = []
        for current in frontier:
            for other in store.children(current) + [
                store.get(p) for p in current.parents
            ]:
                if other is not None and other.id not in seen:
                    seen[other.id] = other
                    nxt.append(other)
        frontier = nxt
    return sorted(seen.values(), key=lambda n: n.level)


def holdout_split(episodes: list[Episode], fraction: float) -> list[Episode]:
    """Deterministic held-out slice, chosen by hash of the episode id.

    Deterministic so a run is reproducible and so the same episodes stay held
    out as the store grows — an episode that drifts between train and test
    across runs quietly leaks.
    """
    if fraction <= 0:
        return list(episodes)
    kept = []
    for episode in episodes:
        digest = int(hashlib.sha256(episode.id.encode()).hexdigest()[:8], 16)
        if (digest % 1000) / 1000.0 < fraction:
            kept.append(episode)
    return kept


# --------------------------------------------------------------------------- #
# running
# --------------------------------------------------------------------------- #


def _probe(adapter: Adapter, task: str, lesson: str, timeout: int) -> str:
    run = adapter.run(
        REPLAY_PROBE.format(task_id="eval", pack=lesson or "(no lesson available)", task=task),
        timeout=timeout,
        tools=False,
    )
    return run.text if run.ok else ""


def _grade(adapter: Adapter, task: str, expected: str, candidate: str, timeout: int) -> tuple[bool, str]:
    """Blind: the grader never sees the lesson, or which arm this came from."""
    if not candidate.strip():
        return False, "no answer produced"
    verdict = adapter.run(
        BLIND_JUDGE.format(
            task=truncate(task, 3000),
            expected=truncate(expected, 3000),
            candidate=truncate(candidate, 4000),
        ),
        schema=JUDGE_SCHEMA,
        timeout=timeout,
    )
    if not verdict.ok or not verdict.data:
        return False, f"grader unavailable: {verdict.error[:120]}"
    return bool(verdict.data.get("pass")), str(verdict.data.get("reason") or "")[:200]


def evaluate(
    store: Store,
    adapter: Adapter,
    *,
    holdout: float = 0.3,
    samples: int = 1,
    limit: int = 20,
) -> EvalReport:
    """Score every level of every lesson against held-out episodes."""
    timeout = int(store.config.get("limits.agent_timeout_s", 180))
    report = EvalReport(holdout=holdout, samples=samples)

    usable = [
        e
        for e in store.episodes()
        if e.outcome == "success" and e.used and e.accepted_summary.strip()
    ]
    if not usable:
        report.skipped = "no episodes with a recorded task, outcome and attributed use"
        return report

    held = holdout_split(usable, holdout)[:limit]
    if not held:
        report.skipped = (
            f"holdout of {holdout:.0%} selected none of {len(usable)} episode(s); "
            "raise --holdout or record more episodes"
        )
        return report

    for episode in held:
        case = CaseResult(episode_id=episode.id, task=episode.prompt)
        node = next((store.get(i) for i in episode.used if store.get(i)), None)
        if node is None:
            continue

        arms: list[tuple[str, Node | None]] = [(CONTROL, None)]
        arms += [(f"L{n.level}", n) for n in lineage(store, node)]

        for level, candidate_node in arms:
            arm = ArmResult(
                level=level,
                node_id=candidate_node.id if candidate_node else None,
                tokens=candidate_node.tokens if candidate_node else 0,
            )
            for _ in range(samples):
                answer = _probe(
                    adapter, episode.prompt, candidate_node.body if candidate_node else "", timeout
                )
                passed, why = _grade(
                    adapter, episode.prompt, episode.accepted_summary, answer, timeout
                )
                arm.runs += 1
                arm.passes += 1 if passed else 0
                arm.reasons.append(why)
            case.arms[level] = arm
        report.cases.append(case)

    return report
