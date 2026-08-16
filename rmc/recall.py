"""Retrieval: pick the lessons a prompt needs, and build a context pack.

Two entry points:

``recall_pack``  — ambient path, run from the prompt hook. Asks the model which
                   remembered lessons bear on this work, walking the tree from
                   the most abstract nodes downward.

``solve_with_descent`` — controlled path, used by replay and evaluation, where
                   RMC owns the loop and can observe a failure, diagnose it, and
                   descend the tree mid-task.

Relevance and repair are both judgements about meaning and are made by the
model (see ``judge.py``). What lives here is the shape of the search and the
budget it may spend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .adapters import Adapter, AgentResult
from .config import Config
from .judge import Budget, Judge, Pick, WalkResult, walk
from .node import Node
from .prompts import DIAGNOSE, DIAGNOSE_SCHEMA, REPLAY
from .selection import Candidate, Diagnosis, select
from .store import Store
from .util import count_tokens, truncate


@dataclass
class Pack:
    """The text injected ahead of a task, plus the bookkeeping to score it later."""

    text: str = ""
    served: list[str] = field(default_factory=list)
    # Titles alongside ids, so the hook can say *what* was recalled. A count
    # tells you RMC fired; it does not let you notice that it fired wrongly.
    titles: list[str] = field(default_factory=list)
    families: list[str] = field(default_factory=list)
    patches: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)  # still fresh in context
    refreshed: list[str] = field(default_factory=list)  # reminded, not repeated
    tokens: int = 0

    def __bool__(self) -> bool:
        return bool(self.text.strip())


# --------------------------------------------------------------------------- #
# family matching
# --------------------------------------------------------------------------- #


def select_lessons(
    store: Store,
    adapter: Adapter,
    prompt: str,
    *,
    limit: int | None = None,
    budget: Budget | None = None,
) -> WalkResult:
    """Ask the model which lessons bear on this prompt, walking abstract → concrete.

    Relevance is a judgement about meaning, so the model makes it. Token overlap
    cannot tell that "retry the failed CI job" and a lesson about retrying HTTP
    calls are unrelated despite sharing their most distinctive word, nor that
    "the deploy is stuck" and a lesson about Argo Rollouts are the same subject
    despite sharing none.

    What the harness contributes is the search *shape*: apexes are the most
    compressed nodes, so the whole top level fits in one question, and we
    descend only into lines the model says it cannot judge from the summary
    alone. Cost tracks depth, not the size of the memory.
    """
    limit = limit if limit is not None else int(store.config.get("recall.max_families", 3))
    roots = store.apexes()
    if not roots:
        # Structural gate, not a judgement: with nothing to recall there is
        # nothing to ask about.
        return WalkResult()

    # Second structural gate, and the one that matters most in practice: if the
    # whole store fits in the context budget there is nothing to *choose*, so
    # asking which lessons to pick is pure waste — a model call, and the latency
    # of one, spent selecting from a set we can afford entirely.
    #
    # This is not a heuristic standing in for judgement. It is the observation
    # that judgement is only needed under scarcity, and early on there is none.
    # Relevance filtering starts mattering when the tree outgrows the budget,
    # and that is exactly when it switches on.
    pack_budget = int(store.config.get("recall.max_pack_tokens", 1200))
    total = sum(n.tokens for n in roots)
    if total <= pack_budget and not store.config.get("recall.always_judge", False):
        result = WalkResult(selected=roots)
        for node in roots:
            result.picks[node.id] = Pick(
                id=node.id,
                verdict="relevant",
                why=f"whole store is {total} tokens, under the {pack_budget} budget — served without filtering",
            )
        return result

    budget = budget or Budget(max_calls=int(store.config.get("recall.judge_calls", 2)))
    judge = Judge(store, adapter, timeout=int(store.config.get("recall.timeout_s", 20)))
    result = walk(
        judge,
        prompt,
        roots,
        expand=store.children,
        budget=budget,
        max_depth=int(store.config.get("recall.max_depth", 2)),
    )

    # Prefer confident hits; fall back to the maybes only if there is room.
    confident = [n for n in result.selected if result.picks.get(n.id, Pick(n.id)).verdict == "relevant"]
    maybes = [n for n in result.selected if n not in confident]
    result.selected = (confident + maybes)[:limit]
    return result


# --------------------------------------------------------------------------- #
# pack construction
# --------------------------------------------------------------------------- #


def render_node(node: Node) -> str:
    heading = node.title.strip() or node.family
    return f"### {heading}  ·  L{node.level}\n{node.body.strip()}"


def recall_pack(
    store: Store,
    prompt: str,
    adapter: Adapter,
    *,
    budget: int | None = None,
    include_patches: bool = True,
    already_served: dict[str, int] | None = None,
    turn: int = 0,
) -> Pack:
    """Build the context pack for a prompt.

    Costs one or two model calls, cached by prompt. That is a deliberate trade:
    injecting the wrong lesson is worse than injecting none, and only the model
    can tell the difference.
    """
    pack = Pack()
    if not store.config.get("recall.enabled", True):
        return pack

    budget = budget or int(store.config.get("recall.max_pack_tokens", 1200))
    chunks: list[str] = []
    used = 0

    selection = select_lessons(store, adapter, prompt)
    pack.reasons = {n.id: selection.why(n.id) for n in selection.selected}

    seen = already_served or {}
    fresh_for = int(store.config.get("recall.stays_fresh_turns", 8))

    for node in selection.selected:
        family = node.family

        # A lesson injected a moment ago is still sitting in the context window
        # verbatim; repeating it buys nothing. But "still present" and "still
        # attended to" are different things — attention over a long context
        # decays, and a lesson from forty turns back is buried in the middle
        # where models attend least. So there are three cases, not two.
        age = turn - seen[node.id] if node.id in seen else None
        if age is not None and age < fresh_for:
            pack.skipped.append(node.id)
            continue
        if age is not None:
            # Present but stale: refresh salience with the one-line form rather
            # than repaying for the body. If the detail is needed again the
            # agent can open the lesson file directly.
            reminder = f"- (recalled earlier) {node.title or node.family}: {node.summary()}"
            cost = count_tokens(reminder)
            if used + cost <= budget:
                chunks.append(reminder)
                pack.refreshed.append(node.id)
                pack.served.append(node.id)
                used += cost
            continue

        rendered = render_node(node)
        cost = count_tokens(rendered)
        if used + cost > budget and chunks:
            break
        chunks.append(rendered)
        used += cost
        pack.served.append(node.id)
        pack.titles.append(node.title or node.family)
        pack.families.append(family)

        # Deltas that previously rescued this node get re-attached cheaply,
        # rather than waiting for the same failure to recur.
        if include_patches:
            for claim in _sticky_patches(store, node):
                claim_cost = count_tokens(claim)
                if used + claim_cost > budget:
                    break
                chunks.append(f"- {claim}")
                pack.patches.append(claim)
                used += claim_cost

        # An unresolved contradiction is raised here, at the moment the user is
        # already thinking about this topic — the way a student asks about a
        # confusion during the relevant lesson, not at a random later time.
        if store.config.get("placement.surface_conflicts", True) and node.conflict:
            note = (
                f"> **Unresolved:** {node.conflict.strip()}\n"
                f"> Memory holds conflicting lessons here. Ask the user to settle it "
                f"if it matters for this task, then run `rmc resolve <node-id>`."
            )
            cost = count_tokens(note)
            if used + cost <= budget:
                chunks.append(note)
                pack.conflicts.append(node.id)
                used += cost

    pack.text = "\n\n".join(chunks).strip()
    pack.tokens = used
    return pack


def _sticky_patches(store: Store, node: Node, *, min_rescues: int = 1) -> list[str]:
    """Delta claims that have rescued this node before.

    A delta that keeps being needed is evidence the compression cut too deep.
    Re-attaching it is the cheap fix; ``compact.repair`` eventually folds it
    back into the body permanently.
    """
    counts: dict[str, int] = {}
    for event in store.read_events("rescue", limit=2000):
        if event.get("node") == node.id and event.get("claim"):
            counts[event["claim"]] = counts.get(event["claim"], 0) + 1
    return [claim for claim, n in sorted(counts.items(), key=lambda kv: -kv[1]) if n >= min_rescues][:3]


# --------------------------------------------------------------------------- #
# controlled loop: run, verify, diagnose, descend
# --------------------------------------------------------------------------- #


@dataclass
class Attempt:
    node_id: str
    pack: str
    ok: bool
    output: str
    detail: str = ""
    candidate: str = ""
    tokens: int = 0


@dataclass
class DescentResult:
    ok: bool
    attempts: list[Attempt] = field(default_factory=list)
    final_pack: str = ""
    rescued_by: Candidate | None = None
    escalated: bool = False
    diagnosis: Diagnosis | None = None

    @property
    def expansions(self) -> int:
        return max(0, len(self.attempts) - 1)


def solve_with_descent(
    store: Store,
    *,
    adapter: Adapter,
    task_id: str,
    task: str,
    family: str,
    verify: Callable[[AgentResult, str], tuple[bool, str]],
    start: Node | None = None,
    cwd: Any = None,
    max_expansions: int | None = None,
) -> DescentResult:
    """Try apex; on failure diagnose, rank candidates, patch, retry.

    This is the descent policy in DESIGN.md §4 executed end to end.
    """
    config: Config = store.config
    max_expansions = (
        max_expansions
        if max_expansions is not None
        else int(config.get("recall.max_expansions", 3))
    )
    timeout = int(config.get("limits.agent_timeout_s", 180))

    node = start or store.apex(family)
    if node is None:
        return DescentResult(ok=False, escalated=True)

    judge = Judge(store, adapter)
    result = DescentResult(ok=False)
    pack_parts = [render_node(node)]
    tried: set[str] = set()

    def attempt(label: str) -> tuple[bool, str, str]:
        pack_text = "\n\n".join(pack_parts)
        run = adapter.run(
            REPLAY.format(task_id=task_id, pack=pack_text, task=task),
            cwd=cwd,
            timeout=timeout,
            tools=False,
        )
        ok, detail = verify(run, pack_text)
        result.attempts.append(
            Attempt(
                node_id=node.id,
                pack=pack_text,
                ok=ok,
                output=truncate(run.text, 4000),
                detail=detail,
                candidate=label,
                tokens=count_tokens(pack_text),
            )
        )
        result.final_pack = pack_text
        return ok, detail, run.text

    ok, detail, output = attempt("apex")
    if ok:
        result.ok = True
        return result

    for _ in range(max_expansions):
        diag = _diagnose(store, adapter, task_id, task, "\n\n".join(pack_parts), output, detail)
        result.diagnosis = diag
        candidates = select(
            node,
            resolve=store.get,
            diag=diag,
            judge=judge,
            config=config,
            task=task,
            exclude=tried,
        )
        candidates = [c for c in candidates if c.label not in tried]
        if not candidates:
            break
        best = candidates[0]
        tried.add(best.label)

        if best.kind == "delta":
            pack_parts.append(f"- {best.text}")
        else:
            pack_parts = [render_node(best.node)] if best.node else pack_parts
            node = best.node or node

        ok, detail, output = attempt(best.label)
        if ok:
            result.ok = True
            result.rescued_by = best
            return result

    # Escalate to the level-0 node: always present, never deleted.
    base = store.base_node(family)
    if base is not None and base.id != node.id:
        result.escalated = True
        pack_parts = [render_node(base)]
        node = base
        ok, detail, output = attempt("escalate:L0")
        result.ok = ok
    return result


def _diagnose(
    store: Store,
    adapter: Adapter,
    task_id: str,
    task: str,
    pack: str,
    output: str,
    complaint: str,
) -> Diagnosis:
    run = adapter.run(
        DIAGNOSE.format(
            task_id=task_id,
            task=truncate(task, 4000),
            pack=truncate(pack, 6000),
            output=truncate(output, 4000),
            complaint=truncate(complaint, 2000),
        ),
        schema=DIAGNOSE_SCHEMA,
        timeout=int(store.config.get("limits.agent_timeout_s", 180)),
    )
    if not run.ok or not run.data:
        # Degrade rather than break: with no diagnosis the scorer falls back to
        # lexical overlap and priors, i.e. roughly stepwise descent.
        return Diagnosis(category="rationale", missing=[complaint][:1], confidence=0.0)
    return Diagnosis.from_dict(run.data)
