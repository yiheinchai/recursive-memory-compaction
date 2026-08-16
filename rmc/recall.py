"""Retrieval: pick the lesson families a prompt needs, and build a context pack.

Two entry points:

``recall_pack``  — ambient path. Runs inside a hook on every real prompt, must
                   be fast and must never call a model. Serves the apex (most
                   compressed) node of each matching family.

``solve_with_descent`` — controlled path. Used by replay/eval and ``rmc solve``,
                   where RMC owns the loop and can therefore observe failure,
                   diagnose it, and descend the tree mid-task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .adapters import Adapter, AgentResult
from .config import Config
from .node import Node
from .prompts import DIAGNOSE, DIAGNOSE_SCHEMA, REPLAY
from .selection import Candidate, Diagnosis, select
from .store import Store
from .util import count_tokens, jaccard, signature, truncate


@dataclass
class Pack:
    """The text injected ahead of a task, plus the bookkeeping to score it later."""

    text: str = ""
    served: list[str] = field(default_factory=list)
    families: list[str] = field(default_factory=list)
    patches: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    tokens: int = 0

    def __bool__(self) -> bool:
        return bool(self.text.strip())


# --------------------------------------------------------------------------- #
# family matching
# --------------------------------------------------------------------------- #


def match_families(
    store: Store, prompt: str, *, limit: int | None = None, min_match: float | None = None
) -> list[tuple[str, float]]:
    """Rank lesson families by lexical similarity to the prompt.

    Deliberately model-free: this runs on every keystroke-to-submit in a real
    session, so it has to cost nothing. Cheap matching plus an apex that is only
    ~100 tokens means a false positive is nearly free, while a false negative
    costs a whole learned lesson.
    """
    limit = limit if limit is not None else int(store.config.get("recall.max_families", 3))
    floor = min_match if min_match is not None else float(store.config.get("recall.min_match", 0.12))

    psig = signature(prompt)
    if not psig:
        return []

    scored: list[tuple[str, float]] = []
    for family in store.families():
        apex = store.apex(family)
        if apex is None:
            continue
        fam_sig = apex.sig | signature(family.replace("-", " ")) | set(apex.tags)
        for child in store.descendants(apex)[:6]:
            fam_sig |= set(child.tags)
        score = max(
            jaccard(psig, fam_sig),
            0.85 * _tag_hit(psig, set(apex.tags) | {family}),
        )
        if score >= floor:
            scored.append((family, score))

    scored.sort(key=lambda kv: -kv[1])
    return scored[:limit]


def _tag_hit(psig: set[str], tags: set[str]) -> float:
    tags = {t.lower().replace("-", " ") for t in tags if t}
    if not tags:
        return 0.0
    hits = sum(1 for t in tags if any(part in psig for part in t.split()))
    return min(1.0, hits / max(1, len(tags)))


# --------------------------------------------------------------------------- #
# pack construction
# --------------------------------------------------------------------------- #


def render_node(node: Node) -> str:
    heading = node.title.strip() or node.family
    return f"### {heading}  ·  L{node.level}\n{node.body.strip()}"


def recall_pack(
    store: Store,
    prompt: str,
    *,
    budget: int | None = None,
    include_patches: bool = True,
) -> Pack:
    """Build the ambient context pack for a prompt. No model calls."""
    pack = Pack()
    if not store.config.get("recall.enabled", True):
        return pack

    budget = budget or int(store.config.get("recall.max_pack_tokens", 1200))
    chunks: list[str] = []
    used = 0

    for family, _score in match_families(store, prompt):
        node = store.apex(family)
        if node is None:
            continue
        rendered = render_node(node)
        cost = count_tokens(rendered)
        if used + cost > budget and chunks:
            break
        chunks.append(rendered)
        used += cost
        pack.served.append(node.id)
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

    task_sig = signature(task)
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
            task_sig=task_sig,
            config=config,
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
