"""Where does a newly learned lesson go?

Growing the tree is not just "append a leaf". When something is learned — by
human teaching or by self-discovery — it has to be reconciled with what is
already known, and there are five genuinely different answers:

| Relation | What it means | What we do |
|---|---|---|
| `duplicate` | already known, no new information | nothing; record the hit |
| `refines` | same topic, adds detail the tree lacks | fold into the L0 node; patch ancestors |
| `contradicts` | same topic, incompatible claim | keep both, mark disputed, ask the human |
| `specialises` | same topic, a distinct case | attach as a sibling; merge-compression may later generalise both |
| `orthogonal` | unrelated | new family — a brand new leaf |

The interesting one is `contradicts`. Silently overwriting is how a memory
system rots: whichever lesson was written last wins, regardless of which is
true. Instead the contradiction is recorded on the node and surfaced **at recall
time**, when the user is already thinking about that topic — the same reason a
student raises a confusion during the lesson it belongs to rather than at random.

Both judgements here are the model's: which remembered lessons cover the same
subject, and how the new one relates to them. Neither is a similarity score —
two lessons about the same service can share no vocabulary, and two that share
their most distinctive word can be about unrelated systems.

Efficiency comes from the tree, not from cheap approximations. Apexes are the
most compressed nodes in the store, so the whole top level fits in one question;
only lines the model cannot judge from a summary are opened further; every
candidate is reconciled in a single call; and verdicts are cached.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .adapters import Adapter
from .node import Node
from .store import Store
from .judge import Budget, Judge
from .util import truncate

RECONCILE_SCHEMA = {
    "type": "object",
    "required": ["match", "relation", "rationale"],
    "properties": {
        "match": {
            "type": "string",
            "description": "Id of the existing lesson this relates to, or empty for none.",
        },
        "relation": {
            "type": "string",
            "enum": ["duplicate", "refines", "contradicts", "specialises", "orthogonal"],
        },
        "rationale": {"type": "string"},
        "question": {
            "type": "string",
            "description": "If contradicts: the single question a human must answer to resolve it.",
        },
        "merged_body": {
            "type": "string",
            "description": "If refines: the existing lesson rewritten to include the new detail.",
        },
    },
}

RECONCILE = """RMC:reconcile

A new lesson has been learned. Decide how it relates to what is already in
memory, so it can be filed correctly instead of blindly appended.

You are shown several existing lessons. Pick the ONE it most relates to and put
its id in `match`, then classify the relation. If it relates to none of them,
set `match` to an empty string and `relation` to `orthogonal`.

Pick exactly one relation:

- `duplicate`   — the new lesson says nothing the existing one does not already
                  say. Wording differences do not count as new information.
- `refines`     — same subject, and the new lesson adds detail, a constraint, or
                  a case the existing one lacks. The two are compatible.
- `contradicts` — same subject, and they cannot both be true. One tells an agent
                  to do something the other forbids, or they state different
                  values for the same thing.
- `specialises` — same general subject, but the new lesson is about a distinct
                  case that deserves to stand alongside rather than merge in.
- `orthogonal`  — different subject; the topical overlap is superficial.

Be strict about `contradicts`: it means genuinely incompatible, not merely
different emphasis. When you do pick it, `question` must be the single question
a human could answer to settle which is right — concrete and answerable in one
sentence, e.g. "Was the port changed to 5433 permanently, or only while
legacy-pg was running?"

For `refines`, return `merged_body`: the matched lesson rewritten to carry the
new detail. Keep every load-bearing claim from both. Do not pad it.

<<<EXISTING
{existing}
EXISTING>>>

<<<NEW
{new}
NEW>>>
"""



@dataclass
class Placement:
    action: str  # new-family | attach-sibling | fold-into | conflict | duplicate
    family: str
    relation: str = "orthogonal"
    target: Node | None = None
    rationale: str = ""
    question: str = ""
    merged_body: str = ""
    consulted: bool = False  # whether a model call was needed

    def describe(self) -> str:
        where = f" -> {self.target.id}" if self.target else ""
        return f"{self.action}[{self.relation}] {self.family}{where}: {self.rationale[:120]}"


@dataclass
class PlacementResult:
    placement: Placement
    node: Node | None = None
    patched: list[str] = field(default_factory=list)


def _cache_key(new_body: str, node_ids: list[str]) -> str:
    digest = hashlib.sha256(new_body.encode("utf-8")).hexdigest()[:12]
    return f"{digest}:{','.join(sorted(node_ids))}"


def related_lessons(
    store: Store, judge: Judge, body: str, *, budget: Budget | None = None
) -> list[Node]:
    """Existing lessons that might be about the same thing, found by tree walk.

    Walking abstract → concrete and asking the model at each level is what
    replaces a similarity score here. Two lessons about the same service can
    share no vocabulary, and two lessons sharing their most distinctive word can
    be about unrelated systems — neither case is visible to token overlap, and
    both are obvious to a reader.

    The tree is what keeps this affordable: apexes are the most compressed nodes
    in the store, so the whole top level fits in a single question, and we only
    open a line when the model says the summary was too abstract to judge from.
    """
    roots = [n for n in (store.apex(f) for f in store.families()) if n is not None]
    if not roots:
        return []

    budget = budget or Budget(max_calls=int(store.config.get("placement.judge_calls", 2)))
    frontier, out, seen = roots, [], set()

    for _ in range(int(store.config.get("placement.max_depth", 2))):
        level = [n for n in frontier if n.id not in seen][:12]
        if not level or not budget.take():
            break
        seen.update(n.id for n in level)
        picks = judge.related(body, level)
        by_id = {n.id: n for n in level}
        nxt: list[Node] = []
        for pick in picks:
            node = by_id.get(pick.id)
            if node is None or not pick.positive:
                continue
            children = store.children(node) if pick.descend else []
            if children:
                nxt.extend(children)
            else:
                out.append(node)
        frontier = nxt
    return out + [n for n in frontier if n.id not in {o.id for o in out}]


def decide(
    store: Store,
    adapter: Adapter,
    *,
    body: str,
    family_hint: str = "",
    consult: bool = True,
) -> Placement:
    """Choose where a new lesson belongs.

    Two judgements, both the model's: *which* remembered lessons are about the
    same thing (found by walking the tree), and *how* the new one relates to
    them. The harness contributes the walk, the budget and the cache.
    """
    judge = Judge(store, adapter)
    candidates = related_lessons(store, judge, body)

    if not candidates:
        # The model found nothing on the same subject — a genuinely new leaf.
        return Placement(
            action="new-family",
            family=family_hint or "general",
            relation="orthogonal",
            rationale="nothing already remembered covers this subject",
        )

    best = candidates[0]
    if not consult:
        return Placement(
            action="attach-sibling",
            family=best.family,
            relation="specialises",
            target=best,
            rationale="related to an existing lesson; reconciliation skipped",
        )

    # One reconciliation call for every candidate the walk surfaced, so cost is
    # flat in the size of the tree and a contradiction with the second-best
    # match is still visible.
    cache = _load_cache(store)
    key = _cache_key(body, [n.id for n in candidates])
    data = cache.get(key)
    consulted = False
    if data is None:
        rendered = "\n\n".join(
            f"[id: {node.id}] {node.title or node.family}\n{truncate(node.body, 1500)}"
            for node in candidates
        )
        run = adapter.run(
            RECONCILE.format(existing=rendered, new=truncate(body, 4000)),
            schema=RECONCILE_SCHEMA,
            timeout=int(store.config.get("limits.agent_timeout_s", 180)),
        )
        if not run.ok or not run.data:
            # Reconciliation is an optimisation, not a gate. If it fails, attach
            # as a sibling: nothing is lost or overwritten, and merge-compression
            # can still generalise the two later.
            return Placement(
                action="attach-sibling",
                family=best.family,
                relation="specialises",
                target=best,
                rationale=f"reconciler unavailable ({run.error[:120]}); attached alongside",
            )
        data = run.data
        consulted = True
        _save_cache(store, key, data)

    matched_id = str(data.get("match") or "").strip()
    target = next((n for n in candidates if n.id == matched_id), None) or best
    relation = str(data.get("relation") or "orthogonal")
    if not matched_id:
        relation = "orthogonal"

    action = {
        "duplicate": "duplicate",
        "refines": "fold-into",
        "contradicts": "conflict",
        "specialises": "attach-sibling",
        "orthogonal": "new-family",
    }.get(relation, "attach-sibling")

    return Placement(
        action=action,
        family=(family_hint or target.family) if action == "new-family" else target.family,
        relation=relation,
        target=target if action != "new-family" else None,
        rationale=str(data.get("rationale") or ""),
        question=str(data.get("question") or ""),
        merged_body=str(data.get("merged_body") or ""),
        consulted=consulted,
    )


def _cache_path(store: Store):
    return store.root / "reconcile-cache.json"


def _load_cache(store: Store) -> dict[str, Any]:
    path = _cache_path(store)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_cache(store: Store, key: str, value: dict[str, Any], *, limit: int = 500) -> None:
    """Remember verdicts so re-running learning never re-pays for the same pair."""
    cache = _load_cache(store)
    cache[key] = value
    if len(cache) > limit:
        for stale in list(cache)[: len(cache) - limit]:
            cache.pop(stale, None)
    try:
        _cache_path(store).write_text(json.dumps(cache), encoding="utf-8")
    except Exception:
        pass


def apply(store: Store, placement: Placement, node: Node) -> PlacementResult:
    """Carry out a placement decision. ``node`` is the freshly minted lesson."""
    result = PlacementResult(placement=placement)

    if placement.action == "duplicate":
        # Nothing to store, but the hit is evidence the existing lesson is
        # pulling its weight, and worth knowing when reading the tree.
        store.log(
            "placement",
            action="duplicate",
            target=placement.target.id if placement.target else None,
            rationale=placement.rationale[:300],
        )
        return result

    if placement.action == "fold-into" and placement.target is not None:
        return _fold(store, placement, result)

    node.family = placement.family
    if placement.action == "conflict" and placement.target is not None:
        node.status = "disputed"
        node.conflict = placement.question or placement.rationale
        store.save_node(node)
        # Mark the incumbent too: whichever is wrong, an agent reading either one
        # should know the question is open.
        target = placement.target
        target.status = "disputed"
        target.conflict = placement.question or placement.rationale
        store.save_node(target)
        store.log(
            "conflict",
            new=node.id,
            existing=target.id,
            question=node.conflict[:300],
            rationale=placement.rationale[:300],
        )
    else:
        store.save_node(node)
        store.log(
            "placement",
            action=placement.action,
            node=node.id,
            family=node.family,
            relation=placement.relation,
        )

    store.invalidate()
    result.node = store.get(node.id)
    return result


def _fold(store: Store, placement: Placement, result: PlacementResult) -> PlacementResult:
    """Merge new detail into the existing L0, and patch everything above it.

    Folding into the detailed node is only half the job. Every ancestor was
    compressed from the *old* body and validated against it, so each is now
    missing the new detail. Rather than invalidating them — which would throw
    away working compressions — the new detail is registered as a rescue on each
    ancestor, so recall re-attaches it immediately and `compact.repair` folds it
    in permanently. The tree keeps working while it catches up.
    """
    target = placement.target
    assert target is not None

    base = store.base_node(target.family) or target
    addition = placement.merged_body.strip()
    if addition:
        base.body = addition
    store.save_node(base)

    detail = truncate(placement.rationale, 240)
    for ancestor in [target, *store.ancestors(base)]:
        if ancestor.id == base.id:
            continue
        store.log("rescue", node=ancestor.id, claim=detail, source="placement")
        result.patched.append(ancestor.id)

    store.invalidate()
    store.log(
        "placement",
        action="fold-into",
        node=base.id,
        patched=result.patched,
        rationale=placement.rationale[:300],
    )
    result.node = store.get(base.id)
    return result


def open_conflicts(store: Store, family: str | None = None) -> list[Node]:
    """Nodes with an unresolved contradiction, for surfacing at recall time."""
    out = [n for n in store.nodes() if n.conflict and n.status == "disputed"]
    if family:
        out = [n for n in out if n.family == family]
    return out


def resolve(store: Store, node_id: str, *, keep: bool = True) -> Node | None:
    """Mark a conflict settled: ``keep`` this node, or archive it."""
    node = store.get(node_id)
    if node is None:
        return None
    node.conflict = ""
    node.status = "active" if keep else "archived"
    store.save_node(node)
    store.invalidate()
    store.log("conflict-resolved", node=node_id, kept=keep)
    return node


