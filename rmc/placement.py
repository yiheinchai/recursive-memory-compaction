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

Cheap first: a lexical shortlist decides whether there is anything to reconcile
at all, and only then is a model asked. Most new lessons in an empty or
unrelated part of the tree cost nothing.
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
from .util import jaccard, signature, truncate

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

{hints}
<<<EXISTING
{existing}
EXISTING>>>

<<<NEW
{new}
NEW>>>
"""

HINT_HEADER = """The following identifiers are given DIFFERENT values by the new lesson and an
existing one. That usually means a contradiction, but not always — check whether
one supersedes the other or whether they apply in different situations:

{lines}
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
    similarity: float = 0.0
    consulted: bool = False  # whether a model call was needed

    def describe(self) -> str:
        where = f" -> {self.target.id}" if self.target else ""
        return f"{self.action}[{self.relation}] {self.family}{where}: {self.rationale[:120]}"


@dataclass
class PlacementResult:
    placement: Placement
    node: Node | None = None
    patched: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# free contradiction pre-filter
# --------------------------------------------------------------------------- #

# `KEY=value`, `KEY: value`, `--flag value` and `--flag=value`.
_ASSIGN_RE = re.compile(
    r"(?:^|[\s`'\"(])(--?[A-Za-z][\w-]{2,}|[A-Z][A-Z0-9_]{3,}|[a-z][\w.]{3,})"
    r"\s*[:=]\s*[`'\"]?([\w./:-]{1,40})"
)

# Values that carry no meaning on their own and would produce noise.
_TRIVIAL = frozenset({"true", "false", "none", "null", "yes", "no", "0", "1", "the", "a"})


def assignments(text: str) -> dict[str, set[str]]:
    """Identifier -> values it is given in this text."""
    out: dict[str, set[str]] = {}
    for key, value in _ASSIGN_RE.findall(text or ""):
        value = value.strip("`'\".,;)")
        if not value or value.lower() in _TRIVIAL:
            continue
        out.setdefault(key, set()).add(value)
    return out


def contradiction_hints(a: str, b: str, *, limit: int = 6) -> list[str]:
    """Identifiers both texts assign, but to different values.

    A free, purely lexical signal. It cannot decide a contradiction on its own —
    two lessons may legitimately use different ports in different environments —
    but it is a strong enough smell to be worth (a) forcing a reconciliation
    call that similarity alone would have skipped, and (b) telling the
    reconciler exactly where to look, which measurably improves its verdicts.
    """
    left, right = assignments(a), assignments(b)
    hints: list[str] = []
    for key in sorted(set(left) & set(right)):
        if left[key] != right[key] and not (left[key] & right[key]):
            hints.append(
                f"- `{key}`: existing says {sorted(left[key])}, new says {sorted(right[key])}"
            )
        if len(hints) >= limit:
            break
    return hints


def _cache_key(new_body: str, node_ids: list[str]) -> str:
    digest = hashlib.sha256(new_body.encode("utf-8")).hexdigest()[:12]
    return f"{digest}:{','.join(sorted(node_ids))}"


def shortlist(store: Store, body: str, family_hint: str = "", *, top: int = 3) -> list[tuple[Node, float]]:
    """Existing apex nodes that might be about the same thing."""
    sig = signature(body)
    if not sig:
        return []
    scored: list[tuple[Node, float]] = []
    for family in store.families():
        apex = store.apex(family)
        if apex is None:
            continue
        score = jaccard(sig, apex.sig)
        if family_hint and family == family_hint:
            # The reflector already proposed a family; trust it enough to always
            # consider, but not enough to skip the reconciliation check.
            score = max(score, 0.5)
        scored.append((apex, score))
    scored.sort(key=lambda kv: -kv[1])
    return scored[:top]


def decide(
    store: Store,
    adapter: Adapter,
    *,
    body: str,
    family_hint: str = "",
    consult: bool = True,
) -> Placement:
    """Choose where a new lesson belongs."""
    floor = float(store.config.get("placement.min_similarity", 0.15))
    top_k = int(store.config.get("placement.candidates", 3))
    candidates = shortlist(store, body, family_hint, top=top_k)

    # Free pre-filter. Two lessons that give the same identifier different values
    # deserve a look even when they share little vocabulary — that is precisely
    # the case a similarity floor misses, and precisely the case that matters.
    flagged: list[tuple[Node, float]] = []
    all_hints: list[str] = []
    for node, score in candidates:
        hints = contradiction_hints(node.body, body)
        if hints:
            all_hints.extend(hints)
            flagged.append((node, max(score, floor)))

    considered = flagged or [c for c in candidates if c[1] >= floor]

    if not considered:
        # Nothing close enough to reconcile with — a genuinely new leaf. The
        # common case early on, and it costs no model call at all.
        return Placement(
            action="new-family",
            family=family_hint or "general",
            relation="orthogonal",
            rationale="no existing lesson is close enough to reconcile with",
            similarity=candidates[0][1] if candidates else 0.0,
        )

    best, score = considered[0]
    if not consult:
        return Placement(
            action="attach-sibling",
            family=best.family,
            relation="specialises",
            target=best,
            rationale="similar to an existing lesson; reconciliation skipped",
            similarity=score,
        )

    # One call for all candidates rather than one per candidate: reconciliation
    # cost stays constant as the tree grows, while coverage improves — a
    # contradiction with the second-best match is no longer invisible.
    cache = _load_cache(store)
    key = _cache_key(body, [n.id for n, _ in considered])
    data = cache.get(key)
    consulted = False
    if data is None:
        rendered = "\n\n".join(
            f"[id: {node.id}] {node.title or node.family}\n{truncate(node.body, 1500)}"
            for node, _ in considered
        )
        hint_block = HINT_HEADER.format(lines="\n".join(all_hints)) if all_hints else ""
        run = adapter.run(
            RECONCILE.format(existing=rendered, new=truncate(body, 4000), hints=hint_block),
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
                similarity=score,
            )
        data = run.data
        consulted = True
        _save_cache(store, key, data)

    matched_id = str(data.get("match") or "").strip()
    target = next((n for n, _ in considered if n.id == matched_id), None) or best
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
        similarity=score,
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
            similarity=round(placement.similarity, 3),
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


def config_defaults() -> dict[str, Any]:
    return {"min_similarity": 0.15, "consult": True}
