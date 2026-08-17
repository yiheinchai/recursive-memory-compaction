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
from .judge import Budget, Judge, Pick, WalkResult, _render, walk
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
    # The judge could not answer. An empty pack then means memory is broken,
    # not that nothing applied — and the two must never look the same to a user
    # deciding whether RMC works.
    degraded: bool = False
    error: str = ""
    # Which selection rules shaped this pack. Carried through to the session
    # record so the reflector can credit the ones that worked and charge the
    # ones that sent the selector somewhere useless — without this the routing
    # layer has outcomes it can never see.
    rules_shown: list[str] = field(default_factory=list)
    rules_used: list[str] = field(default_factory=list)

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
    session_id: str = "",
    cwd: Any = None,
) -> WalkResult:
    """Which lessons bear on this prompt.

    Two selectors, and which one runs is a question about capability rather than
    preference. The agentic one searches the store from a fork of the live
    session and is the only one whose cost does not grow with the store; it
    needs a session to fork and a backend that can fork it. When either is
    missing — the first turn of a session, a non-Claude backend — the judge-walk
    below runs instead. It is the fallback, not a legacy path: it is also the
    measurement baseline every arm of `rmc eval-recall` is compared against.
    """
    from . import select_agent

    can_search, why_not = select_agent.available(store, adapter, session_id)
    if can_search:
        result = select_agent.select(
            store, adapter, prompt, session_id=session_id, cwd=cwd, limit=limit
        )
        # A failed search falls through to the judge rather than serving
        # nothing. The two selectors fail independently, and an empty pack is
        # expensive enough that it is worth spending the second call to avoid
        # one caused by a transient.
        if not result.failed:
            return result
        store.log("select-fallback", reason=result.error[:200])
    elif why_not:
        store.log("select-fallback", reason=why_not)

    return _walk_lessons(store, adapter, prompt, limit=limit, budget=budget)


def _walk_lessons(
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

    # A brand-new store skips the decision, and only a brand-new one.
    #
    # This gate used to be a token budget (1200), justified as "judgement is
    # only needed under scarcity, and early on there is none". The cost half of
    # that was wrong: across 57 prompts on a store that fit the budget, 15,917
    # of ~17,800 injected tokens were never used, and re-judging exactly those
    # served sets kept every lesson that had borne on the work while dropping
    # 55% of the noise. Context that fits is not context that is free — an
    # unrelated lesson spends attention, and the relevance prompt itself says it
    # "can actively mislead".
    #
    # The latency half was right, and does not go away: this runs inside a hook
    # that blocks the user's prompt, at ~5s of CLI startup alone and ~34s on the
    # model that routes well. So the gate survives, sized by *count* rather than
    # tokens — what it waves through should be a few lines, not a page. Above
    # this, the noise is worth five seconds; at or below it, it is not.
    #
    # Scarcity was never the reason to choose. Latency is, and it is a worse
    # reason — see EXPERIMENTS.md §4.4, which records this as unresolved.
    ceiling = int(store.config.get("recall.filter_above", 3))
    if len(roots) <= ceiling:
        result = WalkResult(selected=roots)
        for node in roots:
            result.picks[node.id] = Pick(
                id=node.id,
                verdict="relevant",
                why=f"only {len(roots)} lesson(s) stored — served without a routing call",
            )
        return result

    # The top level must be judged in full, whatever it costs. It is the only
    # level where every lesson is a candidate, so a chunk left unexamined there
    # is a lesson that cannot be retrieved at all — which is what `judge_calls:
    # 2` against 26 apexes and a fanout of 12 actually meant. `judge_calls` now
    # buys *descent*, on top of a complete first pass.
    #
    # Both terms are counts, so the harness owns them.
    fanout = int(store.config.get("recall.fanout", 12))
    min_cacheable = int(store.config.get("recall.min_cacheable_tokens", 1200))
    first_pass = max(1, -(-len(roots) // fanout))
    budget = budget or Budget(
        max_calls=first_pass + int(store.config.get("recall.judge_calls", 2))
    )
    judge = Judge(store, adapter, timeout=int(store.config.get("recall.timeout_s", 20)))

    # Keep the candidate list warm across prompts once it is large enough that
    # re-sending it costs more than seeding it once. Below that the seed call is
    # pure overhead — the list is cheap and the extra round trip is not.
    warm_above = int(store.config.get("recall.warm_prefix_above_tokens", 2000))
    apex_render = "\n\n".join(_render(n) for n in roots)
    if count_tokens(apex_render) >= warm_above:
        import hashlib

        from .router import Router

        judge.router = Router(store)

        # Providers only open a new cache entry for a prefix above a minimum
        # size — around a thousand tokens. A chunk of 12 summaries is ~660, so
        # seeding it writes nothing and every fork reads back the host's system
        # prompt and none of ours. Measured: `ours_cached: 4` on nine calls out
        # of ten. Widening the chunk until it clears the minimum is what makes
        # the mechanism function at all, and chunk width is a count.
        per = max(1, count_tokens(apex_render) // max(1, len(roots)))
        fanout = max(fanout, -(-min_cacheable // per))

        # Named chunk by chunk, exactly as the walk will split them, so each
        # question lands on the conversation already holding its candidates.
        judge.warm_prefixes = {
            hashlib.sha256(
                "\n\n".join(_render(n) for n in roots[i : i + fanout]).encode("utf-8")
            ).hexdigest()[:16]
            for i in range(0, len(roots), fanout)
        }
    result = walk(
        judge,
        prompt,
        roots,
        expand=store.children,
        budget=budget,
        max_depth=int(store.config.get("recall.max_depth", 2)),
        fanout=fanout,
        workers=int(store.config.get("recall.parallel", 4)),
    )

    # An empty pack has two very different causes and they must not look alike
    # to the caller: nothing was relevant, or nothing answered.
    if judge.failures and not result.selected:
        result.failed = True
        result.error = judge.last_error

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
    session_id: str = "",
    cwd: Any = None,
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

    selection = select_lessons(store, adapter, prompt, session_id=session_id, cwd=cwd)
    pack.reasons = {n.id: selection.why(n.id) for n in selection.selected}
    pack.degraded, pack.error = selection.failed, selection.error
    pack.rules_shown = list(selection.rules_shown)
    pack.rules_used = list(selection.rules_used)
    if pack.degraded:
        store.log("recall-degraded", error=pack.error[:200])

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
