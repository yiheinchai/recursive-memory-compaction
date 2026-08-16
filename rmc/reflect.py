"""Turning a finished session into store updates.

Three jobs, in order of cost:

``observe``  — free. Score the session, update the stats of whatever was served,
               and file the episode into the replay corpus.
``descend``  — free. If a served lesson failed and the human said what was
               missing, match that text against the node's delta manifest and
               record a rescue, so the next recall re-attaches the claim.
``mint``     — one model call. Only when the session contains something reusable
               that is not already in the tree.

The ordering matters: the common case (a session that went fine) costs nothing
beyond a few counter increments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adapters import Adapter
from .node import Node
from .prompts import REFLECT, REFLECT_SCHEMA
from .selection import Diagnosis, rank, build_candidates
from .signals import Outcome, SessionFacts, classify, correction_text, excerpt, summarise_work
from .store import Episode, Store
from .util import new_id, signature, truncate


@dataclass
class ObserveResult:
    outcome: Outcome
    episode: Episode | None = None
    updated: list[str] = field(default_factory=list)
    rescues: list[tuple[str, str]] = field(default_factory=list)  # (node_id, claim)


def observe(
    store: Store,
    facts: SessionFacts,
    *,
    session_id: str = "",
    served: list[str] | None = None,
    family_hint: str = "",
    cwd: str = "",
) -> ObserveResult:
    """Score a finished session and fold the result back into the tree."""
    min_tool_calls = int(store.config.get("learning.min_tool_calls", 8))
    min_conf = float(store.config.get("signals.min_confidence", 0.5))

    outcome = classify(facts, min_tool_calls=min_tool_calls)
    result = ObserveResult(outcome=outcome)

    served = served or []
    nodes = [n for n in (store.get(i) for i in served) if n is not None]

    # Below the confidence floor we deliberately do nothing. A noisy label is
    # worse than no label: it poisons both the priors and the replay corpus that
    # every future compression is judged against.
    if outcome.label == "unknown" or outcome.confidence < min_conf:
        store.log(
            "observe",
            session=session_id,
            outcome="unknown",
            confidence=outcome.confidence,
            served=served,
        )
        return result

    for node in nodes:
        node.stats.attempts += 1
        if outcome.label == "success":
            node.stats.successes += 1
        else:
            node.stats.failures += 1
        node.stats.last_used = store.config.get("_now") or None
        store.save_node(node)
        result.updated.append(node.id)

    family = family_hint or (nodes[0].family if nodes else "")
    episode = Episode(
        id=new_id("e"),
        family=family or "default",
        prompt=truncate(facts.first_prompt, 4000),
        outcome=outcome.label,
        confidence=outcome.confidence,
        served=served,
        accepted_summary=summarise_work(facts) if outcome.label == "success" else "",
        session_id=session_id,
        cwd=cwd,
    )
    # Only successful episodes are replayable regression tests; failures are
    # kept for diagnosis but must never become the thing a compression is
    # validated against.
    if outcome.label == "success" or store.config.get("learning.capture_failures", True):
        store.save_episode(episode)
        result.episode = episode

    if outcome.label == "success":
        for node in nodes:
            if episode.id not in node.covers_tasks:
                node.covers_tasks = sorted({*node.covers_tasks, episode.id})
                store.save_node(node)

    if outcome.label == "failure" and nodes:
        result.rescues = descend(store, nodes, facts)

    store.log(
        "observe",
        session=session_id,
        outcome=outcome.label,
        confidence=outcome.confidence,
        served=served,
        episode=episode.id,
        evidence=outcome.evidence[:3],
    )
    return result


def descend(store: Store, nodes: list[Node], facts: SessionFacts) -> list[tuple[str, str]]:
    """Ambient descent: match the human's correction against the delta manifest.

    This is the cheap half of the descent policy. The human has already told us
    what was missing, in words, so there is nothing to diagnose with a model —
    we can treat the correction as the diagnosis and rank the dropped claims
    against it directly.

    A match is recorded as a `rescue` event rather than acted on immediately:
    the session is already over. `recall._sticky_patches` re-attaches the claim
    on the next matching prompt, and `compact.repair` eventually folds it back
    into the body for good.
    """
    correction = correction_text(facts)
    if not correction.strip():
        return []

    diag = Diagnosis(
        category="rationale",  # unknown; scoring falls back to lexical overlap
        missing=[correction],
        wrong_step="",
        confidence=0.4,
    )
    task_sig = signature(facts.first_prompt)
    rescues: list[tuple[str, str]] = []

    for node in nodes:
        if not node.dropped:
            # Nothing was ever dropped from this node, so the gap is genuinely
            # new knowledge, not lost detail. Leave it to `mint`.
            continue
        candidates = rank(
            build_candidates(node, resolve=store.get, strategy="delta-patch"),
            diag=diag,
            task_sig=task_sig,
            config=store.config,
        )
        best = next((c for c in candidates if c.kind == "delta"), None)
        if best is None or best.parts.get("delta", 0.0) <= 0.05:
            continue  # no dropped claim plausibly explains this failure
        store.log("rescue", node=node.id, claim=best.text, score=round(best.score, 4))
        rescues.append((node.id, best.text))

        node.stats.expansions += 1
        hint = truncate(correction, 200)
        if hint not in node.preserve:
            node.preserve = [*node.preserve, hint][-8:]
        store.save_node(node)

    return rescues


# --------------------------------------------------------------------------- #
# minting a level-0 lesson
# --------------------------------------------------------------------------- #


@dataclass
class MintResult:
    created: Node | None = None
    reason: str = ""


def mint(
    store: Store,
    adapter: Adapter,
    facts: SessionFacts,
    *,
    session_id: str = "",
    cwd: Path | None = None,
) -> MintResult:
    """Ask a model whether this session contained a reusable lesson, and file it.

    Deliberately conservative — the prompt tells the model that "no" is the
    expected answer. Every low-value lesson permanently taxes retrieval, because
    it competes for the family match on every future prompt.
    """
    if not store.config.get("learning.enabled", True):
        return MintResult(reason="learning disabled")

    min_tool_calls = int(store.config.get("learning.min_tool_calls", 8))
    if facts.tool_calls < min_tool_calls and not correction_text(facts):
        return MintResult(reason="session too small")

    run = adapter.run(
        REFLECT.format(
            families="\n".join(f"- {f}" for f in store.families()) or "(none yet)",
            excerpt=excerpt(facts),
        ),
        schema=REFLECT_SCHEMA,
        cwd=cwd,
        timeout=int(store.config.get("limits.agent_timeout_s", 180)),
    )
    if not run.ok or not run.data:
        return MintResult(reason=f"reflector failed: {run.error[:200]}")
    if not run.data.get("capture"):
        return MintResult(reason=str(run.data.get("reason") or "nothing worth capturing"))

    body = str(run.data.get("body") or "").strip()
    if not body:
        return MintResult(reason="reflector returned an empty lesson")

    family = _slug(str(run.data.get("family") or "general"))
    node = Node(
        id=new_id("n"),
        family=family,
        body=body,
        level=0,
        title=str(run.data.get("title") or family),
        tags=[_slug(t) for t in (run.data.get("tags") or []) if str(t).strip()][:8],
        origin="reflection",
    )
    store.save_node(node)
    store.invalidate()
    store.log("mint", node=node.id, family=family, session=session_id, tokens=node.tokens)
    return MintResult(created=node, reason="captured")


def _slug(text: str) -> str:
    cleaned = "".join(c if c.isalnum() else "-" for c in str(text).strip().lower())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")[:48] or "general"
