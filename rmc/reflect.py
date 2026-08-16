"""Turning a finished session into store updates.

Three steps, in order of cost:

``observe``  — one judgement. The model reads the session and says how it went,
               whether the human had to steer, and what was worked out by trial.
               Node statistics, the episode record and ambient descent all key
               off that verdict.
``descend``  — free when nothing was dropped; otherwise one judgement about
               which omitted detail the correction was really about.
``mint``     — one judgement. Only when the session contains something reusable,
               followed by reconciliation against what is already known.

A structural gate comes first: a session with two tool calls and no follow-up is
skipped without asking anything. That is a question about size, not meaning, so
it stays in code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adapters import Adapter
from .judge import Judge
from .node import Node
from .prompts import REFLECT, REFLECT_SCHEMA
from .selection import Diagnosis, build_candidates, rank
from .signals import SessionFacts, digest, summarise_work, worth_assessing
from .store import Episode, Store
from .util import new_id, truncate, utcnow


@dataclass
class Outcome:
    """The model's reading of a session."""

    label: str = "unknown"  # success | failure | unknown
    confidence: float = 0.0
    corrected: bool = False
    correction: str = ""
    evidence: list[str] = field(default_factory=list)
    discoveries: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""

    @classmethod
    def from_verdict(cls, raw: dict[str, Any] | None) -> "Outcome":
        raw = raw or {}
        label = str(raw.get("outcome") or "unknown").strip().lower()
        if label not in ("success", "failure", "unknown"):
            label = "unknown"
        try:
            confidence = max(0.0, min(1.0, float(raw.get("confidence") or 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        evidence = raw.get("evidence") or []
        return cls(
            label=label,
            confidence=confidence,
            corrected=bool(raw.get("corrected")),
            correction=str(raw.get("correction") or ""),
            evidence=[str(e) for e in evidence][:6],
            discoveries=[d for d in (raw.get("discoveries") or []) if isinstance(d, dict)],
            summary=str(raw.get("summary") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "corrected": self.corrected,
            "evidence": self.evidence,
        }

    def render_discoveries(self) -> str:
        lines = []
        for found in self.discoveries:
            failed = str(found.get("what_failed") or "").strip()
            why = str(found.get("why_it_failed") or "").strip()
            worked = str(found.get("what_worked") or "").strip()
            attempts = found.get("attempts")
            suffix = f" (after {attempts} attempts)" if isinstance(attempts, int) else ""
            lines.append(f"- tried: {failed}\n  failed because: {why}\n  what worked: {worked}{suffix}")
        return "\n".join(lines)


@dataclass
class ObserveResult:
    outcome: Outcome
    episode: Episode | None = None
    updated: list[str] = field(default_factory=list)
    rescues: list[tuple[str, str]] = field(default_factory=list)  # (node_id, claim)
    skipped: str = ""


def observe(
    store: Store,
    facts: SessionFacts,
    *,
    adapter: Adapter | None = None,
    session_id: str = "",
    served: list[str] | None = None,
    family_hint: str = "",
    cwd: str = "",
) -> ObserveResult:
    """Judge a finished session and fold the result back into the tree."""
    min_tool_calls = int(store.config.get("learning.min_tool_calls", 8))
    min_conf = float(store.config.get("signals.min_confidence", 0.5))
    served = served or []
    nodes = [n for n in (store.get(i) for i in served) if n is not None]

    if not worth_assessing(facts, min_tool_calls=min_tool_calls):
        return ObserveResult(outcome=Outcome(), skipped="session too small to judge")
    if adapter is None:
        return ObserveResult(outcome=Outcome(), skipped="no backend available to judge with")

    verdict = Judge(store, adapter).assess(digest(facts))
    if verdict is None:
        store.log("observe", session=session_id, outcome="unknown", reason="judge unavailable")
        return ObserveResult(outcome=Outcome(), skipped="judge unavailable")

    outcome = Outcome.from_verdict(verdict)
    result = ObserveResult(outcome=outcome)

    # The session outcome and the *lesson's* outcome are different questions. If
    # the human had to steer, the served lesson failed at its job even when the
    # session went on to end well — and that is the case RMC most needs to learn
    # from, so it must not be recorded as a success.
    confident = outcome.label != "unknown" and outcome.confidence >= min_conf

    # Below the confidence floor we deliberately do nothing: a noisy label is
    # worse than no label, because it poisons both the priors and the replay
    # corpus every future compression is judged against. An explicit correction
    # is exempt — it is unambiguous evidence about the lesson however the session
    # ended.
    if not confident and not outcome.corrected:
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
        if outcome.corrected or outcome.label == "failure":
            node.stats.failures += 1
        elif confident:
            node.stats.successes += 1
        node.stats.last_used = utcnow()
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
        accepted_summary=(outcome.summary or summarise_work(facts))
        if outcome.label == "success"
        else "",
        session_id=session_id,
        cwd=cwd,
    )
    # Only successful episodes are replayable regression tests; failures are kept
    # for diagnosis but must never become what a compression is validated against.
    if (confident and outcome.label == "success") or store.config.get(
        "learning.capture_failures", True
    ):
        store.save_episode(episode)
        result.episode = episode

    # Attach as a covering task only when the lesson produced the result on its
    # own. A corrected session's episode stays in the corpus — useful once the
    # node is repaired — but attaching it now would gate every future compression
    # behind a test the node cannot currently pass.
    if confident and outcome.label == "success" and not outcome.corrected:
        for node in nodes:
            if episode.id not in node.covers_tasks:
                node.covers_tasks = sorted({*node.covers_tasks, episode.id})
                store.save_node(node)

    if nodes and (outcome.corrected or outcome.label == "failure"):
        result.rescues = descend(store, nodes, outcome, facts, adapter)

    store.log(
        "observe",
        session=session_id,
        outcome=outcome.label,
        confidence=outcome.confidence,
        corrected=outcome.corrected,
        discoveries=len(outcome.discoveries),
        served=served,
        episode=episode.id,
        evidence=outcome.evidence[:3],
    )
    return result


def descend(
    store: Store,
    nodes: list[Node],
    outcome: Outcome,
    facts: SessionFacts,
    adapter: Adapter | None = None,
) -> list[tuple[str, str]]:
    """Work out which dropped detail the correction was really about.

    The human has already said what went wrong, so there is nothing to diagnose
    — but deciding *which* omitted claim that correction refers to is a
    judgement about meaning, and the model makes it.

    A match is recorded as a `rescue` event rather than acted on immediately:
    the session is already over. `recall._sticky_patches` re-attaches the claim
    on the next matching prompt, and `compact.repair` eventually folds it back
    into the body for good.
    """
    complaint = outcome.correction.strip() or "\n".join(outcome.evidence).strip()
    if not complaint:
        return []

    diag = Diagnosis(
        category="",
        missing=[complaint],
        wrong_step="the agent had to be corrected" if outcome.corrected else "the work was wrong",
        confidence=outcome.confidence,
    )
    judge = Judge(store, adapter) if adapter is not None else None
    rescues: list[tuple[str, str]] = []

    for node in nodes:
        if not node.dropped:
            # Nothing was ever dropped from this node, so the gap is genuinely
            # new knowledge rather than lost detail. Leave it to `mint`.
            continue
        candidates = rank(
            build_candidates(node, resolve=store.get, strategy="delta-patch"),
            diag=diag,
            judge=judge,
            config=store.config,
            task=facts.first_prompt,
        )
        best = next((c for c in candidates if c.kind == "delta"), None)
        if best is None or best.parts.get("judge", 0.0) <= 0.05:
            continue  # the model does not think any dropped claim explains this
        store.log("rescue", node=node.id, claim=best.text, score=round(best.score, 4))
        rescues.append((node.id, best.text))

        node.stats.expansions += 1
        hint = truncate(complaint, 200)
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
    placement: Any = None
    patched: list[str] = field(default_factory=list)


def mint(
    store: Store,
    adapter: Adapter,
    facts: SessionFacts,
    *,
    outcome: Outcome | None = None,
    session_id: str = "",
    cwd: Path | None = None,
) -> MintResult:
    """Decide whether this session contained a reusable lesson, and file it.

    Deliberately conservative — the prompt tells the model that "no" is the
    expected answer. Every low-value lesson permanently taxes retrieval, because
    it competes for attention on every future prompt.
    """
    if not store.config.get("learning.enabled", True):
        return MintResult(reason="learning disabled")

    min_tool_calls = int(store.config.get("learning.min_tool_calls", 8))
    if not worth_assessing(facts, min_tool_calls=min_tool_calls):
        return MintResult(reason="session too small")

    discovered = outcome.render_discoveries() if outcome else ""
    correction = (outcome.correction if outcome else "").strip()

    run = adapter.run(
        REFLECT.format(
            families="\n".join(f"- {f}" for f in store.families()) or "(none yet)",
            correction=correction or "(the human did not correct anything)",
            discovered=discovered or "(nothing was worked out by trial)",
            excerpt=digest(facts, limit=7000),
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
        gist=str(run.data.get("gist") or ""),
        tags=[_slug(t) for t in (run.data.get("tags") or []) if str(t).strip()][:8],
        origin="reflection",
    )

    # Reconcile with what is already known before writing. Appending blindly is
    # how a memory grows contradictions it never notices.
    from . import placement as placement_mod

    decision = placement_mod.decide(
        store,
        adapter,
        body=body,
        family_hint=family,
        consult=bool(store.config.get("placement.consult", True)),
    )
    applied = placement_mod.apply(store, decision, node)

    store.log(
        "mint",
        node=applied.node.id if applied.node else None,
        family=decision.family,
        action=decision.action,
        relation=decision.relation,
        session=session_id,
        tokens=node.tokens,
    )
    if applied.node is None:
        return MintResult(reason=f"{decision.relation}: {decision.rationale}", placement=decision)
    return MintResult(
        created=applied.node,
        reason=decision.describe(),
        placement=decision,
        patched=applied.patched,
    )


def _slug(text: str) -> str:
    cleaned = "".join(c if c.isalnum() else "-" for c in str(text).strip().lower())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")[:48] or "general"
