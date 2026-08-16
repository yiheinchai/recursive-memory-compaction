"""Descent policy — *which child do we go to when the compressed lesson fails?*

The reason this question is hard is that compression normally destroys the
information you would need to invert it, so descent degenerates into trying
children at random and paying full price for the wrong one.

RMC's answer is to make compression record its losses. Every compressed node
carries a delta manifest: the discrete claims that were removed, each attributed
to a descendant that still holds it. Descent is then a ranking problem over that
manifest rather than a search over the tree.

**The ranking itself is a model judgement**, not a similarity score. Whether a
particular omitted claim explains a particular failure is a question about
meaning: "parse the body, not the status code" is the fix for "treated HTTP 200
as success" while sharing almost no vocabulary with it. A lexical matcher gets
that exactly backwards, and a categorical `kind == category` join is a coarse
proxy for the same thing.

Two terms stay in code, because they are *evidence* rather than proxies for
judgement:

- ``prior`` — the Laplace-smoothed rate at which this node has actually rescued
  failures before. That is an observed outcome, not an inference about meaning,
  and it makes descent a contextual bandit over the tree.
- ``cost`` — token count. A measured fact.

    score = w_j · model_usefulness + w_p · prior − w_c · cost
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from .config import Config
from .judge import Judge
from .node import Delta, Node
from .util import count_tokens


@dataclass
class Diagnosis:
    """Structured account of *how* a node failed a task."""

    category: str = "rationale"
    missing: list[str] = field(default_factory=list)
    wrong_step: str = ""
    confidence: float = 0.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "Diagnosis":
        raw = raw or {}
        missing = raw.get("missing") or []
        if isinstance(missing, str):
            missing = [missing]
        return cls(
            category=str(raw.get("category") or "rationale").strip().lower(),
            missing=[str(m) for m in missing],
            wrong_step=str(raw.get("wrong_step") or ""),
            confidence=float(raw.get("confidence") or 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "missing": self.missing,
            "wrong_step": self.wrong_step,
            "confidence": self.confidence,
        }

    def render(self) -> str:
        """The failure, as prose for the model to reason about."""
        parts = []
        if self.wrong_step:
            parts.append(f"What went wrong: {self.wrong_step}")
        if self.missing:
            parts.append("Information the lesson appears to lack:")
            parts.extend(f"  - {m}" for m in self.missing)
        if self.category:
            parts.append(f"Kind of gap: {self.category}")
        return "\n".join(parts) or "The lesson did not lead to the right result."


@dataclass
class Candidate:
    """Something we could add to the context pack to rescue a failed recall."""

    kind: Literal["delta", "node"]
    label: str
    tokens: int
    delta: Delta | None = None
    node: Node | None = None
    score: float = 0.0
    parts: dict[str, float] = field(default_factory=dict)
    why: str = ""

    @property
    def text(self) -> str:
        if self.kind == "delta" and self.delta is not None:
            return self.delta.claim
        return self.node.body if self.node else ""

    def explain(self) -> str:
        bits = " ".join(f"{k}={v:+.3f}" for k, v in self.parts.items())
        return f"{self.score:6.3f}  {self.kind:5s} {self.label:<28s} {bits}"


# --------------------------------------------------------------------------- #
# evidence terms (arithmetic over observed outcomes, not semantic guesses)
# --------------------------------------------------------------------------- #


def prior(node: Node | None, *, explore: str = "posterior", c: float = 0.7, total: int = 1) -> float:
    if node is None:
        return 0.5
    base = node.stats.posterior
    if explore != "ucb":
        return base
    attempts = max(1, node.stats.attempts)
    return min(1.0, base + c * math.sqrt(math.log(max(2, total)) / attempts))


def cost(tokens: int, budget: int) -> float:
    if budget <= 0:
        return 0.0
    return min(1.0, tokens / budget)


# --------------------------------------------------------------------------- #
# candidate construction and ranking
# --------------------------------------------------------------------------- #


def build_candidates(
    node: Node,
    *,
    resolve: Any,
    strategy: str = "delta-patch",
    exclude: set[str] | None = None,
) -> list[Candidate]:
    """Enumerate what could rescue ``node``.

    ``resolve`` maps a node id to a Node (normally ``store.get``).
    """
    exclude = exclude or set()
    out: list[Candidate] = []

    if strategy in ("delta-patch", "delta-jump"):
        for i, delta in enumerate(node.dropped):
            if not delta.claim.strip():
                continue
            key = f"{node.id}#{i}"
            if key in exclude:
                continue
            holder = resolve(delta.holder) if delta.holder else None
            if strategy == "delta-jump" and holder is not None:
                out.append(
                    Candidate(
                        kind="node",
                        label=f"{holder.id}(L{holder.level})",
                        tokens=holder.tokens,
                        node=holder,
                        delta=delta,
                    )
                )
            else:
                out.append(
                    Candidate(
                        kind="delta",
                        label=f"{delta.kind}:{key}",
                        tokens=count_tokens(delta.claim),
                        delta=delta,
                        node=holder,
                    )
                )

    # Always offer the direct children as a fallback: a node whose manifest is
    # empty or unhelpful must still be descendable.
    for child_id in node.derived_from:
        if child_id in exclude:
            continue
        child = resolve(child_id)
        if child is None or child.status == "archived":
            continue
        out.append(
            Candidate(
                kind="node",
                label=f"{child.id}(L{child.level})",
                tokens=child.tokens,
                node=child,
            )
        )
    return out


def rank(
    candidates: list[Candidate],
    *,
    diag: Diagnosis,
    judge: Judge | None,
    config: Config,
    task: str = "",
) -> list[Candidate]:
    """Score candidates. The relevance term is the model's; the rest is evidence.

    With no judge available the relevance term is simply absent, and ranking
    falls back to prior and cost — i.e. "try what has worked before, cheapest
    first". That degrades gracefully rather than silently substituting a
    similarity metric that would look like a judgement without being one.
    """
    w_j = float(config.get("selection.w_judge", 0.60))
    w_p = float(config.get("selection.w_prior", 0.28))
    w_c = float(config.get("selection.w_cost", 0.12))
    explore = str(config.get("selection.explore", "posterior"))
    ucb_c = float(config.get("selection.ucb_c", 0.7))
    budget = int(config.get("recall.max_pack_tokens", 1200))
    total_attempts = sum((c.node.stats.attempts if c.node else 0) for c in candidates) or 1

    usefulness: dict[str, float] = {}
    if judge is not None and candidates:
        failure = diag.render()
        if task:
            failure = f"Task: {task.strip()[:600]}\n\n{failure}"
        usefulness = judge.rank_repairs(
            failure,
            [(c.label, (c.delta.kind if c.delta else "lesson"), c.text) for c in candidates],
        )

    for cand in candidates:
        parts = {
            "judge": w_j * usefulness.get(cand.label, 0.0),
            "prior": w_p * prior(cand.node, explore=explore, c=ucb_c, total=total_attempts),
            "cost": -w_c * cost(cand.tokens, budget),
        }
        cand.parts = parts
        cand.score = sum(parts.values())

    # Ties break toward cheaper, then toward more specific (lower level).
    candidates.sort(
        key=lambda c: (
            -c.score,
            c.tokens,
            c.node.level if c.node else 0,
            c.label,
        )
    )
    return candidates


def select(
    node: Node,
    *,
    resolve: Any,
    diag: Diagnosis,
    judge: Judge | None,
    config: Config,
    task: str = "",
    exclude: set[str] | None = None,
) -> list[Candidate]:
    strategy = str(config.get("recall.strategy", "delta-patch"))
    cands = build_candidates(node, resolve=resolve, strategy=strategy, exclude=exclude)
    return rank(cands, diag=diag, judge=judge, config=config, task=task)
