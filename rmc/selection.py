"""Descent policy — *which child do we go to when the compressed lesson fails?*

The reason this question is hard is that compression normally destroys the
information you would need to invert it, so descent degenerates into trying
children at random and paying full price for the wrong one.

RMC's answer is to make compression record its own losses. Every compressed
node carries a delta manifest: the discrete claims that were removed, each
tagged with a `kind` from a closed vocabulary and attributed to a descendant
that still holds it. The failure diagnosis uses that same closed vocabulary for
its `category`, which gives us a join key. Descent then stops being a search
problem and becomes ranked retrieval over the manifest.

Scoring, per candidate:

    score = w_d·delta_match + w_t·task_affinity + w_p·prior − w_c·cost

`prior` is a Laplace-smoothed success rate, which makes descent a contextual
bandit over the tree: branches that repeatedly rescue failures rise, branches
that never help sink, with no hand-tuning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from .config import Config
from .node import Delta, Node
from .util import count_tokens, jaccard, overlap_coeff, signature


@dataclass
class Diagnosis:
    """Structured account of *how* a node failed a task."""

    category: str = "rationale"
    missing: list[str] = field(default_factory=list)
    wrong_step: str = ""
    confidence: float = 0.0

    @property
    def sig(self) -> set[str]:
        return signature(" ".join(self.missing) + " " + self.wrong_step)

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

    @property
    def text(self) -> str:
        if self.kind == "delta" and self.delta is not None:
            return self.delta.claim
        return self.node.body if self.node else ""

    def explain(self) -> str:
        bits = " ".join(f"{k}={v:+.3f}" for k, v in self.parts.items())
        return f"{self.score:6.3f}  {self.kind:5s} {self.label:<28s} {bits}"


# --------------------------------------------------------------------------- #
# component scores
# --------------------------------------------------------------------------- #


def delta_match(candidate_sig: set[str], candidate_kind: str, diag: Diagnosis) -> float:
    """How well a candidate addresses the diagnosed gap.

    Half the signal is the categorical join (did we drop the *kind* of thing
    that is now missing), half is lexical overlap between the dropped claim and
    the named missing facts. The overlap coefficient rather than Jaccard,
    because a three-word diagnosis should still be able to match a long claim.
    """
    kind_hit = 1.0 if candidate_kind and candidate_kind == diag.category else 0.0
    lexical = overlap_coeff(candidate_sig, diag.sig)
    return 0.5 * kind_hit + 0.5 * lexical


def task_affinity(candidate_sig: set[str], task_sig: set[str]) -> float:
    return jaccard(candidate_sig, task_sig)


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
    task_sig: set[str],
    config: Config,
) -> list[Candidate]:
    w_d = float(config.get("selection.w_delta", 0.45))
    w_t = float(config.get("selection.w_affinity", 0.25))
    w_p = float(config.get("selection.w_prior", 0.20))
    w_c = float(config.get("selection.w_cost", 0.10))
    explore = str(config.get("selection.explore", "posterior"))
    ucb_c = float(config.get("selection.ucb_c", 0.7))
    budget = int(config.get("recall.max_pack_tokens", 1200))
    total_attempts = sum((c.node.stats.attempts if c.node else 0) for c in candidates) or 1

    for cand in candidates:
        if cand.kind == "delta" and cand.delta is not None:
            csig, ckind = cand.delta.sig, cand.delta.kind
        else:
            node = cand.node
            csig = node.sig if node else set()
            # A node inherits the best kind-match among the deltas it holds.
            ckind = ""
            if node is not None and node.dropped:
                kinds = {d.kind for d in node.dropped}
                ckind = diag.category if diag.category in kinds else ""
            if cand.delta is not None:
                ckind = cand.delta.kind
                csig = csig | cand.delta.sig

        parts = {
            "delta": w_d * delta_match(csig, ckind, diag),
            "affinity": w_t * task_affinity(csig, task_sig),
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
    task_sig: set[str],
    config: Config,
    exclude: set[str] | None = None,
) -> list[Candidate]:
    strategy = str(config.get("recall.strategy", "delta-patch"))
    cands = build_candidates(node, resolve=resolve, strategy=strategy, exclude=exclude)
    return rank(cands, diag=diag, task_sig=task_sig, config=config)
