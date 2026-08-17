"""Selection lessons: RMC applied to its own retrieval.

Every other stage of RMC learns from outcomes. Retrieval — the stage that
decides what enters the user's context on every prompt — learned from nothing,
and it is the one measured to be worst: 48% precision after filtering
(EXPERIMENTS §4.1), meaning over half of what recall serves is never used.

A selection lesson is what the reflector writes after watching a session: not
knowledge about the work, but knowledge about *where the knowledge was*. "When
the task touches the integration tests, read `nodes/testing/` before running
anything." The next selector reads that and reaches the right set in fewer tool
calls, or skips a search it now knows is fruitless.

Three properties make this layer different from the lesson tree, and each is
load-bearing:

**They live outside `nodes/`.** If selection lessons were nodes they would be
retrieved by the very mechanism they exist to fix, and would compete with real
lessons for the same injection budget. They are always injected instead, under
their own cap.

**They must be conditioned on a task type.** EXPERIMENTS §4.2 already measured
the unconditioned form: annotating candidates with their track record ("shown
5x, never used") dropped precision to 41% and recall to 81% — worse on both.
The recorded reason is that "low usage is a statement about the distribution of
work, not the lesson". So a rule with no `when` is rejected here rather than
stored and hoped for. That check is also what keeps the layer from growing 1:1
with the store: "n_abc is rarely useful" is one rule per lesson, while "when the
task touches X, look in Y" is one rule per *kind of work*.

**The layer is capped, and the cap is enforced by eviction, not by hope.** The
claim that selection lessons are far fewer than memories is a bet, not a
guarantee. ``growth`` reports the ratio so the bet is visible as a number, and
``fit`` drops the least useful rules when the cap binds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import yamlish
from .util import count_tokens, new_id, utcnow

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .store import Store

ROUTING_DIRNAME = "routing"

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


@dataclass
class Rule:
    """One selection lesson: a condition on the task, and what to do about it."""

    id: str
    # The task type this applies to. Never empty — an unconditioned rule is
    # rejected at mint time, because it is both the form that made retrieval
    # worse when measured and the form that grows with the store.
    when: str
    # What the selector should do when `when` holds: where to look, or what to
    # stop opening.
    then: str = ""
    created: str = field(default_factory=utcnow)
    updated: str = field(default_factory=utcnow)
    # How often this rule was in front of a selector, and how it went. `helped`
    # and `wasted` are what compaction ranks on when the cap binds.
    shown: int = 0
    helped: int = 0
    wasted: int = 0
    origin: str = "reflection"  # reflection | manual
    status: str = "active"
    path: Path | None = None

    @property
    def tokens(self) -> int:
        return count_tokens(self.render())

    @property
    def posterior(self) -> float:
        """Laplace-smoothed rate at which this rule improved a selection."""
        return (self.helped + 1) / (self.helped + self.wasted + 2)

    def render(self) -> str:
        return f"- When {self.when.strip().rstrip('.')}: {self.then.strip()}"

    def to_frontmatter(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "when": self.when,
            "created": self.created,
            "updated": self.updated,
            "origin": self.origin,
            "status": self.status,
            "tokens": self.tokens,
            "stats": {"shown": self.shown, "helped": self.helped, "wasted": self.wasted},
        }

    def to_markdown(self) -> str:
        fm = yamlish.dump(self.to_frontmatter()).rstrip("\n")
        return f"---\n{fm}\n---\n\n{self.then.strip()}\n"

    @classmethod
    def from_markdown(cls, text: str, path: Path | None = None) -> "Rule":
        match = _FRONTMATTER_RE.match(text)
        if not match:
            raise ValueError(f"routing rule has no frontmatter: {path}")
        meta = yamlish.load(match.group(1)) or {}
        if not isinstance(meta, dict):
            raise ValueError(f"routing frontmatter is not a mapping: {path}")
        stats = meta.get("stats") or {}
        return cls(
            id=str(meta.get("id") or ""),
            when=str(meta.get("when") or ""),
            then=match.group(2).strip(),
            created=meta.get("created") or utcnow(),
            updated=meta.get("updated") or utcnow(),
            shown=int(stats.get("shown") or 0),
            helped=int(stats.get("helped") or 0),
            wasted=int(stats.get("wasted") or 0),
            origin=str(meta.get("origin") or "reflection"),
            status=str(meta.get("status") or "active"),
            path=path,
        )

    def touch(self) -> None:
        self.updated = utcnow()


# --------------------------------------------------------------------------- #
# storage
# --------------------------------------------------------------------------- #


def routing_dir(store: "Store") -> Path:
    return store.root / ROUTING_DIRNAME


def load(store: "Store") -> list[Rule]:
    """Every active rule, this store's and the global layer's.

    Layered exactly as nodes are: a rule about where *your* lessons live follows
    you between repos, while a rule about this repo's tests does not.
    """
    rules: dict[str, Rule] = {}
    if getattr(store, "parent", None) is not None:
        rules = {r.id: r for r in load(store.parent)}
    directory = routing_dir(store)
    if directory.is_dir():
        for path in sorted(directory.glob("*.md")):
            try:
                rule = Rule.from_markdown(path.read_text(encoding="utf-8"), path)
            except Exception:
                continue  # one malformed rule must not disable routing
            if rule.id and rule.status == "active":
                rules[rule.id] = rule
    return sorted(rules.values(), key=lambda r: (-r.posterior, r.id))


def save(store: "Store", rule: Rule) -> Path:
    directory = routing_dir(store)
    directory.mkdir(parents=True, exist_ok=True)
    rule.touch()
    path = rule.path or (directory / f"{rule.id}.md")
    path.write_text(rule.to_markdown(), encoding="utf-8")
    rule.path = path
    return path


def delete(rule: Rule) -> None:
    if rule.path and rule.path.exists():
        rule.path.unlink()


def get(store: "Store", rule_id: str) -> Rule | None:
    return next((r for r in load(store) if r.id == rule_id), None)


# --------------------------------------------------------------------------- #
# minting
# --------------------------------------------------------------------------- #


def mint(store: "Store", *, when: str, then: str, origin: str = "reflection") -> Rule | None:
    """Store a new rule, or ``None`` if it is not one.

    The rejection is the point. A rule with no task condition is the exact shape
    EXPERIMENTS §4.2 measured as harmful, and accepting it here on the grounds
    that the model meant well is how that result gets repeated.
    """
    when, then = (when or "").strip(), (then or "").strip()
    if not when or not then:
        return None
    rule = Rule(id=new_id("r"), when=when, then=then, origin=origin)
    existing = load(store)
    if any(_same(rule, other) for other in existing):
        return None
    save(store, rule)
    return rule


def _same(a: Rule, b: Rule) -> bool:
    """Cheap duplicate check on the condition, not the action.

    Two rules that fire on the same task type are a contradiction or a
    restatement, and either way the second should be reconciled rather than
    appended — the failure mode this whole layer exists to avoid is a context
    that grows by accretion.
    """
    return _norm(a.when) == _norm(b.when)


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


# --------------------------------------------------------------------------- #
# injection
# --------------------------------------------------------------------------- #


def fit(rules: list[Rule], budget: int) -> list[Rule]:
    """The rules that fit the cap, best-evidenced first.

    Ranked by the rate at which each has actually improved a selection, so a
    rule that keeps sending the selector to the wrong place falls out of the
    layer on its own rather than waiting for someone to notice it.
    """
    out: list[Rule] = []
    used = 0
    for rule in sorted(rules, key=lambda r: (-r.posterior, -r.helped, r.id)):
        cost = rule.tokens
        if used + cost > budget:
            continue
        out.append(rule)
        used += cost
    return out


def render(rules: list[Rule]) -> str:
    if not rules:
        return ""
    return "\n".join(r.render() for r in rules)


def credit(store: "Store", *, helped: list[str], wasted: list[str], shown: list[str]) -> None:
    """Fold a selection's outcome back into the rules that shaped it."""
    by_id = {r.id: r for r in load(store)}
    touched: set[str] = set()
    for rule_id in shown:
        rule = by_id.get(rule_id)
        if rule is not None:
            rule.shown += 1
            touched.add(rule_id)
    for rule_id in helped:
        rule = by_id.get(rule_id)
        if rule is not None:
            rule.helped += 1
            touched.add(rule_id)
    for rule_id in wasted:
        rule = by_id.get(rule_id)
        if rule is not None:
            rule.wasted += 1
            touched.add(rule_id)
    for rule_id in touched:
        save(store, by_id[rule_id])


# --------------------------------------------------------------------------- #
# the measurement the thesis stands or falls on
# --------------------------------------------------------------------------- #


def growth(store: "Store") -> dict[str, Any]:
    """Selection lessons against memories.

    The design bets that this layer grows with the number of *kinds of work* a
    user does, not with the number of lessons they have accumulated. If the
    ratio does not flatten, the bet is wrong and the whole approach to the long
    tail needs revisiting — so it is reported as a number rather than left to be
    inferred from how things feel.
    """
    rules = load(store)
    nodes = [n for n in store.nodes() if n.status != "archived"]
    return {
        "rules": len(rules),
        "nodes": len(nodes),
        "ratio": (len(rules) / len(nodes)) if nodes else 0.0,
        "tokens": sum(r.tokens for r in rules),
    }
