"""The lesson node: one abstraction level of one lesson, stored as markdown.

Edge directions are named explicitly because "parent" is ambiguous in a tree
that grows upward from detail to abstraction:

    compressed_into -> points UP,   toward less detail (at most one)
    derived_from    -> points DOWN, toward more detail (may be several: merges)

Recall walks *down* ``derived_from``. Learning grows *up* via ``compressed_into``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import yamlish
from .util import count_tokens, signature, utcnow

# Closed vocabulary shared by ``dropped[].kind`` and the diagnoser's
# ``category``. Keeping it closed is what lets descent match a failure to a
# dropped claim without embeddings.
DELTA_KINDS = (
    "parameter",
    "example",
    "precondition",
    "edge-case",
    "rationale",
    "counter-example",
    "procedure-step",
    "naming",
    "reference",
)

STATUSES = ("active", "superseded", "demoted", "disputed", "archived")

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


@dataclass
class Delta:
    """One claim removed by a compression, attributed to a node that still holds it."""

    claim: str
    kind: str = "rationale"
    holder: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"claim": self.claim, "kind": self.kind, "holder": self.holder}

    @classmethod
    def from_dict(cls, raw: Any) -> "Delta":
        if isinstance(raw, str):
            return cls(claim=raw)
        raw = raw or {}
        kind = str(raw.get("kind") or "rationale").strip().lower()
        if kind not in DELTA_KINDS:
            kind = "rationale"
        return cls(
            claim=str(raw.get("claim") or "").strip(),
            kind=kind,
            holder=raw.get("holder") or None,
        )

    @property
    def sig(self) -> set[str]:
        return signature(self.claim)


@dataclass
class Stats:
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    expansions: int = 0
    rescues: int = 0  # times this node fixed a failure of an ancestor
    last_used: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "successes": self.successes,
            "failures": self.failures,
            "expansions": self.expansions,
            "rescues": self.rescues,
            "last_used": self.last_used,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "Stats":
        raw = raw or {}
        return cls(
            attempts=int(raw.get("attempts") or 0),
            successes=int(raw.get("successes") or 0),
            failures=int(raw.get("failures") or 0),
            expansions=int(raw.get("expansions") or 0),
            rescues=int(raw.get("rescues") or 0),
            last_used=raw.get("last_used"),
        )

    @property
    def posterior(self) -> float:
        """Laplace-smoothed success rate: unused children are neither favoured nor buried."""
        return (self.successes + 1) / (self.attempts + 2)


@dataclass
class Node:
    id: str
    family: str
    body: str = ""
    level: int = 0
    title: str = ""
    created: str = field(default_factory=utcnow)
    updated: str = field(default_factory=utcnow)
    derived_from: list[str] = field(default_factory=list)
    compressed_into: str | None = None
    covers_tasks: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    dropped: list[Delta] = field(default_factory=list)
    stats: Stats = field(default_factory=Stats)
    status: str = "active"
    origin: str = "reflection"  # reflection | compression | merge | manual
    # An unresolved contradiction with something already in the tree. Held on
    # the node so it can be surfaced at recall time, when the user is already
    # thinking about this topic, rather than as an out-of-context interruption.
    conflict: str = ""
    preserve: list[str] = field(default_factory=list)  # hints from rejected compressions
    path: Path | None = None

    # ---------------------------------------------------------------- derived
    @property
    def tokens(self) -> int:
        return count_tokens(self.body)

    @property
    def sig(self) -> set[str]:
        return signature(f"{self.title}\n{self.body}\n{' '.join(self.tags)}")

    @property
    def is_apex(self) -> bool:
        return self.compressed_into is None

    def deltas_by_kind(self, kind: str) -> list[Delta]:
        return [d for d in self.dropped if d.kind == kind]

    # ------------------------------------------------------------ serialise
    def to_frontmatter(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "title": self.title,
            "level": self.level,
            "status": self.status,
            "origin": self.origin,
            "conflict": self.conflict,
            "created": self.created,
            "updated": self.updated,
            "tokens": self.tokens,
            "derived_from": list(self.derived_from),
            "compressed_into": self.compressed_into,
            "covers_tasks": list(self.covers_tasks),
            "tags": list(self.tags),
            "preserve": list(self.preserve),
            "dropped": [d.to_dict() for d in self.dropped],
            "stats": self.stats.to_dict(),
        }

    def to_markdown(self) -> str:
        fm = yamlish.dump(self.to_frontmatter()).rstrip("\n")
        return f"---\n{fm}\n---\n\n{self.body.strip()}\n"

    @classmethod
    def from_markdown(cls, text: str, path: Path | None = None) -> "Node":
        match = _FRONTMATTER_RE.match(text)
        if not match:
            raise ValueError(f"node file has no frontmatter: {path}")
        meta = yamlish.load(match.group(1)) or {}
        if not isinstance(meta, dict):
            raise ValueError(f"node frontmatter is not a mapping: {path}")
        body = match.group(2).strip()
        return cls(
            id=str(meta.get("id") or ""),
            family=str(meta.get("family") or "default"),
            body=body,
            level=int(meta.get("level") or 0),
            title=str(meta.get("title") or ""),
            created=meta.get("created") or utcnow(),
            updated=meta.get("updated") or utcnow(),
            derived_from=_as_list(meta.get("derived_from")),
            compressed_into=meta.get("compressed_into") or None,
            covers_tasks=_as_list(meta.get("covers_tasks")),
            tags=_as_list(meta.get("tags")),
            dropped=[Delta.from_dict(d) for d in (meta.get("dropped") or [])],
            stats=Stats.from_dict(meta.get("stats")),
            status=str(meta.get("status") or "active"),
            origin=str(meta.get("origin") or "reflection"),
            conflict=str(meta.get("conflict") or ""),
            preserve=_as_list(meta.get("preserve")),
            path=path,
        )

    def touch(self) -> None:
        self.updated = utcnow()


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None and str(v).strip()]
    return []
