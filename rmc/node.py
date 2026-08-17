"""The lesson node: one abstraction level of one lesson, stored as markdown.

Edge directions are named explicitly because "parent" is ambiguous in a tree
that grows upward from detail to abstraction:

    parents      -> points UP,   toward less detail
    derived_from -> points DOWN, toward more detail

Both are lists, which makes this a DAG rather than a tree, and that is
deliberate. A lesson can be abstracted in more than one direction: compressed
vertically into a terser form of itself, *and* merged sideways with a different
lesson into a shared generalisation. Those are two different abstractions over
the same leaf and both are worth keeping.

While `parents` was a single field, the second abstraction silently destroyed
the first — a merge overwrote the pointer, leaving the earlier parent still
claiming the node as a child while the node no longer acknowledged it. Not a
missing feature; a corrupt graph.

Recall walks *down* ``derived_from``. Learning grows *up* via ``parents``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import yamlish
from .util import count_tokens, utcnow

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


@dataclass
class Stats:
    # Times this lesson was put in front of the model. `attempts` counts the
    # subset that actually bore on the work, so `shown - attempts` is what it
    # has cost in context for nothing — the only record a lesson keeps of its
    # own retrieval history, and the thing the selector needs in order to stop
    # repeating a bad pick.
    shown: int = 0
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    expansions: int = 0
    rescues: int = 0  # times this node fixed a failure of an ancestor
    last_used: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "shown": self.shown,
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
            # Older stores have no `shown`. Falling back to `attempts` says
            # "every time it was served it was used", which is the reading that
            # does not punish a lesson for missing history.
            shown=int(raw.get("shown") or raw.get("attempts") or 0),
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
    # One line, written when the lesson is minted or compressed. This is what
    # the router sees. Relevance is decided over gists, not bodies: sending 700
    # characters per lesson to *choose* which lessons to send is the scaling
    # bug that eats the context it was meant to protect.
    gist: str = ""
    created: str = field(default_factory=utcnow)
    updated: str = field(default_factory=utcnow)
    derived_from: list[str] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)
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
    # Spans a reflection pass *observed* doing work: it watched a session and
    # reported that this specific part of the lesson changed what the agent did.
    #
    # Distinct from `preserve`, and the difference is the direction of the
    # evidence. `preserve` is negative and post-hoc — a compression cut this and
    # replay failed, so put it back. This is positive and observational — this
    # part was seen mattering in real work, before any compression touched it.
    # Compression previously had only the first, which meant it guessed what to
    # cut and found out afterwards; with this it can take the reduction from the
    # parts that have no record of being used.
    #
    # Absence is not evidence of uselessness: a span may simply not have come up
    # yet. That asymmetry is stated in the compressor prompt, because reading
    # this list as "everything else is dead" is the obvious wrong move.
    load_bearing: list[str] = field(default_factory=list)
    path: Path | None = None

    # ---------------------------------------------------------------- derived
    @property
    def tokens(self) -> int:
        return count_tokens(self.body)

    @property
    def is_apex(self) -> bool:
        return not self.parents

    def summary(self, *, limit: int = 240) -> str:
        """The routing view. Prefers the stored gist; degrades to a short head."""
        if self.gist.strip():
            return self.gist.strip()
        head = " ".join(self.body.split())
        return head[: limit - 1] + "…" if len(head) > limit else head

    def deltas_by_kind(self, kind: str) -> list[Delta]:
        return [d for d in self.dropped if d.kind == kind]

    # ------------------------------------------------------------ serialise
    def to_frontmatter(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "title": self.title,
            "gist": self.gist,
            "level": self.level,
            "status": self.status,
            "origin": self.origin,
            "conflict": self.conflict,
            "created": self.created,
            "updated": self.updated,
            "tokens": self.tokens,
            "derived_from": list(self.derived_from),
            "parents": list(self.parents),
            "covers_tasks": list(self.covers_tasks),
            "tags": list(self.tags),
            "preserve": list(self.preserve),
            "load_bearing": list(self.load_bearing),
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
            gist=str(meta.get("gist") or ""),
            created=meta.get("created") or utcnow(),
            updated=meta.get("updated") or utcnow(),
            derived_from=_as_list(meta.get("derived_from")),
            # `compressed_into` is the pre-DAG spelling; read it so existing
            # stores keep working.
            parents=_as_list(meta.get("parents")) or _as_list(meta.get("compressed_into")),
            covers_tasks=_as_list(meta.get("covers_tasks")),
            tags=_as_list(meta.get("tags")),
            dropped=[Delta.from_dict(d) for d in (meta.get("dropped") or [])],
            stats=Stats.from_dict(meta.get("stats")),
            status=str(meta.get("status") or "active"),
            origin=str(meta.get("origin") or "reflection"),
            conflict=str(meta.get("conflict") or ""),
            preserve=_as_list(meta.get("preserve")),
            load_bearing=_as_list(meta.get("load_bearing")),
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
