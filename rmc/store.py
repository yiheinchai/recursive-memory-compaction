"""On-disk store: lesson nodes, replay episodes, telemetry, session state.

Layout (``.rmc/`` in the repo, or ``~/.rmc`` when there is no project store):

    .rmc/
      config.yaml
      nodes/<family>/<id>.md      lesson nodes (the tree)
      episodes/<id>.json          replay corpus — the ambient oracle
      sessions/<session_id>.json  in-flight state: which nodes were served
      events.jsonl                append-only telemetry
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import Config
from .node import Node
from .redact import redact_obj
from .util import new_id, signature, utcnow

STORE_DIRNAME = ".rmc"


def find_store_root(start: Path | None = None) -> Path | None:
    """Nearest ``.rmc`` walking up from ``start``; else the global store if it exists."""
    if env := os.environ.get("RMC_HOME"):
        return Path(env).expanduser()
    cur = (start or Path.cwd()).resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / STORE_DIRNAME).is_dir():
            return candidate / STORE_DIRNAME
    global_store = Path.home() / STORE_DIRNAME
    return global_store if global_store.is_dir() else None


@dataclass
class Episode:
    """A recorded real session, replayable as a regression test.

    This is the ambient oracle. In a scripted harness you write an oracle by
    hand; running inside someone's real repo you do not get that, so instead we
    record what actually happened when the work was accepted, and later ask
    whether a compressed lesson still reproduces it.
    """

    id: str
    family: str
    prompt: str
    outcome: str = "unknown"  # success | failure | unknown
    confidence: float = 0.0
    served: list[str] = None  # node ids injected into that session
    accepted_summary: str = ""  # what the agent ended up doing, once accepted
    check: dict[str, Any] = None  # optional mechanical oracle harvested from session
    created: str = ""
    session_id: str = ""
    cwd: str = ""

    def __post_init__(self) -> None:
        self.served = self.served or []
        self.check = self.check or {}
        self.created = self.created or utcnow()

    @property
    def sig(self) -> set[str]:
        return signature(f"{self.prompt}\n{self.accepted_summary}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "prompt": self.prompt,
            "outcome": self.outcome,
            "confidence": self.confidence,
            "served": self.served,
            "accepted_summary": self.accepted_summary,
            "check": self.check,
            "created": self.created,
            "session_id": self.session_id,
            "cwd": self.cwd,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Episode":
        return cls(
            id=raw.get("id") or new_id("e"),
            family=raw.get("family") or "default",
            prompt=raw.get("prompt") or "",
            outcome=raw.get("outcome") or "unknown",
            confidence=float(raw.get("confidence") or 0.0),
            served=list(raw.get("served") or []),
            accepted_summary=raw.get("accepted_summary") or "",
            check=dict(raw.get("check") or {}),
            created=raw.get("created") or utcnow(),
            session_id=raw.get("session_id") or "",
            cwd=raw.get("cwd") or "",
        )


class Store:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.config = Config.load(self.root / "config.yaml")
        self._nodes: dict[str, Node] | None = None

    # ------------------------------------------------------------------ init
    @classmethod
    def init(cls, base: Path, *, force: bool = False) -> "Store":
        root = Path(base) / STORE_DIRNAME
        if root.exists() and not force:
            return cls(root)
        for sub in ("nodes", "episodes", "sessions"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        store = cls(root)
        if not (root / "config.yaml").exists():
            store.config.save(root / "config.yaml")
        gitignore = root / ".gitignore"
        if not gitignore.exists():
            # Sessions and telemetry are machine-local noise; nodes and episodes
            # are the artefact worth committing and sharing across a team.
            gitignore.write_text("sessions/\nevents.jsonl\n*.lock\n", encoding="utf-8")
        return store

    @classmethod
    def discover(cls, start: Path | None = None) -> "Store | None":
        root = find_store_root(start)
        return cls(root) if root else None

    # ----------------------------------------------------------------- paths
    @property
    def nodes_dir(self) -> Path:
        return self.root / "nodes"

    @property
    def episodes_dir(self) -> Path:
        return self.root / "episodes"

    @property
    def sessions_dir(self) -> Path:
        return self.root / "sessions"

    # ----------------------------------------------------------------- nodes
    def _load_nodes(self) -> dict[str, Node]:
        if self._nodes is not None:
            return self._nodes
        nodes: dict[str, Node] = {}
        if self.nodes_dir.is_dir():
            for path in sorted(self.nodes_dir.rglob("*.md")):
                try:
                    node = Node.from_markdown(path.read_text(encoding="utf-8"), path)
                except Exception:
                    continue  # a malformed node must not take the whole store down
                if node.id:
                    nodes[node.id] = node
        self._nodes = nodes
        return nodes

    def invalidate(self) -> None:
        self._nodes = None

    def nodes(self) -> list[Node]:
        return list(self._load_nodes().values())

    def get(self, node_id: str) -> Node | None:
        return self._load_nodes().get(node_id)

    def families(self) -> list[str]:
        return sorted({n.family for n in self.nodes()})

    def family_nodes(self, family: str, *, active_only: bool = True) -> list[Node]:
        out = [n for n in self.nodes() if n.family == family]
        if active_only:
            out = [n for n in out if n.status in ("active", "demoted")]
        return sorted(out, key=lambda n: (-n.level, n.id))

    def apex(self, family: str) -> Node | None:
        """Highest-level servable node of a family (demoted nodes are skipped)."""
        candidates = [
            n for n in self.family_nodes(family) if n.status == "active" and n.is_apex
        ]
        if not candidates:
            candidates = [n for n in self.family_nodes(family) if n.status == "active"]
        if not candidates:
            return None
        return max(candidates, key=lambda n: (n.level, n.stats.posterior))

    def children(self, node: Node) -> list[Node]:
        """Nodes one step *down* — more detail."""
        return [n for n in (self.get(i) for i in node.derived_from) if n is not None]

    def descendants(self, node: Node, *, _seen: set[str] | None = None) -> list[Node]:
        seen = _seen if _seen is not None else set()
        out: list[Node] = []
        for child in self.children(node):
            if child.id in seen:
                continue
            seen.add(child.id)
            out.append(child)
            out.extend(self.descendants(child, _seen=seen))
        return out

    def ancestors(self, node: Node) -> list[Node]:
        """Nodes upward via ``compressed_into``, nearest first."""
        out: list[Node] = []
        seen = {node.id}
        cur = node
        while cur.compressed_into:
            nxt = self.get(cur.compressed_into)
            if nxt is None or nxt.id in seen:
                break
            seen.add(nxt.id)
            out.append(nxt)
            cur = nxt
        return out

    def base_node(self, family: str) -> Node | None:
        """The level-0 fallback: guaranteed-correct, never deleted."""
        zeros = [n for n in self.family_nodes(family, active_only=False) if n.level == 0]
        if not zeros:
            return None
        return max(zeros, key=lambda n: n.stats.posterior)

    def save_node(self, node: Node) -> Path:
        node.touch()
        directory = self.nodes_dir / node.family
        directory.mkdir(parents=True, exist_ok=True)
        path = node.path or (directory / f"{node.id}.md")
        if self.config.get("privacy.redact", True):
            node.body = redact_obj(node.body)
        path.write_text(node.to_markdown(), encoding="utf-8")
        node.path = path
        if self._nodes is not None:
            self._nodes[node.id] = node
        return path

    def delete_node(self, node: Node) -> None:
        if node.path and node.path.exists():
            node.path.unlink()
        if self._nodes is not None:
            self._nodes.pop(node.id, None)

    # -------------------------------------------------------------- episodes
    def episodes(self, family: str | None = None) -> list[Episode]:
        out: list[Episode] = []
        if not self.episodes_dir.is_dir():
            return out
        for path in sorted(self.episodes_dir.glob("*.json")):
            try:
                out.append(Episode.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except Exception:
                continue
        if family:
            out = [e for e in out if e.family == family]
        return out

    def save_episode(self, episode: Episode) -> Path:
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        payload = episode.to_dict()
        if self.config.get("privacy.redact", True):
            payload = redact_obj(payload)
        path = self.episodes_dir / f"{episode.id}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def regression_set(self, node: Node, *, limit: int | None = None) -> list[Episode]:
        """Successful episodes covering this node *and its whole subtree*.

        Validating a compression only against the episode that triggered it is
        how you end up with a beautifully compressed, useless tree.
        """
        ids = {node.id} | {d.id for d in self.descendants(node)}
        task_ids = set(node.covers_tasks)
        for desc in self.descendants(node):
            task_ids.update(desc.covers_tasks)
        out = [
            e
            for e in self.episodes(node.family)
            if e.outcome == "success" and (e.id in task_ids or set(e.served) & ids)
        ]
        out.sort(key=lambda e: e.created, reverse=True)
        return out[:limit] if limit else out

    # --------------------------------------------------------------- session
    def session_path(self, session_id: str) -> Path:
        safe = "".join(c for c in (session_id or "unknown") if c.isalnum() or c in "-_")
        return self.sessions_dir / f"{safe or 'unknown'}.json"

    def read_session(self, session_id: str) -> dict[str, Any]:
        path = self.session_path(session_id)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def write_session(self, session_id: str, data: dict[str, Any]) -> None:
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.session_path(session_id).write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- events
    def log(self, kind: str, **fields: Any) -> None:
        record = {"ts": utcnow(), "kind": kind, **fields}
        if self.config.get("privacy.redact", True):
            record = redact_obj(record)
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def read_events(self, kind: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        path = self.root / "events.jsonl"
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            if kind is None or row.get("kind") == kind:
                rows.append(row)
        return rows[-limit:]

    # ----------------------------------------------------------------- locks
    def lock(self, name: str, *, stale_s: int = 1800) -> "FileLock":
        return FileLock(self.root / f"{name}.lock", stale_s=stale_s)


class FileLock:
    """Best-effort advisory lock so a background compactor cannot race a hook."""

    def __init__(self, path: Path, *, stale_s: int = 1800) -> None:
        self.path = path
        self.stale_s = stale_s
        self.acquired = False

    def __enter__(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            age = time.time() - self.path.stat().st_mtime
            if age > self.stale_s:
                self.path.unlink(missing_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            self.acquired = True
        except FileExistsError:
            self.acquired = False
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)


def iter_chunks(items: Iterable[Any], size: int) -> Iterator[list[Any]]:
    chunk: list[Any] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
