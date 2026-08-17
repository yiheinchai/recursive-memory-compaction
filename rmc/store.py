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
from .util import new_id, utcnow

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
    # The subset that actually bore on the work, judged in reflection. Serving a
    # lesson is a retrieval decision; *using* one is an outcome, and only the
    # outcome should earn credit or drive abstraction.
    used: list[str] = None
    accepted_summary: str = ""  # what the agent ended up doing, once accepted
    check: dict[str, Any] = None  # optional mechanical oracle harvested from session
    created: str = ""
    session_id: str = ""
    cwd: str = ""

    def __post_init__(self) -> None:
        self.served = self.served or []
        self.used = self.used or []
        self.check = self.check or {}
        self.created = self.created or utcnow()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "prompt": self.prompt,
            "outcome": self.outcome,
            "confidence": self.confidence,
            "served": self.served,
            "used": self.used,
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
            used=list(raw.get("used") or []),
            accepted_summary=raw.get("accepted_summary") or "",
            check=dict(raw.get("check") or {}),
            created=raw.get("created") or utcnow(),
            session_id=raw.get("session_id") or "",
            cwd=raw.get("cwd") or "",
        )


class Store:
    """A lesson store, optionally layered over a global one.

    Knowledge comes at two scopes and both matter. "This repo's integration
    tests need PAYMENTS_PG_PORT" belongs to the project; "always prefer the
    model's judgement over a similarity score" follows you everywhere. A single
    scope forces one of them to be in the wrong place.

    So a project store reads through to ``~/.rmc`` as its parent: lessons from
    both are recalled, and writes always land locally unless the store *is* the
    global one. A local node with the same id shadows the global one.
    """

    def __init__(self, root: Path, parent: "Store | None" = None) -> None:
        self.root = Path(root)
        self.parent = parent
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
        config_path = root / "config.yaml"
        if not config_path.exists():
            # Overrides only — never a snapshot of the defaults. Writing the full
            # default set at init freezes them: the file wins the merge, so a
            # store created today keeps today's numbers forever and silently
            # ignores every later improvement. `rmc config` adds keys here when
            # you actually mean to override one.
            config_path.write_text(
                "# RMC overrides. Anything absent follows the current defaults;\n"
                "# see `rmc config` for the full effective settings.\n"
                "version: 1\n",
                encoding="utf-8",
            )
        gitignore = root / ".gitignore"
        if not gitignore.exists():
            # Sessions and telemetry are machine-local noise; nodes and episodes
            # are the artefact worth committing and sharing across a team.
            # `index.md` is derived from the nodes and rebuilt whenever it falls
            # behind them, so committing it buys nothing and conflicts on every
            # branch that learns anything.
            gitignore.write_text(
                "sessions/\nevents.jsonl\nbackground.log\n*-cache.json\n*.lock\nindex.md\n",
                encoding="utf-8",
            )
        return store

    @classmethod
    def discover(cls, start: Path | None = None, *, layered: bool = True) -> "Store | None":
        root = find_store_root(start)
        if root is None:
            return None
        if not layered:
            return cls(root)
        global_root = Path.home() / STORE_DIRNAME
        if global_root.is_dir() and global_root.resolve() != root.resolve():
            return cls(root, parent=cls(global_root))
        return cls(root)

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
        # Start from the global layer so a local node of the same id shadows it.
        nodes: dict[str, Node] = dict(self.parent._load_nodes()) if self.parent else {}
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
        if self.parent:
            self.parent.invalidate()

    def global_layer(self) -> "Store":
        """The store a cross-project lesson belongs in."""
        return self.parent or self

    def owns(self, node: Node) -> bool:
        """False when this node lives in the global layer, not here."""
        try:
            return node.path is not None and self.root in node.path.parents
        except Exception:
            return True

    def nodes(self) -> list[Node]:
        return list(self._load_nodes().values())

    def get(self, node_id: str) -> Node | None:
        return self._load_nodes().get(node_id)

    def families(self) -> list[str]:
        return sorted({n.family for n in self.nodes()})

    def family_nodes(self, family: str, *, active_only: bool = True) -> list[Node]:
        out = [n for n in self.nodes() if n.family == family]
        if active_only:
            out = [n for n in out if n.status in ("active", "demoted", "disputed")]
        return sorted(out, key=lambda n: (-n.level, n.id))

    def apex(self, family: str) -> Node | None:
        """Highest-level servable node of a family.

        `disputed` nodes are still served. Withholding a contradicted lesson
        would lose the knowledge *and* remove the occasion to ask the user about
        it — recall is exactly when the question is worth raising. It is served
        with its conflict note attached (see `recall.recall_pack`).

        `demoted` nodes — ones that have repeatedly regressed — are skipped as
        apex where a healthy alternative exists, since those failed on merit.
        """
        healthy = [
            n
            for n in self.family_nodes(family)
            if n.status in ("active", "disputed") and n.is_apex
        ]
        if not healthy:
            healthy = [
                n for n in self.family_nodes(family) if n.status in ("active", "disputed")
            ]
        if not healthy:
            return None
        return max(healthy, key=lambda n: (n.level, n.stats.posterior))

    def apexes(self, family: str | None = None) -> list[Node]:
        """Every servable top-of-tree node, across all families.

        A family is not a single lesson. Consolidation deliberately creates
        siblings — two lessons about the same subject that cover distinct cases
        stand alongside each other until a merge-compression generalises them —
        and every one of them must be reachable. Taking only the best node per
        family silently orphans the rest: they are stored, counted, and never
        served again.
        """
        out = [
            n
            for n in self.nodes()
            if n.status in ("active", "disputed") and n.is_apex
            and (family is None or n.family == family)
        ]
        # Deepest (most compressed) first, then by how well each has held up.
        out.sort(key=lambda n: (-n.level, -n.stats.posterior, n.id))
        return out

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
        """Every node above this one, nearest first.

        Breadth-first, because a node may be abstracted several ways at once and
        each line upward is equally real. The seen-set is not paranoia: a merge
        that swept up one of its own ancestors would otherwise loop forever.
        """
        out: list[Node] = []
        seen = {node.id}
        frontier = list(node.parents)
        while frontier:
            nxt: list[str] = []
            for ident in frontier:
                if ident in seen:
                    continue
                seen.add(ident)
                parent = self.get(ident)
                if parent is None:
                    continue
                out.append(parent)
                nxt.extend(parent.parents)
            frontier = nxt
        return out

    def base_node(self, family: str) -> Node | None:
        """The level-0 fallback: guaranteed-correct, never deleted."""
        zeros = [n for n in self.family_nodes(family, active_only=False) if n.level == 0]
        if not zeros:
            return None
        return max(zeros, key=lambda n: n.stats.posterior)

    def save_node(self, node: Node) -> Path:
        """Write a node. Edits to a global-layer node are written back to it,
        so updating a cross-project lesson from inside a repo does not silently
        fork a local copy that then drifts."""
        if self.parent is not None and node.path is not None and not self.owns(node):
            return self.parent.save_node(node)
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
        out: list[Episode] = list(self.parent.episodes() if self.parent else [])
        if not self.episodes_dir.is_dir():
            return [e for e in out if not family or e.family == family]
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

        Deliberately **not** filtered by family. An episode's family is a weak
        label — the first family that happened to be served — while what makes
        an episode a regression test for a node is that the node was *used* in
        it. Filtering by family meant a cross-family merge could never find its
        evidence, which broke the one case merging exists for: two lessons from
        different families that keep being needed together.
        """
        ids = {node.id} | {d.id for d in self.descendants(node)}
        task_ids = set(node.covers_tasks)
        for desc in self.descendants(node):
            task_ids.update(desc.covers_tasks)
        out = [
            e
            for e in self.episodes()
            if e.outcome == "success"
            and (e.id in task_ids or set(e.used or e.served) & ids)
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

    def recent_nudge(self, window_s: int = 1800) -> dict[str, Any] | None:
        """The most recent reflection nudge, if one fired inside the window.

        Used to attribute a capture: one that follows a nudge was *prompted*,
        one that does not was *spontaneous*. That distinction is the only way to
        answer whether the nudge is load-bearing scaffolding or a crutch the
        agent has outgrown — so it is recorded rather than assumed.
        """
        import time
        from datetime import datetime, timezone

        for event in reversed(self.read_events("nudge", limit=50)):
            try:
                when = datetime.strptime(event["ts"], "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc
                )
            except Exception:
                continue
            if time.time() - when.timestamp() <= window_s:
                return event
        return None

    def read_events(self, kind: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        path = self.root / "events.jsonl"
        inherited = self.parent.read_events(kind, limit) if self.parent else []
        if not path.exists():
            return inherited
        rows: list[dict[str, Any]] = list(inherited)
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            if kind is None or row.get("kind") == kind:
                rows.append(row)
        return rows[-limit:]

    # ----------------------------------------------------------------- locks
    def lock(self, name: str, *, stale_s: int = 1800, wait_s: float = 0.0) -> "FileLock":
        return FileLock(self.root / f"{name}.lock", stale_s=stale_s, wait_s=wait_s)


class FileLock:
    """Advisory lock. ``wait_s`` turns it from try-once into wait-then-try.

    Two flavours are needed. A compactor that loses the lock should simply skip —
    whatever the winner does, the loser would have duplicated. A *writer* must
    wait instead: giving up would silently drop a lesson.
    """

    def __init__(self, path: Path, *, stale_s: int = 1800, wait_s: float = 0.0) -> None:
        self.path = path
        self.stale_s = stale_s
        self.wait_s = wait_s
        self.acquired = False

    def _try(self) -> bool:
        if self.path.exists():
            age = time.time() - self.path.stat().st_mtime
            if age > self.stale_s:
                self.path.unlink(missing_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            return False

    def __enter__(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + self.wait_s
        while True:
            if self._try():
                self.acquired = True
                break
            if time.time() >= deadline:
                self.acquired = False
                break
            time.sleep(0.15)
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
