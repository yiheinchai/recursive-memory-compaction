"""The store index: what a selector greps instead of what a router is sent.

Routing used to work by rendering every apex into one question. That is
affordable at 29 lessons and impossible at 5,000: at ~55 tokens per apex, and
with apex count tracking node count roughly 1:1 (EXPERIMENTS §3.4), the
candidate list alone reaches ~225k tokens *per prompt*. No amount of prompt
caching fixes a list that no longer fits.

The way out is to stop sending the list at all. This module writes one line per
lesson to ``.rmc/index.md`` — id, family, level, title, tags, gist, path — and
the selector searches that file with grep rather than reading it into a prompt.

The distinction is the whole scaling argument, so it is worth stating flatly:

    the index is a file that is *searched*, never text that is *injected*.

At 5,000 lessons it is roughly 125k tokens on disk and exactly zero tokens per
prompt. What costs tokens per prompt is the selection-lesson layer, which is
bounded by ``routing.max_tokens`` and does not grow with the store.

**This is a first pass, not the only surface.** The index holds a title and a
one-line summary, so a lesson whose *body* names the exact command or error
string will not match on its summary — and after a verbatim skills migration
most of what a store knows lives in bodies thousands of lines long. The selector
is told to grep ``nodes/`` as well, and to use whatever else the shell offers.
Nothing here is a boundary on where it may look; it is the cheapest place to
look first.

One line per node, one node per line, no wrapping: the format exists to be
matched by `grep`, and a claim that spans two lines is a claim grep will cut in
half. Everything a search might key on is on the same line as the id needed to
open the file.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .node import Node
    from .store import Store

INDEX_NAME = "index.md"

HEADER = """# RMC lesson index

One line per lesson. Search this file, do not read it whole.

    <id> · <family> · L<level> · <title> · [tags] · <gist> → <path to the body>

Open the path at the end of a line when its summary is not enough to decide.
The path is given because lessons come from two places: this project's store and
the cross-project one in your home directory. Both are indexed here, and only
the path says which is which.

    grep -i 'postgres' .rmc/index.md          # by any word in title/tags/gist
    grep '^n_7f2a91' .rmc/index.md            # by id
    grep ' · retry · ' .rmc/index.md          # by family

"""


def line_for(node: "Node", *, root: Path | None = None) -> str:
    """One node, as one greppable line.

    Tags are bracketed so a tag search cannot be satisfied by the same word
    appearing in prose, which is the difference between "this lesson is tagged
    postgres" and "this lesson mentions postgres in passing".

    The path is the node's real one rather than one reconstructed from family
    and id. A store layered over the global one indexes lessons that do not live
    under it at all, and a guessed path sends the selector to a file that is not
    there — which reads exactly like the lesson having been deleted.
    """
    tags = ",".join(node.tags)
    gist = " ".join(node.summary().split())
    title = " ".join((node.title or node.family).split())
    location = ""
    if node.path is not None:
        try:
            location = str(node.path.relative_to(root)) if root else str(node.path)
        except ValueError:
            location = str(node.path)
    return (
        f"{node.id} · {node.family} · L{node.level} · {title} · [{tags}] · {gist}"
        + (f" → {location}" if location else "")
    )


def render(nodes: list["Node"], *, root: Path | None = None) -> str:
    """The whole index, families grouped so a family search reads contiguously."""
    live = [n for n in nodes if n.status != "archived"]
    live.sort(key=lambda n: (n.family, n.level, n.id))
    return HEADER + "\n".join(line_for(n, root=root) for n in live) + "\n"


_ENTRY_RE = re.compile(r"^\S+ · \S+ · L\d+ · ", re.MULTILINE)


def count_lines(text: str) -> int:
    """Entries in an index, ignoring the header.

    A plain substring test counts the header's own format example, which is
    indented and reads exactly like an entry — so the count came out one too
    high and every index looked permanently stale.
    """
    return len(_ENTRY_RE.findall(text or ""))


def path_for(store: "Store") -> Path:
    return store.root / INDEX_NAME


def rebuild(store: "Store") -> Path | None:
    """Write the index for every lesson this store can reach.

    Every lesson, not every local one: a project store reads through to the
    global layer, and a selector that cannot see those lessons cannot retrieve
    them. Paths on each line keep the two distinguishable.

    Best effort by design. A store whose index is stale or missing still works
    — the selector falls back to walking `nodes/` directly, and recall falls
    back to the judge — so a failure here must never be allowed to break the
    write that triggered it.
    """
    try:
        target = path_for(store)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render(store.nodes(), root=store.root), encoding="utf-8")
        return target
    except Exception:
        return None


def _node_dirs(store: "Store") -> list[Path]:
    """Every directory holding lessons this index covers, including the parent's."""
    dirs = [store.nodes_dir]
    parent = getattr(store, "parent", None)
    while parent is not None:
        dirs.append(parent.nodes_dir)
        parent = getattr(parent, "parent", None)
    return dirs


def is_stale(store: "Store") -> bool:
    """Whether the index no longer matches the nodes on disk.

    Checked before every selection rather than rebuilt after every write: a
    rebuild is O(store) and compaction saves several nodes in a row, so writing
    the index on each save is quadratic for no benefit. Detecting staleness here
    costs two stats and gives the same guarantee — the selector never searches a
    picture of a store that has moved on.

    Two conditions, because one is not enough. An mtime alone misses a deletion:
    removing a node makes nothing newer, so the index would keep advertising a
    lesson that is gone. The count catches that.

    Both conditions cover the global layer as well as this store, since the
    index does — a lesson learned in another repo and edited there is exactly
    the kind that would otherwise go stale unnoticed.
    """
    target = path_for(store)
    if not target.exists():
        return True
    try:
        stamp = target.stat().st_mtime
        paths = [p for d in _node_dirs(store) if d.is_dir() for p in d.rglob("*.md")]
        if max((p.stat().st_mtime for p in paths), default=0.0) > stamp:
            return True
        return count_lines(target.read_text(encoding="utf-8")) != len(
            [n for n in store.nodes() if n.status != "archived"]
        )
    except Exception:
        return True
