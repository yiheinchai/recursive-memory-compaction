"""The routing view of a lesson: its title and its one-line gist.

The relevance walk reads title and gist and never the body — sending 700
characters per candidate to decide which candidates to send is the scaling bug
the design exists to avoid. That makes these two fields load-bearing for
retrieval, not decoration.

They were only ever written on two paths: a lesson minted by the reflector, and
a nightly backfill during `dream`. Everything else left them empty or stale:

* `rmc add` wrote neither, so a lesson taught directly routed on a truncated
  head of its own body until a dream happened to run — and dream is gated on
  elapsed time *and* new episodes, so that can be days, or never.
* A fold merged two bodies and kept the survivor's title, so a lesson could
  grow past its own name. One here ended up leading with "the customer's
  experience defines the value of a product" while still advertising itself as
  "walk the whole flow as a stranger before shipping it" — correct content,
  wrong label, and the router only sees the label.

So the summary is written whenever the body it describes changes.
"""

from __future__ import annotations

from typing import Any

from .adapters import Adapter
from .node import Node
from .store import Store
from .util import truncate

SUMMARY = """RMC:summary

Write the routing view of this lesson: the two lines a future agent sees when
deciding whether to open it at all.

`title` — a short noun phrase naming the claim, not the topic. It is what a
reader scans in a list of twenty. Prefer the specific over the general: name the
tool, command, service or failure it concerns.

`gist` — one line, at most 25 words, naming what the lesson is about and *when
it applies*. Do not summarise the advice; identify the situation that should
make someone reach for it.

<<<LESSON
{body}
LESSON>>>
"""

SUMMARY_SCHEMA = {
    "type": "object",
    "required": ["gist"],
    "properties": {
        "title": {"type": "string"},
        "gist": {"type": "string"},
    },
}


def refresh(
    store: Store,
    adapter: Adapter,
    node: Node,
    *,
    force: bool = False,
    save: bool = True,
) -> bool:
    """Fill title and gist from the body. Returns whether anything changed.

    Never destructive: a title the user wrote by hand is kept unless `force`.
    A failed or unavailable model leaves the node exactly as it was — a lesson
    with a stale gist still works, and an invented one poisons retrieval.
    """
    if not force and node.gist.strip() and node.title.strip():
        return False

    run = adapter.run(
        SUMMARY.format(body=truncate(node.body, 3000)),
        schema=SUMMARY_SCHEMA,
        timeout=int(store.config.get("limits.agent_timeout_s", 180)),
    )
    if not run.ok or not run.data:
        return False

    changed = False
    gist = str(run.data.get("gist") or "").strip()
    if gist and (force or not node.gist.strip()):
        node.gist = gist
        changed = True

    title = str(run.data.get("title") or "").strip()
    if title and (force or not node.title.strip()):
        node.title = title
        changed = True

    if changed and save:
        store.save_node(node)
        store.invalidate()
    return changed


def backfill(store: Store, adapter: Adapter, *, limit: int = 20) -> int:
    """Give older lessons the routing view they were written without."""
    filled = 0
    for node in store.nodes():
        if filled >= limit:
            break
        if node.gist.strip() and node.title.strip():
            continue
        if refresh(store, adapter, node):
            filled += 1
    return filled
