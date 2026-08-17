"""Selection by search, run in a fork of the live session.

The judge-walk this replaces asks one question with every apex rendered into it.
That is affordable now and arithmetically impossible later: EXPERIMENTS §8 puts
a 5,000-lesson store at ~225k tokens of candidate list *per prompt*, at which
point the thing that decides what to load no longer fits beside the work. Prompt
caching changes the constant and not the shape.

So selection stops being a question about a rendered list and becomes a search
over a directory. Three consequences, and the third is the one that made the
fork the right process to run it in:

**Cost stops tracking the store.** The index is grepped, never sent. What costs
tokens per prompt is the selection-rule layer, capped by ``routing.max_tokens``.

**The candidate set stops being the apex layer.** EXPERIMENTS §8.2 found half
the store unreachable because a lesson below an apex could only be found by
descending into it. A grep does not care what level a lesson sits at.

**The selector gets the reasoning.** A fork inherits the whole conversation —
the task, the tool calls, what has already been tried. That is a far better
basis for "what does this work need" than the user's opening sentence, and it is
the input the reflection loop is defined over: task + tool calls + reasoning →
selected memories. Running selection anywhere else throws that away.

What this costs is latency, in a hook that blocks the user's prompt, and it is
not a small cost — EXPERIMENTS §4.4 records process startup alone at ~5s. The
answer is not to hope: it is ``recall.selector_max_tool_calls`` bounding the
search, ``recall.selector_timeout_s`` bounding the wait, and selection rules
collapsing the search over time. If that last part does not happen, this is
slower than what it replaced and the measurement should say so.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from . import index as index_mod
from . import routing
from .adapters import Adapter, Session
from .judge import Pick, WalkResult
from .node import Node
from .prompts import SELECT, SELECT_SCHEMA
from .store import Store
from .util import truncate

# Hosts identify a resumable session by UUID. Checking the shape before
# spawning turns a wasted round trip into a string comparison.
_SESSION_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

# Read-only by construction. In headless mode a tool outside the allowlist is
# refused rather than prompted for, so this is the boundary, not a suggestion —
# a selection pass that edited the repo would be a bug of the worst kind, since
# it runs unattended on every prompt.
SEARCH_TOOLS = [
    "Read",
    "Grep",
    "Glob",
    "Bash(grep:*)",
    "Bash(rg:*)",
    "Bash(ls:*)",
    "Bash(cat:*)",
    "Bash(head:*)",
    "Bash(sed:*)",
    "Bash(find:*)",
    "Bash(wc:*)",
]


def available(store: Store, adapter: Adapter, session_id: str) -> tuple[bool, str]:
    """Whether an agentic selection can run at all, and why not if it cannot.

    The reason is returned rather than swallowed because the caller has to
    choose a fallback, and "no session to fork" (normal, first turn) and "the
    backend cannot fork" (a configuration problem) deserve different treatment.
    """
    if str(store.config.get("recall.selector", "agentic")).lower() != "agentic":
        return False, "selector is set to judge"
    if not session_id:
        # First turn: the transcript does not exist yet, so there is nothing to
        # fork and no reasoning to inherit. The judge-walk is the right answer
        # here, not a degraded one.
        return False, "no session to fork yet"
    if not _SESSION_RE.match(session_id):
        # `--resume` requires a real session id and rejects anything else, but
        # only after a process has started — about a second of the user's wait
        # spent discovering something the shape of the string already said.
        return False, f"session id {session_id!r} is not a resumable session"
    if getattr(adapter, "name", "") != "claude":
        return False, f"backend {getattr(adapter, 'name', '?')} cannot fork a session"
    if not adapter.available():
        return False, "backend not on PATH"
    return True, ""


def build_prompt(store: Store, prompt: str, rules: list[routing.Rule]) -> str:
    max_calls = int(store.config.get("recall.selector_max_tool_calls", 6))
    return SELECT.format(
        prompt=truncate(prompt, 4000),
        store=str(store.root),
        rules=routing.render(rules) or "(none yet — this is the first selection)",
        max_calls=max_calls,
    )


def parse_picks(data: dict[str, Any] | None, resolve: Any) -> tuple[list[Node], dict[str, Pick]]:
    """Turn the fork's JSON into nodes, dropping anything that does not exist.

    A selector that searched a stale index can name a lesson that has since been
    archived or compressed away. Silently dropping the unknown id is right: the
    alternative is a crash on every prompt until someone rebuilds the index.
    """
    nodes: list[Node] = []
    picks: dict[str, Pick] = {}
    for raw in (data or {}).get("picks") or []:
        if not isinstance(raw, dict):
            continue
        node_id = str(raw.get("id") or "").strip()
        if not node_id:
            continue
        node = resolve(node_id)
        if node is None or node.status == "archived":
            continue
        if any(n.id == node.id for n in nodes):
            continue
        nodes.append(node)
        picks[node.id] = Pick(
            id=node.id,
            verdict="relevant",
            why=str(raw.get("why") or "").strip(),
        )
    return nodes, picks


def select(
    store: Store,
    adapter: Adapter,
    prompt: str,
    *,
    session_id: str,
    cwd: Path | None = None,
    limit: int | None = None,
) -> WalkResult:
    """Run one selection pass. Never raises; a failure comes back as ``failed``."""
    limit = limit if limit is not None else int(store.config.get("recall.max_families", 3))

    if index_mod.is_stale(store):
        # Cheap, and the alternative is the selector confidently searching a
        # picture of the store as it was. A miss caused by a stale index looks
        # exactly like a lesson that does not exist.
        index_mod.rebuild(store)

    rules = routing.fit(
        routing.load(store) if store.config.get("routing.enabled", True) else [],
        int(store.config.get("routing.max_tokens", 800)),
    )

    # An empty session id means a *cold* selection: a fresh process that
    # searches the store without the conversation in front of it. That is not
    # how selection runs in production and it is not meant to be — it is how
    # `rmc eval-recall` scores the search on its own, separately from whatever
    # the inherited reasoning contributes. Measuring the two together would
    # leave no way to tell which half is working.
    session = Session(id=session_id, resume=True) if session_id else None

    started = time.monotonic()
    run = adapter.run(
        build_prompt(store, prompt, rules),
        schema=SELECT_SCHEMA,
        cwd=cwd,
        timeout=int(store.config.get("recall.selector_timeout_s", 45)),
        session=session,
        allowed_tools=SEARCH_TOOLS,
    )
    elapsed = time.monotonic() - started

    result = WalkResult()
    result.rules_shown = [r.id for r in rules]

    if not run.ok or run.data is None:
        # An outage and "nothing applied" must never look alike to the caller —
        # `recall_notice` renders them differently and the user's trust in the
        # whole system rests on being able to tell.
        result.failed = True
        result.error = (run.error or "selector returned no parseable JSON")[:500]
        store.log(
            "select",
            ok=False,
            error=result.error[:200],
            duration_s=round(elapsed, 2),
            rules=len(rules),
        )
        return result

    nodes, picks = parse_picks(run.data, store.get)
    result.selected = nodes[:limit]
    result.picks = picks
    result.calls = 1
    result.rules_used = [
        str(r) for r in (run.data.get("rules_used") or []) if str(r).strip()
    ]
    result.searched = [str(s) for s in (run.data.get("searched") or []) if str(s).strip()]

    store.log(
        "select",
        ok=True,
        picked=[n.id for n in result.selected],
        searches=len(result.searched),
        rules_shown=len(rules),
        rules_used=result.rules_used,
        cached_in=run.cached_in,
        duration_s=round(elapsed, 2),
    )
    return result
