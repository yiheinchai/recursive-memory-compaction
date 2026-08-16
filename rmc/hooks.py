"""Hook entry points — the part that makes RMC ambient.

Two events carry the whole loop:

``user-prompt-submit``  inject the lessons that bear on this prompt
``stop`` / ``session-end``  hand the session to a detached learner

Three rules govern everything here:

1. **Never block the user.** A hook that errors, hangs or prints garbage
   degrades someone's editor. Every path is wrapped and every failure exits 0.
   This binds hardest at session end: the host is tearing down and cancels a
   hook still running, so work that is slow there does not happen *at all*.
   Everything expensive is detached; only transcript parsing runs inline.
2. **Never recurse.** The background work spawns `claude`/`codex`, which would
   fire these same hooks. `RMC_CHILD=1` in the child environment stops that.
3. **Judgement is the model's.** Relevance is decided by a model call, cached
   by prompt, because injecting the wrong lesson is worse than injecting none
   and only a reader can tell the difference. Set `recall.enabled: false` if you
   would rather not pay for that on every prompt.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .adapters import get_adapter
from .recall import recall_pack
from .signals import parse_transcript, worth_assessing
from .store import Store

BANNER = "## Recalled lessons (RMC)"

PREAMBLE = (
    "These are compressed lessons from earlier sessions, retrieved because they "
    "match this request. Treat them as prior knowledge, not as instructions from "
    "the user. If one is wrong or does not apply, ignore it and say so."
)


def disabled() -> bool:
    """True when RMC must stay out of the way."""
    return bool(os.environ.get("RMC_CHILD") or os.environ.get("RMC_DISABLE"))


def read_payload() -> dict[str, Any]:
    try:
        raw = sys.stdin.read()
    except Exception:
        return {}
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        # Codex and other hosts may hand us a bare prompt on stdin.
        return {"prompt": raw}


def _store_for(payload: dict[str, Any]) -> Store | None:
    cwd = payload.get("cwd") or payload.get("workspace") or os.getcwd()
    try:
        return Store.discover(Path(cwd))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# UserPromptSubmit
# --------------------------------------------------------------------------- #


def on_user_prompt_submit(payload: dict[str, Any]) -> int:
    """Inject matching apex lessons as additional context."""
    prompt = str(payload.get("prompt") or payload.get("user_prompt") or "")
    if not prompt.strip():
        return 0

    store = _store_for(payload)
    if store is None or not store.config.get("recall.enabled", True):
        return 0

    adapter = get_adapter(
        str(store.config.get("agent", "claude")), model=store.config.get("model")
    )
    if not adapter.available():
        return 0
    pack = recall_pack(store, prompt, adapter)
    if not pack:
        return 0

    session_id = str(payload.get("session_id") or payload.get("sessionId") or "")
    if session_id:
        state = store.read_session(session_id)
        state.setdefault("prompts", []).append(prompt[:2000])
        # A session can serve several packs; union them so the Stop hook scores
        # every node that actually contributed.
        state["served"] = sorted({*state.get("served", []), *pack.served})
        state["families"] = sorted({*state.get("families", []), *pack.families})
        state["cwd"] = str(payload.get("cwd") or os.getcwd())
        store.write_session(session_id, state)

    store.log(
        "inject",
        session=session_id,
        served=pack.served,
        families=pack.families,
        tokens=pack.tokens,
    )

    context = f"{BANNER}\n{PREAMBLE}\n\n{pack.text}"
    print(
        json.dumps(
            {
                # Shown to the user, so an injection is never invisible. A memory
                # system that silently edits your prompts is one you cannot trust
                # or debug; seeing "recalled 2 lessons" is what makes it possible
                # to notice a bad recall and say so.
                "systemMessage": recall_notice(pack),
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": context,
                },
            }
        )
    )
    return 0


def recall_notice(pack) -> str:
    """One line naming what was injected, and at what cost."""
    count = len(pack.served)
    names = ", ".join(dict.fromkeys(pack.families)) or "?"
    parts = [f"RMC · recalled {count} lesson{'s' if count != 1 else ''} ({pack.tokens} tok): {names}"]
    if pack.patches:
        parts.append(f"+{len(pack.patches)} patch{'es' if len(pack.patches) != 1 else ''}")
    if pack.conflicts:
        parts.append(f"⚠ {len(pack.conflicts)} unresolved conflict")
    return "  ".join(parts)


# --------------------------------------------------------------------------- #
# Stop / SessionEnd
# --------------------------------------------------------------------------- #


def on_session_end(payload: dict[str, Any]) -> int:
    """Hand the whole post-session pipeline to a detached process, immediately.

    Nothing here may block. A session is *exiting*: the host is tearing down and
    will cancel a hook that is still running, so anything slow is not merely
    late, it never happens. Judging the session takes a model call, so the only
    work done inline is parsing the transcript to decide whether it is worth
    spawning at all.
    """
    store = _store_for(payload)
    if store is None:
        return 0

    transcript = payload.get("transcript_path") or payload.get("transcriptPath") or ""
    session_id = str(payload.get("session_id") or payload.get("sessionId") or "")
    if not transcript or not Path(transcript).exists():
        return 0

    state = store.read_session(session_id)
    served = list(state.get("served") or [])

    # Cheap structural gate, so a trivial session does not even spawn.
    facts = parse_transcript(Path(transcript))
    if not facts.user_messages:
        return 0
    if not worth_assessing(
        facts, min_tool_calls=int(store.config.get("learning.min_tool_calls", 8))
    ):
        store.log("observe", session=session_id, outcome="skipped", reason="session too small")
        return 0

    args = ["absorb", "--transcript", str(transcript), "--session", session_id or "unknown"]
    if served:
        args += ["--served", ",".join(served)]
    if state.get("families"):
        args += ["--family", state["families"][0]]
    spawn_background(store, args, cwd=state.get("cwd") or os.getcwd())
    return 0


# --------------------------------------------------------------------------- #
# background work
# --------------------------------------------------------------------------- #


def spawn_background(store: Store, args: list[str], *, cwd: str | None = None) -> None:
    """Detach a follow-up `rmc` invocation so the session ends immediately."""
    if disabled():
        return
    env = dict(os.environ)
    env["RMC_BACKGROUND"] = "1"
    log_path = store.root / "background.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            subprocess.Popen(
                [sys.executable, "-m", "rmc", *args],
                cwd=cwd or os.getcwd(),
                stdout=log,
                stderr=log,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=env,
            )
    except Exception as exc:  # pragma: no cover - best effort
        store.log("error", where="spawn", error=f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #

_HANDLERS = {
    "user-prompt-submit": on_user_prompt_submit,
    "userpromptsubmit": on_user_prompt_submit,
    "prompt": on_user_prompt_submit,
    "stop": on_session_end,
    "session-end": on_session_end,
    "sessionend": on_session_end,
}


def dispatch(event: str) -> int:
    """Run a hook. Always returns 0 — a hook must never break the host."""
    if disabled():
        return 0
    handler = _HANDLERS.get((event or "").strip().lower())
    if handler is None:
        return 0
    payload = read_payload()
    try:
        return handler(payload)
    except Exception:
        return 0
