"""Hook entry points — the part that makes RMC ambient.

Two events carry the whole loop:

``user-prompt-submit``  inject the lessons that bear on this prompt
``stop``                nudge the agent to reflect, but only after a surprise
``session-end``         hand the session to a detached learner

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
    session_id = str(payload.get("session_id") or payload.get("sessionId") or "")
    state = store.read_session(session_id) if session_id else {}
    turn = int(state.get("turn") or 0) + 1

    pack = recall_pack(
        store,
        prompt,
        adapter,
        already_served=dict(state.get("served_at") or {}),
        turn=turn,
    )
    if not pack:
        if session_id:
            state["turn"] = turn
            store.write_session(session_id, state)
        return 0

    if session_id:
        state.setdefault("prompts", []).append(prompt[:2000])
        # A session can serve several packs; union them so the Stop hook scores
        # every node that actually contributed.
        state["served"] = sorted({*state.get("served", []), *pack.served})
        state["families"] = sorted({*state.get("families", []), *pack.families})
        # When each lesson was last put in front of the model, so a repeat can
        # be skipped while it is still fresh and merely refreshed once it is not.
        served_at = dict(state.get("served_at") or {})
        for node_id in [*pack.served, *pack.skipped]:
            if node_id not in pack.skipped:
                served_at[node_id] = turn
        state["served_at"] = served_at
        state["turn"] = turn
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


def on_pre_compact(payload: dict[str, Any]) -> int:
    """Context is about to be rewritten, so nothing can be assumed present.

    Compaction is lossy: a lesson injected earlier may survive only as a phrase
    in a summary, or not at all. Every record of what the model has already been
    shown is therefore void, and lessons must be servable in full again.
    """
    store = _store_for(payload)
    session_id = str(payload.get("session_id") or payload.get("sessionId") or "")
    if store is None or not session_id:
        return 0
    state = store.read_session(session_id)
    if state.get("served_at"):
        state["served_at"] = {}
        state["served"] = []
        store.write_session(session_id, state)
        store.log("compact-reset", session=session_id)
    return 0


NOTICE_WIDTH = 96


def recall_notice(pack) -> str:
    """One short line. It appears on every prompt, so it must not become noise.

    It names what was recalled, not just how much. A count tells you RMC fired;
    only the titles let you notice it fired *wrongly* — which is the failure
    this line exists to make visible, and the one a bare number hides.
    """
    count = len(pack.served)
    note = f"RMC · {count} lesson{'s' if count != 1 else ''} · {pack.tokens} tok"

    titles = [t.strip() for t in getattr(pack, "titles", []) if t and t.strip()]
    if titles:
        room = NOTICE_WIDTH - len(note)
        shown: list[str] = []
        for i, title in enumerate(titles):
            remaining = len(titles) - i
            # Reserve space for the "+N more" that will follow if we stop here.
            tail = len(f", +{remaining} more") if remaining > 1 else 0
            if len(", ".join(shown + [title])) + tail > room:
                break
            shown.append(title)
        if shown:
            note += " — " + ", ".join(shown)
            left = len(titles) - len(shown)
            if left:
                note += f", +{left} more"
        else:
            note += f" — {titles[0][: max(12, room - 3)]}…"

    if pack.conflicts:
        note += "  ⚠ conflict"
    return note


# --------------------------------------------------------------------------- #
# Stop — the surprise trigger
# --------------------------------------------------------------------------- #

NUDGE = """Automated learning check from the RMC harness — the user did not ask for
this and it is NOT a request to capture anything.

Look back over the turns you just did and find:

  **Every point where the user had to steer you.**

A correction, a rejection, a "no, like this", a restated requirement, a
complaint about quality, a preference you did not know they held. Each one is a
round of the user's time that a better-informed agent would not have cost them.
That — not being wrong in the abstract — is the failure this check exists to
find.

For each, ask what you would have needed to know at the start to make that
correction unnecessary. It is usually one of these, and the first three are the
ones this check has historically missed:

  1. **A standard or preference this user holds** — what they consider good,
     how they want work presented, an aesthetic, a tone, a tool they expect.
  2. **A method they expect applied** — a check to run, a bar to clear, an
     order to work in, a perspective to take.
  3. **A quality bar** — the level of fidelity or polish at which they stop
     pushing back.
  4. **A false belief about a tool, language, platform or library.** This counts
     even though it surfaced as a bug. The test is whether a belief was wrong,
     not whether it arrived with an error message.
  5. **A false belief about this project.**
  6. **A modelling error** — you represented something in a way that is not true
     of it.
{evidence}
Then filter on usefulness rather than worthiness:

  (a) would knowing this at the start have removed at least one round of
      steering?
  (b) does it apply to some task other than this exact file?
  (c) is it still true tomorrow?

If the agent ran `rmc add` itself during these turns, or the user asked why
something was not learned, treat that as a capture an earlier pass should have
made and did not. Say what about the criteria let it through.

Calibrate against the steering, not against a general prior. A session in which
the user never had to redirect you almost certainly teaches nothing — say so and
finish. A session in which they redirected you repeatedly almost certainly does,
and returning nothing from one is this check failing. Do not invent a lesson to
fill a gap; equally, do not talk yourself out of one because it feels small or
obvious in hindsight. The user having to say it is the evidence that it was
neither."""

FAILURE_EVIDENCE = """
For reference, {count} tool call{plural} failed since the last check. Failures are
the loudest material here and rarely the most valuable — an unmet expectation of
the user's, or a belief that was quietly wrong, usually leaves no error at all:

{lines}
"""


def on_turn_end(payload: dict[str, Any]) -> int:
    """Give the agent an occasion to reflect. It decides whether there is a lesson.

    The occasion is deliberately **not** "something failed". An earlier version
    triggered on failed tool calls, which sounds structural and is in fact a
    mechanical proxy for a semantic question — and it biases hard toward the
    cheapest kind of mistake. The expensive errors are conceptual: believing a
    system works one way when it does not. Those produce no error message, no
    non-zero exit, nothing to grep for. During RMC's own development the single
    worst mistake, building retrieval on lexical similarity, ran green the whole
    way; a failure-triggered check would have sat silent through it.

    So the occasion is simply "this turn did enough to be worth a thought",
    which is a question about size, and every question about *worth* is handed
    to the agent — which has the whole conversation in context and is the only
    thing here that can tell a conceptual correction from a typo.

    The nudge costs no extra model call: blocking continues the agent's own turn
    with the reason as input. The cost is one turn per cooldown window, so the
    cooldown is what keeps it from nagging.
    """
    # Never loop on our own continuation.
    if payload.get("stop_hook_active"):
        return 0

    store = _store_for(payload)
    if store is None or not store.config.get("learning.nudge_enabled", True):
        return 0

    transcript = payload.get("transcript_path") or payload.get("transcriptPath") or ""
    if not transcript or not Path(transcript).exists():
        return 0
    session_id = str(payload.get("session_id") or payload.get("sessionId") or "")

    facts = parse_transcript(Path(transcript))
    state = store.read_session(session_id)

    failures = [e for e in facts.tool_events if e.ok is False]
    fresh_failures = failures[int(state.get("nudged_failures") or 0) :]
    new_tools = facts.tool_calls - int(state.get("nudged_tools") or 0)
    new_turns = len(facts.user_messages) - int(state.get("nudged_turns") or 0)

    # Any of these means the turn had substance. None of them claims to know
    # whether it *taught* anything — that is the agent's call, below.
    substantial = (
        new_tools >= int(store.config.get("learning.nudge_after_tool_calls", 12))
        or len(fresh_failures) >= int(store.config.get("learning.min_surprises", 2))
        or new_turns >= int(store.config.get("learning.nudge_after_turns", 3))
    )
    if not substantial or _too_soon(store, state):
        return 0

    state["nudged_failures"] = len(failures)
    state["nudged_tools"] = facts.tool_calls
    state["nudged_turns"] = len(facts.user_messages)
    state["nudged_at"] = _now()
    store.write_session(session_id, state)

    mode = str(store.config.get("learning.nudge_mode", "background")).lower()
    if mode == "fork":
        # Reflect in a copy of the agent's own session: same context, same
        # working memory, but off the main thread entirely.
        #
        # The reason this is affordable is prompt caching. A fork re-sends the
        # conversation as its prefix, but cache reads bill at 0.1x, and the
        # cache is keyed on prefix content rather than session identity — so
        # the fork hits the cache the live session just wrote. Claude Code uses
        # a 1-hour TTL, which comfortably covers the reflection cooldown.
        #
        # It is not the default because 10% of a very large context is still
        # more than a transcript digest, and the digest has proven able to spot
        # conceptual corrections. Choose fork when fidelity matters more than
        # tokens.
        if _spawn_fork(
            store, session_id, state.get("cwd") or os.getcwd(), served=state.get("served") or []
        ):
            store.log("nudge", session=session_id, mode="fork", tools=new_tools)
            return 0
        # Fall through to background rather than silently skipping reflection.
        store.log("nudge", session=session_id, mode="fork-failed", tools=new_tools)
        mode = "background"

    if mode == "background":
        # Reflect *off* the main thread. Interrupting an agent in the middle of
        # a large task is its own cost: it spends a turn, pollutes the working
        # context with meta-cognition, and breaks concentration precisely when
        # concentration is worth most.
        #
        # The transcript is the context, serialised — so a detached process
        # reading it can do the same reflection with no claim on the session at
        # all. The agent is never told this happened.
        args = ["absorb", "--transcript", str(transcript), "--session", session_id or "unknown"]
        if state.get("served"):
            args += ["--served", ",".join(state["served"])]
        spawn_background(store, args, cwd=state.get("cwd") or os.getcwd())
        store.log("nudge", session=session_id, mode="background", tools=new_tools)
        return 0

    store.log(
        "nudge",
        session=session_id,
        mode="block",
        tools=new_tools,
        turns=new_turns,
        failures=len(fresh_failures),
    )

    evidence = ""
    if fresh_failures:
        evidence = FAILURE_EVIDENCE.format(
            count=len(fresh_failures),
            plural="" if len(fresh_failures) == 1 else "s",
            lines="\n".join(
                f"  · `{e.detail[:110]}` → {' '.join((e.output or '').split())[:120]}"
                for e in fresh_failures[-3:]
            ),
        )
    print(json.dumps({"decision": "block", "reason": NUDGE.format(evidence=evidence)}))
    return 0


FORK_PROMPT = """You are a reflection pass running in a fork of this session. The
user cannot see you and is not waiting — the main session carried on without you.
Do not continue the task, do not edit anything, and do not report progress.

You have the whole conversation above. Read it as your own history and find:

  **Every point where the user had to steer you.**

A correction, a rejection, a "no, like this", a restated requirement, a
complaint about quality, a preference you did not know they held. Each one is a
round of the user's time that a better-informed agent would not have cost them.
That — not being wrong in the abstract — is the failure this check exists to
find.

For each, ask what you would have needed to know at the start to make that
correction unnecessary. It is usually one of these, and the first three are the
ones this check has historically missed:

  1. **A standard or preference this user holds** — what they consider good,
     how they want work presented, an aesthetic, a tone, a tool they expect.
  2. **A method they expect applied** — a check to run, a bar to clear, an
     order to work in, a perspective to take.
  3. **A quality bar** — the level of fidelity or polish at which they stop
     pushing back.
  4. **A false belief about a tool, language, platform or library.** This counts
     even though it surfaced as a bug. The test is whether a belief was wrong,
     not whether it arrived with an error message.
  5. **A false belief about this project.**
  6. **A modelling error** — you represented something in a way that is not true
     of it.

Then filter on usefulness rather than worthiness:

  (a) would knowing this at the start have removed at least one round of
      steering?
  (b) does it apply to some task other than this exact file?
  (c) is it still true tomorrow?

If it clears the bar:

    rmc add --family <slug> "<what to do, AND the wrong belief or blind spot \
it replaces>"

Record the blind spot as well as the correction. A lesson that states only the
right answer lets the next agent arrive at the same wrong assumption and merely
recognise the fix afterwards.

**If what you found is a defect in RMC itself** — the reflector missing what it
plainly should have caught, a stage that did not run, a command that reported
success while doing nothing — then the lesson belongs upstream, not only in this
store. Run `rmc report --about "..." --expected "..."`, which writes a redacted
report to disk and sends nothing, then ask the user whether they want it filed
as an issue. Ask; never file it yourself. Five hundred people hitting the same
defect and each fixing it locally teaches the project nothing.

**If the conversation shows the agent running `rmc add` itself, or the user
asking why something was not learned, that is a capture you failed to make.**
Not a lesson already handled — a miss. The evidence was in front of an earlier
reflection pass and it walked past. Read what was added, work out what about
these criteria let it through, and capture *that* in the `reflection` family. A
capture system that cannot see its own misses cannot improve.

**If the user had to tell you something that was already in a lesson you were
served, do not add it again.** That is not a gap in what is known — it is a
lesson that is not landing. Say so on your reply line, naming the node id, and
capture nothing.

Calibrate against the steering, not against a general prior. A session in which
the user never had to redirect you almost certainly teaches nothing — say so and
finish. A session in which they redirected you repeatedly almost certainly does,
and returning nothing from one is this check failing. Do not invent a lesson to
fill a gap; equally, do not talk yourself out of one because it feels small or
obvious in hindsight. The user having to say it is the evidence that it was
neither.

{attribution}Reply with one line when you are done. Nothing else."""


ATTRIBUTION = """Second, these lessons were recalled into this session before the work
started:

{served}

Say which of them actually bore on what happened — a lesson counts only if the
work would plausibly have gone differently without it. Being on-topic is not
being used; being read and found irrelevant is not being used. Be strict, and
note that having been *served* a lesson and then done the opposite means it was
not used.

You are better placed to answer this than anything else in the system: you hold
the actual context, so you can see a principle being applied and not merely a
command being run.

For the ones that were used, also say **what specific work they bore on** and
**what doing it correctly looked like**. That pair turns the use into a
replayable test: a fresh agent given that task and that lesson can be checked
against that outcome, which is the only evidence that ever lets a lesson be
compressed. Describe the actual task — not the session's opening request, which
by now has little to do with what just happened.

    rmc used --session {session} --used <ids> --unused <ids> \
      --task "<the specific work, one sentence>" \
      --outcome "<what doing it right looked like, one sentence>"

"""


def _spawn_fork(store: Store, session_id: str, cwd: str, served: list[str] | None = None) -> bool:
    """Fork the live session for reflection. True if the fork was launched."""
    from .adapters._proc import child_env, which

    if not session_id or which("claude") is None:
        return False

    # Attribution belongs here rather than in the digest pass: the fork has the
    # real conversation, and influence on reasoning is invisible in a digest.
    nodes = [n for n in (store.get(i) for i in (served or [])) if n is not None]
    attribution = (
        ATTRIBUTION.format(
            session=session_id,
            served="\n".join(f"  [{n.id}] {n.title or n.family} — {n.summary()}" for n in nodes),
        )
        if nodes
        else ""
    )
    argv = [
        "claude",
        "--resume",
        session_id,
        "--fork-session",  # new session id, so the live one is never touched
        "-p",
        FORK_PROMPT.format(attribution=attribution),
        "--no-session-persistence",
        "--permission-mode",
        "acceptEdits",
        "--allowedTools",
        "Bash",  # it needs exactly one tool: to run `rmc add`
    ]
    log_path = store.root / "background.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            subprocess.Popen(
                argv,
                cwd=cwd,
                stdout=log,
                stderr=log,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                # child_env sets RMC_CHILD=1: without it the fork fires these
                # same hooks and forks itself, forever.
                env=child_env(),
            )
        return True
    except Exception as exc:  # pragma: no cover - best effort
        store.log("error", where="fork", error=f"{type(exc).__name__}: {exc}")
        return False


def _now() -> float:
    import time

    return time.time()


def _too_soon(store: Store, state: dict[str, Any]) -> bool:
    """Cooldown, lengthened when the agent is evidently not needing the prompt.

    If the last several nudges each produced nothing, that is evidence the agent
    is already capturing what matters on its own — or that this kind of work
    simply has little to teach. Either way, keep interrupting it and the nudge
    becomes noise the agent learns to dismiss. Backing off is measured from
    outcomes, not guessed.
    """
    cooldown = int(store.config.get("learning.nudge_cooldown_s", 900))
    barren = _barren_streak(store)
    threshold = int(store.config.get("learning.nudge_backoff_after", 3))
    if barren >= threshold:
        cooldown *= 2 ** min(4, 1 + barren - threshold)
    last = state.get("nudged_at")
    return isinstance(last, (int, float)) and (_now() - last) < cooldown


def _barren_streak(store: Store) -> int:
    """How many recent nudges in a row were followed by no capture."""
    timeline = [
        e for e in store.read_events(limit=400) if e.get("kind") in ("nudge", "capture")
    ]
    streak = 0
    for event in reversed(timeline):
        if event.get("kind") == "capture":
            break
        streak += 1
    return streak


# --------------------------------------------------------------------------- #
# SessionEnd
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
    # Distinct events: Stop fires per turn (the surprise nudge), SessionEnd once
    # at teardown (the sweep). Aliasing them, as an earlier version did, meant
    # the sweep ran on every turn.
    "pre-compact": on_pre_compact,
    "precompact": on_pre_compact,
    "stop": on_turn_end,
    "turn-end": on_turn_end,
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
