"""Command line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .adapters import available_backends, get_adapter
from .store import Store

# --------------------------------------------------------------------------- #
# output helpers
# --------------------------------------------------------------------------- #

_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def dim(t: str) -> str:
    return _c(t, "2")


def bold(t: str) -> str:
    return _c(t, "1")


def green(t: str) -> str:
    return _c(t, "32")


def red(t: str) -> str:
    return _c(t, "31")


def yellow(t: str) -> str:
    return _c(t, "33")


def die(message: str, code: int = 1) -> int:
    print(red(f"error: {message}"), file=sys.stderr)
    return code


def need_store(args: argparse.Namespace) -> Store | None:
    store = Store.discover(Path(getattr(args, "cwd", None) or os.getcwd()))
    if store is None:
        print(
            red("no RMC store found.")
            + " run "
            + bold("rmc init")
            + " here, or set RMC_HOME.",
            file=sys.stderr,
        )
        return None
    return store


def make_adapter(store: Store, args: argparse.Namespace):
    name = getattr(args, "agent", None) or store.config.get("agent", "claude")
    model = getattr(args, "model", None) or store.config.get("model")
    return get_adapter(name, model=model)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def cmd_init(args: argparse.Namespace) -> int:
    base = Path(args.path or os.getcwd()).resolve()
    store = Store.init(base, force=args.force)
    print(f"{green('initialised')} {store.root}")
    if args.agent:
        store.config.set("agent", args.agent)
        store.config.save(store.root / "config.yaml")
    print(dim(f"  backend: {store.config.get('agent')}  ·  available: {', '.join(available_backends())}"))
    print(dim("  next: rmc install    (wire the hooks so it runs automatically)"))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    store = need_store(args)
    if store is None:
        return 1
    nodes = store.nodes()
    episodes = store.episodes()
    families = store.families()

    print(bold(f"RMC {__version__}") + dim(f"  ·  {store.root}"))
    print(f"  backend    {store.config.get('agent')}  ({', '.join(available_backends())} available)")
    print(f"  families   {len(families)}")
    print(f"  nodes      {len(nodes)}")
    print(
        f"  episodes   {len(episodes)} "
        + dim(f"({sum(1 for e in episodes if e.outcome == 'success')} usable for replay)")
    )
    if nodes:
        total = sum(n.tokens for n in nodes)
        apex = [n for n in nodes if n.is_apex and n.status == "active"]
        apex_tokens = sum(n.tokens for n in apex)
        print(f"  tokens     {total} stored, {apex_tokens} served at apex")
        deepest = max(n.level for n in nodes)
        print(f"  max level  {deepest}")
    _print_reflection_stats(store)

    if not families:
        print(dim("\n  no lessons yet — they appear as you work."))
        return 0

    print()
    print(bold("  family                 apex  lvl   tok   used   ok"))
    for family in families:
        node = store.apex(family)
        if node is None:
            continue
        rate = f"{node.stats.posterior:.0%}"
        print(
            f"  {family[:20]:<20s}  {node.id:>8s}  L{node.level}  {node.tokens:>4d}"
            f"  {node.stats.attempts:>4d}  {rate:>4s}"
        )
    return 0


def _print_reflection_stats(store: Store) -> None:
    """Is the reflection nudge load-bearing, or has the agent outgrown it?

    That is an open question and worth measuring rather than arguing about. The
    number that answers it is how many lessons the agent captured *without*
    being prompted. If that climbs toward all of them, the nudge is scaffolding
    you can take down; if it stays near zero, the nudge is doing the work.
    """
    captures = store.read_events("capture", limit=2000)
    nudges = store.read_events("nudge", limit=2000)

    # Retrieval precision: of the lessons put in front of the model, how many
    # actually bore on the work. Low precision is not a memory problem, it is a
    # recall problem — the store is fine and the router is over-serving.
    episodes = [e for e in store.episodes() if e.served]
    shown = sum(len(e.served) for e in episodes)
    used = sum(len(e.used) for e in episodes)
    if shown:
        print(
            f"  precision  {used}/{shown} served lessons were used  "
            + dim(f"({used / shown:.0%})")
        )

    if not captures and not nudges:
        return

    prompted = sum(1 for c in captures if c.get("prompted"))
    spontaneous = len(captures) - prompted
    share = (spontaneous / len(captures)) if captures else 0.0

    print()
    print(f"  captures   {len(captures)}  " + dim(f"({spontaneous} unprompted, {prompted} after a nudge)"))
    print(f"  nudges     {len(nudges)}  " + dim(f"({len(nudges) - prompted} produced nothing)"))
    if captures:
        verdict = (
            "the agent is capturing on its own"
            if share >= 0.6
            else "the nudge is doing the work"
            if share <= 0.25
            else "mixed"
        )
        print(f"  unprompted {share:.0%}  " + dim(f"— {verdict}"))


def cmd_recall(args: argparse.Namespace) -> int:
    from .recall import recall_pack, select_lessons

    store = need_store(args)
    if store is None:
        return 1
    prompt = args.prompt or sys.stdin.read()
    if not prompt.strip():
        return die("no prompt given (pass --prompt or pipe on stdin)")

    adapter = make_adapter(store, args)
    selection = select_lessons(store, adapter, prompt)
    pack = recall_pack(store, prompt, adapter)
    matches = [(n.family, selection.why(n.id)) for n in selection.selected]
    if args.json:
        print(
            json.dumps(
                {
                    "matches": [{"family": f, "why": w} for f, w in matches],
                    "served": pack.served,
                    "tokens": pack.tokens,
                    "text": pack.text,
                },
                indent=2,
            )
        )
        return 0
    if not matches:
        print(dim("no matching lessons"))
        return 0
    for family, why in matches:
        print(f"{green('match')} {family}  {dim(why[:88])}")
    print()
    print(pack.text or dim("(empty pack)"))
    print()
    print(dim(f"— {pack.tokens} tokens, nodes: {', '.join(pack.served)}"))
    return 0


def cmd_learn(args: argparse.Namespace) -> int:
    from .reflect import Outcome, mint
    from .signals import parse_transcript

    store = need_store(args)
    if store is None:
        return 1
    if not args.transcript:
        return die("--transcript is required")
    path = Path(args.transcript)
    if not path.exists():
        return die(f"no such transcript: {path}")

    facts = parse_transcript(path)
    adapter = make_adapter(store, args)
    from .judge import Judge
    from .signals import digest, worth_assessing

    outcome = None
    if worth_assessing(facts, min_tool_calls=int(store.config.get("learning.min_tool_calls", 8))):
        outcome = Outcome.from_verdict(Judge(store, adapter).assess(digest(facts)))
    result = mint(store, adapter, facts, outcome=outcome, session_id=args.session or "")
    if result.created is None:
        print(dim(f"nothing captured: {result.reason}"))
        return 0
    print(f"{green('captured')} {result.created.id} "
          f"[{result.created.family}] {result.created.tokens} tokens")
    print(dim(f"  {result.created.path}"))
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    """Teach RMC something right now, mid-session.

    The transcript sweep at session end is a safety net for what nobody noticed
    in the moment. This is the live path: the instant the user explains
    something, it goes into the tree — reconciled against what is already known
    — and is available to the very next prompt in the same conversation.
    """
    from .judge import Judge
    from .node import Node
    from .placement import apply, decide
    from .util import new_id

    store = need_store(args)
    if store is None:
        return 1
    body = (args.body or sys.stdin.read()).strip()
    if not body:
        return die("no lesson text (pass it as an argument or pipe it on stdin)")

    adapter = make_adapter(store, args)

    # Where a lesson lands decides whether it can ever be found again. A lesson
    # about a vendor API filed under one repo is invisible from every other one,
    # so nothing downstream — not recall, not co-use, not dreaming — can rescue
    # it. That makes scope a judgement, and it has to happen here.
    target = store
    if args.scope == "global" or (args.scope == "auto" and store.parent is not None):
        if args.scope == "global":
            target = store.global_layer()
        else:
            verdict = Judge(store, adapter).scope(body, repo=Path.cwd().name)
            if verdict and str(verdict.get("scope")) == "global":
                target = store.global_layer()
                print(dim(f"scope: global — {str(verdict.get('why') or '')[:100]}"))
    elif args.scope == "project":
        target = store
    store = target

    family = _slugify(args.family or "general")
    node = Node(
        id=new_id("n"),
        family=family,
        body=body,
        level=0,
        title=args.title or "",
        tags=[_slugify(t) for t in (args.tags or "").split(",") if t.strip()],
        origin="manual",
    )

    # Reconciliation is what stops two reflectors recording the same lesson —
    # but only if each one *sees* what the other wrote. Deciding and writing must
    # therefore be atomic: without this, two reflectors that start together both
    # read a store lacking the lesson, both conclude "new", and both create it.
    #
    # A writer waits rather than skipping. Losing the lock and giving up would
    # silently drop a lesson, which is worse than being slow.
    with store.lock("write", wait_s=90) as lock:
        if not lock.acquired:
            return die("another reflector is holding the write lock; try again")
        store.invalidate()  # pick up anything written while we waited
        decision = decide(
            store,
            adapter,
            body=body,
            family_hint=family,
            consult=not args.no_reconcile,
        )
        result = apply(store, decision, node)

    verb = {
        "new-family": "new lesson",
        "attach-sibling": "added alongside",
        "fold-into": "folded into an existing lesson",
        "duplicate": "already known",
        "conflict": "CONFLICTS with what is remembered",
    }.get(decision.action, decision.action)

    # Attribute the capture. Whether the agent reaches for this on its own or
    # only after being prompted is the one measurement that answers whether the
    # reflection nudge is load-bearing scaffolding or a crutch — so record it
    # instead of forming an opinion about it.
    nudge = store.recent_nudge()
    store.log(
        "capture",
        node=result.node.id if result.node else None,
        family=decision.family,
        action=decision.action,
        prompted=bool(nudge),
    )

    colour = red if decision.action == "conflict" else green
    print(f"{colour(verb)}  {dim(decision.rationale[:110])}")
    if result.node:
        print(f"  {result.node.id} [{result.node.family}] {result.node.tokens} tokens")
    if result.patched:
        print(dim(f"  patched {len(result.patched)} compressed ancestor(s): {', '.join(result.patched)}"))
    if decision.action == "conflict" and decision.question:
        print(f"\n  {yellow('needs your answer:')} {decision.question}")
        print(dim("  settle it with: rmc resolve <node-id> [--drop]"))
    return 0


def _slugify(text: str) -> str:
    cleaned = "".join(c if c.isalnum() else "-" for c in str(text).strip().lower())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")[:48] or "general"


def cmd_absorb(args: argparse.Namespace) -> int:
    """The whole post-session pipeline, run detached: judge, learn, compress.

    Exists as one command rather than three spawns because the steps are
    ordered: compaction is only eligible once `observe` has recorded the
    successes that make a node due. Running them as separate background
    processes raced, and compaction usually lost.
    """
    from .compact import run_due
    from .reflect import mint, observe
    from .signals import parse_transcript

    store = need_store(args)
    if store is None:
        return 1
    path = Path(args.transcript)
    if not path.exists():
        return die(f"no such transcript: {path}")

    adapter = make_adapter(store, args)
    facts = parse_transcript(path)
    served = [s for s in (args.served or "").split(",") if s]

    # Mid-session reflection and the session-end sweep can overlap; only one may
    # write. Losing the lock is fine — whatever the winner learns, the loser
    # would have learned too.
    lock = store.lock("absorb")
    with lock:
        if not lock.acquired:
            print("absorb: another run holds the lock; skipping")
            return 0
        return _absorb(store, adapter, facts, served, args)


def _absorb(store, adapter, facts, served, args) -> int:
    from .compact import run_due
    from .reflect import mint, observe

    state = store.read_session(args.session or "")
    result = observe(
        store,
        facts,
        adapter=adapter,
        attributed=dict(state.get("attributed") or {}),
        session_id=args.session or "",
        served=served,
        family_hint=args.family or "",
    )
    if result.skipped:
        print(f"observe: skipped ({result.skipped})")
        return 0
    print(
        f"observe: {result.outcome.label} conf={result.outcome.confidence:.2f} "
        f"corrected={result.outcome.corrected} rescues={len(result.rescues)}"
    )

    minted = mint(store, adapter, facts, outcome=result.outcome, session_id=args.session or "")
    print(f"learn: {minted.reason[:160]}")

    if result.outcome.label == "success":
        for res in run_due(store, adapter, limit=1):
            state = "accepted" if res.accepted else "rejected"
            print(f"compact: {state} {res.node_id} — {res.reason[:120]}")

    # Consolidation is not a reaction to this session, so it does not belong to
    # it — but this is the one place that already runs detached, holds a lock and
    # is allowed to spend calls. Gated on a clock and on new evidence, so most
    # sessions skip it entirely.
    due, why = dream_due(store)
    if due:
        print(f"dream: running ({why})")
        print(f"dream: {dream(store, adapter, limit=int(store.config.get('dream.limit', 2))).render()}")
    else:
        print(f"dream: skipped — {why}")
    return 0


def cmd_used(args: argparse.Namespace) -> int:
    """Record which recalled lessons actually bore on the work.

    Called by the in-session reflector, which is the best-placed judge there is:
    it holds the real context rather than a digest, so it can see a principle
    being applied and not merely a command being run. The verdict is parked on
    the session and preferred over the digest-based one when the episode is
    finally written.
    """
    store = need_store(args)
    if store is None:
        return 1
    if not args.session:
        return die("--session is required")

    state = store.read_session(args.session)
    verdicts = dict(state.get("attributed") or {})
    for ident in [i for i in (args.used or "").split(",") if i.strip()]:
        verdicts[ident.strip()] = True
    for ident in [i for i in (args.unused or "").split(",") if i.strip()]:
        verdicts[ident.strip()] = False
    state["attributed"] = verdicts
    store.write_session(args.session, state)
    store.log("attributed", session=args.session, verdicts=verdicts, source="in-session")

    hits = [k for k, v in verdicts.items() if v]
    print(f"recorded: {len(hits)} used, {len(verdicts) - len(hits)} not used")
    return 0


def cmd_observe(args: argparse.Namespace) -> int:
    from .reflect import observe
    from .signals import parse_transcript

    store = need_store(args)
    if store is None:
        return 1
    path = Path(args.transcript)
    if not path.exists():
        return die(f"no such transcript: {path}")
    facts = parse_transcript(path)
    served = args.served.split(",") if args.served else []
    result = observe(
        store,
        facts,
        adapter=make_adapter(store, args),
        session_id=args.session or "",
        served=[s for s in served if s],
    )
    if result.skipped:
        print(dim(f"skipped: {result.skipped}"))
        return 0
    colour = {"success": green, "failure": red}.get(result.outcome.label, yellow)
    print(f"{colour(result.outcome.label)} confidence={result.outcome.confidence:.2f}")
    for line in result.outcome.evidence:
        print(dim(f"  · {line}"))
    if result.rescues:
        print(bold("  rescues:"))
        for node_id, claim in result.rescues:
            print(f"    {node_id}: {claim[:100]}")
    return 0


def cmd_compact(args: argparse.Namespace) -> int:
    from .compact import compress_node, due_nodes, merge_candidates, merge_nodes, run_due

    store = need_store(args)
    if store is None:
        return 1
    adapter = make_adapter(store, args)

    if args.node:
        node = store.get(args.node)
        if node is None:
            return die(f"no such node: {args.node}")
        results = [compress_node(store, adapter, node, dry_run=args.dry_run)]
    elif args.merge:
        groups = merge_candidates(store, args.merge, adapter)
        if not groups:
            print(dim(f"no merge candidates in family {args.merge}"))
            return 0
        results = [merge_nodes(store, adapter, g, dry_run=args.dry_run) for g in groups[: args.limit]]
    else:
        pending = due_nodes(store)
        if not pending:
            print(dim("nothing due for compaction"))
            return 0
        if args.list:
            for node in pending:
                print(
                    f"{node.id} [{node.family}] L{node.level} "
                    f"{node.tokens}tok successes={node.stats.successes}"
                )
            return 0
        results = run_due(store, adapter, limit=args.limit, dry_run=args.dry_run)

    if not results:
        print(dim("no compaction ran (another process may hold the lock)"))
        return 0

    for res in results:
        if res.accepted:
            print(
                f"{green('accepted')} {res.node_id} -> "
                f"{res.new_node.id if res.new_node else '(dry-run)'}  "
                f"{res.before_tokens}->{res.after_tokens} tokens "
                f"({res.ratio:.0%}), replay {res.pass_rate:.0%}"
            )
            for delta in res.dropped:
                print(dim(f"    dropped [{delta.kind}] {delta.claim[:90]}"))
        else:
            print(f"{yellow('rejected')} {res.node_id}: {res.reason}")
            for replay in res.replays:
                if not replay.ok:
                    print(dim(f"    failed {replay.episode_id}: {replay.reason[:100]}"))
    return 0


def cmd_dream(args: argparse.Namespace) -> int:
    """Whole-store consolidation, independent of any session."""
    from .compact import co_use_groups, dream

    store = need_store(args)
    if store is None:
        return 1

    if args.log:
        from .compact import dream_logs

        logs = dream_logs(store, limit=args.limit)
        if not logs:
            print(dim("no dreams recorded yet"))
            return 0
        if args.log is True or args.log == "latest":
            print(logs[0].read_text())
            print(dim(f"— {logs[0]}  ({len(logs)} dream(s) on record)"))
            return 0
        for path in logs:
            print(f"  {path.stem}  {dim(str(path))}")
        return 0

    if args.due:
        from .compact import dream_due

        due, why = dream_due(store)
        print(f"{green('due') if due else dim('not due')} — {why}")
        return 0

    if args.list:
        groups = co_use_groups(store)
        if not groups:
            print(dim("no lessons have been used together on successful work yet"))
            return 0
        print(bold("lessons repeatedly used together"))
        for nodes, seen in groups:
            names = ", ".join(f"{n.id}[{n.family}]" for n in nodes)
            print(f"  {seen}×  {names}")
        print(dim("\n  these are the merge candidates — co-use, not similarity"))
        return 0

    adapter = make_adapter(store, args)
    report = dream(store, adapter, limit=args.limit, dry_run=args.dry_run)
    print(report.render())
    for line in report.merged:
        print(f"  {green('merged')}   {line}")
    for line in report.rejected:
        print(f"  {yellow('rejected')} {line}")
    return 0


def cmd_tree(args: argparse.Namespace) -> int:
    store = need_store(args)
    if store is None:
        return 1
    families = [args.family] if args.family else store.families()
    if not families:
        print(dim("no lessons yet"))
        return 0

    for family in families:
        apexes = [n for n in store.family_nodes(family) if n.is_apex]
        print(bold(family))
        seen: set[str] = set()
        for apex in apexes:
            _print_node(store, apex, prefix="  ", args=args, seen=seen)
        print()
    return 0


def _print_node(store: Store, node, *, prefix: str, args, seen: set[str] | None = None) -> None:
    # A node reachable from two parents is printed once, with a pointer the
    # second time. Without this the DAG renders as an exponentially larger tree.
    seen = seen if seen is not None else set()
    if node.id in seen:
        print(f"{prefix}{dim(f'{node.id} (shown above)')}")
        return
    seen.add(node.id)
    flag = {"active": "", "demoted": yellow(" demoted"), "superseded": dim(" superseded")}.get(
        node.status, ""
    )
    title = node.title or node.family
    print(
        f"{prefix}{bold(node.id)} L{node.level} {node.tokens:>4d}tok "
        f"{dim(f'use={node.stats.attempts} ok={node.stats.posterior:.0%}')}{flag}  {title[:44]}"
    )
    if args.verbose:
        for line in node.body.splitlines()[:3]:
            print(f"{prefix}  {dim('│ ' + line[:88])}")
    for delta in node.dropped:
        holder = f" -> {delta.holder}" if delta.holder else ""
        print(f"{prefix}  {dim(f'△ [{delta.kind}] {delta.claim[:64]}{holder}')}")
    for child in store.children(node):
        _print_node(store, child, prefix=prefix + "    ", args=args, seen=seen)


def cmd_trace(args: argparse.Namespace) -> int:
    """Agent's-eye view: every stage of a recall, ending in the literal context.

    RMC edits what the model sees. That should never be something you have to
    take on trust, so this shows the whole path — what was asked, what came
    back, what was injected verbatim, and what the user is told about it.
    """
    from .hooks import BANNER, PREAMBLE, recall_notice
    from .recall import recall_pack, select_lessons

    store = need_store(args)
    if store is None:
        return 1
    prompt = args.prompt or sys.stdin.read()
    if not prompt.strip():
        return die("no prompt given (pass --prompt or pipe on stdin)")

    adapter = make_adapter(store, args)
    width = 74

    def stage(n: int, title: str) -> None:
        print(f"\n{bold(f'{n}. {title}')}")
        print(dim("─" * width))

    stage(1, "the prompt you typed")
    print(f"   {prompt.strip()[:600]}")

    roots = [n for n in (store.apex(f) for f in store.families()) if n is not None]
    stage(2, f"what RMC put in front of the model ({len(roots)} apex lessons)")
    if not roots:
        print(dim("   nothing stored yet — no question is asked at all"))
        return _trace_after(store, adapter, Path(args.after), stage) if args.after else 0
    for node in roots:
        depth = f", {len(node.dropped)} detail(s) beneath" if node.dropped else ""
        print(f"   [{node.id}] {node.title or node.family}  {dim(f'L{node.level}, {node.tokens} tok{depth}')}")
    print(dim(f"\n   these are the most compressed nodes, which is why they all fit in one question"))

    selection = select_lessons(store, adapter, prompt)
    stage(3, "what the model decided")
    if not selection.picks:
        print(dim("   no verdict returned"))
    for node_id, pick in selection.picks.items():
        mark = {"relevant": green("✓ relevant "), "maybe": yellow("~ maybe    ")}.get(
            pick.verdict, dim("· unrelated")
        )
        opened = dim("  → opened for detail") if pick.descend else ""
        print(f"   {mark} {node_id}  {dim(pick.why[:70])}{opened}")
    skipped = [p for p in selection.picks.values() if not p.positive]
    print(
        dim(
            f"\n   {len(skipped)} branch(es) judged irrelevant were never walked further — "
            f"{selection.calls} model call(s) total"
        )
    )

    pack = recall_pack(store, prompt, adapter)
    stage(4, "what is injected into the agent's context, verbatim")
    if not pack:
        print(dim("   (nothing — the agent sees your prompt unchanged)"))
        return _trace_after(store, adapter, Path(args.after), stage) if args.after else 0
    block = f"{BANNER}\n{PREAMBLE}\n\n{pack.text}"
    for line in block.splitlines():
        print(f"   {dim('│')} {line}")
    print(dim(f"   └─ {pack.tokens} tokens"))

    stage(5, "what you see in Claude Code")
    print(f"   {dim('⋯')} Recalling lessons…        {dim('(while the hook runs)')}")
    print(f"   {dim('⋯')} {recall_notice(pack)}")

    stage(6, "so the model's turn begins as")
    print(dim("   <additional-context>"))
    print(dim(f"   {BANNER} … {pack.tokens} tokens of prior knowledge …"))
    print(dim("   </additional-context>"))
    print(f"   {prompt.strip()[:200]}")
    print()
    print(
        dim(
            "   The lessons are framed as prior knowledge, not as instructions from you,\n"
            "   so a stale lesson cannot impersonate a request."
        )
    )

    if args.after:
        return _trace_after(store, adapter, Path(args.after), stage)
    print(dim("\n   pass --after <transcript.jsonl> to also trace what happens at session end"))
    return 0


def _trace_after(store, adapter, path: Path, stage) -> int:
    """The other half: what RMC does once the session is over."""
    from .judge import Judge
    from .reflect import Outcome
    from .signals import digest, parse_transcript, worth_assessing

    if not path.exists():
        return die(f"no such transcript: {path}")
    facts = parse_transcript(path)

    stage(7, "at session end — the facts RMC parsed out")
    print(f"   {len(facts.user_messages)} human turn(s), {facts.tool_calls} tool call(s)")
    for event in facts.tool_events[:6]:
        print(dim(f"   {event.render()[:150]}"))
    if not worth_assessing(facts, min_tool_calls=int(store.config.get("learning.min_tool_calls", 8))):
        print(dim("\n   too small to be worth judging — nothing is asked, nothing is stored"))
        return 0

    stage(8, "what the model made of it")
    outcome = Outcome.from_verdict(Judge(store, adapter).assess(digest(facts)))
    print(f"   outcome    {outcome.label}  (confidence {outcome.confidence:.2f})")
    print(f"   corrected  {outcome.corrected}")
    for line in outcome.evidence[:4]:
        print(dim(f"   · {line}"))
    if outcome.discoveries:
        print(bold("\n   worked out by trial:"))
        print(dim("   " + outcome.render_discoveries().replace("\n", "\n   ")[:700]))
    print(
        dim(
            "\n   A correction counts against the lesson that was served, even when the\n"
            "   session ended well — those are different questions."
        )
    )
    return 0


def cmd_conflicts(args: argparse.Namespace) -> int:
    from .placement import open_conflicts

    store = need_store(args)
    if store is None:
        return 1
    conflicts = open_conflicts(store, args.family)
    if not conflicts:
        print(dim("no unresolved conflicts"))
        return 0
    print(bold(f"{len(conflicts)} unresolved conflict(s)"))
    for node in conflicts:
        print(f"\n  {bold(node.id)} [{node.family}] L{node.level}  {node.title[:50]}")
        print(f"    {yellow('?')} {node.conflict}")
        print(dim(f"    {node.body.splitlines()[0][:90] if node.body else ''}"))
    print(dim("\n  settle with: rmc resolve <node-id> [--drop]"))
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    from .placement import resolve

    store = need_store(args)
    if store is None:
        return 1
    node = resolve(store, args.node, keep=not args.drop)
    if node is None:
        return die(f"no such node: {args.node}")
    verb = "archived" if args.drop else "kept"
    print(f"{green(verb)} {node.id} — conflict cleared")
    return 0


def cmd_hook(args: argparse.Namespace) -> int:
    from .hooks import dispatch

    return dispatch(args.event)


def cmd_install(args: argparse.Namespace) -> int:
    from .install import install

    return install(
        scope=args.scope,
        targets=args.target or ["claude"],
        path=Path(args.path or os.getcwd()),
        dry_run=args.dry_run,
    )


def cmd_uninstall(args: argparse.Namespace) -> int:
    from .install import uninstall

    return uninstall(scope=args.scope, targets=args.target or ["claude"], path=Path(args.path or os.getcwd()))


def cmd_config(args: argparse.Namespace) -> int:
    store = need_store(args)
    if store is None:
        return 1
    if args.key and args.value is not None:
        from .config import _coerce

        store.config.set(args.key, _coerce(args.value))
        store.config.save(store.root / "config.yaml")
        print(f"{green('set')} {args.key} = {store.config.get(args.key)!r}")
        return 0
    if args.key:
        print(json.dumps(store.config.get(args.key), indent=2, default=str))
        return 0
    print(json.dumps(store.config.data, indent=2, default=str))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from .install import status as install_status

    store = Store.discover(Path(os.getcwd()))
    print(bold("backends"))
    for name in ("claude", "codex"):
        adapter = get_adapter(name)
        mark = green("✓") if adapter.available() else red("✗")
        print(f"  {mark} {name}")
    print(bold("\nstore"))
    if store is None:
        print(f"  {red('✗')} none found (run rmc init)")
    else:
        print(f"  {green('✓')} {store.root}")
        print(f"    nodes={len(store.nodes())} episodes={len(store.episodes())}")
    print(bold("\nhooks"))
    for line in install_status():
        print(f"  {line}")
    return 0


def cmd_events(args: argparse.Namespace) -> int:
    store = need_store(args)
    if store is None:
        return 1
    for row in store.read_events(args.kind, limit=args.limit):
        print(json.dumps(row))
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rmc",
        description="Recursive Memory Compaction — lessons that get cheaper the more you use them.",
    )
    parser.add_argument("--version", action="version", version=f"rmc {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_agent_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--agent", choices=["claude", "codex", "mock"], help="execution backend")
        p.add_argument("--model", help="model override")

    p = sub.add_parser("init", help="create a store in this repo")
    p.add_argument("path", nargs="?")
    p.add_argument("--force", action="store_true")
    p.add_argument("--agent", choices=["claude", "codex", "mock"])
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("status", help="overview of the lesson tree")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("recall", help="show the context pack a prompt would get")
    p.add_argument("--prompt", "-p")
    p.add_argument("--json", action="store_true")
    add_agent_flags(p)
    p.set_defaults(func=cmd_recall)

    p = sub.add_parser("learn", help="mint a level-0 lesson from a transcript")
    p.add_argument("--transcript")
    p.add_argument("--session")
    add_agent_flags(p)
    p.set_defaults(func=cmd_learn)

    p = sub.add_parser("add", help="teach RMC something now, without waiting for session end")
    p.add_argument("body", nargs="?", help="the lesson, as instruction to a future agent")
    p.add_argument("--family", help="short slug for the recurring situation it applies to")
    p.add_argument("--title")
    p.add_argument("--tags", help="comma-separated")
    p.add_argument("--no-reconcile", action="store_true", help="skip the consistency check")
    p.add_argument(
        "--scope",
        choices=["auto", "project", "global"],
        default="auto",
        help="auto asks whether the lesson is repo-specific or would apply anywhere",
    )
    add_agent_flags(p)
    p.set_defaults(func=cmd_add)

    p = sub.add_parser(
        "absorb", help="run the whole post-session pipeline (judge, learn, compress)"
    )
    p.add_argument("--transcript", required=True)
    p.add_argument("--session")
    p.add_argument("--served", help="comma-separated node ids that were injected")
    p.add_argument("--family")
    add_agent_flags(p)
    p.set_defaults(func=cmd_absorb)

    p = sub.add_parser("used", help="record which recalled lessons bore on the work")
    p.add_argument("--session", required=True)
    p.add_argument("--used", help="comma-separated node ids that changed what was done")
    p.add_argument("--unused", help="comma-separated node ids that did not")
    p.set_defaults(func=cmd_used)

    p = sub.add_parser("observe", help="judge a transcript and update stats")
    p.add_argument("--transcript", required=True)
    p.add_argument("--session")
    p.add_argument("--served", help="comma-separated node ids that were injected")
    add_agent_flags(p)
    p.set_defaults(func=cmd_observe)

    p = sub.add_parser("compact", help="compress lessons and regression-test the result")
    p.add_argument("--node", help="compress one specific node")
    p.add_argument("--due", action="store_true", help="process the queue (default)")
    p.add_argument("--merge", metavar="FAMILY", help="merge sibling lessons in a family")
    p.add_argument("--list", action="store_true", help="list what is due, do nothing")
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    add_agent_flags(p)
    p.set_defaults(func=cmd_compact)

    p = sub.add_parser(
        "dream", help="consolidate the whole store: fill gists, merge co-used lessons"
    )
    p.add_argument("--list", action="store_true", help="show merge candidates, change nothing")
    p.add_argument(
        "--log",
        nargs="?",
        const=True,
        default=None,
        metavar="all",
        help="read the last dream's report; 'all' lists every recorded dream",
    )
    p.add_argument("--due", action="store_true", help="say whether a dream is due, and why")
    p.add_argument("--limit", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    add_agent_flags(p)
    p.set_defaults(func=cmd_dream)

    p = sub.add_parser("tree", help="visualise the lesson tree")
    p.add_argument("--family")
    p.add_argument("--verbose", "-v", action="store_true")
    p.set_defaults(func=cmd_tree)

    p = sub.add_parser("trace", help="agent's-eye view: every stage of a recall")
    p.add_argument("--prompt", "-p")
    p.add_argument("--after", metavar="TRANSCRIPT", help="also trace what happens at session end")
    add_agent_flags(p)
    p.set_defaults(func=cmd_trace)

    p = sub.add_parser("conflicts", help="lessons that contradict each other")
    p.add_argument("--family")
    p.set_defaults(func=cmd_conflicts)

    p = sub.add_parser("resolve", help="settle a conflict")
    p.add_argument("node")
    p.add_argument("--drop", action="store_true", help="archive this node instead of keeping it")
    p.set_defaults(func=cmd_resolve)

    p = sub.add_parser("hook", help="hook entry point (called by the host agent)")
    p.add_argument("event")
    p.set_defaults(func=cmd_hook)

    p = sub.add_parser("install", help="wire RMC into claude / codex")
    p.add_argument("--target", action="append", choices=["claude", "codex"])
    p.add_argument("--scope", choices=["user", "project"], default="project")
    p.add_argument("--path")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_install)

    p = sub.add_parser("uninstall", help="remove RMC hooks")
    p.add_argument("--target", action="append", choices=["claude", "codex"])
    p.add_argument("--scope", choices=["user", "project"], default="project")
    p.add_argument("--path")
    p.set_defaults(func=cmd_uninstall)

    p = sub.add_parser("config", help="read or write configuration")
    p.add_argument("key", nargs="?")
    p.add_argument("value", nargs="?")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("doctor", help="check the installation")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("events", help="dump the telemetry log")
    p.add_argument("--kind")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_events)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:  # pragma: no cover
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
