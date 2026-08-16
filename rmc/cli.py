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


def cmd_recall(args: argparse.Namespace) -> int:
    from .recall import match_families, recall_pack

    store = need_store(args)
    if store is None:
        return 1
    prompt = args.prompt or sys.stdin.read()
    if not prompt.strip():
        return die("no prompt given (pass --prompt or pipe on stdin)")

    matches = match_families(store, prompt)
    pack = recall_pack(store, prompt)
    if args.json:
        print(
            json.dumps(
                {
                    "matches": [{"family": f, "score": round(s, 4)} for f, s in matches],
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
    for family, score in matches:
        print(f"{green('match')} {family} {dim(f'{score:.3f}')}")
    print()
    print(pack.text or dim("(empty pack)"))
    print()
    print(dim(f"— {pack.tokens} tokens, nodes: {', '.join(pack.served)}"))
    return 0


def cmd_learn(args: argparse.Namespace) -> int:
    from .reflect import mint
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
    result = mint(store, adapter, facts, session_id=args.session or "")
    if result.created is None:
        print(dim(f"nothing captured: {result.reason}"))
        return 0
    print(f"{green('captured')} {result.created.id} "
          f"[{result.created.family}] {result.created.tokens} tokens")
    print(dim(f"  {result.created.path}"))
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
    result = observe(store, facts, session_id=args.session or "", served=[s for s in served if s])
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
        groups = merge_candidates(store, args.merge)
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
        for apex in apexes:
            _print_node(store, apex, prefix="  ", args=args)
        print()
    return 0


def _print_node(store: Store, node, *, prefix: str, args) -> None:
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
        _print_node(store, child, prefix=prefix + "    ", args=args)


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
    p.set_defaults(func=cmd_recall)

    p = sub.add_parser("learn", help="mint a level-0 lesson from a transcript")
    p.add_argument("--transcript")
    p.add_argument("--session")
    add_agent_flags(p)
    p.set_defaults(func=cmd_learn)

    p = sub.add_parser("observe", help="score a transcript and update stats")
    p.add_argument("--transcript", required=True)
    p.add_argument("--session")
    p.add_argument("--served", help="comma-separated node ids that were injected")
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

    p = sub.add_parser("tree", help="visualise the lesson tree")
    p.add_argument("--family")
    p.add_argument("--verbose", "-v", action="store_true")
    p.set_defaults(func=cmd_tree)

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
