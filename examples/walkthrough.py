#!/usr/bin/env python3
"""A complete RMC cycle you can watch, in a throwaway store.

    python3 examples/walkthrough.py                # mock backend: instant, free
    python3 examples/walkthrough.py --agent claude # real backend
    python3 examples/walkthrough.py --agent codex

It walks the whole thesis end to end:

  1. a verbose L0 lesson is minted
  2. real sessions using it are recorded as episodes (the replay corpus)
  3. compression produces L1, validated against that corpus in fresh processes
  4. a task needing the dropped detail FAILS on the compressed lesson
  5. descent matches the failure diagnosis to the delta manifest and rescues it
  6. repeated rescues fold the detail back in — the tree heals where it was cut

With the mock backend every step is deterministic, so this doubles as an
integration test of the live control flow.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rmc.adapters import get_adapter
from rmc.adapters.mock import MockAdapter, MockWorld
from rmc.compact import compress_node, repair
from rmc.node import Node
from rmc.recall import recall_pack, solve_with_descent
from rmc.store import Episode, Store

BOLD, DIM, GREEN, RED, YELLOW, OFF = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def head(n: int, text: str) -> None:
    print(f"\n{BOLD}{n}. {text}{OFF}")
    print(DIM + "─" * 68 + OFF)


LESSON = """When calling flaky remote services in this codebase, follow these rules.

- Retry only idempotent operations. A non-idempotent write must have a dedupe
  key established before the first attempt, otherwise a retry double-writes.
  @idempotent

- Use jittered exponential backoff rather than a fixed delay, so that retries
  from many clients do not synchronise into a thundering herd. @backoff

- S3 is a special case that catches everyone: it can return HTTP 200 with an
  error document in the response body. You must parse the body rather than
  trusting the status code, and treat a parsed error exactly as you would treat
  a 5xx response for the purposes of retrying. @s3-body
"""

EPISODES = [
    (
        "e_http",
        "add retry to the http client in api/client.py",
        {"idempotent"},
        "Retried GET/PUT/DELETE only; POST retries are gated behind an "
        "idempotency key so a retry cannot double-write.",
    ),
    (
        "e_db",
        "make the db writer retry safely on deadlock",
        {"idempotent", "backoff"},
        "Retried on deadlock with jittered exponential backoff, and required a "
        "dedupe key before retrying any non-idempotent write.",
    ),
    (
        "e_queue",
        "the queue consumer needs backoff between attempts",
        {"backoff"},
        "Used jittered exponential backoff between attempts rather than a fixed "
        "delay, to avoid consumers synchronising into a thundering herd.",
    ),
]

# The task that exercises the detail compression will drop.
S3_TASK = ("t_s3", "handle the s3 upload response correctly", {"idempotent", "s3-body"})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="mock", choices=["mock", "claude", "codex"])
    parser.add_argument("--keep", action="store_true", help="keep the temp store")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="rmc-walkthrough-"))
    store = Store.init(tmp)
    world = MockWorld({eid: facts for eid, _, facts, _s in EPISODES} | {S3_TASK[0]: S3_TASK[2]})
    adapter = MockAdapter(world=world) if args.agent == "mock" else get_adapter(args.agent)

    print(f"{BOLD}RMC walkthrough{OFF}  {DIM}backend={args.agent}  store={tmp}{OFF}")

    # ---------------------------------------------------------------- 1
    head(1, "Mint the verbose L0 lesson")
    base = Node(id="n_L0", family="retry", title="Retrying flaky services", body=LESSON, level=0)
    store.save_node(base)
    store.invalidate()
    print(f"   {base.id}  L0  {base.tokens} tokens")

    # ---------------------------------------------------------------- 2
    head(2, "Record real sessions that used it (the replay corpus)")
    for eid, prompt, _facts, summary in EPISODES:
        store.save_episode(
            Episode(
                id=eid,
                family="retry",
                prompt=prompt,
                outcome="success",
                confidence=0.9,
                served=[base.id],
                accepted_summary=summary,
            )
        )
        print(f"   {GREEN}✓{OFF} {eid}  {DIM}{prompt}{OFF}")
    base = store.get(base.id)
    base.covers_tasks = [e[0] for e in EPISODES]
    base.stats.attempts = base.stats.successes = len(EPISODES)
    store.save_node(base)
    store.invalidate()

    # ---------------------------------------------------------------- 3
    head(3, "Compress — validated against that corpus in fresh processes")
    result = compress_node(store, adapter, store.get(base.id))
    if not result.accepted:
        print(f"   {RED}rejected{OFF}: {result.reason}")
        return 1
    apex = result.new_node
    print(
        f"   {GREEN}accepted{OFF}  {result.before_tokens} → {result.after_tokens} tokens "
        f"({result.ratio:.0%}), replay {result.pass_rate:.0%}"
    )
    print(f"   apex is now {apex.id} at L{apex.level}")
    print(f"\n   {BOLD}delta manifest{OFF} {DIM}(what makes descent possible){OFF}")
    for delta in apex.dropped:
        print(f"     △ [{delta.kind}] {delta.claim[:80]} {DIM}-> {delta.holder}{OFF}")

    # ---------------------------------------------------------------- 4
    head(4, "Recall now serves the cheap apex")
    pack = recall_pack(store, "the http client needs retry logic")
    print(f"   {pack.tokens} tokens served {DIM}(was {result.before_tokens}){OFF}")

    # ---------------------------------------------------------------- 5
    head(5, "A task needing the dropped detail — apex fails, descent rescues")
    task_id, task_text, _ = S3_TASK

    def verify(run, pack_text):
        ok, missing = world.solves(task_id, pack_text)
        # Report in the @fact vocabulary so the simulated diagnoser can name the
        # gap the way a real one would name a missing detail.
        return ok, "missing: " + " ".join(f"@{m}" for m in sorted(missing))

    descent = solve_with_descent(
        store,
        adapter=MockAdapter(world=world),  # verification is world-driven either way
        task_id=task_id,
        task=task_text,
        family="retry",
        verify=verify,
    )
    for i, attempt in enumerate(descent.attempts):
        mark = f"{GREEN}pass{OFF}" if attempt.ok else f"{RED}fail{OFF}"
        print(f"   attempt {i + 1}  {mark}  via {attempt.candidate:<18s} {attempt.tokens:>4d} tok")
        if not attempt.ok:
            print(f"             {DIM}{attempt.detail}{OFF}")
    if descent.rescued_by is not None:
        print(f"\n   {GREEN}rescued by{OFF} {descent.rescued_by.label}")
        print(f"   {DIM}{descent.rescued_by.text[:100]}{OFF}")
        print(f"   {DIM}score components: {descent.rescued_by.parts}{OFF}")
    elif descent.escalated:
        print(f"\n   {YELLOW}escalated to L0{OFF} (no delta matched)")

    # ---------------------------------------------------------------- 6
    head(6, "Repeated rescues heal the tree")
    claim = descent.rescued_by.text if descent.rescued_by else ""
    if claim:
        for _ in range(2):
            store.log("rescue", node=apex.id, claim=claim)
        restored = repair(store, store.get(apex.id), min_rescues=2)
        healed = store.get(apex.id)
        print(f"   folded {len(restored)} claim(s) back into {apex.id}")
        print(f"   apex now {healed.tokens} tokens, manifest has {len(healed.dropped)} entries left")

    head(7, "Final tree")
    for node in sorted(store.family_nodes("retry"), key=lambda n: -n.level):
        arrow = "└─" if node.level == 0 else "  "
        print(
            f"   {arrow} {node.id}  L{node.level}  {node.tokens:>4d} tok  "
            f"{DIM}use={node.stats.attempts} ok={node.stats.posterior:.0%}{OFF}"
        )

    print(f"\n{DIM}store: {tmp}{OFF}")
    if not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
