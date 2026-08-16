# RMC — Recursive Memory Compaction

A harness for token-based continual learning. Lessons are learned once in verbose
form, then **recursively compressed each time they are successfully recalled**,
producing a tree of progressively more abstract memories. At retrieval time you
load the most compressed node that still works, and only pay for detail when the
abstraction fails.

Works with **Claude Code** (`claude -p`) and **Codex** (`codex exec`) as
interchangeable execution backends.

---

## The idea

The standard learning cycle produces one artifact and stops:

```
task + human steering  ->  correct output  ->  reflection  ->  lesson
```

RMC keeps going. Every *successful recall* of a lesson is evidence that the
lesson contained slack, so it triggers a compression attempt:

```
task + lesson(L0)   -> correct output -> reflect -> lesson(L1)   ~40% tokens
task + lesson(L1)   -> correct output -> reflect -> lesson(L2)   ~15% tokens
task + lesson(L2)   -> correct output -> reflect -> lesson(L3)   ~5%  tokens
```

The result is a tree, not a chain, because compression also **merges siblings**:
three lessons about retrying HTTP calls, retrying DB writes and retrying S3 puts
collapse into one lesson about idempotent retry. Abstraction and compression are
the same operation viewed from different ends.

Retrieval walks the tree from the apex (most compressed) downward, spending
tokens only as needed:

```
        L3  "Retry idempotent ops with jittered backoff; cap by deadline."   87 tok
        /                          \
      L2 (http)                    L2 (storage)                             ~300 tok each
     /       \                    /            \
   L1        L1                 L1             L1                          ~900 tok each
   |          |                  |              |
   L0        L0                 L0             L0                         ~3000 tok each
```

**The more a memory is used, the cheaper and more general it becomes.** That is
the whole thesis: usage drives abstraction.

---

## The hard part: which child do you descend to?

When the apex lesson fails, something was lost in compression — but compression
normally *destroys the very information you would need to invert it*. Blind
descent means trying children at random and paying full price for the wrong one.

**RMC's answer: compression is required to record what it dropped.**

Every compression emits a **delta manifest** alongside the compressed node — a
list of discrete claims that were removed, each tagged with a kind and attributed
to the descendant that still holds it:

```yaml
dropped:
  - claim: "Backoff constants are 100ms / 400ms / 1.6s, jitter ±25%"
    kind: parameter
    holder: n_7f2a
  - claim: "S3 returns 200 with an error body; must parse body not status"
    kind: edge-case
    holder: n_c41b
```

Descent then stops being a search problem and becomes a **retrieval problem over
deltas, keyed by the failure diagnosis**. When a node fails we ask the verifier
and a diagnosing agent for a structured account of *how* it failed, and match
that against the manifest. Three consequences fall out:

1. **Delta jumping.** Deltas are attributed transitively, so you can jump
   straight to the descendant holding the needed detail instead of walking the
   tree one level at a time.
2. **Delta patching** (the default). You usually do not need the child at all —
   you need three sentences from it. Re-run the apex plus only the matched delta
   claims. Typical repair cost is tens of tokens, not thousands.
3. **Failure is training signal.** A node that fails a task adds that task to its
   ancestors' regression sets, so future compressions are forbidden from dropping
   that detail again. The tree repairs itself where it has been cut too deep.

Full scoring function, budgets and escalation rules are in
[DESIGN.md](DESIGN.md#4-descent-and-selection).

---

## Compression is only accepted if it survives a regression test

The user-facing check from the original sketch — *spawn a subagent, give it the
same prompt plus the compressed lesson, see if you still get the right output* —
is implemented, and hardened against overfitting:

- A candidate compression is validated against the **regression set**: every task
  the node *and its entire subtree* were previously validated on, not just the
  task that triggered the compression.
- Validation runs in a **fresh agent process** with no conversation history, so
  the only thing carrying knowledge is the lesson text itself.
- Acceptance requires both a pass-rate threshold and a real token reduction.
- Rejections are recorded with the failing tasks, and the next compression
  attempt is given those as explicit `preserve:` hints.

---

## It runs by itself

You do not drive RMC. You use Claude Code or Codex normally on your own repos,
and the loop closes in the background:

| When | What happens | Cost |
|---|---|---|
| you submit a prompt | matching apex lessons are injected as context | no model call — pure lexical match |
| the session ends | the outcome is read from what you did next (corrections, approvals, test results) | no model call |
| a correction happened | the correction *is* the diagnosis; it is matched against the delta manifest and the claim is re-attached next time | no model call |
| something reusable happened | one reflection call mints a level-0 lesson | 1 call, detached |
| a lesson has succeeded twice | a compression is attempted and replay-tested | detached, rejects freely |

The expensive steps are detached into a background process, so nothing is ever
added to the latency of your session. Spawned agents run with `RMC_CHILD=1`,
which makes RMC's own hooks no-op — otherwise compression would recursively
trigger compression.

## Install

Requires Python 3.10+ and at least one of `claude` or `codex`. No third-party
Python dependencies.

**As a Claude Code plugin** (recommended — works in every repo):

```
/plugin marketplace add yiheinchai/recursive-memory-compaction
/plugin install rmc@rmc
```

**Or from a clone**, into one repo:

```bash
git clone https://github.com/yiheinchai/recursive-memory-compaction
cd recursive-memory-compaction && ./bin/rmc install --target claude --target codex
```

`./bin/rmc` needs no virtualenv or `pip`; `pip install -e .` also works and puts
`rmc` on your PATH. Add `--scope user` to install globally rather than per-repo.

Undo with `rmc uninstall` — it removes only what it added and leaves your
lessons in place.

## Seeing what it knows

```bash
rmc status                        # families, levels, token cost, success rates
rmc tree --family retry           # the tree, with delta manifests
rmc recall --prompt "add retry"   # exactly what would be injected, and why
rmc compact --list                # what is eligible for compression
rmc doctor                        # backends, store, hook wiring
```

In a session, `/rmc` does the same through the bundled skill.

## Watch the whole cycle

```bash
python3 examples/walkthrough.py            # deterministic, offline, ~1 second
python3 examples/walkthrough.py --agent claude
```

It mints a lesson, records episodes, compresses it, then fails a task that needed
the dropped detail and shows descent recovering it — printing the score
components so you can see *why* each candidate was chosen.

## Tests

```bash
python3 -m unittest discover -s tests
```

29 tests, no dependencies. They run against a simulated knowledge world where a
task is solved iff the required facts are present in the lesson text, so the
compress → fail → descend → rescue cycle is genuinely executed rather than
mocked at the seams.

## Documentation

| Doc | Contents |
|---|---|
| [DESIGN.md](DESIGN.md) | Tree semantics, delta manifests, selection policy, validation protocol, known failure modes |
| [docs/integration.md](docs/integration.md) | How the hooks wire in, and how to run it with Codex |
| [docs/cli.md](docs/cli.md) | Full command reference |

## Status

Research harness, v0.1. It works end to end and is tested, but the store format
is not yet stable and the ambient outcome signals are heuristic — see
[DESIGN.md §8](DESIGN.md#8-failure-modes-this-design-accepts) for what this
design knowingly gets wrong.
