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

Descent then stops being a search problem and becomes a **ranking problem over
deltas**. When a node fails, the verifier and a diagnosing agent produce a
structured account of *how* it failed, and the model — not a similarity metric —
decides which dropped claim explains it. That distinction matters: "parse the
body, not the status code" is the fix for "treated HTTP 200 as success" while
sharing almost no vocabulary with it. Three consequences fall out:

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

## The harness structures; the model judges

A memory system is full of questions like *is this relevant*, *does this
contradict that*, *did this go well*. It is tempting to answer them with
similarity scores and phrase lists, and RMC did at first. That is a mistake:
meaning does not live in token overlap, and a heuristic silently caps the system
at what a bag of words can express while looking like a judgement.

So the split is strict.

**The harness owns structure** — the tree, the traversal loop, the budgets, the
caches, the schemas answers must fit, and the decision of *whether to ask at
all* (an empty store, an exhausted budget, and a two-tool-call session need no
judgement).

**The model owns meaning** — every question above, each behind a JSON schema,
cached so nothing is judged twice.

Efficiency comes from structure rather than from cheap approximations. Because
apexes are the most compressed nodes in the store, the entire top level of the
tree fits in one question; a line is only opened when the model says the summary
was too abstract to decide from. Cost tracks the *depth* of the tree, not its
size — and a branch judged clearly irrelevant is never walked at all.

The two terms still computed in code are evidence, not proxies for judgement:
how often a node has actually rescued a failure before, and how many tokens it
costs.

---

## It runs by itself

You do not drive RMC. You use Claude Code or Codex normally on your own repos,
and the loop closes in the background:

| When | What happens | Cost |
|---|---|---|
| you submit a prompt | the model is asked which remembered lessons bear on it, walking the tree from the most abstract nodes down | 1 call, cached by prompt |
| the session ends | the model reads the session and judges how it went, whether you had to steer, and what was worked out by trial | 1 call, detached |
| a correction happened | the correction *is* the diagnosis; the model picks which dropped detail it was about, and that claim is re-attached next time | 1 call |
| you teach it something | `rmc add` records it immediately, reconciled, available to your next prompt | 1–2 calls |
| something reusable happened | a reflection call mints a level-0 lesson from the transcript | 1 call, detached |
| the new lesson touches known ground | it is reconciled with what is already there — folded in, set alongside, or flagged as a contradiction | 1 call, cached |
| a lesson has succeeded twice | a compression is attempted and replay-tested | detached, rejects freely |

The expensive steps are detached into a background process, so nothing is ever
added to the latency of your session. Spawned agents run with `RMC_CHILD=1`,
which makes RMC's own hooks no-op — otherwise compression would recursively
trigger compression.


### Learning happens live, and does not need you

Teach it something mid-conversation and it lands immediately:

```bash
rmc add --family deploys "Use `kubectl argo rollouts promote`; plain apply fails, the CRD is not registered"
```

That is reconciled against what is already known and available to your very next
prompt. The bundled skill tells the agent to do this the moment you explain
something, so you rarely type it yourself.

Human corrections are the *rarer* source of lessons, though. Most of the time
the environment does the teaching: a command fails, a different one works, a
test rejects an approach. RMC pairs each tool call with its result, so a
session-end sweep can recover what was learned by trial —

```
[Bash] tried `pytest tests/integration` -> failed: could not connect to postgres at :5432
    then `PAYMENTS_PG_PORT=5433 pytest tests/integration` -> worked (after 4 attempts)
```

— and record both the fix *and* the trap. The point is to compress reasoning,
not just text: something that cost four attempts to discover should cost zero
next time.

### New knowledge is reconciled, not appended

A lesson that overlaps something already known is classified before it is
stored: duplicate, refines, contradicts, specialises, or orthogonal. Refinements
fold into the detailed node and patch the compressed ones above it. Orthogonal
knowledge starts a new leaf.

Contradictions are never settled silently — last-write-wins is how a memory
rots. Both lessons stay, both are marked disputed, and a question is raised **at
recall time**, when you are already thinking about that topic:

```
> Unresolved: Is 5434 the new permanent host port mapping for the payments
> postgres container, or was 5433 only temporarily unavailable at the time?
```

`rmc conflicts` lists them; `rmc resolve <id>` settles them.

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

55 tests, no dependencies. Two kinds: structural tests stub the judgements and
assert what the harness does with an answer, and control-flow tests run against
a simulated knowledge world where a task is solved iff the required facts are
present in the lesson text — so compress → fail → descend → rescue genuinely
executes rather than being mocked at the seams.

## Documentation

| Doc | Contents |
|---|---|
| [DESIGN.md](DESIGN.md) | Tree semantics, delta manifests, selection policy, validation protocol, known failure modes |
| [docs/integration.md](docs/integration.md) | How the hooks wire in, and how to run it with Codex |
| [docs/cli.md](docs/cli.md) | Full command reference |

## Status

Research harness, v0.1. It works end to end and is tested, but the store format
is not yet stable and the ambient outcome signals are heuristic — see
[DESIGN.md §8](DESIGN.md#9-failure-modes-this-design-accepts) for what this
design knowingly gets wrong.
