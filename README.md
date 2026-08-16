# RMC — Recursive Memory Compaction

A continual-learning harness for **Claude Code** and **Codex**. Lessons are
learned once in verbose form, then **recursively compressed each time they are
actually used**, producing a graph of progressively more abstract memories. At
retrieval time you load the most compressed node that still works, and only pay
for detail when the abstraction fails.

You do not drive it. You work normally, and it runs in hooks.

**[Read the docs →](https://yiheinchai.com/rmc)**

---

## Contents

| | |
|---|---|
| [The thesis](#the-thesis) | why compression is driven by usage |
| [The loop](#the-loop) | what happens, when, and what it costs |
| [The governing rule](#the-governing-rule) | harness structures, model judges |
| [1. Recall](#1-recall) | choosing what to inject |
| [2. Reflection](#2-reflection) | noticing there was something to learn |
| [3. Attribution](#3-attribution) | which lessons actually mattered |
| [4. Consolidation](#4-consolidation) | where a new lesson goes |
| [5. Compression](#5-compression) | earning a smaller form |
| [6. Descent](#6-descent) | recovering detail when the short form fails |
| [7. Dreaming](#7-dreaming) | whole-store consolidation |
| [Data model](#data-model) | nodes, episodes, the DAG, the store |
| [Install](#install) | plugin or clone |
| [Inspecting it](#inspecting-it) | status, tree, recall, trace |
| [Not built](#not-built-yet) | honest gaps |

---

## The thesis

The standard learning cycle produces one artifact and stops:

```
task + steering  ->  correct output  ->  reflection  ->  lesson
```

RMC keeps going. Every time a lesson is *used* and the work succeeds, that is
evidence the lesson contained slack, so it earns a compression attempt:

```
task + lesson(L0)  -> used, worked -> reflect -> lesson(L1)   ~75% tokens
task + lesson(L1)  -> used, worked -> reflect -> lesson(L2)   ~55%
task + lesson(L2)  -> used, worked -> reflect -> lesson(L3)   ~40%
```

And lessons that are repeatedly useful *together* earn a shared parent, which is
a different kind of abstraction — generalisation across lessons rather than
compression within one.

**The more a memory is used, the cheaper and more general it becomes.** That is
the whole idea: usage drives abstraction.

---

## The loop

| When | What happens | Model calls |
|---|---|---|
| you submit a prompt | matching lessons are injected as context | 0–1, cached |
| context is about to compact | the record of what you were shown is cleared | 0 |
| a substantial turn ends | a reflector runs off-thread: did anything teach us something, and which lessons actually mattered | 1–2, detached |
| you teach it something | `rmc add` records it immediately, reconciled against what is known | 1–2 |
| the session ends | the transcript is judged, lessons minted, compression attempted | 2–4, detached |
| once a day, if there is new evidence | `dream` merges lessons that keep being used together | 1–2 per group |

Everything expensive is **detached** — spawned as a separate process that
outlives the hook. Spawned agents get `RMC_CHILD=1`, which makes RMC's own hooks
no-op; without it a reflector would trigger reflectors forever.

---

## The governing rule

> **The harness owns structure. The model owns meaning.**

The harness owns the graph, the traversal, budgets, caches, the schemas answers
must fit, and *whether to ask at all*. The model owns every question about
meaning, each behind a JSON schema, cached so nothing is judged twice.

There are exactly five semantic calls in the system (`rmc/judge.py`):

| Call | Question |
|---|---|
| `relevance` | which remembered lessons bear on this prompt? |
| `related` | which existing lessons cover the same subject as this new one? |
| `scope` | does this lesson belong to this repo, or everywhere? |
| `assess` | how did this session go, and which lessons actually bore on it? |
| `rank_repairs` | which dropped detail explains this failure? |

Plus four in the compression path: compress, merge, replay-probe, and the replay
judge.

**What stays in code, and why it is not a violation:** counting an *observed
outcome* is evidence — a success rate, a rescue count, how often two lessons
were used together on work that succeeded. Computing how *alike* two texts are
is a proxy standing in for a judgement, and must be a model call. The test is
what question the number answers, not where it came from.

This distinction was learned the hard way. Retrieval was originally Jaccard
similarity, contradictions were found with a `KEY=value` regex, and session
outcomes came from phrase banks with hand-tuned weights. All of it looked
structural and all of it was a lookup table wearing a judgement's clothes.

---

## 1. Recall

Runs on `UserPromptSubmit`. Produces the block injected above your prompt.

### Two structural gates first

Neither is a judgement, so neither costs a call:

1. **Empty store** — nothing to recall, nothing to ask.
2. **Everything fits** — if the whole store is under `recall.max_pack_tokens`,
   there is no *choice* to make, so all of it is served unfiltered. Relevance
   filtering switches on exactly when the store outgrows the budget.

The second gate matters more than it sounds. Without it, recall spent a model
call deciding which two of two lessons to load, and blew the hook's deadline
doing it. With it, recall returns in ~0.1s until the store is genuinely large.

### The walk

Once filtering is needed, the model decides — walking **abstract to concrete**:

1. Start from the **apexes** (nodes with no parent). They are the most
   compressed things in the store, so the whole top level fits in one question.
2. Each candidate is rendered as one line: id, title, and a **gist** — never the
   body. Sending 700 characters per lesson to *choose* which lessons to send is
   the scaling bug that eats the context it was meant to protect.
3. The model returns `relevant` / `maybe` / `unrelated` per candidate, plus
   `descend` when a summary is too abstract to judge from.
4. Only `descend` branches are opened. A branch judged unrelated is never walked.
5. Bounded by `recall.judge_calls` and `recall.max_depth`. Anything unresolved
   when the budget runs out is served rather than dropped.

### Three tiers of re-injection

A lesson already in context should not be paid for twice — but *present* and
*still attended to* are different things, since attention over long context
decays and an early injection ends up in the middle where models attend least.

| State | Action |
|---|---|
| never served | full body |
| served < `stays_fresh_turns` ago (8) | **skipped** — still fresh |
| served longer ago | **refreshed** with its one-line gist, ~20 tokens |

`PreCompact` clears the whole record, because compaction may have deleted the
text being tracked.

### Also in the pack

- **Sticky patches** — dropped claims that previously rescued this node get
  re-attached, rather than waiting for the same failure to recur.
- **Unresolved conflicts** — surfaced here, at the moment you are thinking about
  the topic.

### What you see

```
⋯ Recalling lessons…                                    (while the hook runs)
⋯ RMC · 2 lessons · 512 tok — Retry policy, Cache TTLs   (after injection)
⋯ RMC · 1 lesson · 118 tok — Retry policy · 1 refreshed, 2 already in context
```

The counts distinguish three different things: a lesson injected in full, a
one-line refresher for one that is present but stale, and one skipped entirely
because it is still fresh in context. Only the first is a lesson arriving.

An injection is never silent. A memory system that edits your prompts without
saying so is one you cannot notice going wrong.

---

## 2. Reflection

Runs on `Stop`, per turn. Its job is to **schedule attention**, not to judge.

The occasion is structural — "did this turn have enough substance to be worth a
thought" — measured by `nudge_after_tool_calls` (12), `nudge_after_turns` (3), or
`min_surprises` (2 failed tool calls), subject to `nudge_cooldown_s`.

**The occasion is deliberately not "something failed."** An earlier version
triggered on failed tool calls, which sounds structural but is a mechanical
proxy for a semantic question, and it biases hard toward the cheapest mistakes.
Conceptual errors — believing a system works one way when it does not — exit
zero and pass their tests. During RMC's own development the worst mistake ran
green the whole way.

### Where it runs — `learning.nudge_mode`

| Mode | Reflector sees | Cost | Interrupts you |
|---|---|---|---|
| `background` (default) | a transcript digest, ~3k tokens | one small call | no |
| `fork` | the whole session, inherited | ~0.1× its tokens | no |
| `block` | the agent's own live context | none extra | **yes**, one turn |
| `off` | — | none until session end | no |

`fork` spawns `claude --resume <id> --fork-session` detached. `--fork-session`
allocates a new id, so the live session is never written to. It is affordable
because prompt-cache reads bill at **0.1×** and the cache keys on prefix
*content*, not session identity — the fork hits what the live session just
wrote.

If nudges keep producing nothing, the cooldown backs off automatically
(`nudge_backoff_after`).

---

## 3. Attribution

The question that drives everything downstream: **which served lessons actually
bore on the work?**

Serving is a retrieval decision; using is an outcome. Conflating them breaks two
things at once — an irrelevant lesson that happened to be injected accrues a
record of usefulness it never earned, and serving ten lessons manufactures
forty-five "co-use" pairs out of a single retrieval decision.

Two sources, best first:

1. **The in-session reflector.** The fork holds the real conversation, so it can
   see a principle being applied and not merely a command being run. It reports
   with `rmc used --session <id> --used <ids> --unused <ids>`.
2. **The digest judge.** `assess` is shown the served lessons and the session
   digest and answers `lessons_used` per lesson. Weaker: influence on *reasoning*
   is nearly invisible in a digest of commands, so it under-credits principles.
   The digest now includes the agent's own reasoning for exactly this reason.

Either way the prompt is strict and defaults to false: *being on-topic is not
being used; being read and found irrelevant is not being used; having been
served a lesson and then done the opposite means it was not used.*

**A lesson served but unused is scored neither success nor failure.** It has
three possible causes and only the first is a retrieval problem:

| Cause | Meaning |
|---|---|
| irrelevant | recall over-served |
| redundant | the agent knew it anyway |
| relevant and ignored | the lesson isn't landing — a salience failure |

`rmc status` reports `precision` (used ÷ served).

---

## 4. Consolidation

New knowledge is reconciled with old before it is stored, never appended blindly.

### Scope, decided first

Where a lesson lands decides whether it can ever be found again. A lesson about
a vendor API filed under one repo is invisible from every other one, so nothing
downstream can rescue it — not recall, not co-use, not dreaming.

So `scope` asks: does this depend on *this repository*, or would it be true for
anyone using these tools anywhere? Global lessons go to `~/.rmc`, project
lessons stay local. The error is asymmetric: a global lesson filed locally is
**lost**, a project lesson filed globally is merely noise.

### Then the relation

`related` walks the graph for lessons on the same subject, and `reconcile`
classifies:

| Relation | Action |
|---|---|
| `duplicate` | nothing stored; the hit is recorded |
| `refines` | fold into the matched lesson's own lineage, then patch its ancestors |
| `contradicts` | keep both, mark both `disputed`, attach a question |
| `specialises` | attach as a sibling; a merge may later generalise both |
| `orthogonal` | new family |

**Refinement must reach the apex.** Folding detail into a level-0 node leaves
every ancestor missing it. Rather than invalidating working compressions, the
detail is registered as a rescue on each ancestor, so recall re-attaches it
immediately and `repair` folds it in permanently once it proves necessary.

**Contradictions are never resolved silently.** Last-write-wins is how a memory
rots. Both lessons stay servable, and the question is raised at recall time:

```
> Unresolved: Is 5434 the new permanent host port mapping, or was 5433
> only temporarily unavailable at the time?
```

`rmc conflicts` lists them, `rmc resolve <id> [--drop]` settles them.

---

## 5. Compression

A node becomes eligible when it is an active apex, below `max_level` (6), has at
least `min_successes` (2) recalls **that were attributed as used**, is past its
cooldown, and **has recorded episodes to validate against**. A node with no
episodes is left alone: compressing with no way to check the result is worse
than not compressing.

### The manifest is mandatory

Every compression must declare what it removed, as discrete claims:

```yaml
dropped:
  - claim: "Backoff constants are 100ms / 400ms / 1.6s, jitter ±25%"
    kind: parameter
    holder: n_7f2a
```

A compression that does not record its losses cannot be descended, so the lost
detail is simply gone. A candidate that shrank materially while declaring
nothing is rejected as under-reported.

Deltas are **inherited** — an apex knows about detail several levels below it
without holding the text, which is what makes delta-jumping possible.

### Validation

The check from the original sketch, hardened:

```
R = sample(regression episodes over the node's whole subtree, k=5)
for each: fresh agent process, candidate lesson + the original prompt
accept iff pass_rate >= 1.0  and  tokens <= 0.75 × original
```

- **Fresh process** — otherwise the main agent's memory of the verbose lesson
  leaks in and every compression looks successful.
- **Subtree-wide** — validating only on the triggering episode is how you get a
  beautifully compressed, useless tree.
- **Probe form** — replay asks *how would you approach this*, not *do the work*.
  Asking for the work meant judging scaffolding completeness and truncation
  artefacts rather than the lesson.
- **An unreadable judge is a failure**, never a pass, so an infrastructure blip
  cannot promote a bad compression.
- **Rejections are informative** — failing episodes become `preserve:` hints for
  the next attempt, so the compressor converges instead of thrashing.

### Merging

Two or more lessons become one abstract parent covering all of them. Merges may
span families, and a merge that would swallow one of its own ancestors is
refused — that would make the graph cyclic and every upward walk
non-terminating.

### Repair

A delta that repeatedly rescues the same node is proof the compression cut too
deep, so it is folded permanently back into the body and dropped from the
manifest. The graph heals where it was over-cut.

---

## 6. Descent

*When the compressed lesson fails, which child do you go to?*

Compression normally destroys the information you would need to invert it, which
is why the manifest is mandatory. Descent is then a ranking problem over dropped
claims, not a search over the graph:

```
score = w_judge · model_usefulness + w_prior · rescue_rate − w_cost · tokens
         (0.60)                       (0.28)                  (0.12)
```

- **`model_usefulness`** — one call ranks every candidate 0–1: *how likely is it
  that this detail being absent caused this failure*. A judgement, so the model
  makes it. "Parse the body, not the status code" is the fix for "treated HTTP
  200 as success" while sharing none of its words.
- **`prior`** — Laplace-smoothed rate at which this node has actually rescued
  failures. Observed outcome, so it stays in code. Makes descent a contextual
  bandit over the graph.
- **`cost`** — token count. A measurement.

With no judge available the first term is simply absent and ranking falls back
to "try what has worked, cheapest first" — degrading honestly rather than
substituting a metric that looks like judgement.

Strategies (`recall.strategy`): `delta-patch` (apex + matched claims, default),
`delta-jump` (swap in the holder node), `stepwise` (walk children, ignore the
manifest — the ablation baseline).

Escalation: exhaust candidates → load the level-0 node, which is never deleted.
If that fails too, it is a genuine knowledge gap, not a compression bug.

---

## 7. Dreaming

`rmc dream` — whole-store consolidation, independent of any session. Every other
path reacts to the session in front of it; this one steps back.

It grows abstraction from **co-use**: lessons repeatedly *used together* on work
that succeeded. That signal is the right one because the pair most worth merging
often shares no surface features — a lesson about deployment tooling and one
about cache economics can belong under one abstraction purely because they are
needed together, and no similarity metric will place them near each other.

Counting co-occurrence is evidence and stays in code. Whether a group shares a
generalisable idea is the model's call. Threshold is 2 occurrences; one is
coincidence. Failed sessions do not count.

### When it runs

Dreaming is not a reaction to a session, so no session event is its natural
occasion. It runs on **elapsed time gated by new evidence**: at most once per
`dream.interval_s` (24h), and only when at least `dream.min_new_episodes` (3)
new successful multi-lesson episodes have accumulated since the last pass.
Dreaming over an unchanged store only re-asks questions already answered and
cached.

Both halves of that gate are structural — a clock reading and a count — so
neither belongs to the model. It is invoked from `absorb`, the detached
end-of-session pipeline, because that is the one place that already runs off the
main thread, already holds a lock, and is already allowed to spend calls. Most
sessions skip it in a line.

### Seeing what it did

A dream rewrites the store while nobody is watching, which is exactly the
situation that needs a record. Every pass writes a report to `.rmc/dreams/`:

```
# dream 2026-08-16T16:12:41Z

| | before | after |
|---|---|---|
| nodes | 12 | 13 |
| apexes | 9 | 8 |
| tokens served at apex | 2088 | 1640 |

Examined 3 co-use group(s); wrote 2 gist(s).

## merged
- n_cache+n_deploy (co-used 4x) -> n_9f10

## refused
- n_a+n_b (co-used 3x): merge regression pass-rate 67% < 100%
```

`tokens served at apex` is the number that matters — it is what recall pays on
every prompt.

```bash
rmc dream --due        # is a dream due, and why not
rmc dream --list       # merge candidates from co-use, change nothing
rmc dream --log        # the last dream's report
rmc dream --log all    # every dream on record
rmc dream --dry-run    # generate and validate, write nothing
```

---

## Data model

### Node — one lesson at one level of abstraction

```yaml
id: n_7f2a91
family: retry
title: Retrying flaky services
gist: Retry idempotent remote calls; S3 needs body parsing.   # the routing view
level: 3
status: active            # active | superseded | demoted | disputed | archived
origin: compression       # reflection | compression | merge | manual
conflict: ""              # an unresolved contradiction, surfaced at recall
derived_from: [n_c41b, n_9de1]   # points DOWN, toward detail
parents: [n_aa01]                # points UP, toward abstraction
covers_tasks: [e_4f1a, e_9c22]   # the regression set
dropped: [...]                   # the delta manifest
preserve: [...]                  # hints from rejected compressions
stats: {attempts, successes, failures, expansions, rescues, last_used}
```

**Both edges are lists, so this is a DAG, not a tree.** A lesson can be
abstracted in more than one direction: compressed vertically into a terser form
of itself, *and* merged sideways into a shared generalisation. While the parent
link was a single field, the second silently destroyed the first.

### Episode — a replayable regression test

```yaml
id: e_4f1a
prompt: "add retry to the http client"
outcome: success
served: [n_7f2a, n_c41b]   # what was injected
used:   [n_7f2a]           # what actually bore on the work
accepted_summary: "..."    # what the agent ended up doing
```

This is the **ambient oracle**. Nobody writes YAML oracles for their own repo,
so instead RMC records what happened when work was accepted, and later asks
whether a compressed lesson still reproduces it.

### Store layout

```
.rmc/
  config.yaml          settings
  nodes/<family>/*.md  the graph. Worth committing.
  episodes/*.json      the replay corpus. Worth committing.
  sessions/*.json      per-session scratch. Machine-local.
  events.jsonl         telemetry. Machine-local.
  judge-cache.json     cached judgements. Machine-local.
```

**Two scopes.** If `~/.rmc` exists it is layered under the project store: both
are recalled, new lessons are written to the project one, and editing a global
lesson writes back to it rather than forking a local copy that drifts.

**Privacy.** Everything passes through `redact.py` before touching disk — API
keys, tokens, private keys, card numbers, `secret=…` assignments. Biased toward
over-redaction: a mangled lesson is recoverable, a leaked key is not. RMC never
sends anything anywhere; model calls go through whichever CLI you already have.

---

## Install

Python 3.10+ and at least one of `claude` / `codex`. No third-party
dependencies.

```bash
uv tool install rmc-memory
rmc install --scope user
```

That is the whole thing. `rmc install` wires the hooks and RMC runs in every
repo you open — you do not drive it, you just work. Drop `--scope user` to
limit it to the project you run it from.

Try it without installing anything: `uvx rmc-memory doctor`. Without uv:
`pipx install rmc-memory`, or `pip install --user rmc-memory`.

> The distribution is `rmc-memory` because `rmc` was taken on PyPI. The command
> it installs, and the package you import, are both `rmc`.

**As a Claude Code plugin**, if you would rather not install anything yourself:

```
/plugin marketplace add yiheinchai/rmc
/plugin install rmc@rmc
```

**Codex** gets an `AGENTS.md` block instructing the agent to call `rmc recall`
itself, since Codex's hook schema is less settled. Codex also works as an
execution backend for any RMC install (`rmc config agent codex`).

### Working on RMC itself

```bash
git clone https://github.com/yiheinchai/rmc && cd rmc
python3 -m unittest discover -s tests
./bin/rmc install --scope user
```

`./bin/rmc` runs from a clone with no virtualenv, since the package is
stdlib-only. Installing from a clone also symlinks the command into
`~/.local/bin`; pass `--no-link` to skip that. Note the scope — without
`--scope user` the hooks land in the clone, and every repo you actually work in
gets nothing.

`rmc uninstall` removes only what RMC added, and `rmc doctor` reports what is
wired where.

---

## Inspecting it

```bash
rmc status                     # families, levels, precision, capture stats
rmc tree --family retry        # the graph, with delta manifests
rmc recall --prompt "..."      # what would be injected, and the model's reason
rmc trace --prompt "..."       # every stage, ending in the verbatim block
rmc dream --list               # merge candidates from co-use
rmc conflicts                  # unresolved contradictions
rmc doctor                     # backends, store, hook wiring
rmc report --about "..."       # a redacted defect report; sends nothing
rmc events --kind inject       # raw telemetry
```

`rmc trace` is the agent's-eye view: what was offered to the model, its verdict
and reason for each candidate, the exact injected text, and what you see in
Claude Code. Use it when you want to know precisely what RMC is doing to your
prompts.

Full command reference: [docs/cli.md](docs/cli.md). Wiring details:
[docs/integration.md](docs/integration.md). Design rationale and known failure
modes: [DESIGN.md](DESIGN.md).

---

## Not built yet

Stated plainly, because a harness that hides its gaps is useless:

- **Progressive disclosure.** Recall is one pass. A lesson only recognisable as
  relevant *after* reading another one cannot currently be found — you ask about
  a stuck deploy, retrieve "deploys use Argo Rollouts", and the lesson about
  Argo's dedupe key stays invisible because your prompt never said "Argo". The
  fix is a second pass with the first results in hand.
- **Agent-driven search.** Lessons are plain markdown on disk, so
  `grep -r "argo" .rmc/nodes/` already works — the agent is just never told it
  can.
- **Held-out evaluation.** Compressions are validated against the episodes they
  were tested on. In-sample. There is no measurement of whether a compressed
  lesson generalises to unseen work.
- **Codex hooks.** Instruction block only, not true ambient operation.

## Known limits

- **Attribution is a counterfactual the model cannot run.** "Would this have
  gone differently without the lesson?" is inferred, not measured. Reliable for
  concrete lessons, softer for dispositional ones.
- **Recall costs a call per prompt once the store outgrows the budget.**
  Cached, and `recall.enabled: false` opts out.
- **Reconciliation only compares against apexes**, so a contradiction with
  detail deep in a subtree is missed until descent surfaces it.
- **Merges can over-generalise.** The regression set is a mitigation, not a proof.
- **Compression is not monotone.** A level-4 node is only known to be better on
  the episodes it was validated against.

## Tests

```bash
python3 -m unittest discover -s tests
```

97 tests, no dependencies. Two kinds: *structural* tests stub the judgements and
assert what the harness does with an answer; *control-flow* tests run against a
simulated knowledge world where a task is solved iff the required facts are in
the lesson text, so compress → fail → descend → rescue genuinely executes.
