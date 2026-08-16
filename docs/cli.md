# CLI reference

All commands accept `--agent {claude,codex,mock}` and `--model` where they spawn
an agent. The store is found by walking up from the working directory to the nearest
`.rmc/`. If `~/.rmc` also exists it is layered underneath as a **global** store:
lessons from both are recalled, and new lessons are written to the project one.
Editing a global lesson writes back to it rather than forking a local copy.
`RMC_HOME` overrides the lookup entirely — use it to target the global store
directly, e.g. `RMC_HOME=~/.rmc rmc add ...` for something that should follow
you across every repo.

---

## Setup

### `rmc init [path]`
Create a store. `--force` re-creates missing subdirectories, `--agent` sets the
default backend.

### `rmc install`
Wire RMC into the host agent(s).

| Flag | Meaning |
|---|---|
| `--target claude\|codex` | repeatable; defaults to `claude` |
| `--scope project\|user` | this repo (default) or globally |
| `--dry-run` | print what would be written, write nothing |

Also puts the `rmc` command on PATH, by symlinking `bin/rmc` into
`~/.local/bin`. Hooks never need this — they invoke the package by absolute
path — so without it RMC runs fine while `rmc status` reports command not found.
`--no-link` skips it.

Bootstrap it from a clone with `./bin/rmc install`; every suggestion this
command prints uses an absolute path, because advice that begins "run `rmc`" is
no use to someone who does not have it.

### `rmc uninstall`
Remove only RMC-tagged hooks. Lessons are left in place.

### `rmc report [--about "..."] [--expected "..."] [--days N]`

Writes a redacted defect report to `.rmc/reports/` and prints it, along with
the `gh issue create` command that would file it. **It sends nothing.** RMC
makes no network calls and that is a guarantee, so the transport is your own
`gh`, run deliberately.

It contains counts, timings and non-default settings — the things that localise
a defect to one stage — plus whatever you write in `--about`. It never contains
lesson text, prompts, transcripts or paths from your machine, and everything
passes through the redactor on the way out.

Reflectors are told to run this when they find a defect in RMC itself, and then
to *ask* whether you want it filed. They never file it themselves.

### `rmc doctor`
Which backends are on PATH, whether a store exists, whether hooks are wired.
Start here when RMC seems inert.

---

## Inspection

### `rmc status`
Families, node counts, episode counts, total vs apex-served tokens, and a table
of each family's apex with its level, cost and success rate.

### `rmc tree [--family F] [-v] [--recent] [--limit N]`
The tree, indented from apex down to L0. `△` lines are delta manifest entries —
what a compression dropped and which node still holds it. `-v` includes the
first lines of each lesson body.

`--recent` drops the family grouping and lists newest first, which is the view
that answers "did what I just taught it actually land?" — the default grouping
cannot, since a new lesson sorts wherever its family happens to sit.

Every line carries an age.

### `rmc recall --prompt "..." [--json]`
Exactly what would be injected for that prompt, and the model's stated reason for
each. The tool for answering "why did it think that?" — the answer is a sentence,
not a score.

Reads stdin if `--prompt` is omitted.

### `rmc trace --prompt "..." [--after TRANSCRIPT]`
The agent's-eye view. Walks every stage of a recall and prints the result of
each: the apex lessons put in front of the model, its verdict and reason for
every one (including the branches it judged irrelevant and therefore never
opened), the **verbatim** block that lands in the agent's context, what you see
in Claude Code while it happens, and how the model's turn then begins.

`--after` continues into the other half: the facts parsed out of a finished
session, and what the model made of them — outcome, whether you had to steer,
and what was worked out by trial.

Use it when you want to know exactly what RMC is doing to your prompts. It edits
what the model sees, and that should never be something you take on trust.

### `rmc conflicts [--family F]`
Lessons that contradict each other, with the question that would settle each.
These are also raised inside the recall pack, so you normally meet them while
working rather than by running this.

### `rmc resolve <node-id> [--drop]`
Settle a conflict: keep this lesson (default) or archive it. Clears the disputed
state on that node.

### `rmc events [--kind K] [--limit N]`
Raw telemetry as JSONL. Useful kinds: `inject`, `observe`, `rescue`, `mint`,
`placement`, `conflict`, `conflict-resolved`, `compaction`, `merge`, `repair`,
`error`.

---

## The loop

### `rmc add [body] [--family F] [--title T] [--tags a,b]`
Teach RMC something **now**, without waiting for the session to end. Reads stdin
if no body is given. The lesson is reconciled against what is already known
before being stored, so it may be folded into an existing lesson, set alongside
one, or reported as a contradiction with the question that would settle it.

This is the live path. The transcript sweep at session end is a safety net for
what nobody noticed in the moment; this is for the moment itself, and the lesson
is available to the very next prompt in the same conversation.

`--no-reconcile` stores it without the consistency check.

Deciding and writing happen under a lock, so concurrent reflectors cannot both
conclude "new" about the same lesson. A writer waits for the lock rather than
skipping.

### `rmc observe --transcript PATH [--served ids] [--session id]`
Judge a finished session and fold the result into the tree: update node stats,
file the episode, and work out which dropped detail any correction was about.
Costs one judgement, skipped entirely for sessions too small to teach anything.
Normally invoked by the `SessionEnd` hook.

### `rmc learn --transcript PATH [--session id]`
Ask a model whether the session contained a reusable lesson, and mint a level-0
node if so. Deliberately conservative — "nothing captured" is the common and
correct outcome.

### `rmc dream`
Whole-store consolidation: backfill gists, then merge lessons that keep being
used together. Runs automatically from `absorb` at most once per
`dream.interval_s` and only with `dream.min_new_episodes` of new evidence.

| Flag | Meaning |
|---|---|
| `--due` | say whether a dream is due, and why not |
| `--list` | merge candidates from co-use, change nothing |
| `--log [all]` | read the last dream's report, or list every one |
| `--limit N` | merge groups attempted per pass (default 2) |
| `--dry-run` | generate and validate, write nothing |

Every pass writes a report to `.rmc/dreams/<timestamp>.md` recording what was
examined, what merged, what was refused and why, and the before/after of nodes,
apexes and tokens served at apex.

### `rmc compact`
Compress lessons and regression-test the result.

| Flag | Meaning |
|---|---|
| `--list` | show what is eligible and why, run nothing |
| `--due` | process the queue (default) |
| `--node ID` | compress one specific node |
| `--merge FAMILY` | generalise sibling lessons into one |
| `--limit N` | how many to process (default 1) |
| `--dry-run` | generate and validate, but do not write |

A node is eligible when it is an active apex, below `compaction.max_level`, has
at least `compaction.min_successes` successful recalls, is past its cooldown, and
**has recorded episodes to validate against**. A node with no episodes is left
alone on purpose: compressing with no way to check the result is worse than not
compressing.

Rejection is normal and informative — the failing episodes become `preserve:`
hints for the next attempt.

---

## Configuration

### `rmc config [key] [value]`
No arguments dumps everything. One argument reads a dotted key. Two arguments
set it.

Any key can be overridden per-run by an environment variable:
`recall.max_pack_tokens` → `RMC_RECALL_MAX_PACK_TOKENS`.

| Key | Default | Meaning |
|---|---|---|
| `agent` | `claude` | default backend |
| `recall.enabled` | `true` | inject lessons at all |
| `recall.strategy` | `delta-patch` | `delta-patch`, `delta-jump`, or `stepwise` |
| `recall.max_pack_tokens` | `1200` | ceiling on injected context |
| `recall.max_families` | `3` | lesson families served per prompt |
| `recall.always_judge` | `false` | filter for relevance even when the whole store fits in the budget (it is served unfiltered otherwise — there is nothing to choose) |
| `recall.timeout_s` | `20` | bound on the routing call, kept under the hook deadline |
| `recall.judge_calls` | `2` | model calls the relevance walk may spend |
| `recall.max_depth` | `2` | how far down the tree the walk may look |
| `recall.max_expansions` | `3` | descents before escalating to L0 |
| `selection.w_judge` | `0.60` | weight on the model's usefulness ranking |
| `selection.w_prior` | `0.28` | weight on the observed rescue rate |
| `selection.w_cost` | `0.12` | penalty on token cost |
| `selection.explore` | `posterior` | `ucb` to keep probing rare branches |
| `compaction.min_successes` | `2` | successful recalls before compressing |
| `compaction.max_ratio` | `0.75` | candidate must be ≤75% of the original |
| `compaction.threshold` | `1.0` | required replay pass-rate |
| `compaction.regression_k` | `5` | episodes replayed per validation |
| `compaction.max_level` | `6` | deepest compression level |
| `learning.min_tool_calls` | `8` | below this a session is ignored |
| `learning.nudge_enabled` | `true` | reflect after a substantial turn |
| `learning.nudge_mode` | `background` | `background` — a detached process reflects on the transcript digest; `fork` — reflect inside a fork of the live session, full context, ~0.1× tokens via cache reads; `block` — interrupt the agent to reflect in place; `off` — wait for session end |
| `learning.nudge_after_tool_calls` | `12` | activity that makes a turn worth a thought |
| `learning.nudge_after_turns` | `3` | human turns that do the same |
| `learning.min_surprises` | `2` | failed tool calls that also do |
| `learning.nudge_cooldown_s` | `900` | minimum gap between asks |
| `learning.nudge_backoff_after` | `3` | consecutive fruitless nudges before backing off |
| `dream.enabled` | `true` | run whole-store consolidation at all |
| `dream.interval_s` | `86400` | minimum gap between dreams |
| `dream.min_new_episodes` | `3` | new multi-lesson episodes required before dreaming |
| `dream.limit` | `2` | merge groups attempted per pass |
| `placement.consult` | `true` | ask a model how new knowledge relates to old |
| `placement.judge_calls` | `2` | model calls the relatedness walk may spend |
| `placement.max_depth` | `2` | how far down the walk may look |
| `placement.surface_conflicts` | `true` | raise unresolved contradictions during recall |
| `signals.min_confidence` | `0.5` | floor for acting on an outcome |
| `privacy.redact` | `true` | scrub secrets before writing |
| `limits.agent_timeout_s` | `180` | per spawned agent call |

### Ablations

`recall.strategy` exists so the descent policy can be measured rather than
asserted:

```bash
rmc config recall.strategy stepwise      # baseline: walk children, ignore deltas
rmc config recall.strategy delta-jump    # replace apex with the holder node
rmc config recall.strategy delta-patch   # default: apex + matched claims only
```

---

## Hooks

### `rmc hook <event>`
Reads a JSON payload on stdin. Events: `user-prompt-submit`, `stop` (per turn),
`session-end` (at teardown). Always exits 0. No-ops when `RMC_CHILD` or
`RMC_DISABLE` is set.

You should not need to call this yourself.
