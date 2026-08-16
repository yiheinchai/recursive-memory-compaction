# CLI reference

All commands accept `--agent {claude,codex,mock}` and `--model` where they spawn
an agent. The store is found by walking up from the working directory to the
nearest `.rmc/`, then falling back to `~/.rmc`; `RMC_HOME` overrides both.

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

### `rmc uninstall`
Remove only RMC-tagged hooks. Lessons are left in place.

### `rmc doctor`
Which backends are on PATH, whether a store exists, whether hooks are wired.
Start here when RMC seems inert.

---

## Inspection

### `rmc status`
Families, node counts, episode counts, total vs apex-served tokens, and a table
of each family's apex with its level, cost and success rate.

### `rmc tree [--family F] [-v]`
The tree, indented from apex down to L0. `△` lines are delta manifest entries —
what a compression dropped and which node still holds it. `-v` includes the
first lines of each lesson body.

### `rmc recall --prompt "..." [--json]`
Exactly what would be injected for that prompt, with the family match scores.
The tool for answering "why did it think that?".

Reads stdin if `--prompt` is omitted.

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

### `rmc observe --transcript PATH [--served ids] [--session id]`
Score a finished session and fold the result into the tree: update node stats,
file the episode, and match any correction against delta manifests. No model
call. Normally invoked by the `SessionEnd` hook.

### `rmc learn --transcript PATH [--session id]`
Ask a model whether the session contained a reusable lesson, and mint a level-0
node if so. Deliberately conservative — "nothing captured" is the common and
correct outcome.

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
| `recall.min_match` | `0.12` | similarity a family must clear to be served |
| `recall.max_expansions` | `3` | descents before escalating to L0 |
| `selection.w_delta` | `0.45` | weight on manifest match |
| `selection.w_affinity` | `0.25` | weight on task similarity |
| `selection.w_prior` | `0.20` | weight on the smoothed success rate |
| `selection.w_cost` | `0.10` | penalty on token cost |
| `selection.explore` | `posterior` | `ucb` to keep probing rare branches |
| `compaction.min_successes` | `2` | successful recalls before compressing |
| `compaction.max_ratio` | `0.6` | candidate must be ≤60% of the original |
| `compaction.threshold` | `1.0` | required replay pass-rate |
| `compaction.regression_k` | `5` | episodes replayed per validation |
| `compaction.max_level` | `6` | deepest compression level |
| `learning.min_tool_calls` | `8` | below this a session is ignored |
| `placement.min_similarity` | `0.15` | below this a new lesson is a new leaf, with no model call |
| `placement.consult` | `true` | ask a model how new knowledge relates to old |
| `placement.candidates` | `3` | existing lessons compared in **one** reconciliation call |
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
Reads a JSON payload on stdin. Events: `user-prompt-submit`, `session-end`
(alias `stop`). Always exits 0. No-ops when `RMC_CHILD` or `RMC_DISABLE` is set.

You should not need to call this yourself.
