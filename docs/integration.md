# Integration

How RMC attaches to Claude Code and Codex, and what it writes where.

---

## Claude Code

### As a plugin (recommended)

```
/plugin marketplace add yiheinchai/recursive-memory-compaction
/plugin install rmc@rmc
```

The plugin ships the hooks, the `recursive-memory` skill and the `/rmc` command.
Hook commands resolve through `${CLAUDE_PLUGIN_ROOT}/bin/rmc`, so nothing needs
to be on your `PATH` and no Python environment is required.

### By hand

```bash
rmc install --target claude                 # this repo only
rmc install --target claude --scope user    # every repo
rmc install --target claude --dry-run       # show what it would write
```

This edits `.claude/settings.json` (or `~/.claude/settings.json`) additively.
Each entry RMC adds is tagged `"_rmc": true`, and `rmc uninstall` removes only
tagged entries — hooks you configured yourself are left alone.

### Hooks used

| Event | Command | Purpose | Budget |
|---|---|---|---|
| `UserPromptSubmit` | `rmc hook user-prompt-submit` | serve the lessons that bear on this prompt | 30s; instant while the store fits the budget |
| `Stop` | `rmc hook stop` | after a substantial turn, reflect — off-thread by default | 15s, nothing inline |
| `SessionEnd` | `rmc hook session-end` | parse the transcript, then detach the whole learner | 30s; returns in ~0.1s, always |

`UserPromptSubmit` returns:

```json
{"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                        "additionalContext": "## Recalled lessons (RMC) …"}}
```

Injected text is explicitly framed as prior knowledge rather than user
instruction, so a stale lesson cannot impersonate a request.

### Seeing it happen

An injection is never silent. While the hook runs you see its status line, and
when it injects, the hook returns a `systemMessage` that Claude Code shows you:

```
⋯ Recalling lessons…
⋯ RMC · recalled 2 lessons (312 tok): retry, k8s-deploys   +1 patch
```

Unresolved contradictions are flagged there too. For the full picture — what was
offered to the model, what it decided and why, and the exact text injected — run
`rmc trace --prompt "..."`.

Recall costs a model call on the hot path **only once the store outgrows
`max_pack_tokens`**. Below that everything is served unfiltered, with no call at
all, in milliseconds — there is nothing to choose between. That is deliberate — injecting the
wrong lesson is worse than injecting none, and only a reader can tell the
difference — but it is a real latency cost, so it is cached by prompt and can be
switched off with `rmc config recall.enabled false`.

### Where reflection runs

`learning.nudge_mode` picks the trade:

| Mode | Context the reflector sees | Cost | Interrupts you |
|---|---|---|---|
| `background` (default) | a transcript digest, ~3k tokens | one small call | no |
| `fork` | the whole session, inherited | ~0.1× its tokens via cache reads | no |
| `block` | the agent's own live context | none extra | **yes** — one turn |
| `off` | — | none until session end | no |

`fork` spawns `claude --resume <id> --fork-session` detached. `--fork-session`
allocates a new session id, so the live session is never written to. It is
affordable because prompt-cache reads bill at **0.1×** and the cache keys on
prefix *content* rather than session identity — the fork hits what the live
session just wrote. Claude Code uses a 1-hour cache TTL, which covers the
reflection cooldown comfortably.

It is not the default because 10% of a very large context still exceeds a 3k
digest, and the digest has proven able to identify conceptual corrections rather
than only mechanical ones. Pick `fork` when fidelity matters more than tokens —
the digest necessarily drops nuance, and a conceptual mistake can live in the
nuance.

Every spawned reflector runs with `RMC_CHILD=1`; without it the fork fires these
same hooks and forks itself, forever.

### Failure behaviour

Every hook path is wrapped and returns exit 0. A broken store, an unparseable
transcript or a raised exception all degrade to "inject nothing" rather than
interrupting the session. This is deliberate: a memory system that can break
someone's editor will be uninstalled, and correctly so.

`SessionEnd` gets a stricter rule still. The host is shutting down and will
cancel a hook that is still running, so slow work there is not late — it never
happens. Judging a session takes a model call, so the hook does none of it: it
parses the transcript, decides whether the session is even worth learning from,
and hands everything else to a detached `rmc absorb`. It returns in about 0.1s.

`absorb` runs judge → learn → compress in that order, in one process. They were
briefly three parallel spawns, which raced: compaction is only eligible once the
judgement has recorded the successes that make a node due, so it usually lost.

---

## Codex

Codex's hook schema is less settled than Claude Code's, so the reliable route is
an instruction block appended to `AGENTS.md`:

```bash
rmc install --target codex
```

which appends a marked block telling the agent to run `rmc recall --prompt "…"`
before non-trivial work. If `~/.codex/hooks.json` already exists, a
`UserPromptSubmit` entry is added there as well.

Codex also works as an **execution backend** regardless of how recall is wired —
it is often the better choice for background compaction, since `codex exec
--ephemeral --output-schema` gives native structured output:

```bash
rmc config agent codex
rmc compact --due --agent codex
```

---

## What gets written

```
.rmc/
  config.yaml        settings (see `rmc config`)
  nodes/<family>/    lesson nodes — the tree. Worth committing.
  episodes/          the replay corpus. Worth committing.
  sessions/          per-session scratch. Machine-local.
  events.jsonl       telemetry. Machine-local.
  judge-cache.json   cached judgements. Machine-local.
  background.log     output from detached learning runs (`rmc absorb`).
```

### Two scopes

If `~/.rmc` exists it is layered underneath the project store. Lessons from both
are recalled; new ones are written to the project. This is how a cross-project
principle ("prefer the model's judgement over a similarity score") and a
repo-specific fact ("this suite needs `PAYMENTS_PG_PORT`") each live at the right
scope instead of one being in the wrong place.

```bash
rmc init ~                      # create the global store
RMC_HOME=~/.rmc rmc add "..."   # teach it something that follows you everywhere
```

Editing a global lesson from inside a repo writes back to the global store
rather than forking a local copy that then drifts.

`.rmc/.gitignore` excludes `sessions/` and `events.jsonl` by default, so
committing `.rmc/` shares the lessons and their regression corpus with your team
without the machine-local noise.

### Privacy

Everything written to disk passes through `redact.py` first: API keys, tokens,
private keys, card numbers and `secret=…` assignments are replaced with
`[REDACTED]`, and emails are reduced to their domain. It is deliberately biased
toward over-redaction — a mangled lesson is recoverable, a leaked key is not.

RMC never sends anything anywhere. Model calls go to whichever CLI you already
have configured, with your existing credentials.

---

## Turning it off

```bash
rmc config recall.enabled false       # stop injecting
rmc config compaction.enabled false   # stop compressing
rmc config learning.enabled false     # stop minting lessons
rmc uninstall --target claude         # remove the hooks entirely
```

`RMC_DISABLE=1` in the environment disables everything for a single run, which
is also how spawned child agents avoid recursing.
