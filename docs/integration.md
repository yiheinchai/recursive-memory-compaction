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
| `UserPromptSubmit` | `rmc hook user-prompt-submit` | inject matching apex lessons | 10s, no model call |
| `SessionEnd` | `rmc hook session-end` | score the session, queue learning | 15s, no model call inline |

`UserPromptSubmit` returns:

```json
{"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                        "additionalContext": "## Recalled lessons (RMC) …"}}
```

Injected text is explicitly framed as prior knowledge rather than user
instruction, so a stale lesson cannot impersonate a request.

### Failure behaviour

Every hook path is wrapped and returns exit 0. A broken store, an unparseable
transcript or a raised exception all degrade to "inject nothing" rather than
interrupting the session. This is deliberate: a memory system that can break
someone's editor will be uninstalled, and correctly so.

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
  background.log     output from detached learning runs.
```

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
