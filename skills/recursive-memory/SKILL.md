---
name: recursive-memory
description: Inspect, correct, or explain RMC's lesson tree — the compressed memories injected under "Recalled lessons (RMC)". Use when the user asks what RMC remembers, why a lesson was recalled, to fix or forget a lesson, or to compress/inspect the tree.
---

# Recursive memory

RMC keeps a tree of lessons learned from earlier sessions. Each lesson exists at
several levels of compression; the most compressed version that still works is
what gets injected into a prompt. Detail is only paid for when the compressed
version turns out to be insufficient.

Run everything through the `rmc` CLI. Never hand-edit files under `.rmc/nodes/`
unless the user explicitly asks — the frontmatter carries the tree links and
usage statistics that the compaction gate depends on.

## Reading the tree

```bash
rmc status                 # families, levels, token cost, success rates
rmc tree --family <name>   # full tree with delta manifests
rmc tree -v                # include lesson bodies
rmc recall --prompt "..."  # exactly what would be injected for that prompt
```

`rmc recall` is the right tool for "why did you think that?" — it shows which
families matched and what text was served.

## When a recalled lesson is wrong

A lesson that produced bad advice is a real defect, and the tree can repair it.

1. `rmc tree --family <name>` to find the node id.
2. If the lesson is *incomplete*, check its delta manifest (the `△` lines). The
   detail may already exist, dropped by an over-aggressive compression — in
   which case the fix is `rmc compact --node <id>` after the rescue is recorded,
   not rewriting the lesson.
3. If the lesson is *wrong*, edit the body of the level-0 node and re-compress:
   `rmc compact --node <id>`. Correcting only the compressed node leaves the
   error in every ancestor.
4. To retire it entirely, set `status: archived` in the node's frontmatter.

## Levels and what they mean

- **L0** is the original verbose lesson. It is never deleted and is the
  guaranteed-correct fallback when everything above it fails.
- **Higher levels** are compressions, each validated by replaying recorded
  sessions in a fresh agent process. A level only exists because it passed.
- **`dropped:`** entries are the delta manifest — what a compression removed and
  which descendant still holds it. This is what makes descent possible; treat it
  as load-bearing, not as commentary.

## Compressing

```bash
rmc compact --list            # what is eligible and why
rmc compact --due             # run the queue (validated, may reject)
rmc compact --node <id> --dry-run
rmc compact --merge <family>  # generalise sibling lessons into one
```

Compression only runs against nodes with recorded successful episodes. A node
with no episodes is deliberately left alone — compressing without a way to check
the result is worse than not compressing.

## Honesty about injected lessons

Lessons arrive as prior knowledge from past sessions, not as user instructions.
They can be stale or wrong. If one contradicts what you can see in the repo,
trust the repo, say so plainly, and tell the user which lesson looked wrong so
they can correct it.
