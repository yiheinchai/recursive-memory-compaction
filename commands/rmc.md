---
description: Inspect the RMC lesson tree — status, tree, or what a prompt would recall
argument-hint: "[status | tree [family] | recall <prompt> | compact]"
allowed-tools: Bash(rmc:*)
---

Run the RMC inspection the user asked for, then explain the result in plain
language rather than pasting raw output.

Argument given: `$ARGUMENTS`

- empty or `status` → `rmc status`
- `tree` (optionally a family) → `rmc tree` / `rmc tree --family <family>`
- `recall <prompt>` → `rmc recall --prompt "<prompt>"`
- `compact` → `rmc compact --list` first, and only run `rmc compact --due` if the
  user confirms; compaction spawns agent processes and costs tokens.

When reporting, say what it means: which lessons are being served, how much
context they cost, and whether anything is failing often enough to need
attention (a node whose success rate is below ~50% is a candidate for repair).
