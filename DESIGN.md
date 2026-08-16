# RMC Design

Version 0.1. This document defines the data model, the descent/selection policy,
and the compression validation protocol.

---

## 1. Vocabulary

| Term | Meaning |
|---|---|
| **Lesson** | A reusable piece of procedural knowledge, stored as one markdown file with frontmatter. |
| **Node** | A lesson at one specific abstraction level. |
| **Family** | A set of nodes connected by compression/merge edges; the unit you traverse. |
| **Level** | Integer. `0` = original verbose lesson. Higher = more compressed/abstract. |
| **Apex** | The highest-level node of a family. Loaded first at recall time. |
| **Delta manifest** | The list of claims a compression removed, attributed to the descendants that still hold them. |
| **Regression set** | Every task a node (and its subtree) has been validated against. |
| **Oracle** | The check that decides whether an output is correct, for a given task. |

Edge directions are named explicitly to avoid the usual confusion, since
"parent" is ambiguous when the tree grows upward from detail to abstraction:

- `compressed_into: <id>` — points **up**, toward less detail.
- `derived_from: [<id>, ...]` — points **down**, toward more detail. Plural,
  because a compression may merge several sibling lessons into one abstraction.

Recall walks *down* `derived_from`. Learning grows *up* via `compressed_into`.

---

## 2. Node format

Nodes are markdown files under `.rmc/nodes/<family>/<id>.md`. Frontmatter is
YAML; the body is the lesson text that actually gets injected into a prompt.

```markdown
---
id: n_7f2a91
family: retry
level: 3
created: 2026-08-16T09:41:19Z
tokens: 87
derived_from: [n_c41b02, n_9de110]
compressed_into: null
covers_tasks: [retry-http, retry-s3, retry-db-write]
dropped:
  - claim: "Backoff constants are 100ms / 400ms / 1.6s with ±25% jitter"
    kind: parameter
    holder: n_c41b02
  - claim: "S3 may return HTTP 200 with an error body; parse the body"
    kind: edge-case
    holder: n_9de110
stats:
  attempts: 12
  successes: 10
  expansions: 2
  last_used: 2026-08-16T09:41:19Z
status: active
---

Retry idempotent operations with jittered exponential backoff. Cap total
elapsed time by the caller's deadline, not by attempt count. Never retry a
non-idempotent write without a dedupe key.
```

`kind` is drawn from a closed vocabulary, because the diagnosis matcher keys off
it (§4.2):

`parameter` · `example` · `precondition` · `edge-case` · `rationale` ·
`counter-example` · `procedure-step` · `naming` · `reference`

### 2.1 Why the delta manifest is mandatory

A compression that does not declare what it dropped is unusable for descent: the
system would have to re-read every descendant to find the missing piece, which
costs more than never compressing at all. The compressor prompt therefore emits
`{body, dropped[]}` as a single structured object, and a compression whose
manifest is empty while the token count fell materially is rejected as
under-reported (see `compact.py::_validate_manifest`).

Deltas are **inherited**: when `L3` is built from `L2`, `L3.dropped` includes
both what `L2 -> L3` removed *and* the entries from `L2.dropped`, re-attributed
to their original holders. This is what enables delta jumping — the apex knows
about detail several levels below it without holding the text.

---

## 3. Recall

Recall produces a **context pack**: the text that gets prepended to the task.

```
pack = apex.body
       [+ matched delta claims]      # only after a failure, see §4.3
       [+ expanded node bodies]      # only after delta patching fails
```

Default budget knobs (`.rmc/config.yaml`):

```yaml
recall:
  max_pack_tokens: 1200
  max_expansions: 3          # how many descents before giving up
  strategy: delta-patch      # delta-patch | delta-jump | stepwise
```

- `delta-patch` — apex + matched claims. Cheapest; default.
- `delta-jump` — replace apex with the descendant holding the matched claim.
- `stepwise` — walk `derived_from` one level at a time, ignoring deltas. Baseline
  for ablation.

---

## 4. Descent and selection

This section answers: *when the loaded node fails the task, which child do we go
to next?*

### 4.1 Detect the failure

A node "fails" when the oracle rejects the output produced by an agent that was
given `pack + task`. Oracles are declared per task (§6) and are the only source
of ground truth — the agent's self-assessment is never trusted for this.

### 4.2 Diagnose the failure

On failure the harness makes one cheap structured call to a **diagnoser** agent,
with the task, the pack, the produced output and the oracle's complaint:

```json
{
  "category": "edge-case",
  "missing": ["what the response body looks like on a soft failure"],
  "wrong_step": "assumed non-2xx signals failure",
  "confidence": 0.8
}
```

`category` uses the same closed vocabulary as `dropped[].kind`. This is the join
key that makes matching tractable without embeddings.

### 4.3 Score the candidates

Candidates are the delta entries on the failed node (for `delta-patch` /
`delta-jump`) or the nodes in `derived_from` (for `stepwise`). Each candidate `c`
is scored against diagnosis `D` and task `T`:

```
score(c) = w_d · delta_match(c, D)
         + w_t · task_affinity(c, T)
         + w_p · prior(c)
         - w_c · cost(c)
```

| Term | Definition | Default weight |
|---|---|---|
| `delta_match` | `0.5·[kind(c) == D.category] + 0.5·lexical_overlap(claim(c), D.missing)`. Optionally replaced by a judge call when `selection.judge: true`. | 0.45 |
| `task_affinity` | Jaccard similarity between `T`'s tag/keyword signature and the union of signatures of `c.covers_tasks`. | 0.25 |
| `prior` | Beta posterior mean `(successes + 1) / (attempts + 2)` — Laplace-smoothed so unused children are neither favoured nor buried. | 0.20 |
| `cost` | `tokens(c) / max_pack_tokens`, clipped to `[0,1]`. | 0.10 |

Ties break toward lower `tokens`, then toward lower `level` (more specific).

The `prior` term makes descent a **contextual bandit over the tree**: children
that repeatedly rescue failures rise, and children that never help sink, without
any hand-tuning. `selection.explore: ucb` swaps the posterior mean for a UCB1
bonus when you want the tree to keep probing rarely-used branches.

### 4.4 Budget and escalation

Try candidates in score order, re-running the task after each patch/expansion,
up to `max_expansions`. Then:

1. **Exhausted candidates, still failing** → load the level-0 node for the family
   in full. This is the guaranteed-correct fallback and is always available
   because L0 nodes are never deleted.
2. **L0 also fails** → this is a genuine knowledge gap, not a compression bug.
   Emit a `learning-event` record and hand back to the outer loop, which runs the
   human-steering cycle to mint a new L0.
3. **L0 succeeded but the apex did not** → this is a compression bug. The harness
   writes a **regression record**: task `T` is appended to `covers_tasks` of every
   node on the path from the apex down to the rescuing node, so all future
   compressions of that path must keep passing `T`. If the same node regresses
   `regression.demote_after` times (default 2), it is marked
   `status: demoted` and stops being served as an apex.

Escalation is therefore not just recovery — it is how the tree learns where it
was cut too deep.

---

## 5. Compression

### 5.1 Trigger

A compression attempt is queued when a node reaches
`compaction.min_successes` (default 2) successful recalls since its last
compression attempt. This is the "the more you use it, the more abstract it gets"
mechanic, made concrete.

### 5.2 Generate

One structured call to a **compressor** agent, given the node body, its
`covers_tasks`, and any `preserve:` hints from previous rejections. It must
return `{body, dropped[], rationale}`.

### 5.3 Validate

This is the check from the original design sketch, hardened:

```
R = sample(regression_set(node) , k = compaction.regression_k)      # default 5
for t in R:
    fresh agent process, no history:  candidate.body + task(t)
    oracle(t) -> pass/fail
accept iff  pass_rate >= compaction.threshold        (default 1.0)
       and  tokens(candidate) <= compaction.max_ratio * tokens(node)   (default 0.6)
```

Three properties matter:

- **Fresh process.** Validation must not run in the main agent's context, or the
  main agent's memory of the verbose lesson leaks in and every compression looks
  successful. Adapters spawn a new `claude -p` / `codex exec` per validation.
- **Subtree-wide regression set.** Validating only on the triggering task is how
  you get a tree that is beautifully compressed and useless. The regression set
  is the union over the node's whole subtree.
- **Rejections are informative.** A rejected candidate records which tasks failed;
  those become `preserve:` hints for the next attempt, so the compressor converges
  instead of thrashing.

### 5.4 Merge (sibling generalization)

When two sibling nodes in a family both sit at level `L` and their
`covers_tasks` are disjoint but their bodies are judged to share a procedure, a
**merge compression** is attempted: both bodies in, one level-`L+1` body out,
`derived_from` listing both, and a regression set that is the union of both
subtrees. Merges are what turn the chain into a tree, and they are also the
riskiest operation, so they use a stricter threshold
(`compaction.merge_threshold`, default 1.0 over a larger `k`).

---

## 6. Tasks and oracles

A task is a YAML file:

```yaml
id: retry-http
family: retry
tags: [http, retry, backoff]
prompt: |
  Implement fetch_with_retry(url, deadline_ms) in src/net.py ...
oracle:
  type: command                 # command | contains | regex | json_equals | judge
  command: pytest -q tests/test_net.py
  cwd: ./sandbox
```

Oracle types:

| Type | Passes when |
|---|---|
| `command` | Exit status 0. `{output}` in the command is substituted with a temp file holding the agent's final message. |
| `contains` / `regex` | The agent output matches. |
| `json_equals` | The agent's structured output deep-equals `expected`. |
| `judge` | A separate agent, given a rubric, returns `{"pass": true}`. Used only where mechanical checks are impossible; noisier, so `judge` oracles get `n_votes` (default 3) and majority-vote. |

Without an oracle none of this works — compression validation is exactly the
claim "this shorter text still produces correct behaviour", and "correct" has to
be mechanically decidable to be worth anything.

---

## 7. Backends

`rmc/adapters/` exposes one interface:

```python
class Adapter(Protocol):
    def run(self, prompt: str, *, system: str | None, cwd: Path,
            schema: dict | None, timeout: int) -> AgentResult: ...
```

| Adapter | Invocation | Structured output |
|---|---|---|
| `claude` | `claude -p --output-format json --model <m>` | fenced-JSON contract + parse, validated against schema |
| `codex` | `codex exec --json --output-schema <f> -o <last>` | native `--output-schema` |
| `mock` | in-process, scripted | trivially |

Both real adapters run **read-only and sandboxed by default**
(`--permission-mode plan` / `-s read-only`); a task that needs to write code opts
in per-task via `sandbox: workspace-write`.

The `mock` adapter exists so the entire control flow — descent, scoring,
validation, regression bookkeeping — is testable offline and deterministically,
without burning tokens. The test suite runs against it.

---

## 8. Failure modes this design accepts

Stated plainly, since a research harness that hides its weaknesses is useless:

- **Oracle coverage bounds everything.** Lessons about taste, tone or judgement
  have no cheap oracle, so they can only use `judge` oracles and will compress
  more noisily.
- **Diagnosis quality bounds descent.** If the diagnoser mislabels `category`,
  scoring falls back to lexical overlap and `prior`, which degrades toward
  stepwise walking rather than breaking outright.
- **Merges can over-generalize.** Two procedures that look alike and differ in
  one precondition will merge, and the failure only shows up on a task that
  exercises the precondition. The regression set is the mitigation; it is not a
  proof.
- **Compression is not monotone.** A level-4 node is not guaranteed better than
  level-3 on unseen tasks — only on the regression set. Held-out evaluation
  (`rmc eval --holdout`) is the honest measurement, and is reported separately
  from the accept/reject gate.
