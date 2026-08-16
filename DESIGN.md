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
pack = selected node bodies
       [+ claims that previously rescued them]
       [+ any unresolved conflict, as a question]
```

### 3.1 When there is nothing to choose

Relevance filtering only matters under **scarcity**. If every stored lesson fits
inside `max_pack_tokens`, there is no selection problem: serve them all, ask
nothing, return in milliseconds. Filtering switches on only once the tree
outgrows the budget.

This is not a heuristic standing in for judgement — it is the observation that
judgement is needed to *choose*, and there is no choice to make. It also happens
to be the common case: a young store is small, and that is exactly when you least
want a model call on every prompt.

Getting this wrong is not theoretical. An earlier version asked on every prompt
regardless, and with a 464-token store against a 1200-token budget it spent a
`claude -p` round trip picking two lessons out of two — and blew the hook's
10-second deadline doing it, so the injection was discarded entirely. Recall
went from timing out to 0.09s.

`recall.always_judge: true` forces filtering anyway.

### 3.2 Choosing what to serve, when there is a choice

Which lessons bear on a prompt is a judgement about meaning, so the model makes
it (`judge.relevance`). The harness supplies the search shape:

1. Start from the **apexes** — the most compressed node of each family. Because
   they are compressed, the entire top level of the store fits in one question.
2. Ask once: for each, is it `relevant`, `maybe`, or `unrelated` — and if
   `maybe`, is the summary too abstract to decide from?
3. Descend only into the lines flagged `descend`. A line judged `unrelated` is
   never opened, so an irrelevant branch costs nothing beyond its one line in
   the first question.
4. Stop at `recall.max_depth`, or when `recall.judge_calls` is spent. Anything
   still unresolved when the budget runs out is served rather than dropped.

Cost therefore tracks the **depth** of the tree and the number of plausible
lines, not the number of lessons stored. Verdicts are cached by prompt, so
re-asking the same thing is free.

The structural gate: an empty store asks nothing at all.

### 3.3 Budgets

```yaml
recall:
  max_pack_tokens: 1200
  max_families: 3            # lessons served per prompt
  judge_calls: 2             # model calls the relevance walk may spend
  max_depth: 2               # how far down the walk may look
  timeout_s: 20              # bound on the routing call, under the hook deadline
  always_judge: false        # filter even when everything would fit
  max_expansions: 3          # descents during a failure, see §4
  strategy: delta-patch      # delta-patch | delta-jump | stepwise
```

- `delta-patch` — apex + the matched claims. Cheapest; default.
- `delta-jump` — replace the apex with the descendant holding the matched claim.
- `stepwise` — walk `derived_from` one level at a time, ignoring the manifest.
  Baseline for ablation.

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
`delta-jump`) or the nodes in `derived_from` (for `stepwise`).

```
score(c) = w_j · model_usefulness(c, D) + w_p · prior(c) − w_c · cost(c)
```

| Term | Definition | Default |
|---|---|---|
| `model_usefulness` | One structured call ranks every candidate 0–1: *how likely is it that this detail being absent caused this failure*. | 0.60 |
| `prior` | Laplace-smoothed rate at which this node has actually rescued failures: `(successes + 1) / (attempts + 2)`. | 0.28 |
| `cost` | `tokens(c) / max_pack_tokens`, clipped to `[0,1]`. | 0.12 |

The first term is a judgement and belongs to the model. An earlier version
scored it as `[kind == category] + lexical_overlap(claim, missing)`, which gets
the common case backwards: "parse the body, not the status code" is the fix for
"treated HTTP 200 as success" and shares none of its words.

The other two stay in code because they are not proxies for meaning. `prior` is
an *observed outcome* — evidence — and it makes descent a contextual bandit over
the tree: branches that repeatedly rescue failures rise, branches that never
help sink, with no tuning. `cost` is a measurement. Ties break toward cheaper,
then toward more specific.

With no judge available the first term is simply **absent**, and ranking falls
back to "try what has worked before, cheapest first". That degrades honestly
rather than substituting a metric that would look like judgement without being
one.

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

## 6. Episodes — the ambient oracle

Compression validation is exactly the claim *"this shorter text still produces
correct behaviour"*, so it needs a definition of correct. In a scripted harness
you write oracles by hand. RMC runs inside someone's real repo, where nobody is
going to author a YAML oracle per lesson — so the oracle has to be **harvested**.

Every session that ends well becomes an **episode**: a replayable regression
test.

```json
{
  "id": "e_4f1a",
  "family": "retry",
  "prompt": "add retry to the http client in api/client.py",
  "outcome": "success",
  "confidence": 0.85,
  "served": ["n_7f2a91"],
  "accepted_summary": "added fetch_with_retry with jittered backoff, deadline-capped",
  "check": {}
}
```

Replaying an episode means: fresh agent process, candidate lesson + the original
prompt, then compare against `accepted_summary`. The comparison uses, in order
of preference:

1. a **mechanical check** harvested from the session (`check.type: "contains"` —
   strings that must appear), which is exact and free of model noise;
2. otherwise a **judge** call, told to compare outcomes rather than wording.

A judge that cannot be read is scored as a **failure**, never a pass. Otherwise
an infrastructure blip silently promotes a bad compression.

### 6.1 Deciding the outcome

`signals.py` parses the transcript into facts and nothing more: who said what,
which tool ran with which input, what came back, and what the host itself marked
as a refusal or a meta turn. Tool calls are paired to their results by id, and a
call's success is recorded **only** from what the host reported — an `is_error`
flag or an exit code. When the host says nothing it stays unknown rather than
being guessed from the output text.

`judge.assess` then reads that digest and returns the outcome, whether the human
had to steer, and what was worked out by trial.

This used to be a scoring function over regex phrase banks: `-0.65` for a
"correction" pattern, `+0.6` for an "approval" one. It cannot work. Whether
"actually, let's use the other one" is a correction or a change of mind is a
reading of intent, and a pattern list only matches the surface forms someone
thought of in advance — while looking, in the code, like a decision.

One structural gate remains, and it is not a judgement: `worth_assessing` skips
sessions with almost no activity and no human follow-up, because there is
nothing there to learn regardless of what they say.

### 6.2 Two different questions

The session outcome and the *lesson's* outcome are not the same thing. A session
where the user corrected the agent and then everything worked is:

- a **success** for the episode — the final result was right, so it is a valid
  regression test;
- a **failure** for the served lesson — it was supposed to prevent that mistake.

These are recorded separately. Conflating them makes lessons look good precisely
when they most need repair.

An explicit correction is also exempt from the confidence floor. Corrected-then-
fixed sessions score near zero because their signals cancel, so a naive floor
discards exactly the sessions with the most to teach.

Below the floor, and with no correction, RMC records nothing at all. A noisy
label is worse than no label: it poisons both the priors and the corpus that
every future compression is judged against.

### 6.3 Learning without a human

Most learning does not involve a human at all. The environment corrects the
agent constantly — a command fails, a test rejects an approach, an API behaves
unexpectedly — and that correction is both a **stronger** oracle than "the user
did not object" and always available.

The transcript parser pairs each tool call to its result by id, so
failure-then-success sequences can be recovered as `Discovery` records:

```
[Bash] tried `pytest tests/integration` -> failed: could not connect to postgres at :5432
    then `PAYMENTS_PG_PORT=5433 pytest tests/integration` -> worked (after 4 attempts)
```

These feed three things:

- **a success signal** — recovering from failures unaided scores +0.30, so a
  session with no human in it can still be labelled confidently;
- **priority in the reflection excerpt** — a paired failure→fix is far more
  informative than a raw error log, so it leads;
- **the reflection prompt itself**, which names self-discovery as the primary
  source of lessons and requires the *trap* to be recorded alongside the fix.
  Recording only the fix means the next agent falls into the same trap and merely
  recognises the way out.

An identical retry that succeeds the second time is flakiness, not discovery,
and is excluded — the inputs must differ.

The point is compression of reasoning, not just of text: a lesson that cost four
attempts to discover should cost zero attempts next time.

---

## 7. Consolidation — where new knowledge goes

Growing the tree is not "append a leaf". A newly learned lesson has to be made
consistent with what is already known, or the memory accumulates contradictions
it never notices. `placement.py` classifies the new lesson against the tree:

| Relation | Action | Effect |
|---|---|---|
| `duplicate` | none | record the hit; the existing lesson is pulling its weight |
| `refines` | fold into L0 | merge the detail in, then patch every ancestor |
| `contradicts` | dispute both | keep both, attach a question, ask the human |
| `specialises` | attach sibling | stands alongside; merge-compression may generalise them later |
| `orthogonal` | new family | a genuinely new leaf |

### 7.1 Refinement has to reach the apex

Folding new detail into the level-0 node is only half the job. Every ancestor
was compressed from the *old* body and validated against it, so each is now
missing the new detail — and the apex is what actually gets served.

Rather than invalidating those compressions (throwing away work that still
mostly holds), the new detail is registered as a **rescue** on each ancestor.
Recall re-attaches it immediately via the sticky-patch path, and `compact.repair`
folds it in permanently once it has proven necessary. The tree keeps working
while it catches up. This reuses the descent machinery exactly as-is.

### 7.2 Contradictions are never resolved silently

Last-write-wins means whichever lesson was written most recently is treated as
true, regardless of which one is. So a contradiction instead:

1. marks **both** nodes `disputed` — we do not know which is wrong;
2. stores a generated question specific enough to settle it in one sentence;
3. keeps serving the lesson, with the question attached.

Point 3 matters. Withholding a contradicted lesson loses the knowledge *and*
removes the occasion to ask about it. Surfacing at recall time means the question
arrives when the user is already thinking about that topic — the same reason a
student raises a confusion during the relevant lesson rather than at random.

`rmc conflicts` lists open questions; `rmc resolve <id> [--drop]` settles them.

### 7.3 Keeping reconciliation cheap

Reconciliation runs on every new lesson, so its cost has to stay flat as the
tree grows. It does, through structure rather than shortcuts:

- **A tree walk, not a scan.** Apexes are the most compressed nodes in the
  store, so the whole top level fits in one question. A line is opened only when
  the model says the summary was too abstract to judge from, and a line judged
  clearly unrelated is never walked at all. Cost tracks depth.
- **One reconciliation call for every candidate the walk surfaced.** Constant in
  tree size, and a contradiction with the second-best match stays visible.
- **Cached verdicts.** Re-running learning never re-pays for a pair already
  judged.
- **No call at all when there is nothing to reconcile with.** An empty or
  unrelated region of the tree costs nothing.

An earlier version added a regex pre-filter that flagged `KEY=value` mismatches
to force a check the similarity floor would otherwise skip. It was removed along
with the floor: the walk asks the model directly whether two lessons concern the
same thing in the world, which catches the same contradictions without a pattern
list deciding what counts as one.

The honest limit: reconciliation compares against **apex** nodes, so a
contradiction with detail that exists only deep in a subtree is missed until
descent surfaces it.

## 8. Backends

`rmc/adapters/` exposes one interface:

```python
class Adapter(Protocol):
    def run(self, prompt: str, *, system: str | None, cwd: Path,
            schema: dict | None, timeout: int) -> AgentResult: ...
```

| Adapter | Invocation | Structured output |
|---|---|---|
| `claude` | `claude -p --output-format json --no-session-persistence` | JSON contract in the prompt, recovered by fenced/balanced-brace parsing |
| `codex` | `codex exec --ephemeral -o <file>` | native `--output-schema` |
| `mock` | in-process, scripted | trivially |

Meta-calls (compress, diagnose, judge) are pure text transforms and run with
tools denied (`--disallowedTools` / `-s read-only`); only replay, which has to
actually do the work, gets write access, scoped to the directory it is handed.

Every spawned process gets `RMC_CHILD=1`. RMC's hooks check it and no-op, which
is the only thing stopping a compression run from triggering compression runs.

The `mock` adapter exists so the entire control flow — descent, scoring,
validation, regression bookkeeping — is testable offline and deterministically,
without burning tokens. The test suite runs against it.

---

## 9. Failure modes this design accepts

Stated plainly, since a research harness that hides its weaknesses is useless:

- **The ambient outcome signal is heuristic.** Inferring success from "the user
  did not object" is genuinely weak. It is mitigated by a confidence floor, by
  preferring host metadata over text matching, and by requiring compressions to
  clear a replay gate rather than trusting the signal directly — but a user who
  silently fixes things themselves will teach RMC the wrong lesson.
- **Oracle coverage bounds everything.** Lessons about taste, tone or judgement
  have no mechanical check, so they fall back to judge calls and compress more
  noisily than lessons about procedures.
- **Diagnosis quality bounds descent.** If the diagnoser mislabels `category`,
  scoring falls back to lexical overlap and `prior`, which degrades toward
  stepwise walking rather than breaking outright. During development a
  mis-wired diagnoser silently reduced `delta_match` to zero and descent still
  worked, carried by the prior — which is reassuring for robustness and a good
  argument for asserting on score *components* in tests, not just outcomes.
- **Merges can over-generalize.** Two procedures that look alike and differ in
  one precondition will merge, and the failure only shows up on a task that
  exercises the precondition. The regression set is the mitigation; it is not a
  proof.
- **Compression is not monotone.** A level-4 node is not guaranteed better than
  level-3 on unseen tasks — only on the regression set it was validated against.
  Held-out evaluation is the honest measurement and is not yet implemented, so
  current accept/reject numbers should be read as in-sample.
- **Recall costs a model call per prompt.** Relevance is a judgement, so it is
  paid for on the hot path, cached by prompt. That is a real latency and cost
  trade against the alternative of injecting the wrong lesson, which is worse
  than injecting none. `recall.enabled: false` opts out.
- **Judgement quality now bounds everything.** Replacing heuristics with a model
  moves the ceiling up but also moves the failure mode: a model that
  misjudges relevance is harder to debug than a scoring function you can read.
  This is mitigated by every judgement being cached and logged with its stated
  reason (`rmc recall` prints them), not by pretending it cannot happen.
