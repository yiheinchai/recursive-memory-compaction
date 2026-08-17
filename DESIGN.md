# RMC Design

Version 0.1. This document defines the data model, the descent/selection policy,
and the compression validation protocol.

---

## 1. Vocabulary

| Term | Meaning |
|---|---|
| **Lesson** | A reusable piece of procedural knowledge, stored as one markdown file with frontmatter. |
| **Node** | A lesson at one specific abstraction level. |
| **Family** | A set of nodes connected by compression edges; the unit you traverse. |
| **Level** | Integer. `0` = original verbose lesson. Higher = more compressed/abstract. |
| **Apex** | The highest-level node of a family. The starting point of the `judge` selector's walk. |
| **Delta manifest** | The list of claims a compression removed, attributed to the descendants that still hold them. |
| **Regression set** | Every task a node (and its subtree) has been validated against. |
| **Oracle** | The check that decides whether an output is correct, for a given task. |

Edge directions are named explicitly to avoid the usual confusion, since
"parent" is ambiguous when the tree grows upward from detail to abstraction:

- `compressed_into: <id>` — points **up**, toward less detail.
- `derived_from: [<id>, ...]` — points **down**, toward more detail. Plural,
  because a lesson can be reached from more than one abstraction above it.

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
pack = selection lessons
       + selected node bodies
       [+ claims that previously rescued them]
       [+ any unresolved conflict, as a question]
```

### 3.1 Selection is a search

Which lessons bear on a prompt is a judgement about meaning, so the model makes
it. What changed is *where the model looks from*.

The original design rendered the apex layer into one question. That is
affordable at 29 lessons and arithmetically impossible later: routing costs ~55
tokens per apex, apex count tracks node count at roughly 1:1 (EXPERIMENTS §3.4),
and at 5,000 lessons the candidate list alone is ~225k tokens **per prompt** —
the thing deciding what to load no longer fits beside the work. Prompt caching
changes that constant by an order of magnitude and does not change its shape.

So selection is a **fork of the live session**, given tools:

1. `.rmc/index.md` holds one line per lesson — id, family, level, title, tags,
   gist, path. It is regenerated whenever it falls behind the nodes.
2. The fork greps that index *and* `nodes/` itself, opens what it needs, and
   returns the ids worth loading.
3. The hook injects those bodies.

The index is a first pass, not the search surface. It holds a summary, so a
lesson whose body names the exact command or error string will not match on it —
and after a verbatim skills migration most of what a store knows lives in bodies
thousands of lines long. The selector has a shell and the store is files; the
prompt names the searches that tend to work and explicitly does not restrict it
to them. What it *is* steered away from is reading a very long lesson whole,
which can spend the whole budget on one candidate.

**The index is searched, never sent.** That is the whole scaling property: at
5,000 lessons it is ~125k tokens on disk and 0 tokens per prompt. What costs
tokens per prompt is the selection-lesson layer (§3.4), which is capped.

Three further consequences, in descending order of how obvious they are:

- **The candidate set is no longer the apex layer.** EXPERIMENTS §8.2 found half
  the store unreachable, because a lesson could only be found by descending into
  an apex whose summary looked unpromising. A grep does not care what level a
  lesson sits at.
- **The selector has the reasoning.** A fork inherits the conversation — the
  task, the tool calls, what has already been tried. That is a far better basis
  for "what does this work need" than the user's opening sentence, and it is the
  input the learning loop in §3.4 is defined over.
- **Attribution becomes an observation.** Which lessons the selector opened is
  visible in the transcript as tool calls, rather than being a judge's later
  reconstruction.

### 3.2 What it costs, and the bound on it

Latency, in a hook that blocks the user's prompt. EXPERIMENTS §4.4 puts process
startup alone at ~5s, and an agentic loop is several round trips rather than
one. This is the likeliest way the design fails in daily use, and it will be
felt long before it shows up in a precision number.

Three things bound it, and none of them is optimism:

- `recall.selector_max_tool_calls` (6) — the prompt tells the fork to answer
  with what it has once spent. An unbounded search always has another phrasing
  to try.
- `recall.selector_timeout_s` (60) — past this the selector is abandoned.
- Selection lessons, which are supposed to make the search converge. If they do
  not, this is slower than what it replaced, and §3.4's measurement is what says
  so.

### 3.3 Falling back, and never silently

The selector needs a session to fork and a backend that can fork one. Without
either — the first turn of a session, a non-Claude backend,
`recall.selector: judge` — the apex walk of the original design runs instead. A
selector that fails outright also falls through to it.

The apex walk is therefore kept deliberately, not left behind: it is the
baseline every arm of `rmc eval-recall` is measured against.

What must never happen is an empty pack that looks like a decision when it was
an outage. `Pack.degraded` carries the distinction to `recall_notice`, because a
user who reads a broken selector as "nothing applied" concludes the whole system
does not work — and that conclusion is not recoverable by any later fix.

### 3.4 Selection lessons — RMC applied to its own retrieval

Every other stage of RMC learns from outcomes. Selection learned from nothing,
and it is the stage measured worst: filtering lifts precision from 28% to 48%,
which means **over half of what recall serves is never used** (EXPERIMENTS §4.1).

A selection lesson is what a reflection pass writes after watching a session —
not knowledge about the work, but knowledge about where the knowledge was:

```
- When the task runs the integration tests: read nodes/testing/ before running pytest
```

They live in `.rmc/routing/`, **not** under `nodes/`. If they were nodes they
would be retrieved by the mechanism they exist to fix, and would compete with
real lessons for the same budget. They are always injected, under
`routing.max_tokens`.

**Every rule must be conditioned on a kind of task, and one that is not is
refused at mint time.** This is not tidiness. EXPERIMENTS §4.2 measured the
unconditioned form — annotating candidates with their usage record — and it
dropped precision to 41% and recall to 81%, worse on both, because how often a
lesson is used is a statement about the distribution of work rather than about
the lesson. The unconditioned form is also one rule per lesson, which would make
this layer a second copy of the store.

That last point is the load-bearing one. The design bets that selection lessons
track *kinds of work* rather than lessons, so the injected layer stays bounded
while the store does not. **This is a bet, not a guarantee**, so `rmc status` and
`rmc route` report the ratio of rules to lessons. If it climbs rather than
falls, the approach to the long tail is wrong and needs revisiting — and that
has to be visible as a number rather than inferred.

### 3.5 Budgets

```yaml
recall:
  max_pack_tokens: 1200
  max_families: 3            # lessons served per prompt
  selector: agentic          # agentic | judge
  selector_timeout_s: 60     # bound on the search
  selector_max_tool_calls: 6 # searches before it must answer
  max_expansions: 3          # descents during a failure, see §4
  strategy: delta-patch      # delta-patch | delta-jump | stepwise
routing:
  enabled: true
  max_tokens: 800            # the only per-prompt cost that remains
```

- `delta-patch` — apex + the matched claims. Cheapest; default.
- `delta-jump` — replace the apex with the descendant holding the matched claim.
- `stepwise` — walk `derived_from` one level at a time, ignoring the manifest.
  Baseline for ablation.

The `judge` selector keeps its own budgets — `judge_calls`, `max_depth`,
`fanout`, `filter_above`, `warm_prefix_above_tokens` — documented in
`rmc/config.py`. They apply only when it runs.

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

### 5.4 What the compressor is told to keep

The compressor used to choose what to cut from the lesson text alone, and find
out afterwards — via replay — whether it had been wrong. It is now given the
spans a reflection pass **observed** doing work: the sentences that changed what
an agent did, reported per session and accumulated on the node as
`load_bearing`.

Where that evidence exists, the reduction is taken from everything else. Where
it is absent the compressor is told so explicitly and compresses
conservatively — the asymmetry matters, because absence of evidence is not
evidence of uselessness. A span may simply not have come up yet, and reading the
list as "everything else is dead" is the obvious wrong move.

Replay is unchanged and still decides. "Would 60% of this lesson still work?" is
a counterfactual a reflector can only guess at; validation against the
regression set in a fresh process is what actually answers it. So this is better
input to an existing gate, not a new gate.

**Merging is gone.** Two or more lessons could previously be generalised into a
shared parent. It was removed rather than disabled, on the evidence in
EXPERIMENTS §3: unchecked merges landed at 100–102% of combined size; 8 of 8
attempts landed at 96–115% until the prompt was given an explicit token budget;
and consolidation removed ~2 apexes per pass while capture added them faster, so
the steady state it existed to produce never arrived. The thing merging was for
— holding the apex layer narrow, because recall enumerated it every prompt —
stopped being a cost the moment selection became a search (§3.1).

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
| `refines` | fold into L0 | work the detail in, then patch every ancestor |
| `contradicts` | dispute both | keep both, attach a question, ask the human |
| `specialises` | attach sibling | stands alongside under the more general lesson |
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
  one precondition will be folded together, and the failure only shows up on a task that
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
