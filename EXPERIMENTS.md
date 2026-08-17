# RMC experiment log

Measurements, including the ones that came out negative. Every number here was
produced by a command in this repo against the store RMC keeps on itself, so
each is reproducible and each is small — this is one user's store over about a
month, not a benchmark. Read the effect sizes as directional and the *signs* as
the finding.

Dates are 2026-08-16/17 unless noted.

---

## 1. Setting: what the store looked like

```
nodes        29        episodes 7 (6 usable for replay)
apexes       23-24     families 17
```

Two costs, and they are paid in different places:

| | where | size |
|---|---|---|
| **routing** | a spawned `claude -p` subprocess | 55 tok per apex, 1,311 tok total |
| **injection** | the user's own context window | 313 tok per prompt, mean of 57 |

The distinction matters and was initially got wrong (§2).

---

## 2. Negative result: the headline cost metric was measuring the wrong thing

`_census` reported "tokens served at apex" as the sum of apex **bodies**.
Recall never sends a body to decide what to send — it sends a one-line render
of title and gist.

| | reported | actual |
|---|---|---|
| routing cost / prompt | 8,699 tok | **1,311 tok** |
| per apex | ~400 tok | **55 tok** |

**7x overstatement**, and it had been driving optimisation decisions for a full
session. Every dream report and every scaling projection derived from it was
wrong in the same direction.

*Lesson for the paper: a compaction system's own instrumentation is a load-
bearing component. An unaudited cost metric will be optimised against, and the
work will look successful the whole time.*

---

## 3. Consolidation: co-use is real but unreachable at realistic arity

Abstraction was built from **co-use** — lessons repeatedly used together on work
that succeeded. Measured incidence:

```
episodes                     7
successful                   6
with >=1 used lesson         3
with >=2 used (co-use input) 1     <- the only episode that qualifies
lessons served per prompt    1.2 (mean; 24x one, 7x two)
```

Co-use requires two lessons used in one episode, **recurring**. Recall correctly
serves about one lesson per prompt, so the signal is rare by construction. Over
a month, one qualifying episode. The mechanism is sound and starves.

### 3.1 Width as a second trigger

Added: consolidate when the apex layer exceeds a width, regardless of usage
evidence. Peer set is the whole apex layer, not one family — 13 families holding
one apex each is the same flat layer as one family holding 13, and costs the
router identically.

| | before | after |
|---|---|---|
| apexes | 23 | 21 |
| routing tok | 1,256 | 1,147 |

Two genuine cross-family parents formed (`dogfooding`+`reflection`,
`communication-style`+`dogfooding`). **~9% reduction per pass.**

### 3.2 Negative result: an unchecked merge makes things worse

`merge_nodes` computed its size ratio, printed it in its own accept message
("merged 2 lessons at 102% of combined size"), and accepted regardless. First
real run:

```
n_906531  parent 1169 tok  vs children 1147 tok   (102%)
n_53f0c4  parent 1560 tok  vs children 1561 tok   (100%)
net effect: +809 tokens added to every prompt
```

Compression had the equivalent gate from the start; merging never did. Both
merges were reverted.

### 3.3 The merge prompt needs a stated budget or it never compresses

With the size gate in place and no budget in the prompt, **8 of 8** merge
attempts landed at 96–115% of combined size. The compressor was asked to find
the shared procedure and did — thoroughly. Adding an explicit token budget to
the prompt flipped it: the same candidates then produced parents at ~86% and
~76%, both accepted.

*Nothing in the instruction said "shorter". The model had no reason to infer it,
and the failure is invisible without the ratio check.*

### 3.4 The real generator of the flat layer is placement, not merging

```
placement decisions:  new-family 22 | fold-into 7 | attach-sibling 7 | duplicate 1
nodes with a parent:  5 of 29
```

**60% of captures mint a new apex.** Dream removes ~2 per pass at a cost of 8
subprocesses. Capture adds them faster than consolidation removes them, so
apexes track nodes at roughly 1:1. Merging is repair; placement is construction.
This is the unfixed constraint.

---

## 4. Retrieval: the main result

Built `rmc eval-recall`. Each recorded episode is replayed against **exactly the
candidate set it was served**, and the judge's picks are scored against
`episode.used` (which lessons a fork judged to have borne on the work).

The restriction to the served set is the methodological point: a lesson nobody
was shown could not have been used, so counting its absence from `used` as
evidence against it would manufacture false positives out of the retrieval
decision.

Precision and recall are always reported together. Precision alone is maximised
by serving nothing.

### 4.1 Baseline: what filtering is worth

Production behaviour at the time served **everything** whenever the store fit
the token budget, on the stated reasoning that "judgement is only needed under
scarcity, and early on there is none."

| | serve-everything (production) | judge the same sets |
|---|---|---|
| precision | 28% (16/58) | **48%** (16/33) |
| recall | 100% (by construction) | **100%** |
| noise tokens | 15,917 | **7,146** |

**Filtering removes 55% of injected noise and loses nothing.** The budget was
never the cost. Context that fits is not context that is free — an unrelated
lesson spends attention, and the relevance prompt itself says it "can actively
mislead."

### 4.2 Negative results: four attempts to improve the judge, all worse

| arm | precision | recall | noise tok | verdict |
|---|---|---|---|---|
| **baseline (unchanged judge)** | **48%** | **100%** | 7,146 | best |
| sharpened criteria only | 47% | 88% | 7,133 | worse |
| track record only | 41% | 81% | 8,697 | worse |
| both | 48% | 81% | 6,072 | worse |

*Sharpened criteria* named the dominant failure mode explicitly: a lesson can be
true about the project and change nothing about this turn. Inside a codebase,
every lesson about that codebase is on-topic and almost none are
decision-changing.

*Track record* annotated each candidate with its own retrieval history
("shown 5x, never used") — the harness supplying a count, the model still
judging. The idea is RMC's own philosophy applied to its selector, which had
been the one stage with no feedback path at all.

**Both made it worse, and the combination lost 3 useful lessons.** Every
intervention bought a little noise reduction by trading away recall. The
plausible reading is that a ratio invites the model to treat it as a quality
score, when a lesson is used precisely when the work happens to need it —
low usage is a statement about the distribution of work, not the lesson.

### 4.3 Negative result: a cheap router is worse than no router

Routing is a classification job over ~24 one-line summaries, so a small model
should suffice.

| routing model | precision | recall | noise tok | latency |
|---|---|---|---|---|
| default (large) | **48%** | **100%** | 7,146 | ~34 s |
| sonnet | 35% | 81% | 9,781 | — |
| haiku | 35% | 75% | 8,818 | ~14 s |

Both small models serve **more** noise *and* drop useful lessons. Since
serve-everything has recall 100% by construction, a cheap router is **strictly
worse than not filtering at all**: it costs latency, loses lessons, and does not
even reduce noise.

*This is the most useful negative result here. "Use a small model for retrieval"
is close to folklore, and on this task it inverts.*

### 4.4 Latency is dominated by process startup, not inference

```
bare `claude -p` "say ok", haiku:   16.75 s (cold) / 5.55 s / 5.27 s
full recall, haiku:                 13.9 s
full recall, default model:         34.4 s
```

**~5 s of every call is CLI startup.** Model choice moves the remainder. The
practical floor for a blocking `UserPromptSubmit` hook is therefore ~5 s even
with a free judge, and ~34 s for the only judge that routes well.

This is the unresolved tension: the accurate configuration is too slow to block
a prompt, and the fast configuration is worse than no filtering.

---

## 5. Two caching bugs that silently voided experiments

Both found *because* the eval existed, and both would have invalidated results
indefinitely without it.

1. **Cache not keyed on criteria.** Judgements were keyed on question +
   candidate ids. A full rewrite of the relevance prompt produced a
   byte-identical eval report — the new prompt was never used.
2. **Cache not keyed on the judge.** Switching the routing model from haiku to
   sonnet returned haiku's numbers exactly, down to the token.

Both fixed by folding a criteria fingerprint and the backend/model into the key.
The same failure had already occurred once elsewhere in this system (the nudge
backoff went stale against changed criteria), which suggests it is endemic to
caching model judgements rather than an isolated slip.

*For the paper: any system that caches LLM judgements must key on the full
judgement context — prompt text and model identity — or its own A/B
infrastructure will report null results with complete confidence.*

---

## 6. Config snapshotting silently froze every default

`config.save()` wrote the fully-merged settings tree, so a store kept the
defaults of the day it was created and ignored every subsequent improvement.
This store was still running `compaction.max_ratio: 0.6` and `min_successes: 2`
months after both were retuned — which is why nothing had been compaction-due.

Fixed: the file is overrides-only, and a value equal to its default is dropped
on save (lossless now, correct later).

*Relevant to any self-improving system that ships tuned constants: a
materialised config is a silent fork of the defaults.*

---

## 7. What is measured, what is not

**Measured.** Routing cost; injection cost; precision and recall of retrieval
against observed use; merge size ratios; consolidation rate; latency
decomposition; four judge interventions; three routing models.

**Not measured, and load-bearing.**

- *Does a recalled lesson shorten the next session?* The entire product claim.
  Needs a paired comparison with memory on and off across matched tasks. Nothing
  here speaks to it.
- *Capture quality.* 28 capture events, only 3 lessons from the reflector and 23
  added by hand. Capture has ground truth (the user notices) but no eval.
- *Scaling.* Every number here comes from a 29-node store. The apex layer grows
  ~1:1 with nodes today, so routing cost is currently **linear in lessons**. At
  5,000 lessons that is ~275k routing tokens per prompt — beyond a single
  context window, forcing hard selection over what even reaches the selector.
  Whether merging can hold apex count flat against capture is the open question
  and §3.4 suggests it currently cannot.

---

## 8. Open problem: routing cost is currently linear in lessons

Everything above is measured on a 29-node store. The scaling argument is not,
and it is the one that decides whether the design holds.

**The arithmetic.** Routing costs ~55 tok per apex, and §3.4 shows apex count
tracks node count at roughly 1:1 because 60% of captures mint a new family.

| lessons | apexes (at today's ratio) | routing tok / prompt |
|---|---|---|
| 29 | 24 | 1,311 |
| 500 | ~410 | ~23k |
| 5,000 | ~4,100 | ~225k |

At 5,000 lessons the candidate list alone approaches a full context window, so
the selector cannot be shown everything and something must decide what even
reaches it. That decision is the same problem one level up, and solving it with
a heuristic would abandon the property the whole design rests on.

**The intended answer is that merging holds apex count flat** — capture adds,
consolidation removes, and the top layer reaches a steady state whose width is
set by how many genuinely distinct subjects the user works on rather than by how
many lessons they have accumulated. That is the claim. It is currently false in
practice: §3.1 removes ~2 apexes per pass while §3.4 adds them faster. Whether
the steady state exists at all is unmeasured and is the single most important
open question here.

**A partial answer that does not depend on the steady state.** The candidate
list is nearly identical between consecutive prompts — the same apex layer,
re-sent every time, with only the question changing. That is exactly the shape
prompt caching rewards: send the candidate list as a stable prefix and vary only
the question, and repeat routing calls read the prefix at cache rates rather
than paying for it again. Forking a warm session rather than spawning a cold
`claude -p` would keep that prefix hot across selection runs.

This does not make routing sublinear — the prefix still has to fit — but it
changes the constant by roughly an order of magnitude, and it composes with
whatever fixes the width. It also attacks §4.4 from the other side: a fork
skips the ~5s CLI startup that currently dominates recall latency, which is the
reason filtering is bypassed on small stores at all.

### 8.1 The warm prefix, implemented and measured

Built. The candidate list is seeded into a conversation once and each prompt
branches a `--resume --fork-session` child from it, so the fork answers against
the stored prefix without appending to it.

**It works, with conditions.** A controlled test with a 21,272-token prefix:

```
seed          cache_creation 21,272   cache_read 57,558   (system prompt only)
resume+fork   cache_creation     23   cache_read 78,830   (= 57,558 + 21,272)
```

The entire prefix came back from cache. Integrating it surfaced four things
that each silently reduced the benefit to zero:

1. **The fork was re-sending the candidate list in the question.** The cached
   copy is then paid for and ignored, because the copy the model reads is the
   new one in the prompt. The warm path needs its own prompt that asks the
   question alone.
2. **Warmth cannot be measured with one global baseline.** The host sends a
   ~58k-token system prompt that is cached regardless, dwarfing our prefix, so
   "were any tokens read from cache?" reports a hit on every call including cold
   ones. Worse, a max-based baseline drifts: a stale 65,372 against a real
   57,558 turned every genuine hit negative. Each conversation is now scored
   against the two readings from its own seeding call.
3. **One session cannot hold a chunked list.** A wide apex layer is judged in
   chunks of `fanout`; a single session is reseeded on every chunk and hits
   nothing. One conversation per chunk.
4. **A prefix below the provider's minimum cacheable size is never cached at
   all.** Chunks of 12 summaries are ~660 tokens, under the ~1024 minimum, so
   seeding wrote nothing: `ours_cached: 4` on nine calls in ten. Widening chunks
   until they clear the minimum took it to **50% warm, 25,869 prefix tokens read
   from cache**.

Each of these fails *quietly* — the mechanism appears to run, the logs look
plausible, and nothing is saved. Only the per-conversation measurement
distinguishes them.

At this store size the saving is small and the mechanism stays off below
`recall.warm_prefix_above_tokens`. Its value is entirely at scale, which is
where the design has to hold.

### 8.2 Found while wiring it: half the store was unreachable

The relevance walk read `level = [n for n in frontier if n.id not in seen][:fanout]`
and never revisited the remainder. With 26 apexes and a fanout of 12,
**14 lessons could not be retrieved on any prompt, ever** — and every precision
figure above was computed over the reachable dozen while looking perfectly
healthy.

Two fixes: a wide level is now judged in every chunk rather than the first, and
the call budget is sized so the top level always completes (`judge_calls` buys
*descent* on top of a complete first pass). A chunk skipped for budget is a
recorded decision; truncation was a silent hole.

*The general point: a retrieval metric computed over the candidates the system
chose to consider cannot see candidates it never considered. Coverage has to be
checked separately from precision, and the eval as built would never have found
this.*

---

## 9. Closing the loop: RMC tuning its own retrieval

Every stage of RMC is corrected by an outcome except the one that decides what
gets recalled. Its criteria — the relevance prompt and a handful of constants —
could only change when a person had an idea, and §4.2 is the record of how well
that goes: **five of six hand-written proposals were regressions**, each
plausible, each argued for.

`rmc tune` runs the same loop without a person in it. Measure; show the model
where retrieval was actually wrong, in cases rather than numbers, misses first;
take one proposal; apply it in a sandbox; measure again; keep it only if
**precision and recall are both at least as good**. A trade is a preference and
preferences belong to the user.

Three properties make it safe to leave running:

- **It cannot reach a correctness gate.** Only recall-shaped constants are
  tunable, and only within a range. A tuner that can move the thresholds it is
  scored against can pass its own exam.
- **A rejected change is reverted unconditionally.** Damage arriving labelled as
  an improvement is worse than no loop at all.
- **Failures are remembered and fed back into the next proposal.** A loop that
  forgets re-proposes forever, and the failures are the more informative half:
  that a plausible change made things worse is a fact about this store that
  nothing else records.

Two prerequisites had to exist first, and both are the same bug in §5 wearing a
different hat. Prompts had to become overridable per store — they were the
largest lever on judgement quality and the one part unchangeable without editing
the package, so they were unmeasurable. And `criteria_version()` had to
fingerprint the *resolved* prompt rather than the shipped one, or an override
would answer from the cache of the text it replaced.

**First run, one round.** It proposed something neither the author nor the model
had raised in six manual attempts: classify the work as **BUILDING** (code is
about to change) or **CONVERSING** (the user is thinking aloud and wants an
answer) *before* judging any lesson, on the grounds that the same lesson is
load-bearing in one mode and pure noise in the other.

| | before | after |
|---|---|---|
| precision | 47% | **51%** |
| recall | 77% | **88%** |

Kept, because both improved. This is the first change to RMC's retrieval that
no human proposed.

*The honest caveat: n=1, on one store, scored against six episodes. What the
result supports is that the loop can find and validate a non-obvious change —
not that it will keep doing so.*

## 10. Reproducing

```
rmc status                                  # store shape, routing cost, precision
rmc eval-recall --save NAME                 # score retrieval against observed use
rmc eval-recall --model M --against NAME    # A/B a routing model
rmc dream --dry-run                         # consolidation candidates, no writes
rmc tune --rounds N                         # propose, measure, keep only wins
rmc tune --history                          # every attempt, including the rejected
rmc migrate [--apply]                       # convert a Claude skills library
```

Saved runs live in `.rmc/evals/*.json`.
