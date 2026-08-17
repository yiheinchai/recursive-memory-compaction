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

## 8. Reproducing

```
rmc status                                  # store shape, routing cost, precision
rmc eval-recall --save NAME                 # score retrieval against observed use
rmc eval-recall --model M --against NAME    # A/B a routing model
rmc dream --dry-run                         # consolidation candidates, no writes
```

Saved runs live in `.rmc/evals/*.json`.
