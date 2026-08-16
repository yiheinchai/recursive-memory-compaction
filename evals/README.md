# RMC-Bench

An eval set for **procedural memory under compression** — which is a different
claim from factual recall, and needs a different set.

OpenAI's factual-recall eval asks: *given a fact about the user, is it recalled
and used correctly?* One axis, one failure mode. RMC's claim is larger and more
fragile:

> A lesson can be compressed repeatedly and keep changing behaviour, while the
> retrieval that finds it stays selective as the store grows.

That has four ways to fail, and a set that only measures the first will report
success while the system rots.

---

## The one property every case must have

**A case is worthless unless the model gets it wrong without the lesson.**

If a competent model already does the right thing, the lesson has no lift, and
the case measures the model's prior rather than the memory. Every case here
therefore names its **trap** — the specific wrong thing a good model does by
default — and the set is only valid if the control arm actually falls into it.

`rmc eval` reports control transfer first for exactly this reason. A case whose
control passes should be **deleted, not celebrated**.

---

## The four axes

| Axis | Question | Cases that test it | Fails when |
|---|---|---|---|
| **Transfer** | does the lesson change behaviour at all | `trap`, `detail`, `principle` | control ≈ treatment |
| **Retention** | does it survive compression | all, run per level | L2 < L0 |
| **Retrieval** | is the right lesson found, and only it | `distractor`, `null`, `multi` | wrong lesson served, or right one missed |
| **Cost** | tokens paid per prompt | measured throughout | tokens rise without transfer rising |

The headline result is not a single number. It is a **curve**: transfer against
tokens, across levels. The thesis holds if transfer stays flat while tokens
fall, and is refuted if transfer degrades with depth.

---

## Case kinds

Each kind exists to catch a failure the others cannot see.

**`trap`** — a strong, wrong default the lesson overrides. The purest transfer
test. If a set contains only these, compression looks perfect long after it has
started dropping specifics, because principles survive summarisation easily.

**`detail`** — the lesson carries an unguessable specific: a port, a constant, a
flag. **The most compression-hostile kind**, because a summariser drops exact
values first and the loss is invisible until something breaks. A set without
these will not detect over-compression.

**`principle`** — dispositional; changes *how* a problem is approached rather
than supplying a fact. Tests whether abstraction survives abstraction. These are
also where attribution is hardest, since influence leaves no command trail.

**`multi`** — needs two lessons together, and either alone is insufficient.
Tests retrieval breadth and whether co-use is being observed. A system that
serves the single best lesson scores well everywhere else and fails here.

**`distractor`** — the task shares vocabulary with a lesson that does **not**
apply. Correct behaviour is to ignore it. Catches the over-serving that a
transfer-only eval rewards, since injecting everything maximises transfer.

**`conflict`** — two stored lessons contradict. Correct behaviour is to surface
the contradiction, **not** to pick one silently. Catches last-write-wins rot.

**`null`** — nothing in the store applies. Correct behaviour is to serve
nothing. The only case that punishes a system for being eager.

---

## Scoring

Per case, per level, blind-graded against `expected`:

- `pass` — the candidate would lead to the same actions
- `fail` — omits something essential, contradicts it, or is too vague to act on

For `distractor` and `null`, passing means the candidate does **not** apply the
named lesson. Serving it at all is recorded as a retrieval miss even when the
answer happens to be right — a system that is right by luck is not right.

Aggregate per axis. Report the control arm alongside every number, always.

---

## Deliberate limits

- **Hand-written, so it reflects one author's idea of a good trap.** Cases drawn
  from real captured lessons would be more representative and are not here.
- **Single-turn.** Real sessions are long, and a lesson's influence over twenty
  turns is not the same as over one.
- **The grader is a model**, so grading noise is real. Run `--samples 3` or more
  before believing a difference smaller than ~15 points.
- **No case tests staleness over time**, because that needs a store with history
  rather than a fixture.
