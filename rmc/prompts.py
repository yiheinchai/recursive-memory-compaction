"""Prompt templates for the meta-calls.

Inputs are delimited as ``<<<NAME ... NAME>>>`` blocks so that (a) models do not
confuse instructions with data, and (b) the mock adapter can parse its own
inputs back out without a model. Every template opens with an ``RMC:<kind>``
marker that routes the mock.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# schemas
# --------------------------------------------------------------------------- #

COMPRESS_SCHEMA = {
    "type": "object",
    "required": ["body", "dropped"],
    "properties": {
        "body": {"type": "string", "description": "The compressed lesson."},
        "dropped": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["claim", "kind"],
                "properties": {
                    "claim": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": [
                            "parameter",
                            "example",
                            "precondition",
                            "edge-case",
                            "rationale",
                            "counter-example",
                            "procedure-step",
                            "naming",
                            "reference",
                        ],
                    },
                },
            },
        },
        "rationale": {"type": "string"},
        "lossless": {
            "type": "boolean",
            "description": (
                "True only if the shorter body preserves every claim in the original — "
                "you tightened prose, removed repetition or hedging, and cut nothing a "
                "reader could act on. If you removed any content, this is false and it "
                "must appear in `dropped`."
            ),
        },
        "title": {"type": "string"},
        "gist": {
            "type": "string",
            "description": (
                "One line, at most 25 words, naming what this lesson is about and "
                "when it applies. A future agent reads only this to decide whether "
                "to open the lesson at all, so it must be specific: name the tool, "
                "command or system, not the category."
            ),
        },
    },
}

DIAGNOSE_SCHEMA = {
    "type": "object",
    "required": ["category", "missing", "confidence"],
    "properties": {
        "category": {
            "type": "string",
            "enum": [
                "parameter",
                "example",
                "precondition",
                "edge-case",
                "rationale",
                "counter-example",
                "procedure-step",
                "naming",
                "reference",
            ],
        },
        "missing": {"type": "array", "items": {"type": "string"}},
        "wrong_step": {"type": "string"},
        "confidence": {"type": "number"},
    },
}

JUDGE_SCHEMA = {
    "type": "object",
    "required": ["pass", "reason"],
    "properties": {
        "pass": {"type": "boolean"},
        "reason": {"type": "string"},
        "missing": {"type": "array", "items": {"type": "string"}},
    },
}

REFLECT_SCHEMA = {
    "type": "object",
    "required": ["capture", "reason"],
    "properties": {
        "capture": {"type": "boolean"},
        "reason": {"type": "string"},
        "family": {"type": "string"},
        "title": {"type": "string"},
        "body": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "gist": {
            "type": "string",
            "description": (
                "One line, at most 25 words, naming what this lesson is about and "
                "when it applies. A future agent reads only this to decide whether "
                "to open the lesson at all, so it must be specific: name the tool, "
                "command or system, not the category."
            ),
        },
    },
}

# --------------------------------------------------------------------------- #
# templates
# --------------------------------------------------------------------------- #

COMPRESS = """RMC:compress

You are compressing a reusable lesson so it costs fewer tokens while still
producing correct behaviour in an agent that has never seen the longer version.

Rules:
1. Preserve every load-bearing detail: exact parameters, preconditions, edge
   cases, and anything an agent would get wrong by guessing.
2. Cut narrative, restatement, hedging, and worked examples whose principle is
   already stated. Generalise where several specifics share one rule.
3. You MUST declare everything you removed in `dropped`. Each entry is one
   self-contained claim, written so it can be re-injected verbatim later as a
   patch. An unreported drop is the worst possible failure here: it makes the
   compression impossible to reverse when it turns out to be wrong.
4. If you genuinely removed nothing — you tightened wording, cut repetition or
   hedging, but every actionable claim survives — set `lossless: true` and leave
   `dropped` empty. Say it explicitly; silence is read as an unreported drop,
   because from the outside those look identical.
5. Never invent content that was not in the original.

Target: at most {target_tokens} tokens ({ratio:.0%} of the original).

<<<LESSON
{body}
LESSON>>>

Tasks this lesson must keep working for:
<<<COVERS
{covers}
COVERS>>>

Details a previous compression attempt wrongly dropped — you must keep these:
<<<PRESERVE
{preserve}
PRESERVE>>>
"""

MERGE = """RMC:compress

You are merging sibling lessons into one more abstract lesson that covers all of
them. Find the shared procedure and state it once; keep any divergence that
would change what an agent does.

Everything dropped must be declared in `dropped`, with the claim written so it
can be re-injected verbatim as a patch.

<<<LESSON
{body}
LESSON>>>

Tasks the merged lesson must keep working for:
<<<COVERS
{covers}
COVERS>>>

<<<PRESERVE
{preserve}
PRESERVE>>>
"""

DIAGNOSE = """RMC:diagnose

An agent was given a lesson and a task, and produced the wrong result. Work out
what information the lesson was missing — not what the agent did wrong.

Answer with the single `category` that best describes the gap, and list the
specific missing facts in `missing`. Be concrete: "the retry backoff constants"
beats "more detail about retries". `confidence` is how sure you are that the
lesson (rather than the agent) was at fault.

TASK_ID: {task_id}

<<<TASK
{task}
TASK>>>

<<<LESSON
{pack}
LESSON>>>

<<<OUTPUT
{output}
OUTPUT>>>

<<<MISSING
{complaint}
MISSING>>>
"""

JUDGE = """RMC:judge

You are testing whether a *lesson* still carries the knowledge it used to carry
after being compressed — not whether the candidate is production-ready code.

Pass if the candidate would lead to the same substantive decisions as the
known-good result: same constraints respected, same traps avoided, same key
values. Fail only if it contradicts the expected approach, omits a decision that
would change behaviour, or gets a specific value wrong.

Explicitly ignore, and never fail for: length, formatting, wording, code style,
incompleteness of scaffolding, or truncation of the response. The candidate is a
short probe, not a deliverable. If the expected result is itself only a brief
summary, judge against what it actually claims and nothing more.

TASK_ID: {task_id}

<<<TASK
{task}
TASK>>>

<<<EXPECTED
{expected}
EXPECTED>>>

<<<CONTEXT
{context}
CONTEXT>>>

<<<CANDIDATE
{candidate}
CANDIDATE>>>
"""

REFLECT = """RMC:reflect

Read this session excerpt and decide whether it contains a reusable lesson.

Lessons come from two places, and the second is the more common one:

  1. **The human corrected the agent.** What should it have known?
  2. **The environment corrected the agent.** A command failed and a different
     one worked; a test rejected an approach; an API behaved unexpectedly; a
     long chain of exploration converged on an answer. Nobody had to intervene —
     the codebase, the test suite or the infrastructure taught it.

The `WORKED OUT BY TRIAL` section below is the second kind, already paired as
failed-attempt → what-worked. It is the highest-value material here: the whole
point is that the next agent reaches the answer directly instead of repeating
the detour. When you capture one, write down what to do AND the trap that made
the detour necessary, or the next agent will simply fall into it again.

Capture ONLY if all of these hold:
  (a) it is a reusable fact or method, not a narrative of what happened;
  (b) an agent that did not know it would take a wrong action or waste a run;
  (c) it is not already obvious from the repository's own code or docs;
  (d) it stays true after this task ends.

A trap that is specific to this codebase or this infrastructure passes (c) even
if it would be obvious to someone who had already hit it — that is exactly the
knowledge worth keeping.

`capture: false` is the expected answer for most sessions and needs no
justification. Never invent a lesson to have something to return — every
low-value lesson permanently taxes future retrievals.

If you do capture: write `body` as direct instruction to a future agent
(imperative, no preamble, no "in this session"). Pick a short lowercase
`family` slug naming the recurring situation it applies to, so later lessons
about the same thing land beside it.

Existing families (reuse one if it fits):
<<<FAMILIES
{families}
FAMILIES>>>

<<<WHAT THE HUMAN CORRECTED
{correction}
WHAT THE HUMAN CORRECTED>>>

<<<WORKED OUT BY TRIAL
{discovered}
WORKED OUT BY TRIAL>>>

<<<SESSION
{excerpt}
SESSION>>>
"""

REPLAY = """You are solving a task with the help of a lesson learned from previous work.

TASK_ID: {task_id}

<<<LESSON
{pack}
LESSON>>>

<<<TASK
{task}
TASK>>>

Do the task. If it is a question, answer it directly and completely.
"""

REPLAY_PROBE = """You are being asked how you would approach a task, given a lesson
from previous work on this codebase.

TASK_ID: {task_id}

<<<LESSON
{pack}
LESSON>>>

<<<TASK
{task}
TASK>>>

Describe the approach you would take in at most 150 words. State the decisions
that matter: constraints you would respect, traps you would avoid, and any
specific values or commands you would use. Do not write the implementation.
"""


BLIND_JUDGE = """RMC:judge

Decide whether a candidate answer achieves what a known-good answer achieved for
the same task.

Judge substance, not presentation: different wording, ordering or length is
fine. Pass it if it would lead to the same decisions and the same actions. Fail
it if it omits something the known-good answer treats as essential, contradicts
it, or is so vague that a reader could not act on it.

You are not told how this candidate was produced, and there is nothing to infer
from that. Judge only what is in front of you.

<<<TASK
{task}
TASK>>>

<<<KNOWN-GOOD OUTCOME
{expected}
KNOWN-GOOD OUTCOME>>>

<<<CANDIDATE
{candidate}
CANDIDATE>>>
"""
