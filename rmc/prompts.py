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
        "title": {"type": "string"},
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
4. Never invent content that was not in the original.

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

Decide whether a candidate result is equivalent in substance to a known-good
result for the same task. Judge outcome, not wording: different phrasing,
formatting or ordering is fine. Fail it if the candidate omits a required step,
gets a value wrong, or would lead to different behaviour.

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

Capture ONLY if all of these hold:
  (a) it is a reusable fact or method, not a narrative of what happened;
  (b) an agent that did not know it would take a wrong action or waste a run;
  (c) it is not already obvious from the repository's own code or docs;
  (d) it stays true after this task ends.

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
