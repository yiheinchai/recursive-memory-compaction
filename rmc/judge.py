"""Every semantic judgement in RMC, in one place.

The harness supplies structure — a tree to walk, a budget to spend, a schema to
answer in, a cache so nothing is judged twice. It does not supply answers.
Questions about *meaning* — is this relevant, does this relate to that, is this
a contradiction, did this session go well — are decided by the model.

That line matters. Similarity of meaning does not live in token overlap, and an
intent classifier made of regexes silently caps the system at whatever a bag of
words can express. Where earlier versions of this file's callers used Jaccard
and phrase banks, they now ask.

What stays in code is the part that is genuinely structural:

* **whether to ask at all** — an empty store, an exhausted budget, or a session
  with two tool calls needs no judgement, and asking would be waste;
* **the walk** — which candidates are put in front of the model, and in what
  order, so the number of questions grows with the *depth* of the tree rather
  than its size;
* **the cache** — the same question is never paid for twice.

Efficiency comes from structure, not from cheap approximations of judgement.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .adapters import Adapter
from .node import Node
from .store import Store
from .util import truncate

# --------------------------------------------------------------------------- #
# schemas
# --------------------------------------------------------------------------- #

RELEVANCE_SCHEMA = {
    "type": "object",
    "required": ["picks"],
    "properties": {
        "picks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "verdict"],
                "properties": {
                    "id": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["relevant", "maybe", "unrelated"],
                        "description": (
                            "relevant: would change how the task is done. "
                            "maybe: same area, unclear whether it applies. "
                            "unrelated: would only add noise."
                        ),
                    },
                    "descend": {
                        "type": "boolean",
                        "description": (
                            "True if the summary is too abstract to judge and the "
                            "more detailed versions beneath it should be examined."
                        ),
                    },
                    "why": {"type": "string"},
                },
            },
        }
    },
}

RELEVANCE = """RMC:relevance

You are deciding which remembered lessons, if any, apply to a piece of work
about to be done. Each lesson below is a compressed summary; more detailed
versions may exist beneath it.

For each one give a verdict:
  - `relevant`  — knowing this would change how the work is done.
  - `maybe`     — same general area, but you cannot tell from this summary
                  alone whether it applies. Set `descend: true` if there is
                  more detail available and seeing it would settle the question.
  - `unrelated` — it would only add noise.

Be strict. An irrelevant lesson costs the reader attention and can actively
mislead. Superficial word overlap is not relevance: a lesson about retrying
HTTP calls is unrelated to a request to retry a failed CI job.

Judge what the work actually needs, not what it superficially mentions.

<<<WORK
{question}
WORK>>>

<<<LESSONS
{candidates}
LESSONS>>>
"""


RELATED = """RMC:related

A new lesson has just been learned. Before it is stored it must be checked
against what is already remembered, so that it can be merged, set alongside, or
flagged as a contradiction rather than blindly appended.

For each remembered lesson below, say whether it covers the same ground:
  - `relevant`  — same subject. The two need reconciling, whether they agree,
                  one adds detail, or they contradict each other.
  - `maybe`     — possibly the same subject, but the summary is too abstract to
                  be sure. Set `descend: true` to see the detailed versions.
  - `unrelated` — different subject.

Same subject means *about the same thing in the world* — the same tool, command,
service, constraint or procedure — not merely similar wording. Two lessons that
both set an environment variable for the same service are the same subject even
if they share no other words. Two lessons that both mention "retry" may be about
entirely different systems.

Look especially for lessons that assign a different value to something this new
lesson also assigns: a port, a flag, a path, a command. Those are the
contradictions that matter most and the easiest to miss.

<<<NEW LESSON
{new}
NEW LESSON>>>

<<<REMEMBERED
{candidates}
REMEMBERED>>>
"""


@dataclass
class Pick:
    id: str
    verdict: str = "unrelated"
    descend: bool = False
    why: str = ""

    @property
    def positive(self) -> bool:
        return self.verdict in ("relevant", "maybe")


@dataclass
class Budget:
    """How many judgements this operation may buy."""

    max_calls: int = 3
    spent: int = 0

    def take(self) -> bool:
        if self.spent >= self.max_calls:
            return False
        self.spent += 1
        return True

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.max_calls


# --------------------------------------------------------------------------- #
# the judge
# --------------------------------------------------------------------------- #


class Judge:
    """A cached, budgeted interface to the model's opinion."""

    def __init__(
        self,
        store: Store,
        adapter: Adapter,
        *,
        cache_name: str = "judge-cache",
        use_cache: bool = True,
    ) -> None:
        self.store = store
        self.adapter = adapter
        self.cache_name = cache_name
        self.use_cache = use_cache
        self.calls = 0

    # ------------------------------------------------------------- plumbing
    def _cache_path(self):
        return self.store.root / f"{self.cache_name}.json"

    def _load(self) -> dict[str, Any]:
        if not self.use_cache:
            return {}
        path = self._cache_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _store_cache(self, key: str, value: Any, *, limit: int = 800) -> None:
        if not self.use_cache:
            return
        cache = self._load()
        cache[key] = value
        if len(cache) > limit:
            for stale in list(cache)[: len(cache) - limit]:
                cache.pop(stale, None)
        try:
            self._cache_path().write_text(json.dumps(cache), encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]

    def ask(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        cache_key: str | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any] | None:
        """One structured judgement. Returns None if the model could not answer."""
        if cache_key:
            cached = self._load().get(cache_key)
            if cached is not None:
                return cached
        run = self.adapter.run(
            prompt,
            schema=schema,
            timeout=timeout or int(self.store.config.get("limits.agent_timeout_s", 180)),
        )
        self.calls += 1
        if not run.ok or not run.data:
            return None
        if cache_key:
            self._store_cache(cache_key, run.data)
        return run.data

    # ------------------------------------------------------------- relevance
    def relevance(self, question: str, candidates: list[Node]) -> list[Pick]:
        """Which of these lessons bear on this work?"""
        if not candidates:
            return []
        rendered = "\n\n".join(_render(node) for node in candidates)
        data = self.ask(
            RELEVANCE.format(question=truncate(question, 3000), candidates=rendered),
            RELEVANCE_SCHEMA,
            cache_key=self.key("relevance", question.strip(), *(n.id for n in candidates)),
        )
        if not data:
            return []
        picks: list[Pick] = []
        known = {n.id for n in candidates}
        for raw in data.get("picks") or []:
            if not isinstance(raw, dict):
                continue
            ident = str(raw.get("id") or "")
            if ident not in known:
                continue
            picks.append(
                Pick(
                    id=ident,
                    verdict=str(raw.get("verdict") or "unrelated").strip().lower(),
                    descend=bool(raw.get("descend")),
                    why=str(raw.get("why") or ""),
                )
            )
        return picks


    # ------------------------------------------------------------- sessions
    def assess(self, digest: str) -> dict[str, Any] | None:
        """How did this session go, and what was learned from it?

        Replaces what used to be regex phrase banks with hand-tuned weights.
        Whether "actually, let's use the other one" is a correction or a change
        of mind is a reading of intent, and a pattern list cannot do it — it can
        only match the surface forms someone thought of in advance.
        """
        if not digest.strip():
            return None
        return self.ask(
            ASSESS.format(digest=truncate(digest, 12000)),
            ASSESS_SCHEMA,
            cache_key=self.key("assess", digest.strip()),
        )

    def related(self, new_lesson: str, candidates: list[Node]) -> list[Pick]:
        """Which existing lessons might be about the same thing as this new one?

        Same walk primitive as :meth:`relevance`, different question. Kept
        separate because "would this help with that work" and "is this about the
        same subject" pull apart: two lessons can cover identical ground and
        neither be useful for a given task.
        """
        if not candidates:
            return []
        rendered = "\n\n".join(_render(node) for node in candidates)
        data = self.ask(
            RELATED.format(new=truncate(new_lesson, 3000), candidates=rendered),
            RELEVANCE_SCHEMA,
            cache_key=self.key("related", new_lesson.strip(), *(n.id for n in candidates)),
        )
        if not data:
            return []
        known = {n.id for n in candidates}
        picks: list[Pick] = []
        for raw in data.get("picks") or []:
            if not isinstance(raw, dict) or str(raw.get("id") or "") not in known:
                continue
            picks.append(
                Pick(
                    id=str(raw["id"]),
                    verdict=str(raw.get("verdict") or "unrelated").strip().lower(),
                    descend=bool(raw.get("descend")),
                    why=str(raw.get("why") or ""),
                )
            )
        return picks

    # ----------------------------------------------------------- descent
    def rank_repairs(self, failure: str, options: list[tuple[str, str, str]]) -> dict[str, float]:
        """Which of these dropped details would fix this failure?

        ``options`` is (key, kind, text). Returns key -> 0..1 usefulness.

        This is the core of the descent policy, and it is a semantic question:
        does *this* omitted claim explain *that* failure. Matching the claim's
        `kind` against a diagnosis category and counting shared words is a
        shadow of the real judgement — it cannot tell that "parse the body, not
        the status" addresses "treated a 200 as success".
        """
        if not options:
            return {}
        rendered = "\n\n".join(
            f"[key: {key}] ({kind})\n{truncate(text, 500)}" for key, kind, text in options
        )
        data = self.ask(
            REPAIR.format(failure=truncate(failure, 2500), options=rendered),
            REPAIR_SCHEMA,
            cache_key=self.key("repair", failure.strip(), *(o[0] for o in options)),
        )
        if not data:
            return {}
        out: dict[str, float] = {}
        known = {key for key, _, _ in options}
        for raw in data.get("ranked") or []:
            if not isinstance(raw, dict):
                continue
            key = str(raw.get("key") or "")
            if key in known:
                try:
                    out[key] = max(0.0, min(1.0, float(raw.get("usefulness", 0))))
                except (TypeError, ValueError):
                    continue
        return out


ASSESS_SCHEMA = {
    "type": "object",
    "required": ["outcome", "confidence", "corrected"],
    "properties": {
        "outcome": {
            "type": "string",
            "enum": ["success", "failure", "unknown"],
            "description": "Did the work end in a correct, accepted state?",
        },
        "confidence": {"type": "number", "description": "0..1 in the outcome."},
        "corrected": {
            "type": "boolean",
            "description": "Did the human have to steer the agent away from a wrong approach?",
        },
        "correction": {
            "type": "string",
            "description": "What the human actually corrected, in their terms. Empty if none.",
        },
        "evidence": {"type": "array", "items": {"type": "string"}},
        "discoveries": {
            "type": "array",
            "description": "Things worked out by trial, with no human involvement.",
            "items": {
                "type": "object",
                "required": ["what_failed", "what_worked"],
                "properties": {
                    "what_failed": {"type": "string"},
                    "why_it_failed": {"type": "string"},
                    "what_worked": {"type": "string"},
                    "attempts": {"type": "integer"},
                },
            },
        },
        "summary": {
            "type": "string",
            "description": "What was actually done, if the outcome was success. One or two sentences.",
        },
    },
}

ASSESS = """RMC:assess

Read this record of a finished coding session and judge how it went. You are not
reviewing the work; you are deciding what can be learned from it.

`outcome` — did the session end with the task correctly done? Judge the end
state, not the path taken. Work that failed several times and was then fixed
ended in success.

`corrected` — did the *human* have to steer the agent away from a wrong
approach? This is a separate question from the outcome, and both can be true: a
session can end perfectly precisely because the user intervened. Be careful not
to read a user's clarification, extra request, or change of mind as a
correction; a correction means the agent was going the wrong way.

`discoveries` — what did the agent work out by trial, with no human help? A
command that failed and a different one that worked; a test that rejected an
approach; an API that behaved unexpectedly. For each, record what failed and
*why*, and what worked. These are the most valuable thing in most sessions,
because they let the next agent skip the detour entirely. An identical command
retried until it succeeded is flakiness, not a discovery.

`confidence` — be honest. A short session with no clear signal is `unknown` with
low confidence, and that is a perfectly good answer. A wrong label is worse than
no label, because it is used to decide which memories are trusted.

<<<SESSION
{digest}
SESSION>>>
"""

REPAIR_SCHEMA = {
    "type": "object",
    "required": ["ranked"],
    "properties": {
        "ranked": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["key", "usefulness"],
                "properties": {
                    "key": {"type": "string"},
                    "usefulness": {
                        "type": "number",
                        "description": "0 = irrelevant to this failure, 1 = would clearly fix it.",
                    },
                    "why": {"type": "string"},
                },
            },
        }
    },
}

REPAIR = """RMC:repair

An agent was given a compressed lesson and got the task wrong. Below is how it
failed, and the specific details that were removed when that lesson was
compressed.

Score each removed detail from 0 to 1: how likely is it that *this* detail being
absent is what caused *this* failure? Reason about what the agent would have
done differently had it known each one. Do not reward vocabulary overlap with
the failure text — a detail can share no words and still be the cause.

Most options should score near 0. Only one or two, usually, are the real gap.

<<<FAILURE
{failure}
FAILURE>>>

<<<REMOVED DETAILS
{options}
REMOVED DETAILS>>>
"""


def _render(node: Node, *, limit: int = 700) -> str:
    depth = "" if node.is_apex else f" (a detailed form, L{node.level})"
    detail = f", {len(node.dropped)} details available beneath it" if node.dropped else ""
    header = f"[id: {node.id}] {node.title or node.family}{depth}{detail}"
    return f"{header}\n{truncate(node.body, limit)}"


# --------------------------------------------------------------------------- #
# the walk
# --------------------------------------------------------------------------- #


@dataclass
class WalkResult:
    selected: list[Node] = field(default_factory=list)
    picks: dict[str, Pick] = field(default_factory=dict)
    calls: int = 0
    depth_reached: int = 0

    def why(self, node_id: str) -> str:
        pick = self.picks.get(node_id)
        return pick.why if pick else ""


def walk(
    judge: Judge,
    question: str,
    roots: list[Node],
    *,
    expand: Callable[[Node], list[Node]],
    budget: Budget | None = None,
    max_depth: int = 3,
    fanout: int = 12,
) -> WalkResult:
    """Walk abstract → concrete, asking the model where to look.

    This is the structural answer to "how do we search a growing memory without
    scoring everything". The tree is already ordered by abstraction, so the most
    compressed nodes are both the cheapest to show and the ones that summarise
    the most. One question covers a whole level; we descend only into the lines
    the model says might be related, and only when it says the summary was too
    abstract to decide from.

    Cost therefore tracks the *depth* of the tree and the number of plausible
    lines, not the total number of lessons.
    """
    budget = budget or Budget()
    result = WalkResult()
    frontier = [n for n in roots if n is not None]
    seen: set[str] = set()

    for depth in range(max_depth):
        if not frontier or budget.exhausted:
            break
        # Structural gate, not a judgement: showing the model 200 summaries at
        # once degrades its answer, so a wide level is judged in chunks.
        level = [n for n in frontier if n.id not in seen][:fanout]
        if not level:
            break
        seen.update(n.id for n in level)

        if not budget.take():
            break
        picks = judge.relevance(question, level)
        result.calls += 1
        result.depth_reached = depth
        by_id = {n.id: n for n in level}
        nxt: list[Node] = []

        for pick in picks:
            result.picks[pick.id] = pick
            node = by_id.get(pick.id)
            if node is None or not pick.positive:
                continue
            children = expand(node) if pick.descend else []
            if children:
                # The model said the summary was too abstract to judge from, so
                # look at the detail rather than guessing.
                nxt.extend(children)
            else:
                result.selected.append(node)

        frontier = nxt

    # Anything still on the frontier when the budget ran out was judged
    # plausible but never resolved; keep it rather than silently dropping it.
    for node in frontier:
        if node.id not in {n.id for n in result.selected}:
            result.selected.append(node)
    return result


def relevant_only(result: WalkResult) -> list[Node]:
    return [n for n in result.selected if result.picks.get(n.id, Pick(n.id)).verdict == "relevant"]


def chunked(items: Iterable[Any], size: int) -> list[list[Any]]:
    out, cur = [], []
    for item in items:
        cur.append(item)
        if len(cur) >= size:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out
