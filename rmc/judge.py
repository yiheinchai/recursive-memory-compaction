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
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .adapters import Adapter, Session
from .node import Node
from .store import Store
from .util import count_tokens, truncate

# --------------------------------------------------------------------------- #
# schemas
# --------------------------------------------------------------------------- #

def criteria_version(store: Any = None) -> str:
    """Fingerprint of the prompts that define a judgement, as resolved.

    Cheap, and it makes every cached verdict self-invalidating: change a prompt
    and the old answers become unreachable rather than silently authoritative.

    Resolved, not declared — a store may override any of these on disk, and a
    fingerprint of the shipped text would leave every override answering with
    the cache of the text it replaced. That is the exact failure this function
    was added to prevent, one level up.
    """
    material = "".join(
        prompt(store, name, default)
        for name, default in (
            ("relevance", RELEVANCE),
            ("warm_relevance", WARM_RELEVANCE),
            ("related", RELATED),
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:8]


def prompt(store: Any, name: str, default: str) -> str:
    """The text to use for a judgement, allowing a store-local override.

    Prompts are the largest lever on judgement quality and were the one part of
    the system that could not be changed without editing the package. That made
    them untunable in place, and untunable means unmeasurable: `rmc tune`
    proposes a wording, scores it against recorded outcomes, and keeps it only
    if it wins — none of which is possible while the text is a module constant.

    A malformed or unreadable override falls back to the shipped text rather
    than failing the call. Recall runs in a hook; an experiment must never be
    able to take memory offline.
    """
    if store is None:
        return default
    try:
        path = store.root / "prompts" / f"{name}.md"
        text = path.read_text(encoding="utf-8").strip() if path.exists() else ""
        return text or default
    except Exception:
        return default


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

SEED = """RMC:candidates

Below is the set of remembered lessons this project has accumulated. Questions
about which of them apply to a given piece of work will follow, one at a time.

Read them now. Reply with the single word READY and nothing else.

<<<LESSONS
{candidates}
LESSONS>>>
"""


WARM_RELEVANCE = """RMC:relevance

Decide which of the lessons above apply to the piece of work below. Same rules
as always:

  - `relevant`  — knowing this would change how the work is done.
  - `maybe`     — same general area, but you cannot tell from the summary alone
                  whether it applies. Set `descend: true` if there is more
                  detail available and seeing it would settle the question.
  - `unrelated` — it would only add noise.

Be strict. An irrelevant lesson costs the reader attention and can actively
mislead. Superficial word overlap is not relevance: a lesson about retrying
HTTP calls is unrelated to a request to retry a failed CI job.

Judge what the work actually needs, not what it superficially mentions.

<<<WORK
{question}
WORK>>>
"""


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
        timeout: int | None = None,
    ) -> None:
        self.store = store
        self.adapter = adapter
        self.cache_name = cache_name
        self.use_cache = use_cache
        # A hook has a hard deadline. Bounding the call here means recall
        # degrades to "inject nothing" rather than being killed mid-flight with
        # its output discarded.
        self.timeout = timeout
        self.calls = 0
        self.failures = 0
        self.last_error = ""
        # Chunks of one level are judged concurrently, so the cache is written
        # from several threads. Without this two of them read the same file,
        # each adds its own entry, and the second write erases the first — the
        # judgements are correct but half of them are paid for again next time.
        self._lock = threading.Lock()
        # Optional: keeps the candidate list warm across prompts. Absent for
        # one-off judgements, where there is no prefix worth preserving.
        self.router = None
        # The chunk hashes worth keeping warm: the apex layer, split the same
        # way the walk splits it. Those are identical on every prompt. A walk
        # also asks about the children it descends into, and those sets differ
        # per question, so warming them would reseed on every call and turn the
        # optimisation into pure overhead. The caller names the stable set.
        self.warm_prefixes: set[str] = set()

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
        # Read-modify-write, so it has to be atomic against the other chunks
        # being judged at the same moment.
        with self._lock:
            cache = self._load()
            cache[key] = value
            if len(cache) > limit:
                for stale in list(cache)[: len(cache) - limit]:
                    cache.pop(stale, None)
            try:
                self._cache_path().write_text(json.dumps(cache), encoding="utf-8")
            except Exception:
                pass

    def key(self, *parts: str) -> str:
        """Cache key for one judgement, including the criteria that produced it.

        Keying only on the question and the candidates makes the cache outlive
        the standard it was answered against: sharpen the relevance prompt and
        every previously-asked question keeps returning the old verdict, so the
        change looks like it did nothing. That is not hypothetical — the first
        A/B run of the recall eval came back byte-identical to its baseline
        after a full prompt rewrite, and the rewrite had never been used.

        The backend and model belong in the key for the same reason. Switching
        the routing model and re-running the eval returned numbers identical to
        the previous model's, down to the token — the second model was never
        asked anything. A cached judgement is reusable only while both the
        criteria and the judge behind it hold.

        Same failure as the nudge backoff, which went stale for the same reason
        and was fixed the same way.
        """
        who = f"{getattr(self.adapter, 'name', '?')}:{getattr(self.adapter, 'model', None) or 'default'}"
        return hashlib.sha256(
            "\x1f".join((criteria_version(self.store), who, *parts)).encode("utf-8")
        ).hexdigest()[:16]

    def ask(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        cache_key: str | None = None,
        timeout: int | None = None,
        session: Any = None,
        prefix_tokens: int = 0,
        prefix_hash: str = "",
        router: Any = None,
    ) -> dict[str, Any] | None:
        """One structured judgement. Returns None if the model could not answer."""
        if cache_key:
            cached = self._load().get(cache_key)
            if cached is not None:
                return cached
        run = self.adapter.run(
            prompt,
            schema=schema,
            timeout=timeout
            or self.timeout
            or int(self.store.config.get("limits.agent_timeout_s", 180)),
            **({"session": session} if session is not None else {}),
        )
        if router is not None:
            # Whether the warm prefix actually held is only knowable from what
            # came back, so it is recorded even when the answer was unusable.
            router.record(
                cached_in=run.cached_in,
                prefix_tokens=prefix_tokens,
                prefix_hash=prefix_hash,
            )
        self.calls += 1
        if not run.ok or not run.data:
            # Counted, because "the model said nothing is relevant" and "the
            # model never answered" produce the same empty list and mean
            # opposite things. Since relevance filtering became unconditional,
            # a backend that is down or slow would otherwise switch memory off
            # in complete silence — the worst way for this to fail, because the
            # user concludes RMC does not work rather than that it is broken.
            self.failures += 1
            self.last_error = (run.error or "no parseable answer")[:200]
            return None
        if cache_key:
            self._store_cache(cache_key, run.data)
        return run.data

    # ------------------------------------------------------------- relevance
    def _warm(self, rendered: str):
        """Seed or reuse the conversation holding the candidate list.

        Seeding costs one extra call the first time and whenever the apex layer
        changes. It is worth it only because the list is re-asked about many
        times between changes; if lessons churned every prompt this would be
        pure overhead, which is why the prefix hash drives the decision.
        """
        digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]
        if digest not in self.warm_prefixes:
            return None, 0, ""  # a descent set: asked once, never worth seeding

        # Seeding decides and writes shared router state, and concurrent chunks
        # would otherwise each conclude they must reseed and overwrite one
        # another's session ids — leaving conversations nobody can find again.
        with self._lock:
            session = self.router.session_for(digest)
        if not session.resume:
            seed = self.adapter.run(
                SEED.format(candidates=rendered),
                timeout=self.timeout
                or int(self.store.config.get("limits.agent_timeout_s", 180)),
                session=session,
            )
            self.calls += 1
            if not seed.ok:
                return None, 0, ""  # could not seed; ask the ordinary way
            # The seeding call is not a result to score — it is how this
            # conversation's two reference readings are taken: what was cached
            # before our prefix existed, and how much of ours got written.
            self.router.record(
                cached_in=seed.cached_in,
                created=seed.created_in,
                prefix_tokens=count_tokens(rendered),
                prefix_hash=digest,
                seeded=True,
            )
            session = Session(id=session.id, resume=True)
        return session, count_tokens(rendered), digest

    def relevance(self, question: str, candidates: list[Node]) -> list[Pick]:
        """Which of these lessons bear on this work?

        The candidate list is the same on every prompt and the question is not,
        so when a router is supplied the list is seeded once into a conversation
        and each prompt branches a fork from it. At today's size that saves
        little; the reason it exists is that the list grows with the store and
        re-sending it every prompt is what makes routing unaffordable at scale.
        """
        if not candidates:
            return []
        rendered = "\n\n".join(_render(node) for node in candidates)
        session = prefix_tokens = digest = None
        if self.router is not None:
            session, prefix_tokens, digest = self._warm(rendered)

        # When the conversation already holds the candidates, the question must
        # not carry them again. Repeating them defeats the entire point — the
        # copy in the prompt is new text at full price, and it is the copy the
        # model reads, so the cached one is paid for and ignored.
        text = (
            prompt(self.store, "warm_relevance", WARM_RELEVANCE).format(
                question=truncate(question, 3000)
            )
            if session is not None
            else prompt(self.store, "relevance", RELEVANCE).format(
                question=truncate(question, 3000), candidates=rendered
            )
        )
        data = self.ask(
            text,
            RELEVANCE_SCHEMA,
            cache_key=self.key("relevance", question.strip(), *(n.id for n in candidates)),
            session=session,
            prefix_tokens=prefix_tokens or 0,
            prefix_hash=digest or "",
            router=self.router if session is not None else None,
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


    # --------------------------------------------------------------- scope
    def scope(self, body: str, *, repo: str = "") -> dict[str, Any] | None:
        """Does this lesson belong to this repository, or everywhere?

        Reachability has to be decided when a lesson is written, because
        nothing downstream can repair it. A lesson filed in a project store is
        invisible from every other project — so it can never be retrieved
        there, never used there, and therefore never co-used into a shared
        abstraction. Co-use strengthens links between lessons that are already
        mutually reachable; it cannot create reachability.
        """
        return self.ask(
            SCOPE.format(repo=repo or "(unnamed)", body=truncate(body, 4000)),
            SCOPE_SCHEMA,
            cache_key=self.key("scope", body.strip()),
        )

    # ------------------------------------------------------------- sessions
    def assess(self, digest: str, served: list[Node] | None = None) -> dict[str, Any] | None:
        """How did this session go, and what was learned from it?

        Replaces what used to be regex phrase banks with hand-tuned weights.
        Whether "actually, let's use the other one" is a correction or a change
        of mind is a reading of intent, and a pattern list cannot do it — it can
        only match the surface forms someone thought of in advance.
        """
        if not digest.strip():
            return None
        rendered = (
            "\n".join(f"[{n.id}] {n.title or n.family} — {n.summary()}" for n in (served or []))
            or "(no lessons were recalled for this session)"
        )
        return self.ask(
            ASSESS.format(digest=truncate(digest, 12000), served=rendered),
            ASSESS_SCHEMA,
            cache_key=self.key("assess", digest.strip(), *(n.id for n in (served or []))),
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
            prompt(self.store, "related", RELATED).format(
                new=truncate(new_lesson, 3000), candidates=rendered
            ),
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

    # ------------------------------------------------------------- worth
    def worth_keeping(
        self, original: str, candidate: str, dropped: list[str], measured: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Is this compression an improvement worth keeping?

        Separate from *correctness*, which replay settles mechanically. This is
        the question of worth, and worth has two axes: a candidate can be better
        because it costs less, or because it says the same thing at a higher
        level and so covers cases the original did not. A token count sees only
        the first, and will reject a genuinely better abstraction for saving
        22% instead of 25%.

        The measurements go in as evidence — they are facts and belong in the
        prompt — but the verdict is the model's.
        """
        return self.ask(
            WORTH.format(
                original=truncate(original, 4000),
                candidate=truncate(candidate, 4000),
                dropped="\n".join(f"- {d}" for d in dropped) or "(nothing declared)",
                before=measured.get("before", 0),
                after=measured.get("after", 0),
                ratio=measured.get("ratio", 1.0),
                target=measured.get("target_ratio", 0.75),
            ),
            WORTH_SCHEMA,
            cache_key=self.key("worth", original.strip(), candidate.strip()),
        )

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


SCOPE_SCHEMA = {
    "type": "object",
    "required": ["scope", "why"],
    "properties": {
        "scope": {"type": "string", "enum": ["project", "global"]},
        "why": {"type": "string"},
    },
}

SCOPE = """RMC:scope

Decide where a newly learned lesson should live.

`project` — it depends on *this particular repository*: its code, its layout,
its conventions, its deployment setup, its test fixtures, its team's decisions.
Useless or misleading anywhere else.

`global` — it would be true for anyone using these tools, languages, services or
APIs, on any codebase. Vendor behaviour, protocol quirks, pricing, general
engineering judgement, or how you should work.

The test is not what the lesson was *discovered* in — almost everything is
discovered inside some project. Ask instead: if someone opened a completely
unrelated repository tomorrow and hit this same tool or service, would this
lesson still be right and still be useful? If yes, it is `global`.

Getting this wrong is asymmetric. A global lesson filed as project-scoped
becomes invisible everywhere else and can never be found again from the place it
would have helped most. A project lesson filed as global is merely noise, and
noise is recoverable.

Repository: {repo}

<<<LESSON
{body}
LESSON>>>
"""

WORTH_SCHEMA = {
    "type": "object",
    "required": ["keep", "why", "generality"],
    "properties": {
        "keep": {"type": "boolean"},
        "generality": {
            "type": "string",
            "enum": ["more", "same", "less"],
            "description": "Does the candidate cover more situations than the original?",
        },
        "why": {"type": "string"},
    },
}

WORTH = """RMC:worth

A lesson has been compressed. Decide whether the shorter version is worth
keeping as a new level above the original. This is not about correctness — that
is tested separately by replaying real tasks against it. This is about whether
it is an *improvement*.

A candidate earns its place on either of two axes:

1. **It costs less.** Fewer tokens on every future recall, with nothing
   important lost.
2. **It is more general.** It states at a higher level what the original said
   about one case, so it now covers situations the original did not. This can be
   worth keeping even when the token saving is small — an abstraction that
   applies to five situations instead of one is more valuable per token, and
   token count cannot see that.

Refuse it when: it saves little AND generalises nothing (a reworded copy); or it
became shorter by becoming vaguer, so an agent reading it would no longer know
what to actually do. Vagueness is not generality. "Handle errors properly" is
shorter and broader than a specific retry rule and is worth nothing.

Measured, as evidence — the numbers are facts, the verdict is yours:
  original:  {before} tokens
  candidate: {after} tokens  ({ratio:.0%} of the original)
  the compressor was asked to reach {target:.0%}

<<<ORIGINAL
{original}
ORIGINAL>>>

<<<CANDIDATE
{candidate}
CANDIDATE>>>

<<<DECLARED AS REMOVED
{dropped}
DECLARED AS REMOVED>>>
"""

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
        "lessons_used": {
            "type": "array",
            "description": "Which of the recalled lessons actually bore on the work.",
            "items": {
                "type": "object",
                "required": ["id", "used"],
                "properties": {
                    "id": {"type": "string"},
                    "used": {
                        "type": "boolean",
                        "description": "True only if this lesson changed what was done.",
                    },
                    "how": {"type": "string", "description": "What it changed, if used."},
                },
            },
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

`lessons_used` — the lessons below were injected into this session before the
work started. For each, say whether it actually *bore on* what happened.

Be strict, and default to `false`. A lesson counts as used only if the work
would plausibly have gone differently without it: it supplied a fact that was
acted on, ruled out an approach, or named a constraint that was respected.
Being on-topic is not being used. Being read and found irrelevant is not being
used. If the same result would have followed without it, it was not used.

This matters more than it looks. These verdicts decide which lessons get credit
for the outcome, and which get abstracted together into a shared parent. Marking
everything used means an irrelevant lesson accrues a record of usefulness it did
not earn, and unrelated lessons get merged because they happened to be shown at
the same time.

`confidence` — be honest. A short session with no clear signal is `unknown` with
low confidence, and that is a perfectly good answer. A wrong label is worse than
no label, because it is used to decide which memories are trusted.

<<<RECALLED LESSONS
{served}
RECALLED LESSONS>>>

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


def _ask_all(
    judge: "Judge", question: str, chunks: list[list[Node]], workers: int
) -> list[tuple[list[Node], list[Pick]]]:
    """Put every chunk to the model at once, and return the answers in order.

    Order is preserved so that what gets selected does not depend on which
    subprocess finished first — a recall that returns different lessons run to
    run is not debuggable, and the eval would measure scheduling noise.

    A chunk that raises is treated as a chunk that answered nothing. One failed
    subprocess must not take down a recall that the other chunks answered.
    """
    if len(chunks) <= 1 or workers <= 1:
        return [(chunk, judge.relevance(question, chunk)) for chunk in chunks]

    from concurrent.futures import ThreadPoolExecutor

    def ask(chunk: list[Node]) -> list[Pick]:
        try:
            return judge.relevance(question, chunk)
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=min(workers, len(chunks))) as pool:
        answers = list(pool.map(ask, chunks))
    return list(zip(chunks, answers))


def _render(node: Node) -> str:
    """One compact line per lesson, for deciding which to open.

    Routing used to send 700 characters of body per candidate, which made the
    decision cost grow with the store until choosing what to load cost more than
    loading everything — 1000 lessons came to ~185k tokens spent purely on
    triage. A gist is ~30.
    """
    depth = "" if node.is_apex else f" L{node.level}"
    detail = f" [+{len(node.dropped)} details beneath]" if node.dropped else ""
    return f"[{node.id}]{depth} {node.title or node.family}{detail} — {node.summary()}"


# --------------------------------------------------------------------------- #
# the walk
# --------------------------------------------------------------------------- #


@dataclass
class WalkResult:
    selected: list[Node] = field(default_factory=list)
    picks: dict[str, Pick] = field(default_factory=dict)
    calls: int = 0
    depth_reached: int = 0
    # Set when the judge could not answer. Distinguishes an empty pack that is
    # a decision from one that is an outage.
    failed: bool = False
    error: str = ""

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
    workers: int = 4,
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
        pending = [n for n in frontier if n.id not in seen]
        if not pending:
            break

        # Showing the model two hundred summaries at once degrades its answer,
        # so a wide level is judged in chunks — but *every* chunk, not the first
        # one. This used to read `[:fanout]` and drop the remainder on the
        # floor: with 26 apexes and a fanout of 12, fourteen lessons could never
        # be retrieved on any prompt, ever, and nothing reported it. Precision
        # measured over the reachable dozen looked perfectly healthy.
        #
        # The budget still bounds the work; that is the difference. Running out
        # of budget is a decision to stop looking, recorded and recoverable,
        # while truncation was a silent hole in the store.
        chunks: list[list[Node]] = []
        for start in range(0, len(pending), fanout):
            if not budget.take():
                # Unexamined nodes stay on the frontier, so the tail below keeps
                # them rather than discarding them unseen.
                leftover = pending[start:]
                break
            chunks.append(pending[start : start + fanout])
        else:
            leftover = []

        # The chunks of one level are independent questions about disjoint sets
        # of lessons, so they are asked at once rather than in turn. This is not
        # a micro-optimisation: recall runs in a hook that blocks the user's
        # prompt, each call costs ~15s of which ~5s is process startup, and the
        # number of chunks grows with the store. Sequentially, a store large
        # enough to need six chunks would exceed the recall timeout and serve
        # nothing — so completeness at the top level would cost the whole
        # feature. Concurrency is what makes covering every lesson affordable.
        nxt: list[Node] = list(leftover)
        for chunk in chunks:
            seen.update(n.id for n in chunk)
        for chunk, picks in _ask_all(judge, question, chunks, workers):
            result.calls += 1
            result.depth_reached = depth
            by_id = {n.id: n for n in chunk}

            for pick in picks:
                result.picks[pick.id] = pick
                node = by_id.get(pick.id)
                if node is None or not pick.positive:
                    continue
                children = expand(node) if pick.descend else []
                if children:
                    # The model said the summary was too abstract to judge from,
                    # so look at the detail rather than guessing.
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
