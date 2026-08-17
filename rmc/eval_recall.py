"""Measuring whether recall serves the right lessons.

Every other stage of RMC is checked against something. Compression replays the
episodes the lesson came from; a merge has to reproduce all its children's; a
delta earns its way back by rescuing a failure. Recall — the stage that decides
what enters the user's context on *every single prompt* — was checked against
nothing at all, and it is the one the user actually feels.

What it costs to get wrong is symmetric, and both directions hurt:

* Serve a lesson that does not bear on the work and you have spent the user's
  context on noise. On this store that came to 15,917 tokens across 57 prompts,
  against 313 tokens injected per prompt on average — most of what recall sent
  was never used.
* Drop a lesson that would have helped and the whole product claim fails. The
  point is that the next session is shorter than the last.

The ground truth is `episode.used`: the lessons a fork of the live session
judged to have actually borne on the work. It is imperfect — a judgement, made
once, about a session that has since ended — but it is a real observation of an
outcome rather than an assertion, and it is the same signal attribution already
runs on.

The one methodological care that matters: a lesson is only scored if it was
*served* in that episode. A lesson nobody was shown could not have been used, so
counting its absence from `used` as evidence against it would manufacture false
positives out of the retrieval decision. So each episode is replayed with
exactly the candidate set it saw, and the question is narrow and fair:

    given the lessons that were served, would the judge now drop the ones that
    turned out not to matter, and keep the ones that did?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .adapters import Adapter
from .judge import Judge
from .store import Store
from .util import count_tokens


@dataclass
class EpisodeScore:
    episode: str
    prompt: str
    kept: set[str] = field(default_factory=set)  # judged worth serving
    used: set[str] = field(default_factory=set)  # actually bore on the work
    served: set[str] = field(default_factory=set)  # shown at the time
    tokens: dict[str, int] = field(default_factory=dict)

    @property
    def hits(self) -> set[str]:
        """Used and kept — the lessons recall exists to deliver."""
        return self.kept & self.used

    @property
    def noise(self) -> set[str]:
        """Kept but never used — context spent for nothing."""
        return self.kept - self.used

    @property
    def misses(self) -> set[str]:
        """Used but would now be dropped — the expensive kind of error."""
        return self.used - self.kept

    def tokens_of(self, ids: set[str]) -> int:
        return sum(self.tokens.get(i, 0) for i in ids)


@dataclass
class Report:
    scores: list[EpisodeScore] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    # Which selector produced these numbers. Carried on the report rather than
    # remembered by the caller, because the two arms are scored against
    # different candidate sets and a table that does not say which is which
    # invites exactly the comparison it cannot support.
    arm: str = "judge"
    searches: int = 0

    # -- aggregates ------------------------------------------------------- #
    @property
    def kept(self) -> int:
        return sum(len(s.kept) for s in self.scores)

    @property
    def used(self) -> int:
        return sum(len(s.used) for s in self.scores)

    @property
    def hits(self) -> int:
        return sum(len(s.hits) for s in self.scores)

    @property
    def misses(self) -> int:
        return sum(len(s.misses) for s in self.scores)

    @property
    def precision(self) -> float:
        """Of what recall would serve, how much bears on the work."""
        return self.hits / self.kept if self.kept else 0.0

    @property
    def recall_rate(self) -> float:
        """Of what would have helped, how much recall still delivers.

        Reported alongside precision and never on its own. Precision is trivial
        to maximise by serving nothing, so a change that improves one while
        quietly destroying the other has to be visible in the same table.
        """
        return self.hits / self.used if self.used else 0.0

    @property
    def noise_tokens(self) -> int:
        return sum(s.tokens_of(s.noise) for s in self.scores)

    @property
    def useful_tokens(self) -> int:
        return sum(s.tokens_of(s.hits) for s in self.scores)

    @property
    def baseline_tokens(self) -> int:
        """What was actually spent at the time, to compare against."""
        return sum(s.tokens_of(s.served) for s in self.scores)

    def to_markdown(self) -> str:
        lines = [
            "| episode | served | kept | used | hits | misses | noise tok |",
            "|---|---|---|---|---|---|---|",
        ]
        for s in self.scores:
            lines.append(
                f"| {s.episode} | {len(s.served)} | {len(s.kept)} | {len(s.used)} "
                f"| {len(s.hits)} | {len(s.misses)} | {s.tokens_of(s.noise)} |"
            )
        lines += [
            "",
            f"arm            {self.arm}"
            + (
                "  (candidates: the whole store, searched cold)"
                if self.arm == "agentic"
                else "  (candidates: exactly what the episode was served)"
            ),
            f"precision      {self.precision:.0%}  ({self.hits}/{self.kept} kept lessons bore on the work)",
            f"recall         {self.recall_rate:.0%}  ({self.hits}/{self.used} useful lessons still delivered)",
            f"noise tokens   {self.noise_tokens}",
            f"useful tokens  {self.useful_tokens}",
            f"was spending   {self.baseline_tokens} tokens to deliver the same {self.used} useful lessons",
        ]
        if self.searches:
            lines.append(
                f"searches       {self.searches} across {len(self.scores)} episodes "
                f"({self.searches / max(1, len(self.scores)):.1f} per selection)"
            )
        if self.skipped:
            lines.append(f"skipped        {len(self.skipped)} episode(s) with nothing to score")
        return "\n".join(lines)


ARMS = ("judge", "agentic")


def run(store: Store, adapter: Adapter, *, limit: int = 0, arm: str = "judge") -> Report:
    """Re-run selection over episodes whose outcome we know.

    Two arms, and they do not face the same question — which is the point, and
    also the thing a reader of the numbers has to be told.

    ``judge`` replays each episode against **exactly the lessons it was served**.
    A lesson nobody was shown could not have been used, so counting its absence
    from `used` as evidence against it would manufacture false positives out of
    the retrieval decision. This is the baseline: 48% precision at 100% recall.

    ``agentic`` gives the selector the whole store and lets it search. The
    denominator is different by necessity — the candidate set genuinely is
    everything — and that cuts both ways. Recall can now exceed the judge's,
    because the judge could only ever pick from what an earlier version of
    retrieval had already chosen; and precision is measured against a much
    larger set of things it could have wrongly picked. The two columns are
    comparable as *outcomes* and not as *scores on one test*, so the report
    labels the arm rather than presenting them as like-for-like.

    The agentic arm here is also **cold**: a fresh process, no conversation in
    front of it. In production the selector forks the live session and has the
    task's tool calls and reasoning. So this arm is a floor on what agentic
    selection does, not an estimate of it.
    """
    arm = (arm or "judge").strip().lower()
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm!r} (want {' | '.join(ARMS)})")

    report = Report()
    report.arm = arm
    episodes = [e for e in store.episodes() if e.served and e.used is not None]
    if limit:
        episodes = episodes[:limit]

    for episode in episodes:
        if not episode.prompt.strip():
            report.skipped.append(episode.id)
            continue
        served = [n for n in (store.get(i) for i in episode.served) if n is not None]
        if not served:
            report.skipped.append(episode.id)
            continue

        # `used` may name lessons since compressed away; only score what exists.
        used = {i for i in (episode.used or []) if store.get(i) is not None}

        if arm == "judge":
            # A judge per episode, so nothing is carried between them — and its
            # own cache means re-running the eval on an unchanged store is free.
            picks = Judge(store, adapter).relevance(episode.prompt, served)
            verdicts = {p.id: p.verdict for p in picks}
            candidates = served
            # An unanswered candidate counts as kept. A judge that silently
            # omits a lesson has not decided to drop it, and scoring silence as
            # a decision would flatter any change that makes the model answer
            # less.
            kept = {n.id for n in served if verdicts.get(n.id, "relevant") != "unrelated"}
        else:
            from . import select_agent

            result = select_agent.select(store, adapter, episode.prompt, session_id="")
            if result.failed:
                report.skipped.append(episode.id)
                continue
            candidates = store.nodes()
            kept = {n.id for n in result.selected}
            report.searches += len(result.searched)

        score = EpisodeScore(
            episode=episode.id,
            prompt=episode.prompt,
            served={n.id for n in served},
            used=used,
            kept=kept,
            tokens={n.id: n.tokens for n in candidates},
        )
        report.scores.append(score)

    return report


def compare(before: Report, after: Report) -> str:
    """Two runs side by side, because a single number proves nothing.

    Precision alone is maximised by serving nothing, so any claim about it has
    to be read next to what it cost in lessons no longer delivered.
    """
    rows = [
        ("precision", f"{before.precision:.0%}", f"{after.precision:.0%}"),
        ("recall", f"{before.recall_rate:.0%}", f"{after.recall_rate:.0%}"),
        ("useful lessons delivered", str(before.hits), str(after.hits)),
        ("lessons dropped that helped", str(before.misses), str(after.misses)),
        ("noise tokens", str(before.noise_tokens), str(after.noise_tokens)),
    ]
    width = max(len(r[0]) for r in rows)
    lines = [f"{'':{width}}   {before.arm:>6}    {after.arm:>6}"]
    for name, b, a in rows:
        lines.append(f"{name:{width}}   {b:>6}    {a:>6}")
    if before.arm != after.arm:
        # Said every time the arms differ, because the numbers look directly
        # comparable and are not: the agentic arm chose from the whole store
        # while the judge chose from what retrieval had already narrowed for it.
        lines += [
            "",
            "these arms faced different candidate sets — read them as two outcomes,",
            "not as two scores on the same test",
        ]
    return "\n".join(lines)
