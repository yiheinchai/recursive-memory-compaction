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
            f"precision      {self.precision:.0%}  ({self.hits}/{self.kept} kept lessons bore on the work)",
            f"recall         {self.recall_rate:.0%}  ({self.hits}/{self.used} useful lessons still delivered)",
            f"noise tokens   {self.noise_tokens}",
            f"useful tokens  {self.useful_tokens}",
            f"was spending   {self.baseline_tokens} tokens to deliver the same {self.used} useful lessons",
        ]
        if self.skipped:
            lines.append(f"skipped        {len(self.skipped)} episode(s) with nothing to score")
        return "\n".join(lines)


def run(store: Store, adapter: Adapter, *, limit: int = 0) -> Report:
    """Re-ask the relevance judge about episodes whose outcome we know.

    Each episode is replayed against exactly the lessons it was served, so the
    judge faces the decision it actually faced and the score is not polluted by
    lessons it never had the chance to pick.
    """
    report = Report()
    episodes = [e for e in store.episodes() if e.served and e.used is not None]
    if limit:
        episodes = episodes[:limit]

    for episode in episodes:
        candidates = [n for n in (store.get(i) for i in episode.served) if n is not None]
        if not candidates or not episode.prompt.strip():
            report.skipped.append(episode.id)
            continue

        # A judge per episode, so nothing is carried between them — and its own
        # cache means re-running the eval on an unchanged store is free.
        picks = Judge(store, adapter).relevance(episode.prompt, candidates)
        verdicts = {p.id: p.verdict for p in picks}

        score = EpisodeScore(
            episode=episode.id,
            prompt=episode.prompt,
            served={n.id for n in candidates},
            # `used` may name lessons since merged away; only score what exists.
            used={i for i in (episode.used or []) if store.get(i) is not None},
            # An unanswered candidate counts as kept. A judge that silently
            # omits a lesson has not decided to drop it, and scoring silence as
            # a decision would flatter any change that makes the model answer
            # less.
            kept={n.id for n in candidates if verdicts.get(n.id, "relevant") != "unrelated"},
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
    lines = [f"{'':{width}}   before    after"]
    for name, b, a in rows:
        lines.append(f"{name:{width}}   {b:>6}    {a:>6}")
    return "\n".join(lines)
