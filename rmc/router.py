"""Keeping the routing prefix warm across prompts.

Every routing call asks a different question of the *same* candidate list — the
apex layer, re-rendered and re-sent on every prompt of every session. Today that
list is 1,311 tokens and the waste is affordable. The design has to survive
5,000 lessons, where at the current apex ratio it would be ~225k tokens per
prompt, and re-sending it each time is the difference between a working system
and an unusable one.

Providers already solve this, with prompt caching: an identical prefix is served
from cache at roughly a tenth of the price. The catch is that the prefix must
arrive as *the same conversation*, and a fresh `claude -p` is a new one every
time. So the candidate list is seeded once into a session, and each prompt
branches a throwaway fork from it. The fork answers against the stored prefix
without appending to it, so the next prompt finds the conversation exactly where
it was — one stable prefix, many questions, no transcript growth.

**Warmth is measured, never assumed.** Cache TTLs are not published per request
and vary with load and plan, so any local timer is a guess that will be wrong in
both directions — reseeding a prefix that was still warm, or forking against one
that has long expired. The provider reports `cache_read_input_tokens` on every
response, which is ground truth about whether the last call actually hit. This
module keeps the timer only as a cheap first filter and treats the observed
reads as the authority, widening or narrowing its own window from what it sees.

The catch, found the first time this ran against the real CLI: **most of that
number is not ours.** Claude Code sends a ~65k-token system prompt that the
provider caches regardless of anything RMC does, so a naive "were tokens read
from cache?" test reports a hit on every single call, including the cold ones.
The candidate list — 690 tokens in that run — is a rounding error beside it.

So warmth is measured *per conversation*, against the two numbers its own
seeding call reported: what was already cached before our prefix existed, and
how many tokens of ours were written. A fork has hit when its read exceeds the
first by roughly the second.

A single global baseline was tried first and does not work — it drifts. Take the
maximum and one unusually warm seed poisons it into reporting permanent misses,
which is exactly what happened: a stale 65,372 baseline against a real
system-prompt read of 57,558 made every genuine hit come out negative. The
readings are only comparable within the conversation that produced them.

Which keeps the split: the harness measures and counts, and nothing here is a
judgement about meaning.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from .adapters import Session

# Providers commonly offer a short cache window by default and a long one on
# request. Starting between the two costs at most one wasted fork before the
# observed reads pull the window to wherever it actually is.
DEFAULT_TTL_S = 300
MIN_TTL_S = 60
MAX_TTL_S = 3600


@dataclass
class Warm:
    """One seeded conversation, holding one chunk of the candidate list.

    The two readings from the seeding call are what make warmth measurable
    later. `seed_cached` is what the provider had cached before our prefix
    existed — the host's own system prompt, and nothing of ours. `seed_created`
    is how many tokens of ours it then wrote. A fork has hit when its cache read
    exceeds the first by roughly the second.
    """

    session_id: str
    seeded_at: float
    last_call_at: float
    seed_cached: int = 0
    seed_created: int = 0


@dataclass
class RouterState:
    """What we know about the seeded conversations, persisted between prompts.

    Plural, because a wide apex layer is judged in chunks of `fanout` and each
    chunk is its own stable prefix. A single session would be reseeded on every
    chunk and hit nothing — which is exactly what the first implementation did.
    """

    warm: dict[str, Warm] = field(default_factory=dict)
    # Observed, not configured: the window is widened when a fork this old
    # still hit the cache, and narrowed when one this young missed.
    ttl_s: float = DEFAULT_TTL_S
    hits: int = 0
    misses: int = 0
    tokens_saved: int = 0
    # What the provider caches without our help — chiefly the host agent's own
    # system prompt. A fork has to beat this to have achieved anything.
    baseline_cached: int = 0

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["warm"] = {k: asdict(v) for k, v in self.warm.items()}
        return out

    @classmethod
    def from_dict(cls, raw: Any) -> "RouterState":
        raw = raw if isinstance(raw, dict) else {}
        known = {f for f in cls.__dataclass_fields__}
        state = cls(**{k: v for k, v in raw.items() if k in known and k != "warm"})
        for key, value in (raw.get("warm") or {}).items():
            if isinstance(value, dict):
                state.warm[key] = Warm(**{k: value[k] for k in Warm.__dataclass_fields__ if k in value})
        return state

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class Router:
    """Decides, per prompt, whether to branch the warm prefix or reseed it."""

    def __init__(self, store: Any) -> None:
        self.store = store
        self.path = store.root / "router.json"
        self.state = self._load()

    # -- persistence ------------------------------------------------------ #
    def _load(self) -> RouterState:
        try:
            return RouterState.from_dict(json.loads(self.path.read_text(encoding="utf-8")))
        except Exception:
            # A missing or corrupt file must never block a prompt. The cost of
            # getting this wrong is one cold call.
            return RouterState()

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.state.to_dict(), indent=2), encoding="utf-8")
        except Exception:
            pass

    # -- the decision ----------------------------------------------------- #
    def session_for(self, prefix_hash: str, *, now: float | None = None) -> Session:
        """The session to run the next routing call against.

        Reseeds when the candidate list has changed — a different prefix cannot
        hit the cache anyway, and branching from a stale one would answer using
        lessons that no longer exist. Otherwise reuses, unless the last call is
        older than what the observed reads suggest the window is.
        """
        now = time.time() if now is None else now
        held = self.state.warm.get(prefix_hash)

        # An unknown chunk has no conversation, and a stale one has a
        # conversation whose cache has expired — reseeding is the same cost as
        # forking into a cold prefix, and it restarts the window.
        if held is None or now - held.last_call_at > self.state.ttl_s:
            self._evict(now)
            fresh = Warm(session_id=str(uuid.uuid4()), seeded_at=now, last_call_at=now)
            self.state.warm[prefix_hash] = fresh
            self.state.last_call_at = now
            return Session(id=fresh.session_id, resume=False)

        held.last_call_at = now
        self.state.last_call_at = now
        return Session(id=held.session_id, resume=True)

    def _evict(self, now: float) -> None:
        """Forget conversations whose cache has certainly expired.

        Bounds the file. Nothing is lost — an evicted chunk is reseeded the next
        time it comes up, at the cost of one call it would have paid anyway.
        """
        self.state.warm = {
            k: v
            for k, v in self.state.warm.items()
            if now - v.last_call_at <= MAX_TTL_S
        }

    def record(
        self,
        *,
        cached_in: int,
        prefix_tokens: int,
        prefix_hash: str = "",
        created: int = 0,
        seeded: bool = False,
        now: float | None = None,
    ) -> bool:
        """Note what the provider actually served from cache. Returns whether it hit.

        This is where the timer stops being a guess. A fork that hit tells us
        the window is at least as wide as the gap we just crossed; one that
        missed tells us it is narrower. Both readings move the window, so it
        converges on the provider's real behaviour instead of on a constant
        somebody wrote down.

        `seeded` marks a call whose prefix was new by construction. Those are
        not scored — they are how the baseline is learned.
        """
        now = time.time() if now is None else now
        gap = now - self.state.last_call_at

        held = self.state.warm.get(prefix_hash)
        if seeded:
            if held is not None:
                held.seed_cached = cached_in
                held.seed_created = created
            self.state.baseline_cached = cached_in
            self.state.last_call_at = now
            self.save()
            return False

        # Only the excess over what this conversation's own seed already had
        # cached is ours. Half the written amount tolerates tokenisation drift
        # without accepting the host's system prompt as a success.
        floor = held.seed_cached if held else self.state.baseline_cached
        target = held.seed_created if held and held.seed_created else prefix_tokens
        ours = cached_in - floor
        hit = ours >= max(256, target // 2)
        cached_in = max(0, ours)

        if hit:
            self.state.hits += 1
            self.state.tokens_saved += cached_in
            if gap > self.state.ttl_s * 0.8:
                self.state.ttl_s = min(MAX_TTL_S, max(self.state.ttl_s, gap * 1.5))
        else:
            self.state.misses += 1
            if gap < self.state.ttl_s:
                self.state.ttl_s = max(MIN_TTL_S, min(self.state.ttl_s, gap * 0.8))

        self.state.last_call_at = now
        self.save()
        self.store.log(
            "router",
            hit=hit,
            ours_cached=cached_in,
            baseline=self.state.baseline_cached,
            prefix_tokens=prefix_tokens,
            gap_s=round(gap, 1),
            ttl_s=round(self.state.ttl_s, 1),
        )
        return hit

    # -- reporting -------------------------------------------------------- #
    def summary(self) -> str:
        s = self.state
        if not (s.hits or s.misses):
            return "router cache: no calls yet"
        return (
            f"router cache: {s.hit_rate:.0%} warm ({s.hits} hit / {s.misses} miss), "
            f"{s.tokens_saved} prefix tokens read from cache "
            f"(over a {s.baseline_cached} baseline the host caches anyway), "
            f"window ~{int(s.ttl_s)}s"
        )
