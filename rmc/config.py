"""Configuration, with defaults that make RMC safe to leave switched on."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import yamlish

DEFAULTS: dict[str, Any] = {
    "version": 1,
    "agent": "claude",  # default backend: claude | codex | mock
    "model": None,  # None -> backend default
    "recall": {
        "enabled": True,
        "strategy": "delta-patch",  # delta-patch | delta-jump | stepwise
        "max_pack_tokens": 1200,
        "max_families": 3,  # how many lessons to inject per prompt
        # Model calls the walk may spend *descending*. The first pass over the
        # top level is not drawn from this — it is sized by how many chunks the
        # apex layer needs, because a chunk skipped there is a lesson that can
        # never be retrieved.
        "judge_calls": 2,
        # Lessons shown to the model in one question. Wide levels are split
        # across several; this is about answer quality, not budget.
        "fanout": 12,
        # Chunks of one level put to the model at the same time. They are
        # independent questions about disjoint sets, and each costs ~15s of
        # which ~5s is process startup, so asking them in turn makes the top
        # level unaffordable exactly as the store grows: six chunks
        # sequentially exceeds the recall timeout and serves nothing. This is
        # what makes covering every lesson possible rather than aspirational.
        "parallel": 4,
        "max_depth": 2,  # how far down the tree the walk may look
        "max_expansions": 3,
        # Apexes below which recall serves everything without asking.
        #
        # There used to be no count here, only a token budget, justified as
        # "judgement is only needed under scarcity". That was wrong about cost:
        # measured over 57 prompts on a store that fit the budget, 15,917 of
        # ~17,800 injected tokens went unused, and judging those same sets kept
        # every lesson that mattered while dropping 55% of the noise. Context
        # that fits is not context that is free.
        #
        # It was right about latency, which is the part that cannot be argued
        # away: a routing call costs ~5s of CLI startup alone and ~34s on the
        # model that routes well, in a hook that blocks the user's prompt. So a
        # gate stays — but a small one, sized so that what it lets through is a
        # few lines rather than a page. Three lessons served blind is cheaper
        # than five seconds of waiting; twenty is not.
        "filter_above": 3,
        # Candidate-list size above which routing seeds a reusable conversation
        # instead of re-sending the list every prompt. Providers serve an
        # identical prefix from cache at roughly a tenth of the price, but only
        # within one conversation — and seeding costs an extra round trip, so
        # below this the trick loses money. It is set for where the design has
        # to work rather than where it is: at 5,000 lessons the list runs to
        # ~225k tokens per prompt and re-sending it is not survivable.
        "warm_prefix_above_tokens": 2000,
        # Smallest prefix a provider will open a cache entry for. Below it,
        # seeding writes nothing and every fork misses — so when warming is on,
        # candidate chunks are widened until they clear this. Roughly 1024 for
        # the larger models and 2048 for the smallest; 1200 leaves margin
        # without making the questions much wider than they need to be.
        "min_cacheable_tokens": 1200,
        # Model for the routing decision. Null means "same as everything else",
        # and that is the measured right answer despite looking wasteful.
        #
        # Choosing which of two dozen one-line summaries bear on a prompt looks
        # like a classification job a small model should win, and it blocks the
        # user's prompt, so the incentive to downgrade it is strong. `rmc
        # eval-recall` says don't:
        #
        #     model      precision  recall  noise tok
        #     default        48%     100%      7,146
        #     sonnet         35%      81%      9,781
        #     haiku          35%      75%      8,818
        #
        # Both small models serve *more* noise and drop useful lessons. Since
        # serving everything unfiltered has 100% recall by construction, a cheap
        # router is strictly worse than no router at all — it costs latency,
        # loses lessons, and does not even save context. Set this only with an
        # eval run to back it.
        "model": None,
        # Turns for which an already-injected lesson counts as still present and
        # attended to. Inside it, a repeat is skipped; beyond it, the lesson is
        # refreshed with its one-line gist rather than repeated in full. Context
        # compaction resets this, since compaction may have removed the text.
        "stays_fresh_turns": 8,
        # Bound on the routing call, kept below the hook's own deadline so a
        # slow judgement degrades to "inject nothing" instead of being killed.
        "timeout_s": 20,
        # How selection is made.
        #
        # agentic: fork the live session and let it search the store. The only
        #   option whose cost does not grow with the number of lessons — the
        #   index is grepped, never sent — and the only one that sees the task's
        #   tool calls and reasoning rather than just its opening sentence.
        # judge:   render the apex layer into one question and walk it. Costs
        #   ~55 tokens per apex per prompt, which is ~225k at 5,000 lessons.
        #
        # `judge` is not dead code: it is the baseline every arm of
        # `rmc eval-recall` is compared against, and the automatic fallback
        # whenever there is no session to fork.
        "selector": "agentic",
        # Longer than `timeout_s` because a search is several tool calls rather
        # than one question, and each costs a round trip. This is the number
        # that decides whether selection is felt as lag, so it is also the first
        # thing to lower if it is.
        "selector_timeout_s": 45,
        # Searches the selector may run before it must answer with what it has.
        # An unbounded search is the failure mode here: there is always another
        # phrasing to try, and the user is waiting the whole time.
        "selector_max_tool_calls": 6,
    },
    "routing": {
        # Selection lessons: what the reflector learned about *where the
        # knowledge was*, as opposed to what the knowledge is.
        "enabled": True,
        # The only part of retrieval that costs tokens on every prompt now that
        # the index is searched rather than sent. It is capped rather than
        # trusted to stay small: the claim that selection lessons stay far fewer
        # than lessons is a bet, and `rmc status` reports the ratio so the bet is
        # visible. When the cap binds, the rules with the worst record of
        # improving a selection are the ones that fall out.
        "max_tokens": 800,
    },
    "selection": {
        # The model decides which dropped detail explains a failure; the other
        # two terms are evidence (observed rescue rate) and measurement (tokens),
        # not proxies for judgement.
        "w_judge": 0.60,
        "w_prior": 0.28,
        "w_cost": 0.12,
        "explore": "posterior",  # posterior | ucb
        "ucb_c": 0.7,
    },
    "compaction": {
        "enabled": True,
        # Successful recalls before a compression is attempted. Recall serves
        # about one lesson per prompt, so waiting for a second success can take
        # weeks — and the wait buys little, because the thing that actually
        # protects the lesson is replay against its own episodes, which runs
        # either way. One success is enough of an occasion.
        "min_successes": 1,
        # Candidate must be <= this fraction of the parent's tokens. Measured
        # against real compressors, a single step on an already-dense lesson
        # lands around 0.7; a stricter gate simply rejects everything and leaves
        # you at 100%. What matters is compounding, not per-step depth —
        # 0.75 per level is ~32% of the original after four levels.
        "max_ratio": 0.75,
        "threshold": 1.0,  # required replay pass-rate
        "regression_k": 5,  # episodes replayed per validation
        "max_level": 6,
        "cooldown_s": 900,
    },
    "learning": {
        "enabled": True,
        "min_tool_calls": 8,  # ignore trivial sessions
        "capture_failures": True,
        # Reflection needs an *occasion*, not a verdict. The agent has the
        # judgement to tell a conceptual mistake from a typo, but in flight its
        # attention is on the task — so the harness schedules the look and the
        # agent decides what it sees. These thresholds only ask "did this turn
        # have enough substance to be worth a thought".
        "nudge_enabled": True,
        # background: a detached process reflects on the transcript, and the
        #   agent is never interrupted. Costs nothing on the main thread.
        # block:      interrupt the agent to reflect in its own context. It has
        #   the live sense of what surprised it, but spends a turn and pollutes
        #   the working context — a real cost in the middle of a large task.
        # fork:       reflect inside a fork of the live session — same context,
        #   off the main thread. Costs ~0.1x its tokens thanks to prompt cache
        #   reads, but 10% of a large context still exceeds a digest.
        # off:        no reflection until session end.
        "nudge_mode": "background",
        "nudge_after_tool_calls": 12,
        "nudge_after_turns": 3,
        "min_surprises": 2,  # failed tool calls also make a turn substantial
        "nudge_cooldown_s": 900,
        # If the agent captures on its own, stop interrupting it. Backs the
        # cooldown off after this many nudges in a row that yielded nothing.
        "nudge_backoff_after": 3,
    },
    "placement": {
        "consult": True,  # ask a model how new knowledge relates to old
        "judge_calls": 2,  # levels of tree walk when looking for related lessons
        "max_depth": 2,
        "surface_conflicts": True,  # raise unresolved contradictions at recall
    },
    "signals": {
        "min_confidence": 0.5,  # below this an outcome is recorded as 'unknown'
    },
    "limits": {
        "agent_timeout_s": 180,
        "max_concurrent": 2,
    },
    "privacy": {
        "redact": True,  # scrub secret-shaped strings before anything is stored
    },
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in (over or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


@dataclass
class Config:
    """Effective settings, and the much smaller set the user actually chose.

    The file on disk is *overrides only*. Writing the merged tree instead — the
    obvious implementation, and the one this had — quietly freezes a store at
    the defaults of the day it was created: the file wins the merge, so every
    later improvement to a default is shadowed by a value nobody ever chose.
    This store was still running `compaction.max_ratio: 0.6` and
    `min_successes: 2` months after both were retuned, and nothing surfaced it,
    because from the inside a frozen snapshot is indistinguishable from a
    preference.
    """

    data: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULTS))
    path: Path | None = None
    overrides: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Config":
        if path.exists():
            parsed = yamlish.load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(parsed, dict):
                parsed = {}
            return cls(_deep_merge(DEFAULTS, parsed), path, parsed)
        return cls(copy.deepcopy(DEFAULTS), path, {})

    def save(self, path: Path | None = None) -> None:
        target = path or self.path
        if target is None:
            raise ValueError("no path to save config to")
        target.parent.mkdir(parents=True, exist_ok=True)
        body = _prune(self.overrides, DEFAULTS)
        body.setdefault("version", DEFAULTS["version"])
        target.write_text(
            "# RMC overrides. Anything absent follows the current defaults;\n"
            "# see `rmc config` for the full effective settings.\n"
            + yamlish.dump(body),
            encoding="utf-8",
        )

    def get(self, dotted: str, default: Any = None) -> Any:
        """Read ``a.b.c``. Environment overrides win: RMC_A_B_C."""
        env_key = "RMC_" + dotted.upper().replace(".", "_")
        if env_key in os.environ:
            return _coerce(os.environ[env_key])
        cur: Any = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur if cur is not None else default

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        for target in (self.data, self.overrides):
            cur = target
            for part in parts[:-1]:
                cur = cur.setdefault(part, {})
            cur[parts[-1]] = value


def _prune(overrides: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """Drop entries that merely restate the default.

    A setting equal to its default is not a preference, it is a snapshot: it
    changes nothing today and blocks the improvement tomorrow. Dropping it is
    lossless in the present and correct in the future.
    """
    out: dict[str, Any] = {}
    for key, value in overrides.items():
        fallback = defaults.get(key) if isinstance(defaults, dict) else None
        if isinstance(value, dict):
            nested = _prune(value, fallback if isinstance(fallback, dict) else {})
            if nested:
                out[key] = nested
        elif key not in defaults or value != fallback:
            out[key] = value
    return out


def _coerce(raw: str) -> Any:
    low = raw.strip().lower()
    if low in ("true", "1", "yes", "on"):
        return True
    if low in ("false", "0", "no", "off"):
        return False
    if low in ("null", "none", ""):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw
