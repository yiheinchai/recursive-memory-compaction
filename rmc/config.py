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
        "judge_calls": 2,  # model calls the relevance walk may spend
        "max_depth": 2,  # how far down the tree the walk may look
        "max_expansions": 3,
        # When every stored lesson fits in max_pack_tokens there is nothing to
        # choose between, so recall serves them all without asking. Set this to
        # force relevance filtering even then.
        "always_judge": False,
        # Bound on the routing call, kept below the hook's own deadline so a
        # slow judgement degrades to "inject nothing" instead of being killed.
        "timeout_s": 20,
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
        "min_successes": 2,  # successful recalls before a compression is attempted
        # Candidate must be <= this fraction of the parent's tokens. Measured
        # against real compressors, a single step on an already-dense lesson
        # lands around 0.7; a stricter gate simply rejects everything and leaves
        # you at 100%. What matters is compounding, not per-step depth —
        # 0.75 per level is ~32% of the original after four levels.
        "max_ratio": 0.75,
        "threshold": 1.0,  # required replay pass-rate
        "merge_threshold": 1.0,
        "regression_k": 5,  # episodes replayed per validation
        "max_level": 6,
        "cooldown_s": 900,
    },
    "learning": {
        "enabled": True,
        "min_tool_calls": 8,  # ignore trivial sessions
        "capture_failures": True,
        # Surprise is the trigger for reflection, as it is for a person: a turn
        # where everything worked has nothing to teach. A failed tool call is a
        # fact the host reports, so noticing costs nothing; deciding whether it
        # taught anything is left to the agent.
        "nudge_on_surprise": True,
        "min_surprises": 2,  # failed tool calls before the agent is asked to reflect
        "nudge_cooldown_s": 900,
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
    data: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULTS))
    path: Path | None = None

    @classmethod
    def load(cls, path: Path) -> "Config":
        if path.exists():
            parsed = yamlish.load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(parsed, dict):
                parsed = {}
            return cls(_deep_merge(DEFAULTS, parsed), path)
        return cls(copy.deepcopy(DEFAULTS), path)

    def save(self, path: Path | None = None) -> None:
        target = path or self.path
        if target is None:
            raise ValueError("no path to save config to")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yamlish.dump(self.data), encoding="utf-8")

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
        cur = self.data
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value


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
