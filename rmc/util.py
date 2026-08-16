"""Small shared helpers: ids, timestamps, token estimation, text signatures."""

from __future__ import annotations

import hashlib
import os
import re
import secrets
from datetime import datetime, timezone

# --------------------------------------------------------------------------- #
# ids and time
# --------------------------------------------------------------------------- #


def new_id(prefix: str = "n") -> str:
    return f"{prefix}_{secrets.token_hex(3)}"


def stable_id(prefix: str, *parts: str) -> str:
    """Deterministic id, so re-running the same operation does not fork the tree."""
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:6]
    return f"{prefix}_{digest}"


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# tokens
# --------------------------------------------------------------------------- #

_TOKENIZER = None
_TOKENIZER_TRIED = False


def _tokenizer():
    global _TOKENIZER, _TOKENIZER_TRIED
    if _TOKENIZER_TRIED:
        return _TOKENIZER
    _TOKENIZER_TRIED = True
    if os.environ.get("RMC_NO_TIKTOKEN"):
        return None
    try:  # pragma: no cover - optional accelerant
        import tiktoken

        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _TOKENIZER = None
    return _TOKENIZER


def count_tokens(text: str) -> int:
    """Token count. Uses tiktoken when installed, else a 4-chars-per-token estimate.

    The estimate only needs to be *consistent*, since every place it is used
    compares one count against another (compression ratios, pack budgets).
    """
    if not text:
        return 0
    enc = _tokenizer()
    if enc is not None:  # pragma: no cover - depends on environment
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, round(len(text) / 4))


# --------------------------------------------------------------------------- #
# text signatures (used for task affinity and family matching)
# --------------------------------------------------------------------------- #

_WORD_RE = re.compile(r"[a-z0-9_][a-z0-9_+.#-]{2,}")

_STOPWORDS = frozenset(
    """
    the and for that this with you your are was were will would can could should
    have has had not but from they them their there then than when what which who
    how why into out about over under just like get got make made use used using
    now new all any some more most other another same each every been being does
    did doing done here also because while after before during between very much
    such only own too its it's i'm don't didn't we're let lets please thanks thank
    okay yeah sure need needs needed want wants wanted try tried trying run runs
    file files code line lines add adds added fix fixes fixed change changes
    """.split()
)


def signature(text: str, limit: int = 40) -> set[str]:
    """Bag of salient lowercase terms, used for cheap lexical similarity."""
    words = _WORD_RE.findall((text or "").lower())
    seen: dict[str, int] = {}
    for w in words:
        if w in _STOPWORDS or w.isdigit():
            continue
        seen[w] = seen.get(w, 0) + 1
    ranked = sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))
    return {w for w, _ in ranked[:limit]}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / len(a | b)


def overlap_coeff(a: set[str], b: set[str]) -> float:
    """Overlap coefficient — kinder than Jaccard when one side is much smaller.

    A three-word failure diagnosis should be able to match a long delta claim.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def truncate(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
