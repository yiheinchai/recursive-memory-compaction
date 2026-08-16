"""Small shared helpers: ids, timestamps, token estimation, text signatures."""

from __future__ import annotations

import hashlib
import os
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
    """The real tokenizer, only when explicitly asked for. See `count_tokens`."""
    global _TOKENIZER, _TOKENIZER_TRIED
    if _TOKENIZER_TRIED:
        return _TOKENIZER
    _TOKENIZER_TRIED = True
    if os.environ.get("RMC_TOKENIZER", "").strip().lower() != "tiktoken":
        return None
    try:  # pragma: no cover - optional, off by default
        import tiktoken

        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _TOKENIZER = None
    return _TOKENIZER


def count_tokens(text: str) -> int:
    """Token count: a deterministic 4-chars-per-token estimate by default.

    Every use of this compares one count against another — a compression's
    before/after ratio, a pack against its budget — so what matters far more
    than accuracy is that the *same text always measures the same*.

    It previously used tiktoken whenever the package happened to be importable,
    which broke exactly that. RMC runs across several processes (a hook, a
    detached learner, your shell), and they need not share an interpreter: the
    same lesson measured 302 in one and 240 in another. A compression could then
    be accepted or rejected depending on which Python ran it.

    So the estimate is the default, and it is dependency-free like the rest of
    RMC. Set ``RMC_TOKENIZER=tiktoken`` for true counts — but set it everywhere,
    or the inconsistency comes back.
    """
    if not text:
        return 0
    enc = _tokenizer()
    if enc is not None:  # pragma: no cover - opt-in only
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, round(len(text) / 4))


def truncate(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
