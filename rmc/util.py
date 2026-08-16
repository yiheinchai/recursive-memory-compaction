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


def truncate(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
