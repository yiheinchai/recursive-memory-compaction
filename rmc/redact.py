"""Scrub secret-shaped strings before anything is written to the store.

RMC persists fragments of real sessions, so it will eventually see a token that
was pasted into a prompt. Redaction runs on every write path (episodes, events,
node bodies) rather than at read time, so a secret never lands on disk in the
first place.

This is a best-effort filter, not a guarantee. It is deliberately biased toward
over-redaction: a mangled lesson is recoverable, a leaked key is not.
"""

from __future__ import annotations

import re

PLACEHOLDER = "[REDACTED]"

# Ordered: more specific patterns first, so a known key shape wins over the
# generic high-entropy catch-all.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws-key", re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA)[0-9A-Z]{16}\b")),
    ("github-pat", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("openai", re.compile(r"\bsk-(?:proj-|ant-|live-)?[A-Za-z0-9_-]{20,}\b")),
    ("slack", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b")),
    ("google", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("stripe", re.compile(r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{16,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    ("private-key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL)),
    ("bearer", re.compile(r"(?i)\b(bearer|token|authorization)\s*[:=]\s*['\"]?([A-Za-z0-9._~+/-]{20,}=*)['\"]?")),
    # key=value where the key name smells like a credential
    (
        "assigned-secret",
        re.compile(
            r"(?i)\b([A-Za-z0-9_.-]*(?:secret|passwd|password|api[_-]?key|access[_-]?key|"
            r"private[_-]?key|client[_-]?secret|auth[_-]?token|session[_-]?token)[A-Za-z0-9_.-]*)"
            r"\s*[:=]\s*['\"]?([^\s'\"#,;]{6,})['\"]?"
        ),
    ),
    ("card", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
)

# Emails are pseudonymised rather than dropped, because "the user's email" is
# sometimes load-bearing context in a lesson.
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def redact(text: str, *, keep_emails: bool = False) -> str:
    """Return ``text`` with credential-shaped substrings replaced."""
    if not text:
        return text
    out = text
    for name, pattern in _PATTERNS:
        if name == "card":
            def _card(m: re.Match[str]) -> str:
                digits = re.sub(r"\D", "", m.group(0))
                # Only redact things that actually check out as card numbers,
                # otherwise every long hash and timestamp gets eaten.
                return PLACEHOLDER if 13 <= len(digits) <= 19 and _luhn_ok(digits) else m.group(0)

            out = pattern.sub(_card, out)
        elif name in ("bearer", "assigned-secret"):
            out = pattern.sub(lambda m: f"{m.group(1)}={PLACEHOLDER}", out)
        else:
            out = pattern.sub(PLACEHOLDER, out)
    if not keep_emails:
        out = _EMAIL_RE.sub(lambda m: f"[email:{m.group(0).split('@')[-1]}]", out)
    return out


def redact_obj(obj, *, keep_emails: bool = False):
    """Recursively redact every string in a JSON-shaped structure."""
    if isinstance(obj, str):
        return redact(obj, keep_emails=keep_emails)
    if isinstance(obj, dict):
        return {k: redact_obj(v, keep_emails=keep_emails) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact_obj(v, keep_emails=keep_emails) for v in obj]
    return obj
