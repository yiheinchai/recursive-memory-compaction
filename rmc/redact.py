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

# --------------------------------------------------------------------------- #
# exemptions: evidence that a match is *not* a secret
# --------------------------------------------------------------------------- #
#
# The bias toward over-redaction above is right and stays. These do not loosen
# it — each requires positive evidence that the thing matched cannot be a
# credential, rather than merely failing to look like one.
#
# What forced this: `AUTH_TOKENS_TABLE = "auth-tokens"` in a Terraform config.
# The name contains `AUTH_TOKEN`, so `assigned-secret` fired, and an imported
# infrastructure lesson was stored teaching a table name of `[REDACTED]`. That
# is not a recoverable mangle — it is a lesson that now says something false,
# and nothing downstream can tell.

# A value that is a variable reference cannot be a secret: it is the name of
# where the secret lives. `${var.X}`, `$X`, `{{ x }}`, `<your-key>`.
_INTERPOLATION_RE = re.compile(
    r"^(?:\$\{[^}]*\}|\$[A-Za-z_][A-Za-z0-9_]*|\{\{[^}]*\}\}|<[^>]*>|%[A-Za-z_]+%)"
)

# A name whose last segment names a *resource* is describing where something is,
# not what the credential is. `AUTH_TOKENS_TABLE` is a table; `SESSION_TOKEN` is
# a token. Only the final segment counts, and the list stays short.
_RESOURCE_SUFFIXES = (
    "table",
    "tablename",
    "bucket",
    "queue",
    "topic",
    "arn",
    "url",
    "uri",
    "endpoint",
    "host",
    "hostname",
    "port",
    "region",
    "namespace",
    "prefix",
    "filename",
    "filepath",
    "path",
)


# Configuration keywords. A field whose value is one of these is declaring a
# mode, not carrying a credential — `secrets: inherit` is GitHub Actions syntax,
# and redacting it turned a workflow lesson into one that cannot be followed.
_KEYWORD_VALUES = frozenset(
    {
        "inherit",
        "true",
        "false",
        "none",
        "null",
        "nil",
        "required",
        "optional",
        "enabled",
        "disabled",
        "auto",
        "default",
        "always",
        "never",
    }
)


def _not_a_secret(name: str, value: str) -> bool:
    """Whether a `name = value` match has positively proved itself harmless."""
    value = value.strip().strip("`'\"")
    if _INTERPOLATION_RE.match(value):
        return True
    if value.lower() in _KEYWORD_VALUES:
        return True
    tail = re.split(r"[_.\-]", name.strip().lower())[-1]
    return tail in _RESOURCE_SUFFIXES


# Addresses that are no-reply by construction. They identify nobody, and they
# appear inside literal commands — `git -c user.email="noreply@anthropic.com"`
# — where pseudonymising them leaves a lesson teaching a command that is wrong.
_NOREPLY_RE = re.compile(r"^(?:no[._-]?reply|donot[._-]?reply|do[._-]?not[._-]?reply)$", re.I)


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
            def _assigned(m: re.Match[str]) -> str:
                # Leave the original text alone when the match has proved
                # itself harmless — including its spacing, since a config line
                # is often the point of the lesson.
                if _not_a_secret(m.group(1), m.group(2)):
                    return m.group(0)
                return f"{m.group(1)}={PLACEHOLDER}"

            out = pattern.sub(_assigned, out)
        else:
            out = pattern.sub(PLACEHOLDER, out)
    if not keep_emails:
        def _email(m: re.Match[str]) -> str:
            local, _, domain = m.group(0).partition("@")
            if _NOREPLY_RE.match(local):
                return m.group(0)
            return f"[email:{domain}]"

        out = _EMAIL_RE.sub(_email, out)
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
