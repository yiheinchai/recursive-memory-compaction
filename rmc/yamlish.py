"""A deliberately small YAML subset, so RMC installs with zero dependencies.

Nodes and configs are machine-written by :func:`dump`, which only ever emits the
subset :func:`load` understands. If PyYAML happens to be installed we use it for
*parsing* (so that hand-edits using wider YAML still work), but we always emit
with our own writer to keep output deterministic and diff-friendly.

Supported: mappings, lists, lists-of-mappings, flow lists of scalars, block
scalars (``|``), and the scalars null/bool/int/float/str.
"""

from __future__ import annotations

import re
from typing import Any

try:  # pragma: no cover - depends on the host environment
    import yaml as _pyyaml
except Exception:  # pragma: no cover
    _pyyaml = None


class YamlishError(ValueError):
    """Raised when a document falls outside the supported subset."""


# --------------------------------------------------------------------------- #
# scalars
# --------------------------------------------------------------------------- #

_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?\d+\.\d+([eE][-+]?\d+)?$")


def _parse_scalar(text: str) -> Any:
    text = text.strip()
    if not text:
        return ""
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        inner = text[1:-1]
        if text[0] == '"':
            return inner.replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")
        return inner
    low = text.lower()
    if low in ("null", "~", ""):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    if _INT_RE.match(text):
        return int(text)
    if _FLOAT_RE.match(text):
        return float(text)
    return text


def _strip_comment(line: str) -> str:
    """Drop a trailing ``#`` comment that is not inside quotes."""
    out, quote = [], None
    for i, ch in enumerate(line):
        if quote:
            out.append(ch)
            if ch == quote and (i == 0 or line[i - 1] != "\\"):
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            out.append(ch)
            continue
        if ch == "#" and (i == 0 or line[i - 1] in " \t"):
            break
        out.append(ch)
    return "".join(out).rstrip()


def _split_flow_list(text: str) -> list[Any]:
    body = text.strip()[1:-1].strip()
    if not body:
        return []
    items, cur, quote, depth = [], [], None, 0
    for ch in body:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            cur.append(ch)
            continue
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            items.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    items.append("".join(cur))
    return [_parse_scalar(i) for i in items if i.strip() != ""]


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #


class _Lines:
    def __init__(self, text: str) -> None:
        self.rows: list[tuple[int, str]] = []
        for raw in text.splitlines():
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            self.rows.append((len(raw) - len(raw.lstrip(" ")), raw))
        self.i = 0

    def peek(self) -> tuple[int, str] | None:
        return self.rows[self.i] if self.i < len(self.rows) else None

    def next(self) -> tuple[int, str]:
        row = self.rows[self.i]
        self.i += 1
        return row


def _read_block_scalar(lines: _Lines, parent_indent: int, raw_text: str) -> str:
    """Consume an indented block after a ``|`` / ``>`` marker."""
    fold = raw_text.strip().startswith(">")
    chunks: list[str] = []
    block_indent: int | None = None
    while (row := lines.peek()) is not None:
        indent, raw = row
        if raw.strip() and indent <= parent_indent:
            break
        lines.next()
        if block_indent is None:
            block_indent = indent
        chunks.append(raw[block_indent:] if len(raw) >= block_indent else raw.strip())
    if fold:
        return " ".join(c.strip() for c in chunks if c.strip())
    return "\n".join(chunks)


def _parse_value_inline(lines: _Lines, indent: int, text: str) -> Any:
    text = text.strip()
    if text.startswith("|") or text.startswith(">"):
        return _read_block_scalar(lines, indent, text)
    if text.startswith("[") and text.endswith("]"):
        return _split_flow_list(text)
    if text.startswith("{") and text.endswith("}"):
        body = text[1:-1].strip()
        if not body:
            return {}
        out: dict[str, Any] = {}
        for piece in _split_flow_list("[" + body + "]"):
            if isinstance(piece, str) and ":" in piece:
                k, _, v = piece.partition(":")
                out[k.strip()] = _parse_scalar(v)
        return out
    return _parse_scalar(text)


def _parse_block(lines: _Lines, indent: int) -> Any:
    row = lines.peek()
    if row is None:
        return None
    if row[1].lstrip().startswith("- "):
        return _parse_list(lines, indent)
    return _parse_map(lines, indent)


def _parse_list(lines: _Lines, indent: int) -> list[Any]:
    items: list[Any] = []
    while (row := lines.peek()) is not None:
        cur_indent, raw = row
        if cur_indent < indent:
            break
        stripped = _strip_comment(raw).strip()
        if not stripped:
            lines.next()
            continue
        if not stripped.startswith("-"):
            break
        lines.next()
        rest = stripped[1:].strip()
        if not rest:
            nxt = lines.peek()
            items.append(_parse_block(lines, nxt[0]) if nxt and nxt[0] > cur_indent else None)
            continue
        # "- key: value" starts an inline mapping that may continue on next lines
        if ":" in rest and not rest.startswith(("[", "{", '"', "'")):
            key, _, val = rest.partition(":")
            entry: dict[str, Any] = {}
            item_indent = cur_indent + (len(stripped) - len(stripped[1:].lstrip()))
            if val.strip():
                entry[key.strip()] = _parse_value_inline(lines, cur_indent, val)
            else:
                nxt = lines.peek()
                entry[key.strip()] = (
                    _parse_block(lines, nxt[0]) if nxt and nxt[0] > cur_indent else None
                )
            while (nrow := lines.peek()) is not None and nrow[0] >= item_indent:
                nstripped = _strip_comment(nrow[1]).strip()
                if nstripped.startswith("-") or ":" not in nstripped:
                    break
                lines.next()
                k2, _, v2 = nstripped.partition(":")
                if v2.strip():
                    entry[k2.strip()] = _parse_value_inline(lines, nrow[0], v2)
                else:
                    nxt = lines.peek()
                    entry[k2.strip()] = (
                        _parse_block(lines, nxt[0]) if nxt and nxt[0] > nrow[0] else None
                    )
            items.append(entry)
            continue
        items.append(_parse_value_inline(lines, cur_indent, rest))
    return items


def _parse_map(lines: _Lines, indent: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    while (row := lines.peek()) is not None:
        cur_indent, raw = row
        if cur_indent < indent:
            break
        stripped = _strip_comment(raw).strip()
        if not stripped:
            lines.next()
            continue
        if stripped.startswith("- "):
            break
        if ":" not in stripped:
            raise YamlishError(f"cannot parse line: {raw!r}")
        lines.next()
        key, _, val = stripped.partition(":")
        key = key.strip().strip("\"'")
        if val.strip():
            out[key] = _parse_value_inline(lines, cur_indent, val)
            continue
        nxt = lines.peek()
        if nxt is None:
            out[key] = None
            continue
        nxt_indent, nxt_raw = nxt
        # A block child is either indented further, or — for lists — a sibling-
        # indented "- " run, which is legal YAML and what many editors produce.
        is_flush_list = nxt_indent == cur_indent and nxt_raw.lstrip().startswith("- ")
        if nxt_indent > cur_indent or is_flush_list:
            out[key] = _parse_block(lines, nxt_indent)
        else:
            out[key] = None
    return out


def load(text: str) -> Any:
    """Parse a YAML document (subset). Uses PyYAML when available."""
    if _pyyaml is not None:
        try:
            return _pyyaml.safe_load(text)
        except Exception:  # fall through to the subset parser
            pass
    lines = _Lines(text)
    if lines.peek() is None:
        return {}
    return _parse_block(lines, lines.peek()[0])


# --------------------------------------------------------------------------- #
# emitter
# --------------------------------------------------------------------------- #

_PLAIN_SAFE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_ ./@+-]*$")


def _emit_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "":
        return '""'
    reserved = text.lower() in ("null", "true", "false", "~", "yes", "no", "on", "off")
    if reserved or not _PLAIN_SAFE.match(text) or _INT_RE.match(text) or _FLOAT_RE.match(text):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'
    return text


def _emit(value: Any, indent: int, out: list[str]) -> None:
    pad = " " * indent
    if isinstance(value, dict):
        if not value:
            out[-1] += " {}"
            return
        for key, val in value.items():
            if isinstance(val, dict) and val:
                out.append(f"{pad}{key}:")
                _emit(val, indent + 2, out)
            elif isinstance(val, list) and val:
                out.append(f"{pad}{key}:")
                _emit(val, indent + 2, out)
            elif isinstance(val, list):
                out.append(f"{pad}{key}: []")
            elif isinstance(val, dict):
                out.append(f"{pad}{key}: {{}}")
            elif isinstance(val, str) and "\n" in val:
                out.append(f"{pad}{key}: |")
                for line in val.split("\n"):
                    out.append(f"{pad}  {line}" if line else "")
            else:
                out.append(f"{pad}{key}: {_emit_scalar(val)}")
        return
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                keys = list(item.items())
                if not keys:
                    out.append(f"{pad}- {{}}")
                    continue
                first_key, first_val = keys[0]
                if isinstance(first_val, (dict, list)) and first_val:
                    out.append(f"{pad}- {first_key}:")
                    _emit(first_val, indent + 4, out)
                else:
                    out.append(f"{pad}- {first_key}: {_emit_scalar(first_val)}")
                for k, v in keys[1:]:
                    if isinstance(v, (dict, list)) and v:
                        out.append(f"{pad}  {k}:")
                        _emit(v, indent + 4, out)
                    else:
                        out.append(f"{pad}  {k}: {_emit_scalar(v)}")
            else:
                out.append(f"{pad}- {_emit_scalar(item)}")
        return
    out.append(f"{pad}{_emit_scalar(value)}")


def dump(value: Any) -> str:
    """Serialise to the YAML subset. Deterministic key order (insertion order)."""
    out: list[str] = []
    _emit(value, 0, out)
    return "\n".join(out) + "\n"
