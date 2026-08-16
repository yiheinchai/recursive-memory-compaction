"""Codex backend: ``codex exec``, using its native ``--output-schema``."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from . import AgentResult, extract_json
from ._proc import run_cmd, which


class CodexAdapter:
    name = "codex"

    def __init__(self, *, model: str | None = None, binary: str = "codex") -> None:
        self.model = model
        self.binary = binary

    def available(self) -> bool:
        return which(self.binary) is not None

    def run(
        self,
        prompt: str,
        *,
        system: str | None = None,
        cwd: Path | None = None,
        schema: dict[str, Any] | None = None,
        tools: bool = False,
        timeout: int = 180,
    ) -> AgentResult:
        if not self.available():
            return AgentResult(ok=False, error=f"{self.binary} not on PATH", backend=self.name)

        tmpdir = Path(tempfile.mkdtemp(prefix="rmc-codex-"))
        last_msg = tmpdir / "last.txt"
        argv = [
            self.binary,
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "-o",
            str(last_msg),
        ]
        if self.model:
            argv += ["-m", self.model]
        # Sandbox: meta-calls are pure text transforms and get read-only; replay
        # needs to edit files, so it gets workspace-write scoped to `cwd`.
        argv += ["-s", "workspace-write" if tools else "read-only"]
        if cwd:
            argv += ["-C", str(cwd)]

        schema_file: Path | None = None
        if schema:
            schema_file = tmpdir / "schema.json"
            schema_file.write_text(json.dumps(_strict(schema)), encoding="utf-8")
            argv += ["--output-schema", str(schema_file)]

        full_prompt = prompt if not system else f"{system}\n\n---\n\n{prompt}"

        code, out, err, dur = run_cmd(argv, cwd=cwd, timeout=timeout, stdin=full_prompt)

        text = ""
        if last_msg.exists():
            text = last_msg.read_text(encoding="utf-8").strip()
        if not text:
            text = _last_agent_message(out) or out.strip()

        if code != 0 and not text:
            return AgentResult(
                ok=False,
                error=(err or f"exit {code}").strip()[:2000],
                duration_s=dur,
                backend=self.name,
                raw=out[:4000],
            )

        data = extract_json(text) if schema else None
        if schema and data is None:
            return AgentResult(
                ok=False,
                text=text,
                error="model did not return parseable JSON",
                duration_s=dur,
                backend=self.name,
                raw=out[:4000],
            )
        return AgentResult(
            ok=True,
            text=text,
            data=data,
            duration_s=dur,
            backend=self.name,
            raw=out[:4000],
        )


def _strict(schema: dict[str, Any]) -> dict[str, Any]:
    """Codex's schema mode is happier with explicit additionalProperties."""
    out = dict(schema)
    if out.get("type") == "object":
        out.setdefault("additionalProperties", False)
        props = out.get("properties")
        if isinstance(props, dict):
            out["properties"] = {k: _strict(v) if isinstance(v, dict) else v for k, v in props.items()}
    if out.get("type") == "array" and isinstance(out.get("items"), dict):
        out["items"] = _strict(out["items"])
    return out


def _last_agent_message(stdout: str) -> str:
    """Recover the final message from ``--json`` JSONL, if -o produced nothing."""
    best = ""
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        msg = row.get("msg") if isinstance(row.get("msg"), dict) else row
        if not isinstance(msg, dict):
            continue
        if msg.get("type") in ("agent_message", "assistant_message", "item.completed"):
            for key in ("message", "text", "content"):
                val = msg.get(key)
                if isinstance(val, str) and val.strip():
                    best = val.strip()
    return best
