"""RMC improving its own retrieval, without anyone supplying the ideas.

Every other stage of RMC learns from outcomes. Compression is kept or rejected
by replay; a delta earns its way back by rescuing failures; consolidation is
driven by what got used together. The *judgement criteria* — the prompts and
constants that decide what gets recalled — were the one part that could only
change when a human thought of something.

That is the gap this closes. The loop is the same one a person runs by hand:

    measure  ->  read the failures  ->  propose one change  ->  measure again
             ->  keep it only if it wins  ->  remember the attempt either way

and it is worth automating because a person running it is wrong most of the
time. In the session that motivated this, five of six hand-written proposals
were regressions — each plausible, each argued for, each worse. The value is not
that the model has better ideas than the person. It is that neither of them gets
to decide, and the ones that lose are written down so nobody tries them twice.

The split is unchanged. The harness measures, applies, scores and reverts — all
mechanical. The model reads the failures and proposes what to change. The
recorded outcome decides. Nothing here asks a model whether its own idea worked.

**Only strict improvements are kept.** Precision is trivially bought by serving
less and recall by serving more, so a change has to be at least as good on both
before it counts. A trade is a preference, and preferences are the user's.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import eval_recall
from .adapters import Adapter
from .judge import RELEVANCE, WARM_RELEVANCE, prompt as resolve_prompt

PROPOSE = """RMC:tune

You are improving the retrieval stage of a memory system for coding agents. On
every prompt it decides which remembered lessons to put into the agent's
context. Two ways to be wrong, and both cost the user:

  - **noise** — a lesson was served and turned out not to bear on the work. It
    spent context and attention, and an irrelevant lesson can actively mislead.
  - **misses** — a lesson would have helped and was dropped. This is worse:
    the entire product claim is that the next session is shorter than the last.

Here is how the current criteria actually performed, scored against what a
later judgement said had really borne on the work.

CURRENT SCORE
{score}

WHERE IT WENT WRONG
{failures}

THE CRITERIA IN USE
<<<PROMPT
{prompt}
PROMPT>>>

TUNABLE CONSTANTS
{knobs}

ALREADY TRIED, AND THE RESULT
{history}

Propose exactly ONE change.

Read the failures before theorising. A change that does not follow from a
specific failure above is a guess, and guesses have a poor record here — of six
hand-written proposals to this same prompt, five made it worse, every one of
them plausible. The common shape of the mistake was buying a little less noise
by dropping lessons that mattered.

Do not repeat anything under ALREADY TRIED, and do not propose a variation whose
mechanism is the same as one that failed there.

If `kind` is `prompt`, return the complete replacement text in `text` — it is
written to disk verbatim and must keep every `{{placeholder}}` the original had.
If `kind` is `config`, give the dotted `key` and the new `value`.

`hypothesis` states what you expect to change and why, in one sentence, so that
it reads as a prediction that can turn out wrong.
"""

PROPOSE_SCHEMA = {
    "type": "object",
    "required": ["kind", "hypothesis"],
    "properties": {
        "kind": {"type": "string", "enum": ["prompt", "config"]},
        "text": {"type": "string", "description": "Full replacement prompt, when kind is prompt."},
        "key": {"type": "string", "description": "Dotted config key, when kind is config."},
        "value": {"description": "New value for that key."},
        "hypothesis": {"type": "string"},
    },
}

# Constants the loop may move, and the range it may move them in. Anything that
# could disable retrieval, spend unbounded money, or weaken a correctness gate
# is deliberately absent: a tuning loop must not be able to reach the checks
# that decide whether its own output was any good.
KNOBS: dict[str, tuple[float, float]] = {
    "recall.max_families": (1, 8),
    "recall.judge_calls": (1, 6),
    "recall.max_depth": (1, 4),
    "recall.fanout": (6, 40),
    "recall.max_pack_tokens": (400, 4000),
    "recall.stays_fresh_turns": (2, 40),
}


@dataclass
class Attempt:
    """One proposal and what measuring it showed."""

    at: str
    kind: str
    target: str
    hypothesis: str
    before: dict[str, float] = field(default_factory=dict)
    after: dict[str, float] = field(default_factory=dict)
    kept: bool = False
    verdict: str = ""

    def line(self) -> str:
        move = (
            f"precision {self.before.get('precision', 0):.0%} -> {self.after.get('precision', 0):.0%}, "
            f"recall {self.before.get('recall', 0):.0%} -> {self.after.get('recall', 0):.0%}"
        )
        return f"- [{self.kind}: {self.target}] {self.hypothesis} — {move}. {self.verdict}"


def _score(report: eval_recall.Report) -> dict[str, float]:
    return {
        "precision": report.precision,
        "recall": report.recall_rate,
        "hits": float(report.hits),
        "noise_tokens": float(report.noise_tokens),
    }


def _describe(report: eval_recall.Report) -> str:
    s = _score(report)
    return (
        f"precision {s['precision']:.0%}, recall {s['recall']:.0%}, "
        f"{int(s['hits'])} useful lessons delivered, {int(s['noise_tokens'])} noise tokens"
    )


def _failures(store: Any, report: eval_recall.Report, limit: int = 6) -> str:
    """The actual mistakes, in the model's own terms rather than as a number.

    A score says the stage is wrong; only the cases say how. Both directions are
    shown, and misses are shown first because they are the expensive kind and a
    proposal that ignores them will read as an improvement while making the
    product worse.
    """
    lines: list[str] = []
    for s in report.scores:
        if not (s.misses or s.noise):
            continue
        lines.append(f'\nWork: "{s.prompt.strip()[:180]}"')
        for ident in sorted(s.misses):
            node = store.get(ident)
            lines.append(f"  MISS  would now drop, but it did help: {_name(node, ident)}")
        for ident in sorted(s.noise):
            node = store.get(ident)
            cost = s.tokens.get(ident, 0)
            lines.append(f"  NOISE served, never used ({cost} tok): {_name(node, ident)}")
        if len(lines) > limit * 4:
            break
    return "\n".join(lines) or "(no recorded failures — nothing to learn from yet)"


def _name(node: Any, ident: str) -> str:
    if node is None:
        return ident
    return f"{node.title or node.family} — {node.summary(limit=110)}"


def _knobs(store: Any) -> str:
    out = []
    for key, (low, high) in KNOBS.items():
        out.append(f"- `{key}` = {store.config.get(key)!r} (allowed {low}..{high})")
    return "\n".join(out)


class Ledger:
    """Every attempt, including — especially — the ones that failed.

    A loop that forgets its failures re-proposes them forever, and the failures
    are the more informative half: that a plausible change made things worse is
    a fact about this store's retrieval that nothing else records.
    """

    def __init__(self, store: Any) -> None:
        self.path = Path(store.root) / "tune.json"

    def all(self) -> list[Attempt]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []
        known = set(Attempt.__dataclass_fields__)
        return [Attempt(**{k: v for k, v in a.items() if k in known}) for a in raw]

    def add(self, attempt: Attempt) -> None:
        entries = self.all() + [attempt]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([a.__dict__ for a in entries], indent=2), encoding="utf-8"
        )

    def history(self, limit: int = 12) -> str:
        entries = self.all()[-limit:]
        return "\n".join(a.line() for a in entries) or "(nothing tried yet)"


class Sandbox:
    """Applies a candidate change so it can be measured, and undoes it.

    Reverting has to be exact and unconditional. A tuning loop that leaves a
    rejected change behind is worse than no tuning loop, because the damage
    arrives labelled as an improvement.
    """

    def __init__(self, store: Any) -> None:
        self.store = store
        self.undo: Any = None

    def apply_prompt(self, name: str, text: str) -> None:
        path = self.store.root / "prompts" / f"{name}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        previous = path.read_text(encoding="utf-8") if path.exists() else None
        path.write_text(text, encoding="utf-8")
        self.undo = lambda: (
            path.write_text(previous, encoding="utf-8") if previous is not None
            else path.unlink(missing_ok=True)
        )

    def apply_config(self, key: str, value: Any) -> None:
        previous = self.store.config.get(key)
        self.store.config.set(key, value)
        self.store.config.save(self.store.root / "config.yaml")

        def restore() -> None:
            self.store.config.set(key, previous)
            self.store.config.save(self.store.root / "config.yaml")

        self.undo = restore

    def revert(self) -> None:
        if self.undo:
            self.undo()
            self.undo = None


def _validate(store: Any, data: dict[str, Any]) -> tuple[str, str, str]:
    """Reject a proposal the loop must not be allowed to apply.

    Not a judgement about whether the idea is good — that is what measuring is
    for. This only refuses changes that would break the call outright or reach
    outside the range the harness is willing to explore.
    """
    kind = str(data.get("kind") or "")
    if kind == "config":
        key = str(data.get("key") or "")
        if key not in KNOBS:
            return "", "", f"{key or '(none)'} is not a tunable constant"
        low, high = KNOBS[key]
        try:
            value = float(data.get("value"))
        except (TypeError, ValueError):
            return "", "", f"{data.get('value')!r} is not a number"
        if not low <= value <= high:
            return "", "", f"{key}={value} is outside {low}..{high}"
        return "config", key, ""

    if kind == "prompt":
        text = str(data.get("text") or "")
        if len(text) < 200:
            return "", "", "replacement prompt is too short to be a real one"
        # A prompt missing a placeholder throws at format time, which would show
        # up as "the judge could not answer" — i.e. as a catastrophic score for
        # reasons that have nothing to do with the idea being tested.
        name = "warm_relevance" if "{candidates}" not in text else "relevance"
        required = ["{question}"] + ([] if name == "warm_relevance" else ["{candidates}"])
        missing = [ph for ph in required if ph not in text]
        if missing:
            return "", "", f"replacement prompt is missing {', '.join(missing)}"
        return "prompt", name, ""

    return "", "", f"unknown change kind {kind!r}"


def run(
    store: Any,
    adapter: Adapter,
    *,
    rounds: int = 1,
    dry_run: bool = False,
) -> list[Attempt]:
    """Propose, measure, and keep only what wins. Returns every attempt."""
    ledger = Ledger(store)
    attempts: list[Attempt] = []
    baseline = eval_recall.run(store, adapter)
    if not baseline.scores:
        return attempts

    for _ in range(rounds):
        current = resolve_prompt(store, "relevance", RELEVANCE)
        run_out = adapter.run(
            PROPOSE.format(
                score=_describe(baseline),
                failures=_failures(store, baseline),
                prompt=current,
                knobs=_knobs(store),
                history=ledger.history(),
            ),
            schema=PROPOSE_SCHEMA,
            timeout=int(store.config.get("limits.agent_timeout_s", 180)),
        )
        if not run_out.ok or not run_out.data:
            break

        data = run_out.data
        kind, target, why_not = _validate(store, data)
        attempt = Attempt(
            at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            kind=kind or str(data.get("kind") or "?"),
            target=target or str(data.get("key") or "?"),
            hypothesis=str(data.get("hypothesis") or "").strip(),
            before=_score(baseline),
        )
        if why_not:
            attempt.verdict = f"not applied: {why_not}"
            attempts.append(attempt)
            ledger.add(attempt)
            continue

        if dry_run:
            attempt.verdict = "dry run — proposed but not measured"
            attempts.append(attempt)
            continue

        sandbox = Sandbox(store)
        try:
            if kind == "prompt":
                sandbox.apply_prompt(target, str(data.get("text")))
            else:
                sandbox.apply_config(target, _coerce(data.get("value")))
            store.invalidate()
            after = eval_recall.run(store, adapter)
            attempt.after = _score(after)

            # Strictly better on both, or it goes back. Anything else is a
            # trade, and a trade is the user's call rather than the loop's.
            wins = (
                after.precision >= baseline.precision
                and after.recall_rate >= baseline.recall_rate
                and (after.precision, -after.noise_tokens)
                > (baseline.precision, -baseline.noise_tokens)
            )
            if wins:
                attempt.kept = True
                attempt.verdict = "kept — better on precision without giving up recall"
                baseline = after
                sandbox.undo = None
            else:
                attempt.verdict = _why_rejected(baseline, after)
                sandbox.revert()
        finally:
            sandbox.revert()
            store.invalidate()

        attempts.append(attempt)
        ledger.add(attempt)

    return attempts


def _why_rejected(before: eval_recall.Report, after: eval_recall.Report) -> str:
    if after.recall_rate < before.recall_rate:
        lost = before.hits - after.hits
        return f"reverted — dropped {lost} lesson(s) that had helped"
    if after.precision < before.precision:
        return "reverted — served more noise for the same lessons"
    return "reverted — no measurable improvement"


def _coerce(value: Any) -> Any:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return int(number) if number.is_integer() else number
