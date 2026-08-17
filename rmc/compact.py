"""Compression: generate a shorter lesson, then earn the right to keep it.

The acceptance gate is the check from the original design sketch — spawn a fresh
agent, give it the task plus the compressed lesson, see whether you still get the
right output — hardened in three ways:

1. **Fresh process.** Validation runs in a new `claude -p` / `codex exec`, so the
   only thing carrying knowledge is the lesson text. Validating inside the main
   agent's context leaks its memory of the verbose lesson and every compression
   looks successful.
2. **Subtree-wide regression set.** Validating only against the episode that
   triggered the compression is how you get a beautifully compressed, useless
   tree. The set is the union over the node's whole subtree.
3. **Rejections are informative.** A rejected candidate records which episodes
   failed; those become `preserve:` hints for the next attempt, so the
   compressor converges instead of thrashing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adapters import Adapter
from .judge import Judge
from .node import Delta, Node
from .prompts import (
    COMPRESS,
    COMPRESS_SCHEMA,
    JUDGE,
    JUDGE_SCHEMA,
    REPLAY_PROBE,
)
from .store import Episode, Store
from .util import count_tokens, stable_id, truncate


@dataclass
class ReplayOutcome:
    episode_id: str
    ok: bool
    reason: str = ""


@dataclass
class CompactionResult:
    node_id: str
    accepted: bool
    reason: str = ""
    new_node: Node | None = None
    before_tokens: int = 0
    after_tokens: int = 0
    replays: list[ReplayOutcome] = field(default_factory=list)
    dropped: list[Delta] = field(default_factory=list)
    generality: str = "same"  # more | same | less — the second axis of worth
    warnings: list[str] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        return (self.after_tokens / self.before_tokens) if self.before_tokens else 1.0

    @property
    def pass_rate(self) -> float:
        if not self.replays:
            return 0.0
        return sum(1 for r in self.replays if r.ok) / len(self.replays)


# --------------------------------------------------------------------------- #
# eligibility
# --------------------------------------------------------------------------- #


def due_nodes(store: Store) -> list[Node]:
    """Nodes that have earned a compression attempt.

    This is the "the more you use it, the more abstract it gets" mechanic made
    concrete: successful recalls since the last attempt are the trigger.
    """
    if not store.config.get("compaction.enabled", True):
        return []
    min_successes = int(store.config.get("compaction.min_successes", 1))
    max_level = int(store.config.get("compaction.max_level", 6))
    cooldown = int(store.config.get("compaction.cooldown_s", 900))
    now = time.time()

    last_attempt: dict[str, float] = {}
    for event in store.read_events("compaction", limit=2000):
        node_id = event.get("node")
        if node_id:
            last_attempt[node_id] = max(last_attempt.get(node_id, 0.0), _epoch(event.get("ts")))

    out = []
    for node in store.nodes():
        if node.status != "active" or not node.is_apex or node.level >= max_level:
            continue
        if node.stats.successes < min_successes:
            continue
        if now - last_attempt.get(node.id, 0.0) < cooldown:
            continue
        if len(store.regression_set(node)) == 0:
            continue  # nothing to validate against; compressing blind is worse than not
        out.append(node)
    out.sort(key=lambda n: -n.stats.successes)
    return out


def _epoch(ts: Any) -> float:
    if not isinstance(ts, str):
        return 0.0
    try:
        from datetime import datetime

        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").timestamp()
    except Exception:
        return 0.0


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #


def replay_episode(
    store: Store,
    adapter: Adapter,
    episode: Episode,
    lesson_body: str,
    *,
    cwd: Path | None = None,
) -> ReplayOutcome:
    """Re-run one recorded episode against a candidate lesson, in a fresh process.

    Uses the *probe* form rather than asking for the work to be redone. Replay is
    testing whether the compressed lesson still transfers its knowledge, so a
    short statement of approach is both a fairer and a far cheaper signal than a
    full implementation — which would otherwise be judged on scaffolding
    completeness and truncation artefacts rather than on the lesson.
    """
    timeout = int(store.config.get("limits.agent_timeout_s", 180))

    run = adapter.run(
        REPLAY_PROBE.format(task_id=episode.id, pack=lesson_body, task=episode.prompt),
        cwd=cwd,
        timeout=timeout,
        tools=False,
    )
    if not run.ok:
        return ReplayOutcome(episode.id, False, f"agent error: {run.error[:200]}")

    # A mechanical check harvested from the original session beats a judge.
    check = episode.check or {}
    if check.get("type") == "contains":
        needles = check.get("values") or []
        missing = [n for n in needles if n.lower() not in run.text.lower()]
        return ReplayOutcome(episode.id, not missing, f"missing: {missing}" if missing else "")

    verdict = adapter.run(
        JUDGE.format(
            task_id=episode.id,
            task=truncate(episode.prompt, 4000),
            expected=truncate(episode.accepted_summary, 4000),
            context=lesson_body,
            candidate=truncate(run.text, 4000),
        ),
        schema=JUDGE_SCHEMA,
        timeout=timeout,
    )
    if not verdict.ok or not verdict.data:
        # An unreadable judge must not be scored as a pass — that would let
        # compressions through on infrastructure failure.
        return ReplayOutcome(episode.id, False, f"judge unavailable: {verdict.error[:160]}")
    return ReplayOutcome(
        episode.id,
        bool(verdict.data.get("pass")),
        str(verdict.data.get("reason") or "")[:300],
    )


def validate(
    store: Store,
    adapter: Adapter,
    lesson_body: str,
    episodes: list[Episode],
    *,
    cwd: Path | None = None,
) -> list[ReplayOutcome]:
    return [replay_episode(store, adapter, e, lesson_body, cwd=cwd) for e in episodes]


# --------------------------------------------------------------------------- #
# compression
# --------------------------------------------------------------------------- #


def compress_node(
    store: Store,
    adapter: Adapter,
    node: Node,
    *,
    cwd: Path | None = None,
    dry_run: bool = False,
) -> CompactionResult:
    config = store.config
    ratio = float(config.get("compaction.max_ratio", 0.6))
    threshold = float(config.get("compaction.threshold", 1.0))
    k = int(config.get("compaction.regression_k", 5))

    episodes = store.regression_set(node, limit=k)
    result = CompactionResult(node_id=node.id, accepted=False, before_tokens=node.tokens)

    if not episodes:
        result.reason = "no regression episodes; refusing to compress blind"
        return result

    target = max(24, int(node.tokens * ratio))
    run = adapter.run(
        COMPRESS.format(
            body=node.body,
            covers="\n".join(f"- {truncate(e.prompt, 240)}" for e in episodes) or "(none)",
            preserve="\n".join(f"- {p}" for p in node.preserve) or "(none)",
            # What reflection actually watched this lesson do. Without it the
            # compressor is choosing what to cut from the text alone, which is
            # why compression used to be a guess that replay either caught or
            # did not.
            load_bearing="\n".join(f"- {p}" for p in node.load_bearing)
            or "(nothing observed yet — compress conservatively)",
            target_tokens=target,
            ratio=ratio,
        ),
        schema=COMPRESS_SCHEMA,
        timeout=int(config.get("limits.agent_timeout_s", 180)),
    )
    if not run.ok or not run.data:
        result.reason = f"compressor failed: {run.error[:200]}"
        return result

    body = str(run.data.get("body") or "").strip()
    if not body:
        result.reason = "compressor returned an empty lesson"
        return result

    dropped = [Delta.from_dict(d) for d in (run.data.get("dropped") or [])]
    dropped = [d for d in dropped if d.claim.strip()]
    result.after_tokens = count_tokens(body)
    result.dropped = dropped

    ok, why = _validate_manifest(node, body, dropped, bool(run.data.get("lossless")))
    if not ok:
        result.reason = why
        store.log("compaction", node=node.id, accepted=False, reason=why)
        return result

    # Worth is a judgement, not a threshold. A candidate can earn its place by
    # costing less *or* by being more general — saying at a higher level what the
    # original said about one case. A ratio sees only the first axis, so a
    # genuinely better abstraction was being refused for saving 22% instead of
    # 25%. The measurements go to the judge as evidence; the verdict is its own.
    #
    # Correctness is a different question and stays mechanical: replay, below.
    verdict = Judge(store, adapter).worth_keeping(
        node.body,
        body,
        [d.claim for d in dropped],
        {
            "before": node.tokens,
            "after": result.after_tokens,
            "ratio": result.ratio,
            "target_ratio": ratio,
        },
    )
    if verdict is not None and not verdict.get("keep", True):
        result.reason = f"not worth keeping: {str(verdict.get('why') or '')[:200]}"
        store.log(
            "compaction",
            node=node.id,
            accepted=False,
            reason=result.reason,
            ratio=round(result.ratio, 3),
            generality=verdict.get("generality"),
        )
        return result
    result.generality = str((verdict or {}).get("generality") or "same")
    if result.after_tokens > node.tokens * ratio:
        # Advisory now, not fatal — recorded so a store full of marginal
        # compressions is visible rather than silently accumulating.
        result.warnings.append(
            f"reduction below target: {result.after_tokens}/{node.tokens} tokens "
            f"({result.ratio:.0%} vs {ratio:.0%}) — kept for generality: "
            f"{str((verdict or {}).get('why') or '')[:120]}"
        )

    result.replays = validate(store, adapter, body, episodes, cwd=cwd)
    if result.pass_rate < threshold:
        failures = [r for r in result.replays if not r.ok]
        result.reason = f"regression pass-rate {result.pass_rate:.0%} < {threshold:.0%}"
        if not dry_run:
            _record_rejection(store, node, failures)
        store.log(
            "compaction",
            node=node.id,
            accepted=False,
            reason=result.reason,
            failed=[f.episode_id for f in failures],
        )
        return result

    result.accepted = True
    result.reason = f"accepted at {result.ratio:.0%} of original, {result.pass_rate:.0%} replay pass"
    if dry_run:
        return result

    result.new_node = _promote(
        store, node, body, dropped, run.data.get("title"), episodes, run.data.get("gist")
    )
    store.log(
        "compaction",
        node=node.id,
        accepted=True,
        new_node=result.new_node.id,
        level=result.new_node.level,
        before=result.before_tokens,
        after=result.after_tokens,
        pass_rate=result.pass_rate,
    )
    return result


def _validate_manifest(
    node: Node, body: str, dropped: list[Delta], lossless: bool = False
) -> tuple[bool, str]:
    """Reject compressions that hid what they removed.

    A manifest-free compression is worse than no compression: it cannot be
    descended, so the lost detail is simply gone.

    But shrinking is not the same as dropping. A compressor that tightens
    wording and cuts repetition legitimately has nothing to declare, and an
    earlier version rejected exactly those — refusing the safest compressions
    for being honest. Silence and a genuine no-loss claim look identical from
    outside, so the compressor is made to say which it is. Replay still gates
    the claim; asserting `lossless` falsely fails there instead, where the
    dishonesty is at least legible.
    """
    before, after = node.tokens, count_tokens(body)
    if after >= before:
        return False, "candidate is not smaller than the original"
    shrink = (before - after) / max(1, before)
    if shrink >= 0.15 and not dropped and not lossless:
        return False, (
            f"manifest under-reported: dropped {shrink:.0%} of tokens, declared nothing "
            f"and did not claim losslessness"
        )
    return True, ""


def _promote(
    store: Store,
    node: Node,
    body: str,
    dropped: list[Delta],
    title: Any,
    episodes: list[Episode],
    gist: Any = None,
) -> Node:
    """Write the compressed node and re-link the family."""
    # New losses are held by `node`; inherited losses keep their original holder,
    # which is what lets the apex jump straight to detail several levels down.
    manifest: list[Delta] = []
    for delta in dropped:
        manifest.append(Delta(claim=delta.claim, kind=delta.kind, holder=delta.holder or node.id))
    seen = {d.claim for d in manifest}
    for inherited in node.dropped:
        if inherited.claim not in seen:
            manifest.append(inherited)
            seen.add(inherited.claim)

    new = Node(
        id=stable_id("n", node.id, body[:400]),
        family=node.family,
        body=body,
        level=node.level + 1,
        title=str(title or node.title or node.family),
        gist=str(gist or node.gist or ""),
        derived_from=[node.id],
        covers_tasks=sorted({*node.covers_tasks, *(e.id for e in episodes)}),
        tags=list(node.tags),
        dropped=manifest,
        origin="compression",
    )
    store.save_node(new)

    # Append: a node compressed after being merged (or vice versa) keeps both
    # abstractions. Assigning here is what used to orphan the earlier parent.
    if new.id not in node.parents:
        node.parents.append(new.id)
    store.save_node(node)
    store.invalidate()
    return new


def _record_rejection(store: Store, node: Node, failures: list[ReplayOutcome]) -> None:
    """Turn a failed compression into hints for the next one."""
    hints = list(node.preserve)
    for failure in failures:
        episode = next((e for e in store.episodes(node.family) if e.id == failure.episode_id), None)
        hint = truncate(
            failure.reason or (episode.prompt if episode else failure.episode_id), 200
        )
        if hint and hint not in hints:
            hints.append(hint)
    node.preserve = hints[-8:]
    store.save_node(node)


# --------------------------------------------------------------------------- #
# repair
# --------------------------------------------------------------------------- #


def repair(store: Store, node: Node, *, min_rescues: int = 2) -> list[str]:
    """Fold repeatedly-needed deltas back into the body.

    A delta that keeps rescuing the same node is proof the compression cut too
    deep. Rather than paying to re-attach it on every recall, we permanently
    restore it and drop it from the manifest — the tree healing where it was
    over-cut.
    """
    counts: dict[str, int] = {}
    for event in store.read_events("rescue", limit=5000):
        if event.get("node") == node.id and event.get("claim"):
            counts[event["claim"]] = counts.get(event["claim"], 0) + 1

    restored = [claim for claim, n in counts.items() if n >= min_rescues]
    if not restored:
        return []

    additions = "\n".join(f"- {claim}" for claim in restored)
    node.body = f"{node.body.rstrip()}\n{additions}"
    node.dropped = [d for d in node.dropped if d.claim not in set(restored)]
    node.stats.rescues += len(restored)
    store.save_node(node)
    store.log("repair", node=node.id, restored=restored)
    return restored


def run_due(
    store: Store,
    adapter: Adapter,
    *,
    limit: int = 1,
    cwd: Path | None = None,
    dry_run: bool = False,
) -> list[CompactionResult]:
    """Process the compaction queue. Lock-guarded so hooks cannot race it."""
    results: list[CompactionResult] = []
    with store.lock("compact") as lock:
        if not lock.acquired:
            return results
        for node in due_nodes(store)[:limit]:
            repair(store, node)
            results.append(compress_node(store, adapter, node, cwd=cwd, dry_run=dry_run))
    return results


def _slug(text: str) -> str:
    cleaned = "".join(c if c.isalnum() else "-" for c in str(text).strip().lower())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")[:48] or "general"


GIST = """RMC:gist

Write one line, at most 25 words, that names what this lesson is about and when
it applies. A future agent reads only this line to decide whether to open the
lesson at all, so be specific: name the tool, command, service or system it
concerns rather than its category. Do not summarise the advice, identify the
situation.

<<<LESSON
{body}
LESSON>>>
"""

GIST_SCHEMA = {
    "type": "object",
    "required": ["gist"],
    "properties": {"gist": {"type": "string"}},
}


def _backfill_gists(store: Store, adapter: Adapter, *, limit: int = 20) -> int:
    """Give older lessons the routing view they were written without.

    One implementation, in summary.py, shared with the add and fold paths — a
    second copy here would drift from whatever those write.
    """
    from .summary import backfill

    return backfill(store, adapter, limit=limit)
