"""Bringing an existing skills library into RMC.

People who need RMC have usually already built a worse version of it by hand.
The common shape is a directory of Claude skills grown by introspection: an
agent notices it learned something, writes a `SKILL.md`, and a companion skill
keeps an index of them. It works, and it has three costs RMC exists to remove.

* **Recall is manual.** A skill fires when its `description` matches what the
  user typed, or when the agent remembers to reach for it. Knowledge that is
  not recognised is not retrieved, and nobody finds out.
* **Nothing consolidates.** Twenty skills that share a procedure stay twenty
  skills. The library grows monotonically and the index grows with it.
* **Nothing is scored.** A skill that has never once changed an outcome is
  indistinguishable from the one that saves an hour a week.

Migration is therefore not a file conversion. A skill is a *document*, often
hundreds of lines holding many separate claims, and a lesson is one claim with
the situation that summons it. Splitting one into the other is a judgement, so
the model does it; this module supplies the files, the structure and the
routing.

Three decisions shape the result.

**Not everything should come across.** A skills library contains both knowledge
and the machinery that captured it — `introspect`, `create-skill`,
`sync-skills`, an index file. Importing the machinery would fill the store with
lessons about how to maintain a system the user is leaving. Those are reported
as superseded, not converted.

**Nothing is deleted.** Migration only ever adds. What to remove afterwards is
the user's call, made once they can see that RMC actually recalls the same
knowledge, and the report ends by naming the candidates rather than acting.

**Everything goes through normal placement.** Imported lessons are reconciled,
deduplicated and conflict-checked exactly like any other, because a library
built over months contains contradictions, and a bulk path that skipped that
would import both sides in silence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

SPLIT = """RMC:migrate

A team has been keeping engineering knowledge as "skills": markdown documents
an agent loads when it recognises the situation in the document's description.
They are moving to a memory system that retrieves automatically, and this
document has to be turned into what that system stores.

The unit here is a **lesson**: one claim, plus the situation that should make a
future agent reach for it. A skill is usually several of those in a trench
coat — a runbook with three traps and a piece of hard-won environment
knowledge — so split it. Do not split further than the claims go: two paragraphs
that only make sense together are one lesson.

First decide whether this document should come across at all.

Skip it if it is **machinery for capturing knowledge rather than knowledge**:
skills whose subject is writing skills, indexing them, syncing them, reflecting
in order to produce them, or handing off between sessions. The new system does
all of that itself, so importing them would fill it with instructions for
maintaining the system being replaced. Set `verdict` to `superseded` and say in
`reason` which part replaces it.

Skip it also if there is no durable knowledge in it — a stub, a pure
index, or a file that only points at other files. Use `empty`.

Otherwise `import`, and write the lessons.

For each lesson:

- `body` — the claim, written as instruction to a future agent, in enough
  detail to act on without the original document. Keep the specifics that make
  it worth having: the exact command, the flag, the constant, the error string,
  the path. A lesson stripped to a principle is not worth retrieving. Where the
  skill records a trap or a failed approach, keep that too — knowing what does
  not work is most of the value.
- `title` — a short noun phrase naming the claim, not the topic.
- `gist` — one line, at most 25 words, naming *when this applies*. The skill's
  own `description` field was written for exactly this purpose; reuse what is
  good in it. Identify the situation, do not summarise the advice.
- `family` — a short kebab-case slug for the recurring situation. Prefer an
  existing subject over inventing one per lesson.

Two things to leave behind. Anything about how to invoke the skill itself
("run /deploy-schema", "this skill uses X") describes the old system's
plumbing. And anything that was true only of one incident, unless the document
presents it as a rule.

SKILL FILE: {path}

<<<SKILL
{body}
SKILL>>>
"""

SPLIT_SCHEMA = {
    "type": "object",
    "required": ["verdict"],
    "properties": {
        "verdict": {"type": "string", "enum": ["import", "superseded", "empty"]},
        "reason": {"type": "string"},
        "lessons": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["body", "gist"],
                "properties": {
                    "body": {"type": "string"},
                    "title": {"type": "string"},
                    "gist": {"type": "string"},
                    "family": {"type": "string"},
                },
            },
        },
    },
}


@dataclass
class Skill:
    path: Path
    name: str
    description: str
    body: str

    @property
    def lines(self) -> int:
        return self.body.count("\n") + 1


@dataclass
class Outcome:
    skill: Skill
    verdict: str = ""
    reason: str = ""
    imported: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    error: str = ""


def _frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Pull the YAML-ish header off a skill file.

    Deliberately shallow: only `name` and `description` are wanted, both are
    scalars, and a real parser would drag in a dependency for two fields. A
    header that does not parse is not an error — the body is what matters.
    """
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    head, body = text[3:end], text[end + 4 :]

    fields: dict[str, str] = {}
    key = ""
    for line in head.splitlines():
        match = re.match(r"^([a-zA-Z_-]+):\s*(.*)$", line)
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()
            fields[key] = "" if value == ">" or value == "|" else value
        elif key and line.strip():
            # Folded scalars (`description: >`) continue on indented lines.
            fields[key] = (fields[key] + " " + line.strip()).strip()
    return fields, body.strip()


def discover(root: Path) -> Iterator[Skill]:
    """Every skill under a directory, in a stable order.

    Worktrees and vendored copies are excluded. A checkout under
    `.claude/worktrees/` holds a full second copy of the library, and importing
    both would double every lesson and then ask the model to reconcile each pair
    against its own twin.
    """
    skip = ("/worktrees/", "/node_modules/", "/.git/")
    for path in sorted(root.rglob("SKILL.md")):
        if any(part in str(path) for part in skip):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        fields, body = _frontmatter(text)
        if not body.strip():
            continue
        yield Skill(
            path=path,
            name=fields.get("name") or path.parent.name,
            description=fields.get("description", ""),
            body=body,
        )


def default_roots() -> list[Path]:
    """Where Claude keeps skills: this project, then the user's own."""
    found = []
    for candidate in (Path.cwd() / ".claude" / "skills", Path.home() / ".claude" / "skills"):
        if candidate.is_dir():
            found.append(candidate)
    return found


def split(adapter: Any, store: Any, skill: Skill) -> tuple[str, str, list[dict[str, str]]]:
    """Ask the model what, if anything, this document should become."""
    from .util import truncate

    run = adapter.run(
        SPLIT.format(path=skill.path.name, body=truncate(skill.body, 24000)),
        schema=SPLIT_SCHEMA,
        timeout=int(store.config.get("limits.agent_timeout_s", 180)),
    )
    if not run.ok or not run.data:
        return "", (run.error or "no answer")[:160], []
    data = run.data
    lessons = [
        lesson
        for lesson in (data.get("lessons") or [])
        if isinstance(lesson, dict) and str(lesson.get("body") or "").strip()
    ]
    return str(data.get("verdict") or ""), str(data.get("reason") or ""), lessons


def absorb(store: Any, adapter: Any, lesson: dict[str, str]) -> tuple[str, str]:
    """Store one lesson through the ordinary placement path.

    Bulk import is exactly where reconciliation matters most: a library grown
    over months has near-duplicates and outright contradictions in it, and a
    fast path that appended everything would import both sides of a
    disagreement without noticing. Returns (action, node id).
    """
    from .node import Node
    from .placement import apply, decide
    from .summary import refresh
    from .util import new_id

    family = re.sub(r"[^a-z0-9]+", "-", str(lesson.get("family") or "general").lower()).strip("-")
    node = Node(
        id=new_id("n"),
        family=family or "general",
        body=str(lesson["body"]).strip(),
        level=0,
        title=str(lesson.get("title") or "").strip(),
        gist=str(lesson.get("gist") or "").strip(),
        origin="migrated",
    )
    with store.lock("write", wait_s=90) as lock:
        if not lock.acquired:
            return "locked", ""
        store.invalidate()
        decision = decide(store, adapter, body=node.body, family_hint=node.family)
        result = apply(store, decision, node)
        if result.node is not None and not result.node.gist.strip():
            refresh(store, adapter, result.node)
    return decision.action, (result.node.id if result.node else "")


def run(
    store: Any,
    adapter: Any,
    roots: list[Path],
    *,
    apply_changes: bool = False,
    limit: int = 0,
) -> list[Outcome]:
    """Convert a skills library. Reads everything; writes only if asked."""
    skills = [s for root in roots for s in discover(root)]
    seen: set[str] = set()
    unique = []
    for skill in skills:
        # The same library is often installed both per-project and globally.
        if skill.name in seen:
            continue
        seen.add(skill.name)
        unique.append(skill)
    if limit:
        unique = unique[:limit]

    outcomes: list[Outcome] = []
    for skill in unique:
        outcome = Outcome(skill=skill)
        verdict, reason, lessons = split(adapter, store, skill)
        outcome.verdict, outcome.reason = verdict, reason
        if not verdict:
            outcome.error = reason
            outcomes.append(outcome)
            continue

        if verdict == "import" and apply_changes:
            for lesson in lessons:
                action, ident = absorb(store, adapter, lesson)
                label = f"{lesson.get('title') or lesson.get('family') or ident}"
                if action == "duplicate":
                    outcome.duplicates.append(label)
                elif action == "conflict":
                    outcome.conflicts.append(label)
                else:
                    outcome.imported.append(f"{label} [{ident}]")
        elif verdict == "import":
            outcome.imported = [
                str(lesson.get("title") or lesson.get("family") or "untitled")
                for lesson in lessons
            ]
        outcomes.append(outcome)
    return outcomes
