"""Bringing an existing skills library into RMC.

People who need RMC have usually already built a worse version of it by hand.
The common shape is a directory of Claude or Codex skills grown by
introspection: an agent notices it learned something, writes a `SKILL.md`, and a
companion skill keeps an index of them. It works, and it has three costs RMC
exists to remove.

* **Recall is manual.** A skill fires when its `description` matches what the
  user typed, or when the agent remembers to reach for it. Knowledge that is
  not recognised is not retrieved, and nobody finds out.
* **Nothing consolidates.** Twenty skills that share a procedure stay twenty
  skills, and each keeps every line it ever accumulated.
* **Nothing is scored.** A skill that has never once changed an outcome is
  indistinguishable from the one that saves an hour a week.

**Migration is a file conversion, and deliberately nothing more.** One skill
becomes one lesson: the body copied byte for byte, `description` becoming the
gist, `name` the title, the directory name the family. No model call, no
splitting, no rewriting. A library of 5,000 lines imports for the cost of
reading 5,000 lines off disk.

This is a reversal, and the reasoning behind the previous design is worth
recording because it was not silly — it was aimed at the wrong constraint.

The old path asked a model to split each skill into atomic lessons, on the
argument that a skill is several claims in a trench coat while a lesson is one
claim plus its trigger. That was true, and it cost a model call per skill, and
every one of those calls was a chance to paraphrase away the exact flag, the
exact error string, the exact constant — the things that make a lesson worth
retrieving at all. Twenty-four skills became a hundred and twenty-two lessons
that no longer said quite what the originals said.

The constraint it was solving was retrieval: a long document was expensive to
route and hard to match. Both halves of that have since gone away.

* Selection is a **search** now, not a rendered candidate list, so a long lesson
  costs nothing to route past. Length stopped being a retrieval tax.
* Compaction is driven by **observed use**, so a lesson that turns out to be
  four-fifths padding gets cut down by evidence rather than by a guess made at
  import time, before anything is known about which parts matter.

So the right thing to import is the original, unedited, and let the system that
measures which parts do work be the thing that shortens it. Import cheaply,
condense on evidence.

**Nothing is deleted.** Migration only ever adds. What to remove afterwards is
the user's call, made once they can see that RMC recalls the same knowledge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

# Skills whose subject is *writing and maintaining skills*. Importing these
# fills a new memory with instructions for operating the system being replaced.
#
# A name list, not a judgement — which is why it is visible here, printed in the
# report, and overridable with `--all`. The alternative was a model call per
# skill to decide, which is most of the cost this rewrite removes, spent on the
# easiest question in the process.
#
# **The list is deliberately narrow, and errs toward importing.** A first draft
# included `sync-repos` and `handoff` on the strength of their names; the first
# clones git repositories and the second writes a runbook for a human, and both
# are exactly the kind of hard-won procedure worth keeping. Wrongly importing
# something costs a lesson nobody retrieves. Wrongly skipping one loses
# knowledge silently, and the user has no reason to go looking for it. So a name
# only belongs here if it cannot plausibly mean anything but skill-writing.
CAPTURE_MACHINERY = (
    "create-skill",
    "skill-creator",
    "sync-skills",
    "introspect",
    "capture-knowledge",
)


@dataclass
class Skill:
    path: Path
    name: str
    description: str
    body: str

    @property
    def lines(self) -> int:
        return self.body.count("\n") + 1

    @property
    def slug(self) -> str:
        """The directory name, which is already kebab-case and stable.

        Preferred over the `name` field, which is prose ("Terraform Migrate")
        and varies in style across a library.
        """
        return _slug(self.path.parent.name or self.name)

    @property
    def siblings(self) -> list[Path]:
        """Files beside the skill that it may refer to.

        A skill with a `references/` directory or a script next to it is a
        document with attachments, and a copy that says nothing about them
        produces a lesson citing files the reader cannot find.
        """
        try:
            return sorted(
                p
                for p in self.path.parent.rglob("*")
                if p.is_file() and p != self.path and not p.name.startswith(".")
            )
        except Exception:
            return []


@dataclass
class Outcome:
    skill: Skill
    verdict: str = ""  # import | superseded | empty
    reason: str = ""
    imported: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    error: str = ""


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", str(text).strip().lower())
    return cleaned.strip("-")[:48] or "general"


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
            fields[key] = "" if value in (">", "|") else value
        elif key and line.strip():
            # Folded scalars (`description: >`) continue on indented lines.
            fields[key] = (fields[key] + " " + line.strip()).strip()
    return fields, body.strip()


def discover(root: Path) -> Iterator[Skill]:
    """Every skill under a directory, in a stable order.

    Worktrees and vendored copies are excluded. A checkout under
    `.claude/worktrees/` holds a full second copy of the library, and importing
    both would double every lesson.
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


HOSTS = (".claude", ".codex")


def candidate_roots(cwd: Path | None = None, home: Path | None = None) -> list[Path]:
    """Everywhere a skills library might live, whether or not it does.

    Both hosts, because a library assembled for one is usually the same
    knowledge as the library assembled for the other, and a migration that
    silently covered half of it would look like it had finished.
    """
    cwd = cwd or Path.cwd()
    home = home or Path.home()
    return [base / host / "skills" for base in (cwd, home) for host in HOSTS]


def default_roots(cwd: Path | None = None, home: Path | None = None) -> list[Path]:
    """The candidates that actually exist."""
    return [c for c in candidate_roots(cwd, home) if c.is_dir()]


def to_node(skill: Skill) -> Any:
    """One skill, as one lesson, with the body untouched.

    The only thing added is a provenance line. It is not decoration: a skill
    that ships a `references/` directory refers to those files by relative path,
    and a copy of the document alone leaves the reader with citations that
    resolve to nothing.
    """
    from .node import Node
    from .util import new_id

    body = skill.body.strip()
    siblings = skill.siblings
    note = f"\n\n---\nImported verbatim from `{skill.path}`."
    if siblings:
        note += (
            f" {len(siblings)} file(s) beside it hold detail this document refers to"
            " — read them from that directory when it cites one."
        )
    return Node(
        id=new_id("n"),
        family=skill.slug,
        body=body + note,
        level=0,
        title=skill.name.strip() or skill.slug,
        # The skill's `description` was written to answer "when should an agent
        # reach for this", which is exactly what a gist is for. Copied whole
        # rather than shortened: it is what the selector's search matches
        # against, and every trigger word dropped is a way the lesson stops
        # being findable.
        gist=" ".join(skill.description.split()),
        origin="migrated",
    )


def absorb(store: Any, node: Any) -> tuple[str, str]:
    """Write one lesson straight to the store.

    No placement call. Reconciliation exists because a lesson minted from a
    session may restate or contradict something already known — but a skills
    library is a set of distinct documents the user already curated, and asking
    a model to compare each against the whole store is a call per skill for a
    question the directory structure has already answered. The dedup that does
    apply here is by skill name, and that is a string comparison in ``run``.

    What is left undone is honest and recoverable: reflection reconciles these
    lessons as they are actually used, which is when there is evidence about
    which of two overlapping lessons is the one that works.
    """
    with store.lock("write", wait_s=90) as lock:
        if not lock.acquired:
            return "locked", ""
        store.invalidate()
        store.save_node(node)
    return "new", node.id


def run(
    store: Any,
    adapter: Any = None,
    roots: list[Path] | None = None,
    *,
    apply_changes: bool = False,
    limit: int = 0,
    include_machinery: bool = False,
) -> list[Outcome]:
    """Convert a skills library. Reads everything; writes only if asked.

    ``adapter`` is accepted and unused. Migration costs no model calls at all
    now, and keeping the parameter means the CLI and any existing caller do not
    have to know that.
    """
    skills = [s for root in (roots or []) for s in discover(root)]
    seen: set[str] = set()
    unique: list[Skill] = []
    for skill in skills:
        # The same library is often installed both per-project and globally.
        key = skill.slug
        if key in seen:
            continue
        seen.add(key)
        unique.append(skill)
    if limit:
        unique = unique[:limit]

    outcomes: list[Outcome] = []
    for skill in unique:
        outcome = Outcome(skill=skill)
        if not include_machinery and skill.slug in CAPTURE_MACHINERY:
            outcome.verdict = "superseded"
            outcome.reason = "captures knowledge rather than holding it; RMC does this itself"
            outcomes.append(outcome)
            continue

        node = to_node(skill)
        outcome.verdict = "import"
        label = f"{node.title} [{node.family}]"
        if apply_changes:
            action, ident = absorb(store, node)
            if action == "locked":
                outcome.error = "another process holds the store lock"
            else:
                outcome.imported.append(f"{label} ({node.tokens} tok) [{ident}]")
        else:
            outcome.imported.append(f"{label} ({node.tokens} tok)")
        outcomes.append(outcome)
    return outcomes
