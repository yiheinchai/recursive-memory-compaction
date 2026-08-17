"""End-to-end tests. Run with: python3 -m unittest discover -s tests

Two kinds of test here, and the split mirrors the architecture:

* **Structure** — the walk, the budget, the cache, how a verdict is plumbed into
  the store. Judgements are stubbed with a router, so these assert what the
  harness does with an answer, never what the answer is.
* **Control flow** — compress, fail, descend, rescue. These use ``MockWorld``,
  where a task is solved iff the required ``@fact`` tokens are present in the
  lesson, so the whole cycle really executes rather than being mocked at the
  seams.

Nothing here asserts on lexical similarity, because nothing in RMC computes it.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rmc import yamlish  # noqa: E402
from rmc.adapters.mock import MockAdapter, MockWorld  # noqa: E402
from rmc.compact import compress_node, due_nodes, repair  # noqa: E402
from rmc.judge import Budget, Judge, Pick, walk  # noqa: E402
from rmc.node import Delta, Node  # noqa: E402
from rmc.recall import recall_pack, select_lessons, solve_with_descent  # noqa: E402
from rmc.redact import redact  # noqa: E402
from rmc.reflect import Outcome, observe  # noqa: E402
from rmc.selection import Diagnosis, build_candidates, rank  # noqa: E402
from rmc.signals import SessionFacts, ToolEvent, digest, parse_transcript, worth_assessing  # noqa: E402
from rmc.store import Episode, Store  # noqa: E402


def router(payload):
    """A MockAdapter whose every judgement is a fixed answer."""
    return MockAdapter(router=lambda prompt, schema: payload)


def counting_router(payload, log: list):
    def _r(prompt, schema):
        log.append(prompt)
        return payload

    return MockAdapter(router=_r)


class StoreCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.store = Store.init(self.base)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def add_node(self, **kw) -> Node:
        node = Node(**kw)
        self.store.save_node(node)
        self.store.invalidate()
        return self.store.get(node.id)

    def add_episode(
        self, ident, family, prompt, *, outcome="success", served=(), used=None, summary="done"
    ) -> Episode:
        ep = Episode(
            id=ident,
            family=family,
            prompt=prompt,
            outcome=outcome,
            confidence=0.9,
            served=list(served),
            used=list(served if used is None else used),
            accepted_summary=summary,
        )
        self.store.save_episode(ep)
        return ep


# --------------------------------------------------------------------------- #
# storage primitives
# --------------------------------------------------------------------------- #


class TestYamlish(unittest.TestCase):
    def test_roundtrip_without_pyyaml(self) -> None:
        original = yamlish._pyyaml
        yamlish._pyyaml = None
        try:
            doc = {
                "id": "n_1",
                "level": 3,
                "ok": True,
                "ratio": 0.6,
                "none": None,
                "list": ["a", "b"],
                "empty": [],
                "deep": {"a": {"b": 1}},
                "dropped": [
                    {"claim": "keep: 100ms, 400ms", "kind": "parameter", "holder": "n_0"},
                    {"claim": "second", "kind": "edge-case", "holder": None},
                ],
                "block": "one\ntwo",
            }
            self.assertEqual(yamlish.load(yamlish.dump(doc)), doc)
        finally:
            yamlish._pyyaml = original

    def test_reserved_words_are_quoted(self) -> None:
        original = yamlish._pyyaml
        yamlish._pyyaml = None
        try:
            doc = {"a": "null", "b": "true", "c": "123", "d": "yes"}
            self.assertEqual(yamlish.load(yamlish.dump(doc)), doc)
        finally:
            yamlish._pyyaml = original


class TestNode(StoreCase):
    def test_markdown_roundtrip(self) -> None:
        node = Node(
            id="n_abc",
            family="retry",
            title="Retry rules",
            body="Retry idempotent ops.\n@backoff-constants 100ms/400ms",
            level=2,
            derived_from=["n_x"],
            covers_tasks=["e1"],
            tags=["retry", "http"],
            dropped=[Delta("exact constants", "parameter", "n_x")],
            conflict="which delay?",
        )
        path = self.store.save_node(node)
        loaded = Node.from_markdown(path.read_text(), path)
        self.assertEqual(loaded.id, "n_abc")
        self.assertEqual(loaded.derived_from, ["n_x"])
        self.assertEqual(loaded.dropped[0].kind, "parameter")
        self.assertEqual(loaded.conflict, "which delay?")

    def test_posterior_is_laplace_smoothed(self) -> None:
        node = Node(id="n_1", family="f")
        self.assertAlmostEqual(node.stats.posterior, 0.5)
        node.stats.attempts, node.stats.successes = 8, 8
        self.assertGreater(node.stats.posterior, 0.85)


class TestRedaction(unittest.TestCase):
    def test_scrubs_credentials(self) -> None:
        text = (
            "export GITHUB_TOKEN=ghp_abcdefghij0123456789ABCDEFGHIJKLMNOP\n"
            "api_key = 'sk-proj-abcdefghijklmnopqrstuvwxyz012345'\n"
            "AKIAIOSFODNN7EXAMPLE"
        )
        out = redact(text)
        self.assertNotIn("ghp_abcdefghij0123456789", out)
        self.assertNotIn("sk-proj-abcdefghijkl", out)
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", out)

    def test_keeps_ordinary_prose_and_numbers(self) -> None:
        text = "Retry after 100ms then 400ms; commit 4050898 fixed it."
        self.assertEqual(redact(text), text)


# --------------------------------------------------------------------------- #
# transcript parsing — facts only, no classification
# --------------------------------------------------------------------------- #


class TestTranscriptParsing(unittest.TestCase):
    def write(self, rows) -> Path:
        import json

        tmp = Path(tempfile.mkdtemp()) / "t.jsonl"
        tmp.write_text("\n".join(json.dumps(r) for r in rows))
        return tmp

    def test_host_metadata_separates_human_turns_from_harness_turns(self) -> None:
        path = self.write(
            [
                {"type": "user", "message": {"role": "user", "content": "do the thing"}},
                {"type": "user", "isMeta": True, "message": {"role": "user", "content": "/goal blah"}},
                {
                    "type": "user",
                    "toolUseResult": {"is_error": False},
                    "message": {"role": "user", "content": [{"type": "tool_result", "content": "ok"}]},
                },
                {"type": "user", "toolDenialKind": "reject", "message": {"role": "user", "content": "no"}},
            ]
        )
        facts = parse_transcript(path)
        self.assertEqual(facts.user_messages, ["do the thing"])
        self.assertTrue(facts.denied)
        self.assertEqual(len(facts.tool_outputs), 1)

    def test_tool_calls_pair_to_results_by_id(self) -> None:
        path = self.write(
            [
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "id": "a", "name": "Bash", "input": {"command": "pytest"}}
                        ],
                    },
                },
                {
                    "type": "user",
                    "toolUseResult": {"is_error": True},
                    "message": {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "a", "content": "boom"}],
                    },
                },
            ]
        )
        facts = parse_transcript(path)
        self.assertEqual(len(facts.tool_events), 1)
        self.assertEqual(facts.tool_events[0].detail, "pytest")
        self.assertIs(facts.tool_events[0].ok, False)

    def test_explicit_is_error_false_records_success(self) -> None:
        """Presence, not truthiness — `is_error: false` says the call worked."""
        path = self.write(
            [
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": "a", "name": "Bash", "input": {"command": "ls"}}],
                    },
                },
                {
                    "type": "user",
                    "toolUseResult": {"is_error": False},
                    "message": {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "a", "content": "fine"}],
                    },
                },
            ]
        )
        self.assertIs(parse_transcript(path).tool_events[0].ok, True)

    def test_ok_stays_unknown_when_the_host_says_nothing(self) -> None:
        """Better an honest unknown than a regex guessing from output text."""
        path = self.write(
            [
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": "a", "name": "Bash", "input": {"command": "ls"}}],
                    },
                },
                {
                    "type": "user",
                    "toolUseResult": {},
                    "message": {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "a", "content": "error: nope"}],
                    },
                },
            ]
        )
        self.assertIsNone(parse_transcript(path).tool_events[0].ok)

    def test_repeated_standing_instruction_is_recorded_once(self) -> None:
        rows = [{"type": "user", "message": {"role": "user", "content": "always use tabs"}}] * 4
        self.assertEqual(len(parse_transcript(self.write(rows)).user_messages), 1)


class TestStructuralGate(unittest.TestCase):
    """Whether to ask is structural; what the answer is, is not."""

    def test_tiny_session_is_not_worth_judging(self) -> None:
        facts = SessionFacts(user_messages=["hi"], assistant_messages=["hello"], tool_calls=1)
        self.assertFalse(worth_assessing(facts))

    def test_a_human_follow_up_always_makes_it_worth_judging(self) -> None:
        facts = SessionFacts(user_messages=["do x", "no, do y"], tool_calls=1)
        self.assertTrue(worth_assessing(facts))

    def test_digest_preserves_order_of_events(self) -> None:
        facts = SessionFacts(
            user_messages=["run tests", "no, use the other port"],
            assistant_messages=["done"],
            tool_calls=2,
            tool_events=[
                ToolEvent("Bash", "pytest", "refused", False, "1"),
                ToolEvent("Bash", "PG_PORT=5433 pytest", "42 passed", True, "2"),
            ],
        )
        facts.first_prompt = facts.user_messages[0]
        facts.last_assistant = "done"
        text = digest(facts)
        self.assertLess(text.index("pytest"), text.index("PG_PORT=5433"))
        self.assertIn("no, use the other port", text)
        self.assertIn("FAILED", text)


# --------------------------------------------------------------------------- #
# the judge: structure around the judgement
# --------------------------------------------------------------------------- #


class TestJudge(StoreCase):
    def test_verdicts_are_cached(self) -> None:
        log: list = []
        judge = Judge(self.store, counting_router({"picks": []}, log))
        node = self.add_node(id="n_1", family="f", body="b")
        for _ in range(3):
            judge.relevance("same question", [node])
        self.assertEqual(len(log), 1)

    def test_unknown_ids_are_discarded(self) -> None:
        """The model must not be able to invent a node id we then act on."""
        node = self.add_node(id="n_real", family="f", body="b")
        judge = Judge(
            self.store,
            router({"picks": [{"id": "n_hallucinated", "verdict": "relevant"}]}),
        )
        self.assertEqual(judge.relevance("q", [node]), [])

    def test_unusable_answer_degrades_to_nothing(self) -> None:
        class Broken:
            ok = False
            data = None
            text = ""
            error = "boom"

        judge = Judge(self.store, MockAdapter(router=lambda p, s: Broken()))
        node = self.add_node(id="n_2", family="f", body="b")
        self.assertEqual(judge.relevance("q", [node]), [])


class TestWalk(StoreCase):
    def build_two_levels(self):
        child = self.add_node(id="n_child", family="f", body="detail", level=0)
        apex = self.add_node(
            id="n_apex", family="f", body="abstract", level=1, derived_from=[child.id]
        )
        child.parents = [apex.id]
        self.store.save_node(child)
        self.store.invalidate()
        return self.store.get("n_apex"), self.store.get("n_child")

    def test_descends_only_when_the_model_asks(self) -> None:
        apex, child = self.build_two_levels()
        log: list = []
        judge = Judge(
            self.store,
            counting_router({"picks": [{"id": "n_apex", "verdict": "relevant", "descend": False}]}, log),
        )
        result = walk(judge, "q", [apex], expand=self.store.children)
        self.assertEqual([n.id for n in result.selected], ["n_apex"])
        self.assertEqual(len(log), 1, "no second level should be examined")

    def test_descend_replaces_the_summary_with_its_detail(self) -> None:
        apex, child = self.build_two_levels()

        def route(prompt, schema):
            # First level asks to go deeper; second level accepts the child.
            if "n_apex" in prompt:
                return {"picks": [{"id": "n_apex", "verdict": "maybe", "descend": True}]}
            return {"picks": [{"id": "n_child", "verdict": "relevant", "descend": False}]}

        result = walk(judge_for(self.store, route), "q", [apex], expand=self.store.children)
        self.assertEqual([n.id for n in result.selected], ["n_child"])
        self.assertEqual(result.calls, 2)

    def test_unrelated_lines_are_not_opened(self) -> None:
        apex, _ = self.build_two_levels()
        log: list = []
        judge = Judge(
            self.store,
            counting_router({"picks": [{"id": "n_apex", "verdict": "unrelated", "descend": True}]}, log),
        )
        result = walk(judge, "q", [apex], expand=self.store.children)
        self.assertEqual(result.selected, [])
        self.assertEqual(len(log), 1)

    def test_budget_stops_the_walk_but_keeps_what_was_found(self) -> None:
        apex, child = self.build_two_levels()
        judge = judge_for(
            self.store,
            lambda p, s: {"picks": [{"id": "n_apex", "verdict": "maybe", "descend": True}]},
        )
        result = walk(judge, "q", [apex], expand=self.store.children, budget=Budget(max_calls=1))
        # The child was reached but never judged; dropping it silently would be
        # worse than serving something plausible.
        self.assertEqual([n.id for n in result.selected], ["n_child"])


def judge_for(store, route):
    return Judge(store, MockAdapter(router=route), use_cache=False)


# --------------------------------------------------------------------------- #
# recall
# --------------------------------------------------------------------------- #


class TestChunksAreJudgedTogether(StoreCase):
    """A wide level is several independent questions, so it is one round trip.

    Each call costs ~15s of which ~5s is process startup, and the number of
    chunks grows with the store. Asked in turn, a store big enough to need six
    chunks exceeds the recall timeout and serves nothing — so completeness at
    the top level would cost the whole feature.
    """

    def nodes(self, n):
        return [self.add_node(id=f"n_{i}", family=f"f{i}", body=f"Lesson {i}.") for i in range(n)]

    def test_every_chunk_is_asked(self) -> None:
        from rmc.judge import Budget, Judge, walk

        roots = self.nodes(9)
        asked: list[int] = []

        def route(prompt, schema):
            asked.append(prompt.count("[n_"))
            return {"picks": []}

        result = walk(Judge(self.store, MockAdapter(router=route)), "work", roots,
                      expand=lambda n: [], budget=Budget(max_calls=5), fanout=4, workers=4)
        self.assertEqual(sorted(asked), [1, 4, 4], "9 lessons across 3 chunks")
        self.assertEqual(result.calls, 3)

    def test_order_does_not_depend_on_which_finished_first(self) -> None:
        """A recall that returns different lessons run to run is not
        debuggable, and an eval over it measures scheduling noise."""
        from rmc.judge import Budget, Judge, walk

        roots = self.nodes(8)

        def route(prompt, schema):
            import re
            return {"picks": [{"id": i, "verdict": "relevant"}
                              for i in re.findall(r"n_\d+", prompt)]}

        runs = [
            [n.id for n in walk(Judge(self.store, MockAdapter(router=route)), "work", roots,
                                expand=lambda n: [], budget=Budget(max_calls=6),
                                fanout=3, workers=4).selected]
            for _ in range(3)
        ]
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(runs[1], runs[2])

    def test_one_failing_chunk_does_not_lose_the_others(self) -> None:
        from rmc.judge import Budget, Judge, walk

        roots = self.nodes(6)

        def route(prompt, schema):
            if "n_0" in prompt:
                raise RuntimeError("subprocess died")
            import re
            return {"picks": [{"id": i, "verdict": "relevant"}
                              for i in re.findall(r"n_\d+", prompt)]}

        result = walk(Judge(self.store, MockAdapter(router=route)), "work", roots,
                      expand=lambda n: [], budget=Budget(max_calls=4), fanout=3, workers=4)
        self.assertEqual([n.id for n in result.selected], ["n_3", "n_4", "n_5"])

    def test_budget_still_bounds_a_very_wide_level(self) -> None:
        """Concurrency makes coverage affordable; it does not make it free."""
        from rmc.judge import Budget, Judge, walk

        roots = self.nodes(20)
        calls: list[int] = []
        adapter = MockAdapter(router=lambda p, s: (calls.append(1), {"picks": []})[1])
        walk(Judge(self.store, adapter), "work", roots,
             expand=lambda n: [], budget=Budget(max_calls=2), fanout=4, workers=4)
        self.assertEqual(len(calls), 2)


class TestSkillMigration(StoreCase):
    """People who need RMC have usually built a worse version of it by hand.

    A skills library is months of work, so the conversion has to be honest
    about what it is doing: never delete, never import the machinery being
    replaced, and never bulk-append past the reconciliation that catches the
    contradictions such a library inevitably contains.
    """

    def skill(self, name, text):
        d = self.base / "skills" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(text, encoding="utf-8")
        return self.base / "skills"

    def test_frontmatter_and_body_are_separated(self) -> None:
        from rmc.migrate import _frontmatter

        fields, body = _frontmatter(
            "---\nname: deploy\ndescription: Ship it. Use when deploying.\n---\n\n# Deploy\nSteps."
        )
        self.assertEqual(fields["name"], "deploy")
        self.assertIn("Use when deploying", fields["description"])
        self.assertTrue(body.startswith("# Deploy"))

    def test_a_folded_description_is_joined(self) -> None:
        """`description: >` spreads the trigger text over several lines, and
        the trigger is the most valuable field in the file — it is already a
        gist."""
        from rmc.migrate import _frontmatter

        fields, _ = _frontmatter(
            "---\nname: x\ndescription: >\n  Use when the user says\n  reflect or retro.\n---\nBody."
        )
        self.assertEqual(fields["description"], "Use when the user says reflect or retro.")

    def test_a_file_with_no_frontmatter_still_migrates(self) -> None:
        from rmc.migrate import discover

        root = self.skill("plain", "# Just a heading\n\nAnd some knowledge.")
        found = [s for s in discover(root) if s.name == "plain"]
        self.assertEqual(len(found), 1)
        self.assertIn("knowledge", found[0].body)

    def test_worktree_copies_are_not_imported_twice(self) -> None:
        """A worktree holds a full second copy of the library. Importing both
        doubles every lesson and then asks the model to reconcile each against
        its own twin."""
        from rmc.migrate import discover

        self.skill("real", "---\nname: real\n---\nContent.")
        wt = self.base / "skills" / ".claude" / "worktrees" / "wt" / "skills" / "real"
        wt.mkdir(parents=True, exist_ok=True)
        (wt / "SKILL.md").write_text("---\nname: real\n---\nContent.", encoding="utf-8")
        self.assertEqual([s.name for s in discover(self.base / "skills")], ["real"])

    def test_an_empty_skill_is_skipped(self) -> None:
        from rmc.migrate import discover

        root = self.skill("hollow", "---\nname: hollow\n---\n")
        self.assertEqual([s.name for s in discover(root) if s.name == "hollow"], [])

    def test_planning_writes_nothing(self) -> None:
        """A skills library is months of work; the conversion is shown before
        it happens."""
        from rmc.migrate import run

        root = self.skill("deploy", "---\nname: deploy\n---\nAlways set PG_PORT first.")
        adapter = MockAdapter(router=lambda prompt, schema: {
            "verdict": "import",
            "lessons": [{"body": "Set PG_PORT before deploying.", "gist": "when deploying",
                         "title": "Set PG_PORT", "family": "deploy"}],
        })
        before = len(list(self.store.nodes()))
        outcomes = run(self.store, adapter, [root], apply_changes=False)
        self.store.invalidate()
        self.assertEqual(len(list(self.store.nodes())), before)
        self.assertEqual(outcomes[0].verdict, "import")
        self.assertEqual(len(outcomes[0].imported), 1)

    def test_applying_routes_through_normal_placement(self) -> None:
        """Bulk import is where reconciliation matters most — a library grown
        over months contains both sides of some disagreements."""
        from rmc.migrate import run

        root = self.skill("deploy", "---\nname: deploy\n---\nAlways set PG_PORT first.")

        def route(prompt, schema):
            if "RMC:migrate" in prompt:
                return {"verdict": "import", "lessons": [
                    {"body": "Set PG_PORT before deploying.", "gist": "when deploying",
                     "title": "Set PG_PORT", "family": "deploy"}]}
            return {"picks": []}

        run(self.store, MockAdapter(router=route), [root], apply_changes=True)
        self.store.invalidate()
        stored = [n for n in self.store.nodes() if n.origin == "migrated"]
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0].family, "deploy")
        self.assertTrue(stored[0].gist, "a migrated lesson must be routable immediately")

    def test_the_source_skill_is_never_touched(self) -> None:
        """Whether to retire a skill is the user's call, made after seeing RMC
        recall the same knowledge."""
        from rmc.migrate import run

        root = self.skill("deploy", "---\nname: deploy\n---\nAlways set PG_PORT first.")
        path = root / "deploy" / "SKILL.md"
        original = path.read_text()

        def route(prompt, schema):
            if "RMC:migrate" in prompt:
                return {"verdict": "import", "lessons": [
                    {"body": "Set PG_PORT first.", "gist": "when deploying", "family": "deploy"}]}
            return {"picks": []}

        run(self.store, MockAdapter(router=route), [root], apply_changes=True)
        self.assertEqual(path.read_text(), original)

    def test_capture_machinery_is_reported_not_imported(self) -> None:
        """Importing the skills that captured skills would fill the store with
        instructions for maintaining the system being replaced."""
        from rmc.migrate import run

        root = self.skill("introspect", "---\nname: introspect\n---\nReflect and write a skill.")
        adapter = MockAdapter(router=lambda prompt, schema: {
            "verdict": "superseded", "reason": "RMC captures automatically"})
        outcomes = run(self.store, adapter, [root], apply_changes=True)
        self.store.invalidate()
        self.assertEqual(outcomes[0].verdict, "superseded")
        self.assertEqual([n for n in self.store.nodes() if n.origin == "migrated"], [])

    def test_a_model_failure_is_reported_not_swallowed(self) -> None:
        from rmc.migrate import run

        root = self.skill("deploy", "---\nname: deploy\n---\nSomething.")
        outcomes = run(self.store, MockAdapter(router=lambda p, s: None), [root])
        self.assertTrue(outcomes[0].error)


class TestSelfTuning(StoreCase):
    """RMC changing its own retrieval criteria, and being stopped when wrong.

    Every other stage is corrected by outcomes; the criteria that decide what
    gets recalled could only change when a person had an idea, and people are
    bad at this — of six hand-written proposals to the relevance prompt, five
    made retrieval worse. So what matters here is not that it proposes, but
    that it cannot keep a change that did not win, and cannot leave one behind.
    """

    def test_only_a_tunable_constant_may_be_moved(self) -> None:
        """A loop that can reach the correctness gates can pass its own exam."""
        from rmc.tune import _validate

        _, _, why = _validate(self.store, {"kind": "config", "key": "compaction.threshold", "value": 0.5})
        self.assertIn("not a tunable", why)

    def test_a_constant_outside_its_range_is_refused(self) -> None:
        from rmc.tune import _validate

        _, _, why = _validate(self.store, {"kind": "config", "key": "recall.max_families", "value": 500})
        self.assertIn("outside", why)

    def test_a_prompt_missing_its_placeholders_is_refused(self) -> None:
        """It would throw at format time and score as 'the judge could not
        answer' — a catastrophic result for reasons unrelated to the idea."""
        from rmc.tune import _validate

        _, _, why = _validate(self.store, {"kind": "prompt", "text": "Decide relevance. " * 40})
        self.assertIn("missing", why)

    def test_a_valid_prompt_is_accepted(self) -> None:
        from rmc.tune import _validate

        kind, target, why = _validate(
            self.store,
            {"kind": "prompt", "text": "Judge the work. " * 30 + "{question} {candidates}"},
        )
        self.assertEqual((kind, target, why), ("prompt", "relevance", ""))

    def test_a_reverted_config_change_leaves_nothing_behind(self) -> None:
        """Damage arriving labelled as an improvement is worse than no loop."""
        from rmc.tune import Sandbox

        before = self.store.config.get("recall.max_families")
        box = Sandbox(self.store)
        box.apply_config("recall.max_families", 7)
        self.assertEqual(self.store.config.get("recall.max_families"), 7)
        box.revert()
        self.assertEqual(self.store.config.get("recall.max_families"), before)

    def test_a_reverted_prompt_change_leaves_nothing_behind(self) -> None:
        from rmc.tune import Sandbox

        path = self.store.root / "prompts" / "relevance.md"
        box = Sandbox(self.store)
        box.apply_prompt("relevance", "experimental {question} {candidates}")
        self.assertTrue(path.exists())
        box.revert()
        self.assertFalse(path.exists(), "an experiment must not survive its own rejection")

    def test_an_override_is_actually_used(self) -> None:
        from rmc.judge import RELEVANCE, prompt

        (self.store.root / "prompts").mkdir(parents=True, exist_ok=True)
        (self.store.root / "prompts" / "relevance.md").write_text("mine {question} {candidates}")
        self.assertEqual(prompt(self.store, "relevance", RELEVANCE), "mine {question} {candidates}")

    def test_an_override_changes_the_criteria_fingerprint(self) -> None:
        """Otherwise every cached verdict answers with the text it replaced, and
        the experiment reports a confident null — the same failure that made a
        full prompt rewrite return a byte-identical eval report."""
        from rmc.judge import criteria_version

        before = criteria_version(self.store)
        (self.store.root / "prompts").mkdir(parents=True, exist_ok=True)
        (self.store.root / "prompts" / "relevance.md").write_text("different {question} {candidates}")
        self.assertNotEqual(criteria_version(self.store), before)

    def test_an_unreadable_override_falls_back_rather_than_failing(self) -> None:
        """Recall runs in a hook. An experiment must never take memory offline."""
        from rmc.judge import RELEVANCE, prompt

        (self.store.root / "prompts").mkdir(parents=True, exist_ok=True)
        (self.store.root / "prompts" / "relevance.md").write_text("   ")
        self.assertEqual(prompt(self.store, "relevance", RELEVANCE), RELEVANCE)

    def test_failed_attempts_are_remembered(self) -> None:
        """A loop that forgets its failures re-proposes them forever, and the
        failures are the more informative half."""
        from rmc.tune import Attempt, Ledger

        ledger = Ledger(self.store)
        ledger.add(Attempt(at="t", kind="config", target="recall.max_depth",
                           hypothesis="deeper finds more", kept=False,
                           before={"precision": 0.48, "recall": 1.0},
                           after={"precision": 0.4, "recall": 0.8},
                           verdict="reverted — dropped 3 lesson(s) that had helped"))
        self.assertIn("deeper finds more", ledger.history())
        self.assertIn("reverted", ledger.history())


class TestWarmRoutingPrefix(StoreCase):
    """The candidate list is identical between prompts; only the question moves.

    Re-sending it every time is affordable at 1,311 tokens and fatal at the
    ~225k the same layer would reach at 5,000 lessons. Providers cache an
    identical prefix, but only within one conversation, so the list is seeded
    once and each prompt branches a fork from it.
    """

    def router(self):
        from rmc.router import Router

        return Router(self.store)

    def seed(self, r, key, *, cached=58000, created=4000, now=0):
        r.session_for(key, now=now)
        r.record(cached_in=cached, created=created, prefix_tokens=created,
                 prefix_hash=key, seeded=True, now=now)

    def test_the_first_call_seeds_and_the_next_branches(self) -> None:
        r = self.router()
        first = r.session_for("hash-a", now=1000)
        self.assertFalse(first.resume, "nothing to branch from yet")
        r.record(cached_in=0, prefix_tokens=4000, prefix_hash="hash-a",
                 seeded=True, now=1000)
        second = r.session_for("hash-a", now=1010)
        self.assertTrue(second.resume)
        self.assertEqual(second.id, first.id, "same conversation, so the prefix matches")

    def test_each_chunk_keeps_its_own_conversation(self) -> None:
        """A wide apex layer is judged in chunks and each is a stable prefix.

        One shared session would be reseeded on every chunk and hit nothing,
        which is what the first version did.
        """
        r = self.router()
        self.seed(r, "chunk-1", now=0)
        self.seed(r, "chunk-2", now=1)
        a = r.session_for("chunk-1", now=2)
        b = r.session_for("chunk-2", now=3)
        self.assertTrue(a.resume and b.resume)
        self.assertNotEqual(a.id, b.id)

    def test_the_host_system_prompt_is_not_counted_as_our_hit(self) -> None:
        """The host sends a ~58k system prompt that is cached no matter what we
        do. Counting it reports 100% warm while delivering nothing."""
        r = self.router()
        self.seed(r, "hash-a", cached=58000, created=4000, now=0)
        self.assertFalse(r.record(cached_in=58004, prefix_tokens=4000,
                                  prefix_hash="hash-a", now=1))
        self.assertTrue(r.record(cached_in=62000, prefix_tokens=4000,
                                 prefix_hash="hash-a", now=2))

    def test_warmth_is_judged_per_conversation_not_globally(self) -> None:
        """A single global baseline drifts. Take the maximum and one unusually
        warm seed poisons it into reporting permanent misses — which happened:
        a stale 65,372 against a real 57,558 made every genuine hit negative."""
        r = self.router()
        self.seed(r, "big", cached=65000, created=20000, now=0)
        self.seed(r, "small", cached=57000, created=4000, now=1)
        self.assertTrue(
            r.record(cached_in=61000, prefix_tokens=4000, prefix_hash="small", now=2),
            "judged against its own seed, not against the other conversation's",
        )

    def test_a_changed_candidate_list_reseeds(self) -> None:
        """A different prefix cannot hit the cache, and branching from a stale
        one would answer using lessons that no longer exist."""
        r = self.router()
        first = r.session_for("hash-a", now=1000)
        r.record(cached_in=9000, prefix_tokens=4000, now=1000)
        second = r.session_for("hash-b", now=1005)
        self.assertFalse(second.resume)
        self.assertNotEqual(second.id, first.id)

    def test_a_hit_across_a_long_gap_widens_the_window(self) -> None:
        """Cache TTLs are not published per request, so the window is learned
        from what the provider actually served rather than written down."""
        r = self.router()
        r.session_for("hash-a", now=0)
        before = r.state.ttl_s
        r.record(cached_in=8000, prefix_tokens=4000, now=before * 0.95)
        self.assertGreater(r.state.ttl_s, before)

    def test_a_miss_inside_the_window_narrows_it(self) -> None:
        r = self.router()
        r.session_for("hash-a", now=0)
        r.record(cached_in=8000, prefix_tokens=4000, now=10)
        wide = r.state.ttl_s
        r.record(cached_in=0, prefix_tokens=4000, now=10 + wide * 0.5)
        self.assertLess(r.state.ttl_s, wide)

    def test_a_stale_prefix_is_reseeded_rather_than_branched(self) -> None:
        r = self.router()
        r.session_for("hash-a", now=0)
        r.record(cached_in=8000, prefix_tokens=4000, now=1)
        later = r.session_for("hash-a", now=1 + r.state.ttl_s + 1)
        self.assertFalse(later.resume)

    def test_a_trickle_of_cached_tokens_is_not_a_hit(self) -> None:
        """The system prompt is cached regardless. Counting that as success
        would report a warm prefix while the part worth caching missed every
        time."""
        r = self.router()
        r.session_for("hash-a", now=0)
        self.assertFalse(r.record(cached_in=200, prefix_tokens=4000, now=1))
        self.assertTrue(r.record(cached_in=3000, prefix_tokens=4000, now=2))

    def test_what_was_learned_survives_a_reseed(self) -> None:
        """The conversation expires; what we found out about how long
        conversations stay warm does not."""
        r = self.router()
        r.session_for("hash-a", now=0)
        r.record(cached_in=8000, prefix_tokens=4000, now=1)
        learned, hits = r.state.ttl_s, r.state.hits
        r.session_for("hash-b", now=2)
        self.assertEqual(r.state.ttl_s, learned)
        self.assertEqual(r.state.hits, hits)

    def test_a_corrupt_state_file_never_blocks_a_prompt(self) -> None:
        (self.store.root / "router.json").write_text("{not json", encoding="utf-8")
        self.assertTrue(self.router().session_for("hash-a", now=0).id)


class TestRecall(StoreCase):
    def setUp(self) -> None:
        super().setUp()
        # These fixtures hold two or three lessons, which is exactly the range
        # the brand-new-store bypass covers. Turn it off: what is under test is
        # the decision, not the shortcut around it.
        self.store.config.set("recall.filter_above", 0)

    def test_a_store_past_the_bypass_is_filtered(self) -> None:
        """Fitting the budget was never a reason to skip choosing.

        This used to serve everything unfiltered whenever the store fit, on the
        reasoning that judgement is only needed under scarcity. Measured on a
        real store that always fit: 15,917 of ~17,800 injected tokens went
        unused, and asking the judge about exactly those sets kept every lesson
        that had mattered while dropping 55% of the noise. Context that fits is
        not context that is free.
        """
        self.add_node(id="n_a", family="retry", body="Retry idempotent calls.")
        self.add_node(id="n_b", family="graphql", body="Batch queries.")
        log: list = []
        adapter = counting_router({"picks": [{"id": "n_a", "verdict": "relevant"}]}, log)
        pack = recall_pack(self.store, "retry the call", adapter)
        self.assertEqual(len(log), 1, "a small store still gets a decision")
        self.assertEqual(pack.served, ["n_a"], "the unrelated lesson is not free just because it fits")

    def test_a_brand_new_store_skips_the_routing_call(self) -> None:
        """The one case where serving blind beats deciding.

        Not because the noise is free — it is not — but because a routing call
        costs ~5s of CLI startup inside a hook that blocks the user's prompt,
        and three lessons served blind is a few lines. The gate is sized so that
        what it waves through stays small.
        """
        self.store.config.set("recall.filter_above", 3)
        self.add_node(id="n_a", family="retry", body="Retry idempotent calls.")
        self.add_node(id="n_b", family="graphql", body="Batch queries.")
        log: list = []
        pack = recall_pack(self.store, "anything", counting_router({"picks": []}, log))
        self.assertEqual(sorted(pack.served), ["n_a", "n_b"])
        self.assertEqual(log, [], "no model call while the store is this small")

    def test_the_bypass_closes_as_soon_as_the_store_grows(self) -> None:
        self.store.config.set("recall.filter_above", 3)
        for i in range(4):
            self.add_node(id=f"n_{i}", family=f"f{i}", body=f"Lesson {i}.")
        log: list = []
        adapter = counting_router({"picks": [{"id": "n_0", "verdict": "relevant"}]}, log)
        pack = recall_pack(self.store, "the first thing", adapter)
        self.assertEqual(len(log), 1)
        self.assertEqual(pack.served, ["n_0"])

    def test_filtering_switches_on_once_the_store_outgrows_the_budget(self) -> None:
        self.store.config.set("recall.max_pack_tokens", 20)
        self.add_node(id="n_a", family="retry", body="Retry idempotent calls. " * 20)
        self.add_node(id="n_b", family="graphql", body="Batch queries. " * 20)
        log: list = []
        adapter = counting_router({"picks": [{"id": "n_a", "verdict": "relevant"}]}, log)
        pack = recall_pack(self.store, "retry the call", adapter)
        self.assertEqual(len(log), 1, "scarcity is what makes the judgement necessary")
        self.assertEqual(pack.served, ["n_a"])

    def test_serves_what_the_model_selects(self) -> None:
        self.add_node(id="n_r", family="retry", title="Retry", body="Retry idempotent calls.", level=2)
        self.add_node(id="n_g", family="graphql", title="GraphQL", body="Batch queries.", level=1)
        adapter = router({"picks": [{"id": "n_r", "verdict": "relevant", "why": "same subject"}]})

        pack = recall_pack(self.store, "the client needs retries", adapter)
        self.assertIn("Retry idempotent calls", pack.text)
        self.assertNotIn("Batch queries", pack.text)
        self.assertEqual(pack.served, ["n_r"])
        self.assertEqual(pack.reasons["n_r"], "same subject")

    def test_nothing_selected_means_nothing_injected(self) -> None:
        self.add_node(id="n_r", family="retry", body="Retry idempotent calls.")
        pack = recall_pack(self.store, "what colour should the logo be", router({"picks": []}))
        self.assertFalse(pack)

    def test_sibling_lessons_are_all_reachable(self) -> None:
        """Consolidation creates siblings on purpose; none may be orphaned.

        Taking only the best node per family silently stranded the rest — they
        stayed stored, counted in `rmc status`, and were never served again.
        """
        self.add_node(id="n_a", family="deploy", body="Use the argo plugin.")
        self.add_node(id="n_b", family="deploy", body="Staging deploys need approval.")
        self.add_node(id="n_c", family="tests", body="Set PG_PORT first.")

        reachable = {n.id for n in self.store.apexes()}
        self.assertEqual(reachable, {"n_a", "n_b", "n_c"})

        keeps = router({"picks": [
            {"id": "n_a", "verdict": "relevant"},
            {"id": "n_b", "verdict": "relevant"},
            {"id": "n_c", "verdict": "relevant"},
        ]})
        pack = recall_pack(self.store, "deploy staging", keeps)
        self.assertEqual(sorted(pack.served), ["n_a", "n_b", "n_c"])

    def test_compressed_nodes_outrank_their_sources(self) -> None:
        """An apex list must lead with the cheapest useful summary."""
        base = self.add_node(id="n_v", family="f", body="verbose original", level=0)
        apex = self.add_node(id="n_s", family="f", body="short", level=1, derived_from=[base.id])
        base.parents = [apex.id]
        self.store.save_node(base)
        self.store.invalidate()
        # The source is no longer an apex, so only the compression is served.
        self.assertEqual([n.id for n in self.store.apexes()], ["n_s"])

    def test_empty_store_asks_nothing(self) -> None:
        log: list = []
        recall_pack(self.store, "anything", counting_router({"picks": []}, log))
        self.assertEqual(log, [], "no lessons means no question to ask")

    def test_conflict_is_surfaced_with_the_lesson(self) -> None:
        self.add_node(
            id="n_c",
            family="db",
            body="Use port 5433.",
            status="disputed",
            conflict="Is 5434 permanent?",
        )
        pack = recall_pack(
            self.store,
            "run the tests",
            router({"picks": [{"id": "n_c", "verdict": "relevant"}]}),
        )
        self.assertIn("Unresolved", pack.text)
        self.assertEqual(pack.conflicts, ["n_c"])

    def test_previously_rescued_claims_are_reattached(self) -> None:
        self.add_node(id="n_p", family="f", body="Short.", dropped=[Delta("the missing bit", "parameter")])
        self.store.log("rescue", node="n_p", claim="the missing bit")
        pack = recall_pack(
            self.store, "q", router({"picks": [{"id": "n_p", "verdict": "relevant"}]})
        )
        self.assertIn("the missing bit", pack.text)


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #


class TestSelection(StoreCase):
    def make_apex(self) -> Node:
        return self.add_node(
            id="n_apex",
            family="retry",
            body="Retry idempotent operations.",
            level=1,
            dropped=[
                Delta("prefer table-driven tests for the parser", "example", None),
                Delta("S3 returns 200 with an error body", "edge-case", None),
                Delta("the deploy pipeline caches node_modules", "reference", None),
            ],
        )

    def test_the_model_decides_which_repair_applies(self) -> None:
        apex = self.make_apex()
        candidates = build_candidates(apex, resolve=self.store.get, strategy="delta-patch")
        target = next(c.label for c in candidates if "S3" in c.text)
        adapter = router({"ranked": [{"key": target, "usefulness": 1.0}]})
        ranked = rank(
            candidates,
            diag=Diagnosis(missing=["the upload silently succeeded"]),
            judge=Judge(self.store, adapter),
            config=self.store.config,
        )
        self.assertEqual(ranked[0].label, target)
        self.assertGreater(ranked[0].parts["judge"], 0)

    def test_without_a_judge_it_falls_back_to_evidence_not_a_similarity_score(self) -> None:
        apex = self.make_apex()
        ranked = rank(
            build_candidates(apex, resolve=self.store.get, strategy="delta-patch"),
            diag=Diagnosis(missing=["s3 error body"]),
            judge=None,
            config=self.store.config,
        )
        # No judgement term at all — not a guess dressed up as one.
        self.assertEqual(ranked[0].parts["judge"], 0.0)
        self.assertTrue(all(c.parts["judge"] == 0.0 for c in ranked))
        # Cheapest wins on the remaining terms.
        self.assertEqual(ranked[0].tokens, min(c.tokens for c in ranked))

    def test_children_are_offered_when_the_manifest_is_empty(self) -> None:
        self.add_node(id="n_c", family="f", body="detail", level=0)
        apex = self.add_node(id="n_p", family="f", body="abstract", level=1, derived_from=["n_c"])
        cands = build_candidates(apex, resolve=self.store.get, strategy="delta-patch")
        self.assertTrue(any(c.kind == "node" and c.node.id == "n_c" for c in cands))


# --------------------------------------------------------------------------- #
# control flow: compress, fail, descend, rescue
# --------------------------------------------------------------------------- #


class TestCompaction(StoreCase):
    def build_family(self) -> Node:
        body = (
            "When calling flaky remote services, follow these rules carefully.\n\n"
            "- Retry only idempotent operations; a non-idempotent write needs a "
            "dedupe key established before the first attempt. @idempotent\n\n"
            "- Use jittered exponential backoff rather than a fixed delay, so that "
            "retries from many clients do not synchronise. @backoff\n\n"
            "- S3 is a special case: it can return HTTP 200 with an error document "
            "in the response body, so you must parse the body rather than trusting "
            "the status code, and treat a parsed error exactly as you would treat a "
            "5xx response for the purposes of retrying. @s3-body"
        )
        node = self.add_node(id="n_base", family="retry", title="Retry", body=body, level=0)
        self.add_episode("e1", "retry", "retry the http call", served=["n_base"])
        self.add_episode("e2", "retry", "add backoff to the client", served=["n_base"])
        node.covers_tasks = ["e1", "e2"]
        self.store.save_node(node)
        self.store.invalidate()
        return self.store.get("n_base")

    def world(self) -> MockWorld:
        return MockWorld({"e1": {"idempotent"}, "e2": {"idempotent", "backoff"}})

    def test_accepted_when_the_regression_set_still_passes(self) -> None:
        node = self.build_family()
        result = compress_node(self.store, MockAdapter(world=self.world()), node)
        self.assertTrue(result.accepted, result.reason)
        self.assertTrue(any("@s3-body" in d.claim for d in result.dropped))
        self.assertEqual(result.pass_rate, 1.0)
        self.assertEqual(self.store.apex("retry").id, result.new_node.id)

    def test_rejected_when_it_drops_a_needed_fact(self) -> None:
        node = self.build_family()
        world = MockWorld({"e1": {"idempotent", "s3-body"}, "e2": {"idempotent", "s3-body"}})
        result = compress_node(self.store, MockAdapter(world=world), node)
        self.assertFalse(result.accepted)
        self.assertEqual(result.pass_rate, 0.0)
        self.assertEqual(self.store.get("n_base").parents, [])
        self.assertTrue(self.store.get("n_base").preserve)

    def test_a_modest_saving_survives_if_it_generalises(self) -> None:
        """Worth has two axes. A ratio only sees one, and would refuse a better
        abstraction for saving 22% instead of 25%."""
        node = self.build_family()

        def route(prompt, schema):
            if "RMC:worth" in prompt:
                return {"keep": True, "generality": "more",
                        "why": "states the rule at a level covering all three cases"}
            if "RMC:compress" in prompt:
                # Barely under target, but broader.
                return {"body": "Retry idempotent remote calls with jittered backoff; "
                                "parse response bodies, not status codes. @idempotent @backoff",
                        "dropped": [], "lossless": True}
            return {"pass": True, "reason": "ok"}

        result = compress_node(self.store, MockAdapter(router=route), node)
        self.assertTrue(result.accepted, result.reason)
        self.assertEqual(result.generality, "more")

    def test_a_reworded_copy_is_refused_by_the_judge_not_the_ratio(self) -> None:
        node = self.build_family()

        def route(prompt, schema):
            if "RMC:worth" in prompt:
                return {"keep": False, "generality": "same",
                        "why": "reworded, saves little and covers no new case"}
            if "RMC:compress" in prompt:
                return {"body": "Retry things carefully. @idempotent", "dropped": [],
                        "lossless": True}
            return {"pass": True, "reason": "ok"}

        result = compress_node(self.store, MockAdapter(router=route), node)
        self.assertFalse(result.accepted)
        self.assertIn("not worth keeping", result.reason)

    def test_a_poor_ratio_warns_rather_than_blocking(self) -> None:
        node = self.build_family()
        # Only marginally shorter than the original, which is the case that used
        # to be fatal and should now merely be noted.
        long_body = " ".join(node.body.split()[: int(len(node.body.split()) * 0.85)])

        def route(prompt, schema):
            if "RMC:worth" in prompt:
                return {"keep": True, "generality": "more", "why": "broader"}
            if "RMC:compress" in prompt:
                return {"body": long_body, "dropped": [], "lossless": True}
            return {"pass": True, "reason": "ok"}

        result = compress_node(self.store, MockAdapter(router=route), node)
        self.assertTrue(result.accepted, result.reason)
        self.assertTrue(result.warnings, "a weak reduction must still be visible")
        self.assertIn("below target", result.warnings[0])

    def test_manifest_under_reporting_is_rejected(self) -> None:
        node = self.build_family()
        adapter = MockAdapter(
            router=lambda prompt, schema: (
                {"body": "Retry things.", "dropped": []}
                if "RMC:compress" in prompt
                else {"pass": True, "reason": "ok"}
            )
        )
        result = compress_node(self.store, adapter, node)
        self.assertFalse(result.accepted)
        self.assertIn("under-reported", result.reason)

    def test_refuses_to_compress_without_a_regression_set(self) -> None:
        node = self.add_node(id="n_lonely", family="solo", body="A lesson. @x", level=0)
        result = compress_node(self.store, MockAdapter(world=MockWorld()), node)
        self.assertFalse(result.accepted)
        self.assertIn("refusing to compress blind", result.reason)

    def test_due_requires_successes_and_episodes(self) -> None:
        node = self.build_family()
        self.assertEqual(due_nodes(self.store), [])
        node.stats.attempts, node.stats.successes = 3, 3
        self.store.save_node(node)
        self.store.invalidate()
        self.assertEqual([n.id for n in due_nodes(self.store)], ["n_base"])


class TestDescent(StoreCase):
    def test_the_dropped_fact_is_found_past_distractors(self) -> None:
        base = self.add_node(
            id="n_d0",
            family="retry",
            body="Retry idempotent ops. @idempotent\nS3 returns 200 with error bodies. @s3-body",
            level=0,
        )
        apex = self.add_node(
            id="n_d1",
            family="retry",
            body="Retry idempotent ops. @idempotent",
            level=1,
            derived_from=[base.id],
            dropped=[
                Delta("prefer table-driven tests for the parser", "example", base.id),
                Delta("the deploy pipeline caches node_modules", "reference", base.id),
                Delta("S3 returns 200 with error bodies. @s3-body", "edge-case", base.id),
            ],
        )
        base.parents = [apex.id]
        self.store.save_node(base)
        self.store.invalidate()

        world = MockWorld({"t_s3": {"idempotent", "s3-body"}})
        adapter = MockAdapter(world=world)

        def verify(run, pack):
            ok, missing = world.solves("t_s3", pack)
            return ok, "missing: " + " ".join(f"@{m}" for m in sorted(missing))

        result = solve_with_descent(
            self.store,
            adapter=adapter,
            task_id="t_s3",
            task="handle the s3 upload response",
            family="retry",
            verify=verify,
        )
        self.assertTrue(result.ok)
        self.assertIsNotNone(result.rescued_by)
        self.assertIn("@s3-body", result.final_pack)
        # Right on the first descent, past two distractors, and because the
        # judgement term chose it — not because it was the only option left.
        self.assertEqual(len(result.attempts), 2)
        self.assertFalse(result.attempts[0].ok)
        self.assertGreater(result.rescued_by.parts["judge"], 0.0)

    def test_escalates_to_level_zero_when_no_delta_helps(self) -> None:
        base = self.add_node(id="n_e0", family="f", body="Full lesson. @a @b", level=0)
        apex = self.add_node(
            id="n_e1", family="f", body="Short lesson. @a", level=1, derived_from=[base.id], dropped=[]
        )
        base.parents = [apex.id]
        self.store.save_node(base)
        self.store.invalidate()

        world = MockWorld({"t": {"a", "b"}})

        def verify(run, pack):
            ok, missing = world.solves("t", pack)
            return ok, "missing: " + " ".join(f"@{m}" for m in sorted(missing))

        result = solve_with_descent(
            self.store,
            adapter=MockAdapter(world=world),
            task_id="t",
            task="do the thing",
            family="f",
            verify=verify,
        )
        self.assertTrue(result.ok)
        self.assertIn("@b", result.final_pack)


class TestMultipleParents(StoreCase):
    """A leaf can be abstracted in more than one direction at once.

    Compressing a lesson into a terser form of itself, and merging it sideways
    with another lesson into a shared generalisation, are different
    abstractions over the same leaf. Both are worth keeping — and while the
    parent link was a single field, the second silently destroyed the first.
    """

    def world(self):
        return MockWorld({"e1": {"a"}, "e2": {"a"}})

    def build(self):
        leaf = self.add_node(
            id="n_leaf",
            family="f",
            level=0,
            body=(
                "A long original lesson with several parts. @a\n\n"
                "- A first supporting point that carries real detail and takes room. @b\n\n"
                "- A second supporting point, also long, also detailed, also here. @c"
            ),
        )
        self.add_episode("e1", "f", "do the thing", served=["n_leaf"])
        self.add_episode("e2", "f", "do the other thing", served=["n_leaf"])
        leaf.covers_tasks = ["e1", "e2"]
        self.store.save_node(leaf)
        self.store.invalidate()
        return self.store.get("n_leaf")

    def test_a_merge_after_a_compression_keeps_both_parents(self) -> None:
        from rmc.compact import compress_node, merge_nodes

        leaf = self.build()
        adapter = MockAdapter(world=self.world())

        compressed = compress_node(self.store, adapter, leaf)
        self.assertTrue(compressed.accepted, compressed.reason)
        first_parent = compressed.new_node.id
        self.assertEqual(self.store.get("n_leaf").parents, [first_parent])

        # Now merge the leaf sideways with an unrelated sibling. The pair is
        # deliberately lopsided and has nothing to compress, so the size gate is
        # lifted — what is under test here is parentage, not economy.
        self.store.config.set("compaction.merge_ratio", 2.0)
        other = self.add_node(id="n_side", family="f", level=0, body="Another lesson. @a")
        merged = merge_nodes(self.store, adapter, [self.store.get("n_leaf"), other])
        self.assertTrue(merged.accepted, merged.reason)

        parents = self.store.get("n_leaf").parents
        self.assertIn(first_parent, parents, "the compression must survive the merge")
        self.assertIn(merged.new_node.id, parents)
        self.assertEqual(len(parents), 2)

    def test_ancestors_walks_every_line_upward(self) -> None:
        a = self.add_node(id="n_a", family="f", body="a")
        p1 = self.add_node(id="n_p1", family="f", body="p1", level=1, derived_from=["n_a"])
        p2 = self.add_node(id="n_p2", family="f", body="p2", level=1, derived_from=["n_a"])
        top = self.add_node(id="n_top", family="f", body="top", level=2, derived_from=["n_p1"])
        a.parents = [p1.id, p2.id]
        p1.parents = [top.id]
        self.store.save_node(a)
        self.store.save_node(p1)
        self.store.invalidate()

        found = {n.id for n in self.store.ancestors(self.store.get("n_a"))}
        self.assertEqual(found, {"n_p1", "n_p2", "n_top"})

    def test_a_node_with_any_parent_is_not_an_apex(self) -> None:
        child = self.add_node(id="n_c", family="f", body="c")
        parent = self.add_node(id="n_p", family="f", body="p", level=1, derived_from=["n_c"])
        child.parents = [parent.id]
        self.store.save_node(child)
        self.store.invalidate()
        self.assertEqual([n.id for n in self.store.apexes()], ["n_p"])

    def test_a_merge_that_would_form_a_cycle_is_refused(self) -> None:
        """Nothing else prevents a merge swallowing its own ancestor, and an
        upward walk over a cyclic graph never terminates."""
        from rmc.compact import merge_nodes

        child = self.add_node(id="n_ch", family="f", body="child. @a")
        parent = self.add_node(id="n_pa", family="f", body="parent. @a", level=1, derived_from=["n_ch"])
        child.parents = [parent.id]
        self.store.save_node(child)
        self.store.invalidate()

        result = merge_nodes(
            self.store, MockAdapter(world=self.world()),
            [self.store.get("n_ch"), self.store.get("n_pa")],
        )
        self.assertFalse(result.accepted)
        self.assertIn("cycle", result.reason)

    def test_legacy_stores_still_load(self) -> None:
        """`compressed_into` is the pre-DAG spelling and is still on disk."""
        path = self.store.nodes_dir / "f"
        path.mkdir(parents=True, exist_ok=True)
        (path / "n_old.md").write_text(
            "---\nid: n_old\nfamily: f\ncompressed_into: n_parent\n---\n\nbody\n"
        )
        self.store.invalidate()
        self.assertEqual(self.store.get("n_old").parents, ["n_parent"])
        self.assertFalse(self.store.get("n_old").is_apex)


class TestRepair(StoreCase):
    def test_repeated_rescues_fold_back_into_the_body(self) -> None:
        node = self.add_node(
            id="n_rep",
            family="f",
            body="Short lesson.",
            level=1,
            dropped=[Delta("the missing constant is 1.6s", "parameter", None)],
        )
        for _ in range(2):
            self.store.log("rescue", node=node.id, claim="the missing constant is 1.6s")
        restored = repair(self.store, node, min_rescues=2)
        self.assertEqual(restored, ["the missing constant is 1.6s"])
        reloaded = self.store.get("n_rep")
        self.assertIn("1.6s", reloaded.body)
        self.assertEqual(reloaded.dropped, [])


# --------------------------------------------------------------------------- #
# observe: plumbing a verdict into the tree
# --------------------------------------------------------------------------- #


class TestObserve(StoreCase):
    def facts(self, **kw) -> SessionFacts:
        base = dict(
            user_messages=["do the thing"],
            assistant_messages=["done"],
            tool_calls=14,
            first_prompt="do the thing",
            last_assistant="done",
        )
        base.update(kw)
        return SessionFacts(**base)

    def verdict(self, **kw):
        payload = {"outcome": "success", "confidence": 0.9, "corrected": False}
        payload.update(kw)
        return router(payload)

    def test_success_updates_stats_and_files_an_episode(self) -> None:
        node = self.add_node(id="n_o", family="retry", body="Retry stuff.", level=1)
        result = observe(self.store, self.facts(), adapter=self.verdict(), served=[node.id])
        self.assertEqual(result.outcome.label, "success")
        self.assertEqual(self.store.get("n_o").stats.successes, 1)
        self.assertEqual(result.episode.outcome, "success")

    def test_a_corrected_session_counts_against_the_lesson(self) -> None:
        """Success for the episode, failure for the lesson that should have prevented it."""
        node = self.add_node(
            id="n_o4",
            family="deploy",
            body="Deploy with kubectl apply.",
            level=1,
            dropped=[Delta("use the argo rollouts plugin", "procedure-step", None)],
        )
        adapter = self.verdict(corrected=True, correction="use the argo rollouts plugin, not kubectl")
        result = observe(self.store, self.facts(), adapter=adapter, served=[node.id])

        self.assertEqual(result.outcome.label, "success")
        reloaded = self.store.get("n_o4")
        self.assertEqual(reloaded.stats.failures, 1)
        self.assertEqual(reloaded.stats.successes, 0)
        self.assertEqual(result.episode.outcome, "success")

    def test_only_lessons_that_were_used_get_credit(self) -> None:
        """An irrelevant lesson that happened to be injected must not accrue a
        record of usefulness — it would eventually earn a compression it never
        deserved."""
        helpful = self.add_node(id="n_used", family="f", body="the one that mattered")
        noise = self.add_node(id="n_noise", family="f", body="shown, irrelevant")
        adapter = self.verdict(
            lessons_used=[
                {"id": "n_used", "used": True, "how": "named the constraint"},
                {"id": "n_noise", "used": False},
            ]
        )
        result = observe(
            self.store, self.facts(), adapter=adapter, served=[helpful.id, noise.id]
        )
        self.assertEqual(self.store.get("n_used").stats.successes, 1)
        self.assertEqual(self.store.get("n_noise").stats.attempts, 0)
        self.assertEqual(result.episode.used, ["n_used"])

    def test_an_unused_lesson_is_not_scored_as_a_failure_either(self) -> None:
        """It was not wrong, it was irrelevant. That is a retrieval miss."""
        noise = self.add_node(id="n_n2", family="f", body="irrelevant")
        adapter = self.verdict(
            corrected=True, lessons_used=[{"id": "n_n2", "used": False}]
        )
        observe(self.store, self.facts(), adapter=adapter, served=[noise.id])
        node = self.store.get("n_n2")
        self.assertEqual(node.stats.failures, 0)
        self.assertEqual(node.stats.attempts, 0)

    def test_an_in_session_verdict_beats_the_digest_verdict(self) -> None:
        """The reflector with real context outranks the one reading a digest.

        Influence on *reasoning* is invisible in a digest of commands, so a
        digest-based judge under-credits principles. When something that held
        the actual conversation has already answered, use its answer.
        """
        a = self.add_node(id="n_a", family="f", body="principle")
        b = self.add_node(id="n_b", family="f", body="other")
        # The digest-based judge says only n_b helped...
        adapter = self.verdict(
            lessons_used=[{"id": "n_a", "used": False}, {"id": "n_b", "used": True}]
        )
        # ...but the in-session reflector saw n_a shape the approach.
        result = observe(
            self.store,
            self.facts(),
            adapter=adapter,
            attributed={"n_a": True, "n_b": False},
            served=[a.id, b.id],
        )
        self.assertEqual(self.store.get("n_a").stats.successes, 1)
        self.assertEqual(self.store.get("n_b").stats.attempts, 0)
        self.assertEqual(result.episode.used, ["n_a"])

    def test_the_fork_is_asked_to_attribute_what_it_was_served(self) -> None:
        from rmc.hooks import ATTRIBUTION, FORK_PROMPT

        prompt = FORK_PROMPT.format(
            attribution=ATTRIBUTION.format(session="s1", served="  [n_x] Retry — retry idempotently")
        )
        self.assertIn("rmc used --session s1", prompt)
        self.assertIn("n_x", prompt)
        # The prompt is hard-wrapped, so compare on collapsed whitespace.
        self.assertIn("Being on-topic is not being used", " ".join(prompt.split()))

    def test_low_confidence_and_no_correction_changes_nothing(self) -> None:
        node = self.add_node(id="n_o3", family="f", body="x", level=0)
        adapter = self.verdict(outcome="unknown", confidence=0.1)
        observe(self.store, self.facts(), adapter=adapter, served=[node.id])
        self.assertEqual(self.store.get("n_o3").stats.attempts, 0)

    def test_a_correction_is_acted_on_even_at_low_confidence(self) -> None:
        """Corrected-then-fixed sessions score near zero; they must not be dropped."""
        node = self.add_node(id="n_o5", family="f", body="x", level=0)
        adapter = self.verdict(outcome="unknown", confidence=0.2, corrected=True, correction="wrong tool")
        observe(self.store, self.facts(), adapter=adapter, served=[node.id])
        self.assertEqual(self.store.get("n_o5").stats.failures, 1)

    def test_tiny_session_is_skipped_without_asking(self) -> None:
        log: list = []
        adapter = counting_router({"outcome": "success", "confidence": 1.0, "corrected": False}, log)
        result = observe(self.store, SessionFacts(user_messages=["hi"], tool_calls=1), adapter=adapter)
        self.assertIn("too small", result.skipped)
        self.assertEqual(log, [])


# --------------------------------------------------------------------------- #
# placement / consolidation
# --------------------------------------------------------------------------- #


class TestPlacement(StoreCase):
    BODY = "Retry idempotent HTTP calls with jittered exponential backoff."

    def seed(self) -> Node:
        return self.add_node(id="n_seed", family="retry", title="Retry", body=self.BODY, level=0)

    def reconciler(self, relation: str, match: str = "n_seed", related=True, **extra):
        """Answers the walk's `related` question, then the reconcile question."""

        def route(prompt, schema):
            if "RMC:related" in prompt:
                verdict = "relevant" if related else "unrelated"
                return {"picks": [{"id": "n_seed", "verdict": verdict}]}
            return {"match": match, "relation": relation, "rationale": f"mock says {relation}", **extra}

        return MockAdapter(router=route)

    def test_unrelated_lesson_starts_a_new_leaf(self) -> None:
        from rmc.placement import decide

        self.seed()
        decision = decide(
            self.store,
            self.reconciler("orthogonal", related=False),
            body="Figma exports need the viewBox stripped.",
            family_hint="svg-assets",
        )
        self.assertEqual(decision.action, "new-family")

    def test_empty_store_needs_no_judgement(self) -> None:
        from rmc.placement import decide

        log: list = []
        decision = decide(
            self.store, counting_router({"picks": []}, log), body="anything", family_hint="new"
        )
        self.assertEqual(decision.action, "new-family")
        self.assertEqual(log, [])

    def test_refinement_folds_into_the_base_node(self) -> None:
        from rmc.placement import apply, decide

        seed = self.seed()
        merged = self.BODY + " Cap total elapsed time by the caller's deadline."
        decision = decide(
            self.store,
            self.reconciler("refines", merged_body=merged),
            body="Retries must be capped by the caller's deadline.",
            family_hint="retry",
        )
        self.assertEqual(decision.action, "fold-into")
        result = apply(self.store, decision, Node(id="n_new", family="retry", body="ignored"))
        self.assertIn("deadline", self.store.get(seed.id).body)
        self.assertIsNone(self.store.get("n_new"))

    def test_refinement_patches_ancestors_so_the_apex_is_not_left_stale(self) -> None:
        from rmc.placement import apply, decide

        base = self.seed()
        apex = self.add_node(
            id="n_apex2", family="retry", body="Retry idempotently.", level=1, derived_from=[base.id]
        )
        base.parents = [apex.id]
        self.store.save_node(base)
        self.store.invalidate()

        def route(prompt, schema):
            if "RMC:related" in prompt:
                return {"picks": [{"id": "n_apex2", "verdict": "relevant"}]}
            return {"match": "n_apex2", "relation": "refines", "rationale": "adds a deadline cap",
                    "merged_body": self.BODY + " Cap by deadline."}

        decision = decide(self.store, MockAdapter(router=route), body="Cap by deadline.", family_hint="retry")
        result = apply(self.store, decision, Node(id="n_x", family="retry", body="b"))
        self.assertIn(apex.id, result.patched)

    def test_a_refinement_never_overwrites_an_unrelated_sibling(self) -> None:
        """Folding must follow the matched lesson's own lineage, not pick the
        family's best-scoring node. Getting this wrong destroys the victim's
        body while leaving its title and id intact — two nodes, one text."""
        from rmc.placement import apply, decide

        target = self.seed()  # n_seed, family "retry"
        bystander = self.add_node(
            id="n_other", family="retry", body="An unrelated lesson that must survive."
        )
        bystander.stats.attempts, bystander.stats.successes = 9, 9  # best posterior
        self.store.save_node(bystander)
        self.store.invalidate()

        decision = decide(
            self.store,
            self.reconciler("refines", merged_body=self.BODY + " Cap by deadline."),
            body="Cap retries by the caller's deadline.",
            family_hint="retry",
        )
        apply(self.store, decision, Node(id="n_new", family="retry", body="ignored"))

        self.assertIn("deadline", self.store.get(target.id).body)
        self.assertEqual(
            self.store.get("n_other").body, "An unrelated lesson that must survive."
        )

    def test_contradiction_disputes_both_and_asks_a_question(self) -> None:
        from rmc.placement import apply, decide, open_conflicts

        seed = self.seed()
        decision = decide(
            self.store,
            self.reconciler("contradicts", question="Fixed delay or jittered backoff?"),
            body="Always retry with a fixed 1s delay.",
            family_hint="retry",
        )
        self.assertEqual(decision.action, "conflict")
        apply(self.store, decision, Node(id="n_conflict", family="retry", body="Fixed 1s delay.", level=0))
        self.assertEqual(self.store.get(seed.id).status, "disputed")
        self.assertEqual(self.store.get("n_conflict").status, "disputed")
        self.assertEqual({n.id for n in open_conflicts(self.store)}, {seed.id, "n_conflict"})

    def test_duplicate_writes_nothing(self) -> None:
        from rmc.placement import apply, decide

        self.seed()
        decision = decide(self.store, self.reconciler("duplicate"), body="Retry idempotently.", family_hint="retry")
        self.assertEqual(decision.action, "duplicate")
        result = apply(self.store, decision, Node(id="n_dup", family="retry", body="x"))
        self.assertIsNone(result.node)
        self.assertIsNone(self.store.get("n_dup"))

    def test_reconciler_failure_degrades_to_attaching_alongside(self) -> None:
        from rmc.placement import decide

        self.seed()

        class Broken:
            ok = False
            data = None
            text = ""
            error = "boom"

        def route(prompt, schema):
            if "RMC:related" in prompt:
                return {"picks": [{"id": "n_seed", "verdict": "relevant"}]}
            return Broken()

        decision = decide(self.store, MockAdapter(router=route), body="Retry, and log attempts.", family_hint="retry")
        self.assertEqual(decision.action, "attach-sibling")

    def test_resolving_clears_the_conflict(self) -> None:
        from rmc.placement import resolve

        self.add_node(id="n_r1", family="retry", body=self.BODY, status="disputed", conflict="which one?")
        resolve(self.store, "n_r1", keep=True)
        node = self.store.get("n_r1")
        self.assertEqual(node.status, "active")
        self.assertEqual(node.conflict, "")

    def test_all_candidates_reconciled_in_one_call(self) -> None:
        from rmc.placement import decide

        for i in range(3):
            self.add_node(id=f"n_c{i}", family=f"retry{i}", title="Retry", body="Retry calls.", level=0)
        reconcile_calls: list = []

        def route(prompt, schema):
            if "RMC:related" in prompt:
                return {"picks": [{"id": f"n_c{i}", "verdict": "relevant"} for i in range(3)]}
            reconcile_calls.append(prompt)
            return {"match": "n_c1", "relation": "specialises", "rationale": "distinct case"}

        decision = decide(self.store, MockAdapter(router=route), body="Retry websockets slower.", family_hint="retry0")
        self.assertEqual(len(reconcile_calls), 1)
        self.assertEqual(decision.target.id, "n_c1")


class TestLayering(unittest.TestCase):
    """A project store reads through to a global one; writes stay local."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.global_store = Store.init(base / "home")
        self.project = Store(Store.init(base / "repo").root, parent=self.global_store)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_global_lessons_are_visible_from_a_project(self) -> None:
        self.global_store.save_node(Node(id="n_g", family="style", body="Prefer judgement."))
        self.global_store.invalidate()
        self.project.invalidate()
        self.assertIsNotNone(self.project.get("n_g"))
        self.assertIn("style", self.project.families())

    def test_new_lessons_are_written_locally(self) -> None:
        self.global_store.save_node(Node(id="n_g2", family="style", body="global"))
        self.project.invalidate()
        path = self.project.save_node(Node(id="n_l", family="repo", body="local"))
        self.assertIn(self.project.root.name, str(path))
        self.assertIsNone(self.global_store.get("n_l"))

    def test_editing_a_global_lesson_writes_back_to_it(self) -> None:
        """Otherwise a repo silently forks a cross-project lesson and it drifts."""
        self.global_store.save_node(Node(id="n_g3", family="style", body="original"))
        self.project.invalidate()
        node = self.project.get("n_g3")
        node.body = "revised"
        self.project.save_node(node)
        self.global_store.invalidate()
        self.assertEqual(self.global_store.get("n_g3").body, "revised")
        self.assertFalse((self.project.root / "nodes" / "style").exists())

    def test_a_local_node_shadows_a_global_one_with_the_same_id(self) -> None:
        self.global_store.save_node(Node(id="n_same", family="style", body="global version"))
        local = Node(id="n_same", family="style", body="local version")
        local.path = self.project.nodes_dir / "style" / "n_same.md"
        self.project.save_node(local)
        self.project.invalidate()
        self.assertEqual(self.project.get("n_same").body, "local version")


class TestReflectionTrigger(StoreCase):
    """The harness schedules the *look*; the agent decides what it sees.

    The occasion must not be "something failed". Conceptual mistakes — believing
    a system works one way when it does not — produce no error message at all,
    and they are the expensive ones. A failure-gated trigger would sit silent
    through exactly the lessons worth having.
    """

    def transcript(self, results: list[bool]) -> Path:
        import json

        rows = [{"type": "user", "message": {"role": "user", "content": "do the thing"}}]
        for i, ok in enumerate(results):
            rows.append(
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": {"command": f"cmd{i}"}}
                        ],
                    },
                }
            )
            rows.append(
                {
                    "type": "user",
                    "toolUseResult": {"is_error": not ok},
                    "message": {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "out"}],
                    },
                }
            )
        path = Path(self.tmp.name) / "t.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows))
        return path

    def use_blocking_mode(self) -> None:
        """Opt into interrupting the agent; the default reflects off-thread."""
        self.store.config.set("learning.nudge_mode", "block")
        self.store.config.save(self.store.root / "config.yaml")

    def fire(self, path: Path, session: str = "s", **extra):
        import io
        import json
        from contextlib import redirect_stdout

        from rmc.hooks import on_turn_end

        payload = {"session_id": session, "cwd": str(self.base), "transcript_path": str(path), **extra}
        buf = io.StringIO()
        with redirect_stdout(buf):
            on_turn_end(payload)
        out = buf.getvalue().strip()
        return json.loads(out) if out else None

    def test_fires_on_substantial_work_even_when_nothing_failed(self) -> None:
        """The case a failure-gated trigger misses, and the reason it was wrong."""
        self.use_blocking_mode()
        result = self.fire(self.transcript([True] * 14))
        self.assertIsNotNone(result, "a long clean turn can still contain a conceptual error")
        self.assertEqual(result["decision"], "block")

    def test_asks_what_the_user_had_to_steer(self) -> None:
        """The trigger is observed steering, not a judgement about wrongness.

        A user correction is evidence sitting in the transcript; "was I wrong
        about anything" is a judgement the reflector has to reach for, and it
        reliably answered no on sessions full of corrections.
        """
        self.use_blocking_mode()
        reason = self.fire(self.transcript([True] * 14))["reason"]
        self.assertIn("had to steer you", reason)
        self.assertIn("teaches nothing", reason.lower(), "a null result stays permitted")

    def test_puts_the_most_missed_kinds_first(self) -> None:
        """Preferences, methods and quality bars lead.

        They are what the user repeats every session until someone writes them
        down, and they were the kinds the previous framing could not see at
        all — it only asked about mistaken beliefs.
        """
        self.use_blocking_mode()
        reason = self.fire(self.transcript([True] * 14))["reason"]
        preference = reason.index("standard or preference this user holds")
        belief = reason.index("false belief about a tool")
        self.assertLess(preference, belief, "the kinds most often missed must lead")

    def test_a_wrong_belief_still_counts_when_it_arrived_as_a_bug(self) -> None:
        """Surfacing as a bug must not demote a wrong belief.

        This is how the costliest lesson of a long debugging session gets
        dropped: it looks mechanical, so a taxonomy that discounts mechanical
        failures discards it.
        """
        self.use_blocking_mode()
        reason = self.fire(self.transcript([True] * 14))["reason"]
        self.assertIn("even though it surfaced as a bug", reason)

    def test_silent_on_a_trivial_turn(self) -> None:
        self.use_blocking_mode()
        self.assertIsNone(self.fire(self.transcript([True, True])))

    def test_fork_mode_copies_the_session_and_never_touches_the_original(self) -> None:
        """A fork inherits the context; --fork-session keeps the live one safe.

        It is affordable because cache reads bill at 0.1x and the cache keys on
        prefix content, so the fork hits what the live session just wrote.
        """
        import rmc.hooks as hooks

        self.store.config.set("learning.nudge_mode", "fork")
        self.store.config.save(self.store.root / "config.yaml")

        launched: list = []
        original = hooks.subprocess.Popen

        class FakePopen:
            def __init__(self, argv, **kw):
                launched.append((argv, kw))

        hooks.subprocess.Popen = FakePopen
        try:
            result = self.fire(self.transcript([True] * 14), session="sess-abc")
        finally:
            hooks.subprocess.Popen = original

        self.assertIsNone(result, "the agent must not be interrupted")
        self.assertTrue(launched)
        argv, kw = launched[0]
        self.assertEqual(argv[:2], ["claude", "--resume"])
        self.assertIn("--fork-session", argv)
        self.assertIn("sess-abc", argv)
        # Without RMC_CHILD the fork fires these hooks and forks itself forever.
        self.assertEqual(kw["env"].get("RMC_CHILD"), "1")
        self.assertTrue(kw.get("start_new_session"))

    def test_fork_falls_back_rather_than_skipping_reflection(self) -> None:
        import rmc.hooks as hooks

        self.store.config.set("learning.nudge_mode", "fork")
        self.store.config.save(self.store.root / "config.yaml")

        spawned: list = []
        orig_fork, orig_bg = hooks._spawn_fork, hooks.spawn_background
        hooks._spawn_fork = lambda *a, **k: False
        hooks.spawn_background = lambda store, args, cwd=None: spawned.append(args)
        try:
            self.fire(self.transcript([True] * 14))
        finally:
            hooks._spawn_fork, hooks.spawn_background = orig_fork, orig_bg
        self.assertTrue(spawned, "a failed fork must degrade to the digest path")
        self.assertEqual(spawned[0][0], "absorb")

    def test_default_mode_reflects_off_thread_without_interrupting(self) -> None:
        """Interrupting an agent mid-task costs a turn and pollutes its context.

        The transcript is the context serialised, so a detached process can do
        the same reflection with no claim on the session at all.
        """
        import rmc.hooks as hooks

        spawned: list = []
        original = hooks.spawn_background
        hooks.spawn_background = lambda store, args, cwd=None: spawned.append(args)
        try:
            result = self.fire(self.transcript([True] * 14))
        finally:
            hooks.spawn_background = original

        self.assertIsNone(result, "the agent must not be interrupted")
        self.assertTrue(spawned, "but the reflection must still happen")
        self.assertEqual(spawned[0][0], "absorb")

    def test_failures_still_count_as_substance(self) -> None:
        self.use_blocking_mode()
        result = self.fire(self.transcript([True, False, False]))
        self.assertIsNotNone(result)
        self.assertIn("cmd1", result["reason"])
        self.assertIn("rarely the most valuable", result["reason"])

    def test_does_not_re_fire_for_work_already_raised(self) -> None:
        self.use_blocking_mode()
        path = self.transcript([False, False])
        self.assertIsNotNone(self.fire(path))
        self.assertIsNone(self.fire(path), "the same turn must not nag twice")

    def test_never_loops_on_its_own_continuation(self) -> None:
        self.assertIsNone(
            self.fire(self.transcript([False, False]), stop_hook_active=True)
        )

    def test_backs_off_when_nudges_keep_yielding_nothing(self) -> None:
        """If the agent captures on its own, stop interrupting it."""
        from rmc.hooks import _barren_streak, criteria_version

        now = criteria_version()
        for _ in range(4):
            self.store.log("nudge", session="s", criteria=now)
        self.assertGreaterEqual(_barren_streak(self.store), 4)
        self.store.log("capture", node="n_x", prompted=True)
        self.assertEqual(_barren_streak(self.store), 0, "a capture resets the streak")

    def test_changing_the_criteria_releases_the_backoff(self) -> None:
        """Barren nudges are evidence about the criteria in force at the time.

        Counting across a criteria change is how a fixed reflector stays
        punished for the broken one it replaced: six fruitless nudges had taken
        the cooldown from 15 minutes to 4 hours here, and rewriting the prompts
        that caused them did not release it, so the fix could never run.
        """
        from rmc.hooks import _barren_streak

        for _ in range(6):
            self.store.log("nudge", session="s", criteria="old12345")
        self.assertEqual(
            _barren_streak(self.store), 0,
            "nudges judged under superseded criteria must not throttle the new ones",
        )

    def test_unversioned_history_is_not_counted(self) -> None:
        """Events predating versioning say nothing about the current criteria."""
        from rmc.hooks import _barren_streak

        for _ in range(5):
            self.store.log("nudge", session="s")
        self.assertEqual(_barren_streak(self.store), 0)

    def test_the_cooldown_cannot_grow_past_an_hour(self) -> None:
        """Beyond about an hour a periodic check is off, not throttled — and a
        long busy session is exactly the one that needs it."""
        base, threshold = 900, 3
        worst = base * (2 ** min(2, 1 + 99 - threshold))
        self.assertLessEqual(worst, 3600, "backoff must stay within an hour")

    def test_can_be_switched_off(self) -> None:
        self.store.config.set("learning.nudge_enabled", False)
        self.store.config.save(self.store.root / "config.yaml")
        self.assertIsNone(self.fire(self.transcript([False, False])))


class TestConcurrentReflectors(StoreCase):
    """Two reflectors may overlap. Neither may record the same lesson twice.

    The defence is reconciliation, not scheduling: a time-based rule fails
    whenever a reflector outlives its window, whereas asking "is this already
    known?" is correct however the two runs interleave. What it requires is that
    decide-and-write be atomic, so the second reflector sees the first's write.
    """

    def test_a_writer_waits_rather_than_dropping_the_lesson(self) -> None:
        held = self.store.lock("write")
        held.__enter__()
        try:
            self.assertTrue(held.acquired)
            # A non-waiting caller gives up immediately...
            with self.store.lock("write") as impatient:
                self.assertFalse(impatient.acquired)
        finally:
            held.__exit__()
        # ...and once released, the lock is takeable again.
        with self.store.lock("write", wait_s=1) as after:
            self.assertTrue(after.acquired)

    def test_a_stale_lock_does_not_wedge_the_store(self) -> None:
        """A reflector killed mid-write must not block every future one."""
        import os
        import time

        path = self.store.root / "write.lock"
        path.write_text("99999")
        os.utime(path, (time.time() - 7200, time.time() - 7200))
        with self.store.lock("write", stale_s=60) as lock:
            self.assertTrue(lock.acquired)

    def test_the_second_reflector_sees_the_first_and_reconciles(self) -> None:
        """The actual defence: a duplicate is judged, not raced."""
        from rmc.placement import apply, decide

        self.add_node(id="n_first", family="deploy", body="Use the argo plugin to promote.")

        def route(prompt, schema):
            if "RMC:related" in prompt:
                return {"picks": [{"id": "n_first", "verdict": "relevant"}]}
            return {"match": "n_first", "relation": "duplicate", "rationale": "already known"}

        decision = decide(
            self.store, MockAdapter(router=route), body="Promote with the argo plugin.",
            family_hint="deploy",
        )
        result = apply(self.store, decision, Node(id="n_second", family="deploy", body="dup"))
        self.assertEqual(decision.action, "duplicate")
        self.assertIsNone(self.store.get("n_second"), "the duplicate must not be stored")


class TestCoUse(StoreCase):
    """Abstraction is built from what got used together, not what reads alike.

    A long tail stays flat unless something merges it, and the useful merge is
    often between lessons with nothing in common on the surface. Co-use is the
    only signal RMC has that speaks to that, and it was already being recorded.
    """

    def setUp(self) -> None:
        super().setUp()
        for ident, family in (("n_a", "deploy"), ("n_b", "deploy"), ("n_c", "caching")):
            self.add_node(id=ident, family=family, body=f"lesson {ident}")

    def co_used(self, ident, used, outcome="success", served=None) -> None:
        self.add_episode(
            ident, "x", "did some work", outcome=outcome,
            served=served if served is not None else used, used=used,
        )

    def test_one_co_occurrence_nominates_a_pair(self) -> None:
        """Two lessons used together in one episode is already uncommon —
        recall serves about one lesson per prompt. Requiring it twice made the
        signal unreachable, and nomination is cheap: the model still has to
        agree they are one procedure, and the merge still has to reproduce both
        their episodes."""
        from rmc.compact import co_use_groups

        self.co_used("e1", ["n_a", "n_b"])
        groups = co_use_groups(self.store)
        self.assertEqual([sorted(n.id for n in nodes) for nodes, _ in groups],
                         [["n_a", "n_b"]])

    def test_the_co_use_floor_is_configurable(self) -> None:
        from rmc.compact import co_use_groups

        self.store.config.set("compaction.min_co_use", 2)
        self.co_used("e1", ["n_a", "n_b"])
        self.assertEqual(co_use_groups(self.store), [])
        self.co_used("e2", ["n_a", "n_b"])
        self.assertEqual(len(co_use_groups(self.store)), 1)

    def test_repeated_co_use_becomes_a_merge_candidate(self) -> None:
        from rmc.compact import co_use_groups

        self.co_used("e1", ["n_a", "n_b"])
        self.co_used("e2", ["n_a", "n_b"])
        groups = co_use_groups(self.store)
        self.assertTrue(groups)
        self.assertEqual({n.id for n in groups[0][0]}, {"n_a", "n_b"})

    def test_co_use_crosses_families(self) -> None:
        """The cross-cutting index: a pair no family structure would ever group."""
        from rmc.compact import co_use_groups

        self.co_used("e1", ["n_a", "n_c"])
        self.co_used("e2", ["n_a", "n_c"])
        groups = co_use_groups(self.store)
        families = {n.family for n in groups[0][0]}
        self.assertEqual(families, {"deploy", "caching"})

    def test_lessons_shown_but_not_used_are_not_co_used(self) -> None:
        """The distinction that matters: serving is a retrieval decision, using
        is an outcome. Counting everything shown manufactures associations —
        serve ten lessons and you invent forty-five pairs."""
        from rmc.compact import co_use_groups

        self.co_used("e1", used=["n_a"], served=["n_a", "n_b", "n_c"])
        self.co_used("e2", used=["n_a"], served=["n_a", "n_b", "n_c"])
        self.assertEqual(co_use_groups(self.store), [], "only one lesson actually bore on the work")

    def test_the_used_subset_is_what_forms_the_abstraction(self) -> None:
        from rmc.compact import co_use_groups

        self.co_used("e1", used=["n_a", "n_c"], served=["n_a", "n_b", "n_c"])
        self.co_used("e2", used=["n_a", "n_c"], served=["n_a", "n_b", "n_c"])
        groups = co_use_groups(self.store)
        self.assertEqual({n.id for n in groups[0][0]}, {"n_a", "n_c"})

    def test_failed_sessions_are_not_evidence_of_belonging(self) -> None:
        from rmc.compact import co_use_groups

        self.co_used("e1", ["n_a", "n_b"], outcome="failure")
        self.co_used("e2", ["n_a", "n_b"], outcome="failure")
        self.assertEqual(co_use_groups(self.store), [])

    def test_a_triple_is_also_evidence_about_each_pair(self) -> None:
        from rmc.compact import co_use_groups

        self.co_used("e1", ["n_a", "n_b", "n_c"])
        self.co_used("e2", ["n_a", "n_b", "n_c"])
        found = {frozenset(n.id for n in nodes) for nodes, _ in co_use_groups(self.store)}
        self.assertIn(frozenset({"n_a", "n_b", "n_c"}), found)
        self.assertIn(frozenset({"n_a", "n_c"}), found, "pairs recur under other companions")


class TestAMergeMustPayForItself(StoreCase):
    """The apex layer is what recall enumerates on every prompt, so a parent
    bigger than the children it stands in front of makes every prompt more
    expensive — the opposite of why we merged.

    This went unchecked: merge computed the ratio, printed it in its own accept
    message ("merged 2 lessons at 102% of combined size"), and accepted anyway.
    Compression had the equivalent gate from the start; merging never did.
    """

    def pair(self, reply_body: str):
        a = self.add_node(id="n_a", family="f", level=0, body="First lesson with detail. @a")
        b = self.add_node(id="n_b", family="f", level=0, body="Second lesson with detail. @b")
        for ident, node in (("e1", "n_a"), ("e2", "n_b")):
            self.add_episode(ident, "f", f"work on {node}", served=[node])
            got = self.store.get(node)
            got.covers_tasks = [ident]
            self.store.save_node(got)
        self.store.invalidate()
        adapter = MockAdapter(router=lambda prompt, schema: {
            "body": reply_body, "title": "Parent", "gist": "a parent", "dropped": [],
        })
        return [self.store.get("n_a"), self.store.get("n_b")], adapter

    def test_a_parent_no_smaller_than_its_children_is_refused(self) -> None:
        from rmc.compact import merge_nodes

        nodes, adapter = self.pair(
            "First lesson with detail. @a Second lesson with detail. @b "
            "And some further words that make this no cheaper than the two apart."
        )
        result = merge_nodes(self.store, adapter, nodes)
        self.assertFalse(result.accepted)
        self.assertIn("does not pay for itself", result.reason)

    def test_a_parent_that_actually_abstracts_goes_on_to_replay(self) -> None:
        from rmc.compact import merge_nodes

        nodes, adapter = self.pair("Both lessons. @a @b")
        result = merge_nodes(self.store, adapter, nodes)
        self.assertLess(result.after_tokens, result.before_tokens)
        self.assertIn("pass-rate", result.reason, "the size gate let it through to replay")

    def test_the_size_gate_runs_before_replay(self) -> None:
        """Replay is the expensive check. No point paying for it to validate a
        parent that could not be kept whatever it says."""
        from rmc.compact import merge_nodes

        nodes, adapter = self.pair(
            "First lesson with detail. @a Second lesson with detail. @b "
            "And some further words that make this no cheaper than the two apart."
        )
        result = merge_nodes(self.store, adapter, nodes)
        self.assertEqual(result.replays, [])


class TestConfigIsOverridesOnly(StoreCase):
    """A store must not be frozen at the defaults of the day it was created.

    The file wins the merge, so anything written into it shadows the default
    forever. Writing the whole tree therefore means a store silently keeps
    months-old numbers and nothing surfaces it — which is exactly what happened
    here with `compaction.max_ratio` and `min_successes`.
    """

    def path(self):
        return self.store.root / "config.yaml"

    def test_saving_writes_only_what_was_chosen(self) -> None:
        from rmc.config import Config

        self.store.config.set("compaction.min_successes", 4)
        self.store.config.save(self.path())
        written = yamlish.load(self.path().read_text())
        self.assertEqual(written.get("compaction"), {"min_successes": 4})
        self.assertNotIn("recall", written, "an untouched section is not a choice")

    def test_a_value_equal_to_the_default_is_not_recorded(self) -> None:
        from rmc.config import DEFAULTS

        self.store.config.set("compaction.max_ratio", DEFAULTS["compaction"]["max_ratio"])
        self.store.config.save(self.path())
        written = yamlish.load(self.path().read_text())
        self.assertNotIn("compaction", written)

    def test_a_later_default_reaches_an_existing_store(self) -> None:
        """The whole point: improve a default, and stores that never chose
        otherwise pick it up."""
        from rmc.config import Config

        self.store.config.set("compaction.min_successes", 4)
        self.store.config.save(self.path())
        reloaded = Config.load(self.path())
        self.assertEqual(reloaded.get("compaction.min_successes"), 4, "the choice survives")
        self.assertEqual(
            reloaded.get("compaction.max_ratio"),
            Config().get("compaction.max_ratio"),
            "everything else follows the current default",
        )

    def test_an_explicit_choice_survives_a_round_trip(self) -> None:
        from rmc.config import Config

        self.store.config.set("agent", "codex")
        self.store.config.save(self.path())
        self.assertEqual(Config.load(self.path()).get("agent"), "codex")


class TestWidthDrivenConsolidation(StoreCase):
    """The long tail has to consolidate without ever having been co-used.

    Co-use needs two lessons used in one episode, twice. Recall serves about one
    lesson per prompt, so a store can accumulate a hundred one-off lessons and
    never produce a single co-use group — while every one of those apexes is
    enumerated on every prompt. Width is the second trigger, and it is a count,
    so the harness owns it; which apexes are the same procedure stays a
    judgement the model makes.
    """

    def widen(self, n: int, family: str = "deploy") -> None:
        for i in range(n):
            self.add_node(id=f"n_{family}_{i}", family=family, body=f"lesson {i}")

    def test_width_alone_makes_a_dream_due(self) -> None:
        from rmc.compact import dream_due

        self.widen(13)
        due, why = dream_due(self.store)
        self.assertTrue(due, why)
        self.assertIn("apexes", why)

    def test_many_families_are_the_same_flat_layer_as_one(self) -> None:
        """Thirteen families of one apex cost the router exactly what one
        family of thirteen costs — so the gate cannot be per-family."""
        from rmc.compact import dream_due

        for i in range(13):
            self.add_node(id=f"n_{i}", family=f"family_{i}", body=f"lesson {i}")
        due, why = dream_due(self.store)
        self.assertTrue(due, why)

    def test_a_narrow_store_with_no_co_use_is_not_due(self) -> None:
        from rmc.compact import dream_due

        self.widen(3)
        due, why = dream_due(self.store)
        self.assertFalse(due)
        self.assertIn("within", why)

    def test_a_child_does_not_count_toward_width(self) -> None:
        from rmc.compact import dream_due

        self.widen(13)
        for i in range(1, 13):
            child = self.store.get(f"n_deploy_{i}")
            child.parents = ["n_deploy_0"]
            self.store.save_node(child)
        self.store.invalidate()
        due, why = dream_due(self.store)
        self.assertFalse(due, why)

    def test_dream_asks_the_model_across_families(self) -> None:
        """The peer set is the apex layer, not one family — otherwise the
        thirteen-singleton store has nothing any pass can reach."""
        import rmc.compact as compact

        for i in range(13):
            self.add_node(id=f"n_{i}", family=f"family_{i}", body=f"lesson {i}")
        seen: list[list[str]] = []

        def spy(store, family=None, adapter=None, *, peers=None):
            seen.append(sorted(n.id for n in (peers or [])))
            return []

        original = compact.merge_candidates
        compact.merge_candidates = spy
        try:
            compact.dream(self.store, MockAdapter(), dry_run=True)
        finally:
            compact.merge_candidates = original

        self.assertEqual(len(seen), 1)
        self.assertEqual(len(seen[0]), 13, "every apex was a candidate, whatever its family")

    def test_a_narrow_layer_is_left_alone(self) -> None:
        import rmc.compact as compact

        self.widen(4)
        called = []
        original = compact.merge_candidates
        compact.merge_candidates = lambda *a, **k: called.append(1) or []
        try:
            compact.dream(self.store, MockAdapter(), dry_run=True)
        finally:
            compact.merge_candidates = original
        self.assertEqual(called, [], "nothing to pay for, nothing to consolidate")

    def test_the_coldest_apexes_go_first(self) -> None:
        """A lesson that keeps getting used is earning its place at the top;
        one nothing has touched is pure routing tax."""
        from rmc.compact import _coldest

        self.widen(3)
        hot = self.store.get("n_deploy_0")
        hot.stats.attempts = 9
        self.store.save_node(hot)
        self.store.invalidate()
        order = [n.id for n in _coldest(self.store, self.store.apexes())]
        self.assertEqual(order[-1], "n_deploy_0")

    def test_one_pass_asks_about_a_bounded_slice(self) -> None:
        """Judging costs a call per anchor, so a huge store cannot have all of
        it considered every night."""
        from rmc.compact import _coldest

        self.widen(60)
        self.assertEqual(len(_coldest(self.store, self.store.apexes())), 24)

    def test_an_over_wide_group_is_not_attempted(self) -> None:
        """One episode that used nine lessons nominates all nine. Attempting
        that spends the whole ration on the least likely candidate."""
        import rmc.compact as compact

        self.widen(9, "deploy")
        peers = [self.store.get(f"n_deploy_{i}") for i in range(9)]
        self.add_episode("e1", "x", "work",
                         served=[n.id for n in peers], used=[n.id for n in peers])
        attempts: list[int] = []

        def merge(store, adapter, nodes, dry_run=False):
            attempts.append(len(nodes))
            return compact.CompactionResult(node_id=nodes[0].id, accepted=False, reason="test")

        original = (compact.merge_nodes, compact.merge_candidates)
        compact.merge_nodes = merge
        compact.merge_candidates = lambda *a, **k: []
        try:
            compact.dream(self.store, MockAdapter(), limit=4, dry_run=True)
        finally:
            compact.merge_nodes, compact.merge_candidates = original

        self.assertTrue(attempts, "the pairs underneath are still candidates")
        self.assertTrue(all(n <= 5 for n in attempts), attempts)

    def spy_merges(self, verdicts, *, limit=2, groups=6):
        """Run a dream whose every merge answers from `verdicts`, and report
        which groups it tried."""
        import rmc.compact as compact

        self.widen(13, "deploy")
        peers = [self.store.get(f"n_deploy_{i}") for i in range(13)]
        pairs = [[peers[i], peers[i + 1]] for i in range(0, groups * 2, 2)]
        tried: list[list[str]] = []
        answers = list(verdicts)

        def candidates(store, family=None, adapter=None, *, peers=None):
            return pairs

        def merge(store, adapter, nodes, dry_run=False):
            tried.append([n.id for n in nodes])
            ok = answers.pop(0) if answers else False
            return compact.CompactionResult(
                node_id=nodes[0].id, accepted=ok, reason="test",
                new_node=nodes[0] if ok else None,
            )

        original = (compact.merge_candidates, compact.merge_nodes)
        compact.merge_candidates, compact.merge_nodes = candidates, merge
        try:
            compact.dream(self.store, MockAdapter(), limit=limit, dry_run=True)
        finally:
            compact.merge_candidates, compact.merge_nodes = original
        return tried

    def test_accepted_merges_are_rationed(self) -> None:
        """Rewriting the tree is the irreversible thing, so it is the one
        rationed tightly."""
        tried = self.spy_merges([True, True, True, True], limit=2)
        self.assertEqual(len(tried), 2)

    def test_a_rejection_does_not_spend_the_ration(self) -> None:
        """Two rejections used to end a pass with thirty-four candidates
        untried — the store stayed flat and the log said work had been done."""
        tried = self.spy_merges([False, False, True, True], limit=2)
        self.assertEqual(len(tried), 4, "kept looking past the rejections")

    def test_one_lesson_does_not_soak_up_the_whole_pass(self) -> None:
        """A single episode that used nine lessons nominates all thirty-six of
        their pairs, and every pair of one node sorts together — so a pass
        would re-ask about the same lesson against eight partners."""
        import rmc.compact as compact

        self.widen(9, "deploy")
        peers = [self.store.get(f"n_deploy_{i}") for i in range(9)]
        self.add_episode("e1", "x", "work",
                         served=[n.id for n in peers], used=[n.id for n in peers])
        tried: list[list[str]] = []

        def merge(store, adapter, nodes, dry_run=False):
            tried.append([n.id for n in nodes])
            return compact.CompactionResult(node_id=nodes[0].id, accepted=False, reason="test")

        original = (compact.merge_nodes, compact.merge_candidates)
        compact.merge_nodes = merge
        compact.merge_candidates = lambda *a, **k: []
        try:
            compact.dream(self.store, MockAdapter(), limit=2, dry_run=True)
        finally:
            compact.merge_nodes, compact.merge_candidates = original

        flat = [i for pair in tried for i in pair]
        self.assertEqual(len(flat), len(set(flat)), f"a lesson was asked about twice: {tried}")

    def test_looking_is_still_bounded(self) -> None:
        self.store.config.set("dream.max_attempts", 3)
        tried = self.spy_merges([False] * 6, limit=2)
        self.assertEqual(len(tried), 3)


class TestAnEmptyPackSaysWhy(StoreCase):
    """Nothing relevant and nothing answering are opposite facts.

    They produce an identical empty pack. Before relevance filtering ran on
    every prompt this barely mattered — a small store was served unfiltered, so
    an outage was visible as lessons vanishing. Now a backend that is down is
    indistinguishable from a quiet day, and the user concludes RMC does not
    work rather than that it is broken.
    """

    def setUp(self) -> None:
        super().setUp()
        self.store.config.set("recall.filter_above", 0)
        self.add_node(id="n_a", family="f", title="Retry", body="Retry idempotent calls.")

    def test_a_judge_that_answers_nothing_relevant_is_not_an_outage(self) -> None:
        pack = recall_pack(self.store, "unrelated work", router({"picks": []}))
        self.assertFalse(pack.degraded)
        self.assertEqual(pack.served, [])

    def test_a_judge_that_cannot_answer_is_reported(self) -> None:
        from rmc.hooks import recall_notice

        broken = MockAdapter(router=lambda prompt, schema: None)
        pack = recall_pack(self.store, "any work", broken)
        self.assertTrue(pack.degraded)
        self.assertIn("could not reach the recall judge", recall_notice(pack))

    def test_the_outage_is_logged_where_a_report_can_find_it(self) -> None:
        broken = MockAdapter(router=lambda prompt, schema: None)
        recall_pack(self.store, "any work", broken)
        self.assertTrue(list(self.store.read_events("recall-degraded", limit=5)))


class TestReInjection(StoreCase):
    """A lesson already in context should not be paid for twice — but "present"
    and "still attended to" are different, so there are three cases."""

    def setUp(self) -> None:
        super().setUp()
        self.node = self.add_node(
            id="n_r", family="f", title="Retry", gist="Retry idempotently.", body="Long body. " * 40
        )

    def pack(self, **kw):
        keeps = router({"picks": [{"id": "n_r", "verdict": "relevant"}]})
        return recall_pack(self.store, "do the thing", keeps, **kw)

    def test_first_sight_serves_the_full_lesson(self) -> None:
        pack = self.pack(already_served={}, turn=1)
        self.assertIn("Long body", pack.text)
        self.assertEqual(pack.served, ["n_r"])

    def test_a_recent_lesson_is_not_repeated(self) -> None:
        pack = self.pack(already_served={"n_r": 5}, turn=7)
        self.assertEqual(pack.skipped, ["n_r"])
        self.assertNotIn("Long body", pack.text)
        self.assertEqual(pack.tokens, 0)

    def test_a_distant_lesson_is_refreshed_by_gist_not_repeated(self) -> None:
        """Cheap salience, not a second full payment."""
        pack = self.pack(already_served={"n_r": 1}, turn=40)
        self.assertEqual(pack.refreshed, ["n_r"])
        self.assertIn("Retry idempotently", pack.text)
        self.assertNotIn("Long body", pack.text)
        self.assertLess(pack.tokens, 40)

    def test_compaction_makes_everything_servable_again(self) -> None:
        """After compaction the lesson text may simply be gone."""
        import io
        from contextlib import redirect_stdout

        from rmc.hooks import on_pre_compact

        self.store.write_session("s", {"served_at": {"n_r": 3}, "served": ["n_r"], "turn": 3})
        with redirect_stdout(io.StringIO()):
            on_pre_compact({"session_id": "s", "cwd": str(self.base)})
        self.assertEqual(self.store.read_session("s")["served_at"], {})


class TestDreamSchedule(StoreCase):
    """Consolidation runs on a clock and on evidence, and leaves an account.

    Both gates are structural — a clock reading and a count of episodes. Neither
    asks what anything means, so neither belongs to the model.
    """

    def seed_evidence(self, n: int) -> None:
        for ident, family in (("n_a", "deploy"), ("n_b", "caching")):
            if not self.store.get(ident):
                self.add_node(id=ident, family=family, body=f"lesson {ident}")
        for i in range(n):
            self.add_episode(f"e{i}", "x", "work", served=["n_a", "n_b"], used=["n_a", "n_b"])

    def test_not_due_without_new_evidence(self) -> None:
        from rmc.compact import dream_due

        self.store.config.set("dream.min_new_episodes", 3)
        self.seed_evidence(1)
        due, why = dream_due(self.store)
        self.assertFalse(due)
        self.assertIn("need 3", why)

    def test_due_once_enough_has_accumulated(self) -> None:
        from rmc.compact import dream_due

        self.store.config.set("dream.min_new_episodes", 3)
        self.seed_evidence(3)
        due, why = dream_due(self.store)
        self.assertTrue(due, why)

    def test_a_single_multi_lesson_episode_is_enough_by_default(self) -> None:
        """Over a month of real use this store produced one such episode. A
        floor of three meant co-use could never fire on its own."""
        from rmc.compact import dream_due

        self.seed_evidence(1)
        due, why = dream_due(self.store)
        self.assertTrue(due, why)

    def test_not_due_again_inside_the_interval(self) -> None:
        """A clock reading, not a judgement."""
        from rmc.compact import dream_due

        self.seed_evidence(3)
        self.store.log("dream", episodes_seen=3)
        due, why = dream_due(self.store)
        self.assertFalse(due)
        self.assertIn("interval", why)

    def test_can_be_switched_off(self) -> None:
        from rmc.compact import dream_due

        self.store.config.set("dream.enabled", False)
        self.seed_evidence(9)
        self.assertFalse(dream_due(self.store)[0])

    def test_a_dream_writes_a_readable_account(self) -> None:
        """Nobody watched it happen, so it has to say what it did."""
        from rmc.compact import dream, dream_logs

        self.seed_evidence(3)
        report = dream(self.store, MockAdapter(world=MockWorld()), limit=1)

        logs = dream_logs(self.store)
        self.assertTrue(logs, "the dream must leave a record")
        text = logs[0].read_text()
        self.assertIn("before", text)
        self.assertIn("routing tokens per prompt", text)
        # Before/after is the point: it shows what changed, not just what ran.
        self.assertIn("nodes", report.before)
        self.assertIn("routing_tokens", report.after)

    def test_a_refused_merge_is_recorded_with_its_reason(self) -> None:
        from rmc.compact import dream, dream_logs

        self.seed_evidence(3)
        broken = MockAdapter(router=lambda p, s: {"body": "", "dropped": []})
        dream(self.store, broken, limit=1)
        self.assertIn("refused", dream_logs(self.store)[0].read_text())


class TestPerUseCredit(StoreCase):
    """Usage happens per turn, so it must be counted per turn.

    Crediting once per session means a lesson leaned on six times scores one,
    and a lesson used all day in a session that never ends scores nothing —
    which is not "the more a memory is used, the cheaper it becomes".
    """

    def run_used(self, **kw):
        import argparse

        from rmc.cli import cmd_used

        args = argparse.Namespace(
            session="s1", used=None, unused=None, task=None, outcome=None, cwd=str(self.base)
        )
        for k, v in kw.items():
            setattr(args, k, v)
        import io
        from contextlib import redirect_stdout

        with redirect_stdout(io.StringIO()):
            cmd_used(args)
        self.store.invalidate()

    def test_each_use_is_credited_immediately(self) -> None:
        node = self.add_node(id="n_u", family="f", body="a lesson")
        for _ in range(3):
            self.run_used(used="n_u")
        reloaded = self.store.get("n_u")
        self.assertEqual(reloaded.stats.successes, 3)
        self.assertEqual(reloaded.stats.attempts, 3)

    def test_a_use_records_a_replayable_episode_from_the_real_task(self) -> None:
        """Session-end episodes used the session's *opening* prompt, so nothing a
        lesson was actually applied to was ever recorded."""
        node = self.add_node(id="n_e", family="f", body="a lesson")
        self.run_used(
            used="n_e",
            task="the user asked four distinct questions in one message",
            outcome="answered with a table mapping each question to its answer",
        )
        episodes = self.store.episodes()
        self.assertEqual(len(episodes), 1)
        self.assertIn("four distinct questions", episodes[0].prompt)
        self.assertEqual(episodes[0].used, ["n_e"])
        # And that episode is what makes the lesson compressible at all.
        self.assertEqual(len(self.store.regression_set(self.store.get("n_e"))), 1)

    def test_without_a_task_no_episode_is_invented(self) -> None:
        self.add_node(id="n_n", family="f", body="a lesson")
        self.run_used(used="n_n")
        self.assertEqual(self.store.episodes(), [])

    def test_repeated_use_makes_a_lesson_due_for_compression(self) -> None:
        """The whole point: usage should drive compression."""
        from rmc.compact import due_nodes

        self.add_node(id="n_d", family="f", body="a lesson worth compressing " * 8)
        self.assertEqual(due_nodes(self.store), [])
        for i in range(2):
            self.run_used(used="n_d", task=f"task {i}", outcome="done right")
        self.assertEqual([n.id for n in due_nodes(self.store)], ["n_d"])

    def test_session_end_does_not_credit_again_what_was_banked(self) -> None:
        node = self.add_node(id="n_b", family="f", body="a lesson")
        self.run_used(used="n_b")
        self.assertEqual(self.store.get("n_b").stats.successes, 1)

        observe(
            self.store,
            SessionFacts(
                user_messages=["do it"], assistant_messages=["done"], tool_calls=14,
                first_prompt="do it", last_assistant="done",
            ),
            adapter=router({"outcome": "success", "confidence": 0.9, "corrected": False,
                            "lessons_used": [{"id": "n_b", "used": True}]}),
            attributed={"n_b": True},
            banked={"n_b": 1},
            served=["n_b"],
        )
        self.assertEqual(
            self.store.get("n_b").stats.successes, 1, "per-use count must not be topped up"
        )


class TestRoutingCost(StoreCase):
    def test_routing_sends_a_gist_not_the_body(self) -> None:
        """Deciding what to load must not cost more than loading it.

        The router used to send 700 characters of body per candidate, so triage
        grew with the store: ~185k tokens to choose among 1000 lessons.
        """
        from rmc.judge import _render

        body = "A very long lesson body. " * 200
        node = self.add_node(id="n_g", family="f", title="Deploys", gist="Promote with argo, never kubectl apply.", body=body)
        rendered = _render(node)
        self.assertIn("Promote with argo", rendered)
        self.assertNotIn("A very long lesson body. A very long", rendered)
        self.assertLess(len(rendered), 300)

    def test_a_lesson_without_a_gist_still_routes_cheaply(self) -> None:
        node = self.add_node(id="n_ng", family="f", title="T", body="x " * 2000)
        self.assertLess(len(node.summary()), 300)


class TestEval(StoreCase):
    """The eval has to be able to deliver bad news, or it is a demo."""

    def build_chain(self):
        base = self.add_node(id="n_l0", family="f", level=0, body="Full lesson. @fact @detail")
        top = self.add_node(id="n_l1", family="f", level=1, body="Short. @fact", derived_from=["n_l0"])
        base.parents = [top.id]
        self.store.save_node(base)
        self.store.invalidate()
        for i in range(4):
            self.add_episode(f"e{i}", "f", f"task {i}", served=["n_l0"], used=["n_l0"],
                             summary="did it with the fact")
        return base, top

    def grader(self, world):
        """Answers the probe from the lesson, then grades blind."""
        def route(prompt, schema):
            if "RMC:judge" in prompt:
                import re
                m = re.search(r"<<<CANDIDATE\n(.*?)\nCANDIDATE>>>", prompt, re.DOTALL)
                return {"pass": "@fact" in (m.group(1) if m else ""), "reason": "checked"}
            m = __import__("re").search(r"<<<LESSON\n(.*?)\nLESSON>>>", prompt, __import__("re").DOTALL)
            return m.group(1) if m else ""
        return MockAdapter(router=route)

    def test_it_measures_a_control_arm(self) -> None:
        """Without one you measure the model's prior, not the lesson."""
        from rmc.evaluate import CONTROL, evaluate

        self.build_chain()
        report = evaluate(self.store, self.grader(None), holdout=1.0, samples=1)
        self.assertTrue(report.cases)
        self.assertIn(CONTROL, report.cases[0].arms)
        self.assertEqual(report.aggregate(CONTROL).rate, 0.0, "no lesson, no fact, no pass")
        self.assertEqual(report.aggregate("L0").rate, 1.0)

    def test_it_reports_lift_and_flags_a_useless_lesson(self) -> None:
        """If the lesson does not beat the control, retention is meaningless."""
        from rmc.evaluate import evaluate

        self.build_chain()
        always = MockAdapter(router=lambda p, s: (
            {"pass": True, "reason": "ok"} if "RMC:judge" in p else "an answer"
        ))
        report = evaluate(self.store, always, holdout=1.0, samples=1)
        self.assertAlmostEqual(report.lift, 0.0)
        self.assertIn("barely beats no lesson", report.render())

    def test_a_lossy_compression_shows_up_as_lost_transfer(self) -> None:
        """The falsifiable claim: if L1 drops what mattered, the eval says so."""
        from rmc.evaluate import evaluate

        base, top = self.build_chain()
        top.body = "Short, and missing the point."  # no @fact
        self.store.save_node(top)
        self.store.invalidate()

        report = evaluate(self.store, self.grader(None), holdout=1.0, samples=1)
        self.assertEqual(report.aggregate("L0").rate, 1.0)
        self.assertEqual(report.aggregate("L1").rate, 0.0, "the eval must catch this")

    def test_the_grader_never_sees_the_lesson(self) -> None:
        """The ordinary replay judge takes the lesson as context, which would
        identify the control arm by its absence."""
        from rmc.evaluate import evaluate

        seen: list = []

        def route(prompt, schema):
            if "RMC:judge" in prompt:
                seen.append(prompt)
                return {"pass": True, "reason": "ok"}
            return "answer"

        self.build_chain()
        evaluate(self.store, MockAdapter(router=route), holdout=1.0, samples=1)
        self.assertTrue(seen)
        for prompt in seen:
            self.assertNotIn("Full lesson", prompt)
            self.assertNotIn("LESSON>>>", prompt)

    def test_holdout_is_deterministic(self) -> None:
        from rmc.evaluate import holdout_split

        self.build_chain()
        eps = self.store.episodes()
        first = [e.id for e in holdout_split(eps, 0.5)]
        second = [e.id for e in holdout_split(eps, 0.5)]
        self.assertEqual(first, second, "an episode drifting between runs leaks")

    def test_it_writes_nothing_to_the_store(self) -> None:
        """An eval that mutates what it measures is measuring itself."""
        from rmc.evaluate import evaluate

        base, _ = self.build_chain()
        before = base.stats.to_dict()
        evaluate(self.store, self.grader(None), holdout=1.0, samples=1)
        self.store.invalidate()
        self.assertEqual(self.store.get("n_l0").stats.to_dict(), before)
        self.assertEqual(len(self.store.episodes()), 4)

    def test_small_samples_are_labelled_as_not_a_result(self) -> None:
        from rmc.evaluate import evaluate

        self.build_chain()
        report = evaluate(self.store, self.grader(None), holdout=1.0, samples=1)
        self.assertIn("NOT A RESULT", report.render())


class TestHooks(StoreCase):
    def test_recursion_guard(self) -> None:
        import os

        from rmc.hooks import dispatch

        os.environ["RMC_CHILD"] = "1"
        try:
            self.assertEqual(dispatch("user-prompt-submit"), 0)
        finally:
            os.environ.pop("RMC_CHILD", None)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestCliOnPath(unittest.TestCase):
    """Wiring hooks does not make `rmc` runnable in a shell.

    Hooks invoke the package by absolute path, so an install could report
    success while `rmc status` — which the README tells you to run — failed
    with command not found.
    """

    def setUp(self) -> None:
        from rmc import install as inst
        self.inst = inst
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _env(self, path_value: str, home: str):
        return mock.patch.dict(os.environ, {"PATH": path_value, "HOME": home}, clear=False)

    def test_shim_is_found_from_the_clone(self) -> None:
        shim = self.inst.shim_path()
        self.assertIsNotNone(shim, "bin/rmc should be discoverable from the package")
        self.assertTrue(os.access(shim, os.X_OK), "and it must be executable")

    def test_link_dir_prefers_a_writable_dir_already_on_path(self) -> None:
        good = Path(self.tmp) / ".local" / "bin"
        good.mkdir(parents=True)
        with mock.patch.object(Path, "home", staticmethod(lambda: Path(self.tmp))), \
             self._env(str(good), self.tmp):
            self.assertEqual(self.inst.link_dir(), good)

    def test_link_creates_a_working_command(self) -> None:
        home = Path(self.tmp)
        with mock.patch.object(Path, "home", staticmethod(lambda: home)), \
             self._env("/usr/bin:/bin", self.tmp), \
             mock.patch.object(self.inst.shutil, "which", return_value=None):
            notes = self.inst.link_cli()
        link = home / ".local" / "bin" / "rmc"
        self.assertTrue(link.is_symlink(), notes)
        self.assertEqual(link.resolve(), self.inst.shim_path().resolve())
        self.assertTrue(any("not on PATH" in n for n in notes),
                        "a link into a dir that is not on PATH must say so")

    def test_advice_is_given_when_the_command_is_missing(self) -> None:
        home = Path(self.tmp)          # nothing linked here yet
        with mock.patch.object(Path, "home", staticmethod(lambda: home)), \
             self._env("/usr/bin:/bin", self.tmp), \
             mock.patch.object(self.inst.shutil, "which", return_value=None):
            advice = self.inst.cli_advice()
        self.assertTrue(advice)
        self.assertTrue(any("ln -s" in line for line in advice))
        self.assertTrue(any("Hooks are unaffected" in line for line in advice),
                        "must not imply the hooks are broken too")

    def test_advice_after_linking_never_tells_you_to_run_rmc(self) -> None:
        """The bootstrap trap: advice for someone without the command must not
        require the command. Once linked, the only gap is PATH itself."""
        home = Path(self.tmp)
        with mock.patch.object(Path, "home", staticmethod(lambda: home)), \
             self._env("/usr/bin:/bin", self.tmp), \
             mock.patch.object(self.inst.shutil, "which", return_value=None):
            self.inst.link_cli()
            advice = self.inst.cli_advice()
        text = "\n".join(advice)
        self.assertIn("export PATH=", text, "must say how to finish the job")
        self.assertNotIn("ln -s", text, "it is already linked; do not repeat that step")
        self.assertNotIn("rmc install", text, "cannot ask them to run the missing command")

    def test_no_advice_when_already_on_path(self) -> None:
        with mock.patch.object(self.inst.shutil, "which", return_value="/somewhere/rmc"):
            self.assertEqual(self.inst.cli_advice(), [])

    def test_existing_unrelated_file_is_not_clobbered(self) -> None:
        home = Path(self.tmp)
        target = home / ".local" / "bin"
        target.mkdir(parents=True)
        (target / "rmc").write_text("someone else's script")
        with mock.patch.object(Path, "home", staticmethod(lambda: home)), \
             self._env("/usr/bin:/bin", self.tmp), \
             mock.patch.object(self.inst.shutil, "which", return_value=None):
            notes = self.inst.link_cli()
        self.assertEqual((target / "rmc").read_text(), "someone else's script")
        self.assertTrue(any("already exists" in n for n in notes))


class TestCaptureAttribution(unittest.TestCase):
    """Who captured a lesson is the signal; whether a nudge preceded it is not.

    The metric this replaces counted captures that followed no nudge and read a
    high share as the agent having outgrown the scaffolding. A session where the
    user has to ask "why did you not learn that?" and the agent then runs
    `rmc add` by hand scored perfectly on it — the worst outcome reported as the
    best.
    """

    def test_both_reflectors_treat_a_hand_added_lesson_as_their_own_miss(self) -> None:
        from rmc import hooks
        flat = lambda t: " ".join(t.split())      # prompts are hard-wrapped
        self.assertIn("rmc add", hooks.FORK_PROMPT)
        self.assertIn("a capture you failed to make", flat(hooks.FORK_PROMPT))
        self.assertIn("rmc add", hooks.NUDGE)
        self.assertIn("should have made and did not", flat(hooks.NUDGE))

    def test_fork_prompt_routes_the_meta_lesson_somewhere(self) -> None:
        from rmc import hooks
        self.assertIn("reflection", hooks.FORK_PROMPT,
                      "a miss about missing needs a family to land in")

    def test_a_reflector_capture_is_marked_as_such(self) -> None:
        with mock.patch.dict(os.environ, {"RMC_CHILD": "1"}, clear=False):
            self.assertEqual(
                "reflector" if os.environ.get("RMC_CHILD") else "session", "reflector")

    def test_a_session_capture_is_marked_as_such(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "RMC_CHILD"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(
                "reflector" if os.environ.get("RMC_CHILD") else "session", "session")


class TestDefectReport(unittest.TestCase):
    """A report is written, never sent. RMC's no-egress promise is on the docs
    page, and it is worth more than the convenience of filing automatically."""

    def setUp(self) -> None:
        from rmc import report
        from rmc.store import Store
        self.report = report
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        Store.init(Path(self.tmp))
        self.store = Store.discover(Path(self.tmp))

    def test_report_module_makes_no_network_calls(self) -> None:
        src = (Path(__file__).resolve().parents[1] / "rmc" / "report.py").read_text()
        for banned in ("urllib", "requests", "http.client", "socket.", "urlopen"):
            self.assertNotIn(banned, src, f"{banned} would break the no-egress promise")

    def test_secrets_in_the_description_are_redacted(self) -> None:
        body = self.report.build(self.store, "died with token=ghp_AAAABBBBCCCCDDDDEEEEFFFF1234")
        self.assertNotIn("ghp_AAAABBBBCCCCDDDDEEEEFFFF1234", body)
        self.assertIn("[REDACTED]", body)

    def test_no_lesson_text_leaks_into_a_report(self) -> None:
        """Node ids are opaque; lesson bodies are the user's own work and often
        describe their codebase."""
        from rmc.node import Node
        secret = "the acme billing service lives behind an internal proxy"
        self.store.save_node(Node(id="n_zzz", family="f", title="Internal", body=secret))
        body = self.report.build(self.store, "capture is not firing")
        self.assertNotIn(secret, body)
        self.assertNotIn("Internal", body)

    def test_it_writes_to_disk_and_hands_back_a_command(self) -> None:
        path = self.report.write(self.store, "reflector captured nothing")
        self.assertTrue(path.exists())
        cmd = self.report.gh_command(path, 'a "quoted" title')
        self.assertIn("gh issue create", cmd)
        self.assertIn(str(path), cmd)
        self.assertNotIn('"a "quoted" title"', cmd, "quotes must not break the command")

    def test_the_reflector_is_told_to_ask_rather_than_file(self) -> None:
        from rmc import hooks
        flat = " ".join(hooks.FORK_PROMPT.split())
        self.assertIn("rmc report", flat)
        self.assertIn("ask the user whether they want it filed", flat)
        self.assertIn("never file it yourself", flat)


class TestTreeRecency(unittest.TestCase):
    """`rmc tree` groups by family, so "what did it just learn?" is unanswerable
    from it — the newest lesson lands wherever its family sorts, looking exactly
    like one from last month."""

    def test_age_is_short_and_degrades_quietly(self) -> None:
        from rmc.cli import _age
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        fmt = lambda d: (now - d).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.assertEqual(_age(fmt(timedelta(seconds=5))), "just now")
        self.assertIn("m ago", _age(fmt(timedelta(minutes=20))))
        self.assertIn("h ago", _age(fmt(timedelta(hours=5))))
        self.assertIn("d ago", _age(fmt(timedelta(days=3))))
        self.assertEqual(_age(""), "", "a missing stamp must not raise")
        self.assertEqual(_age("not a date"), "", "nor must a malformed one")


class TestRoutingView(unittest.TestCase):
    """Title and gist are what the relevance walk reads — never the body. That
    makes them load-bearing for retrieval, so they have to describe the body
    that currently exists, not the one that existed when the node was named."""

    def setUp(self) -> None:
        from rmc import summary
        from rmc.store import Store
        self.summary = summary
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        Store.init(Path(self.tmp))
        self.store = Store.discover(Path(self.tmp))
        self.calls = 0

    def _adapter(self, payload):
        from rmc.adapters.mock import MockAdapter

        def route(prompt, schema):
            self.calls += 1
            return payload
        return MockAdapter(router=route)

    def _node(self, **kw):
        from rmc.node import Node
        base = dict(id="n_t", family="f", body="a lesson about ports", title="", gist="")
        base.update(kw)
        return Node(**base)

    def test_a_missing_gist_is_filled(self) -> None:
        node = self._node()
        adapter = self._adapter({"gist": "when a port is refused", "title": "Ports"})
        self.assertTrue(self.summary.refresh(self.store, adapter, node))
        self.assertEqual(node.gist, "when a port is refused")
        self.assertEqual(node.title, "Ports")

    def test_an_existing_title_is_not_overwritten(self) -> None:
        node = self._node(title="Written by a human")
        adapter = self._adapter({"gist": "g", "title": "Model's title"})
        self.summary.refresh(self.store, adapter, node)
        self.assertEqual(node.title, "Written by a human")
        self.assertEqual(node.gist, "g", "but a missing gist is still filled")

    def test_force_refreshes_both_after_a_fold(self) -> None:
        """A fold rewrites the body and keeps the survivor's name, so the label
        has to be reissued or the node advertises what it used to be about."""
        node = self._node(title="Stale", gist="stale")
        adapter = self._adapter({"gist": "new gist", "title": "New title"})
        self.assertTrue(self.summary.refresh(self.store, adapter, node, force=True))
        self.assertEqual((node.title, node.gist), ("New title", "new gist"))

    def test_nothing_is_written_when_the_model_is_unavailable(self) -> None:
        """A stale gist still routes; an invented one poisons retrieval."""
        node = self._node(title="Kept", gist="kept")
        adapter = self._adapter(None)          # model returns nothing usable
        self.assertFalse(self.summary.refresh(self.store, adapter, node, force=True))
        self.assertEqual((node.title, node.gist), ("Kept", "kept"))

    def test_complete_nodes_cost_nothing(self) -> None:
        node = self._node(title="t", gist="g")
        self.assertFalse(self.summary.refresh(self.store, self._adapter({}), node))
        self.assertEqual(self.calls, 0, "a complete node must not cost a model call")


class TestRecallNotice(unittest.TestCase):
    """The line above every prompt. A count proves RMC fired; only the titles
    let you notice it fired on the wrong lessons."""

    def _pack(self, titles, tokens=100, conflicts=(), refreshed=0, skipped=0):
        from rmc.recall import Pack
        p = Pack()
        p.titles = list(titles)
        # served covers full injections and refreshers alike
        p.served = ([f"n{i}" for i in range(len(titles))]
                    + [f"r{i}" for i in range(refreshed)])
        p.refreshed = [f"r{i}" for i in range(refreshed)]
        p.skipped = [f"s{i}" for i in range(skipped)]
        p.tokens = tokens
        p.conflicts = list(conflicts)
        return p

    def test_a_refresher_is_not_counted_as_a_lesson(self) -> None:
        """A full body and a one-line reminder are materially different, and
        reporting them as one number is why '2 lessons' could appear with a
        single lesson visible."""
        from rmc.hooks import recall_notice
        note = recall_notice(self._pack(["Fix identified defects"], 118, refreshed=1))
        self.assertIn("1 lesson ", note)
        self.assertNotIn("2 lessons", note)
        self.assertIn("1 refreshed", note)

    def test_skips_are_named_so_a_zero_is_explicable(self) -> None:
        """Nothing injected is a decision, not a failure — say why."""
        from rmc.hooks import recall_notice
        note = recall_notice(self._pack([], 0, skipped=3))
        self.assertIn("0 lessons", note)
        self.assertIn("3 already in context", note)

    def test_it_names_what_was_recalled(self) -> None:
        from rmc.hooks import recall_notice
        note = recall_notice(self._pack(["Retrying flaky calls"], 298))
        self.assertIn("Retrying flaky calls", note)
        self.assertIn("298 tok", note)

    def test_it_stays_on_one_line(self) -> None:
        from rmc.hooks import recall_notice, NOTICE_WIDTH
        long = ["A lesson with a deliberately overlong title " + str(i) for i in range(6)]
        note = recall_notice(self._pack(long, 1200))
        self.assertLessEqual(len(note), NOTICE_WIDTH + 24, note)
        self.assertNotIn("\n", note)

    def test_it_says_how_many_it_did_not_name(self) -> None:
        from rmc.hooks import recall_notice
        note = recall_notice(self._pack(
            ["Retry policy", "Cache TTLs", "Deploy rollback",
             "Idempotency keys everywhere", "Parsing bodies not statuses"], 900))
        self.assertIn("more", note, "a truncated list must admit what it hid")

    def test_one_very_long_title_is_elided_not_dropped(self) -> None:
        from rmc.hooks import recall_notice
        note = recall_notice(self._pack(["x" * 200], 740))
        self.assertIn("…", note)
        self.assertIn("xxx", note)

    def test_a_conflict_still_shows(self) -> None:
        from rmc.hooks import recall_notice
        self.assertIn("conflict", recall_notice(self._pack(["Retry"], 10, ["n1"])))

    def test_no_titles_degrades_to_the_old_line(self) -> None:
        from rmc.hooks import recall_notice
        from rmc.recall import Pack
        p = Pack(); p.served = ["n1"]; p.tokens = 12
        self.assertEqual(recall_notice(p), "RMC · 1 lesson · 12 tok")
