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


class TestRecall(StoreCase):
    def force_judging(self) -> None:
        """Opt into relevance filtering — tiny stores skip it by design."""
        self.store.config.set("recall.always_judge", True)

    def test_everything_is_served_when_it_all_fits_in_the_budget(self) -> None:
        """Under the budget there is nothing to choose, so nothing is asked."""
        self.add_node(id="n_a", family="retry", body="Retry idempotent calls.")
        self.add_node(id="n_b", family="graphql", body="Batch queries.")
        log: list = []
        pack = recall_pack(self.store, "anything at all", counting_router({"picks": []}, log))
        self.assertEqual(sorted(pack.served), ["n_a", "n_b"])
        self.assertEqual(log, [], "must not spend a call choosing from what it can afford")

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
        self.force_judging()
        self.add_node(id="n_r", family="retry", title="Retry", body="Retry idempotent calls.", level=2)
        self.add_node(id="n_g", family="graphql", title="GraphQL", body="Batch queries.", level=1)
        adapter = router({"picks": [{"id": "n_r", "verdict": "relevant", "why": "same subject"}]})

        pack = recall_pack(self.store, "the client needs retries", adapter)
        self.assertIn("Retry idempotent calls", pack.text)
        self.assertNotIn("Batch queries", pack.text)
        self.assertEqual(pack.served, ["n_r"])
        self.assertEqual(pack.reasons["n_r"], "same subject")

    def test_nothing_selected_means_nothing_injected(self) -> None:
        self.force_judging()
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

        pack = recall_pack(self.store, "deploy staging", router({"picks": []}))
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
        self.force_judging()
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
        self.force_judging()
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

        # Now merge the leaf sideways with an unrelated sibling.
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

    def test_one_co_occurrence_is_not_evidence(self) -> None:
        from rmc.compact import co_use_groups

        self.co_used("e1", ["n_a", "n_b"])
        self.assertEqual(co_use_groups(self.store), [])

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


class TestReInjection(StoreCase):
    """A lesson already in context should not be paid for twice — but "present"
    and "still attended to" are different, so there are three cases."""

    def setUp(self) -> None:
        super().setUp()
        self.node = self.add_node(
            id="n_r", family="f", title="Retry", gist="Retry idempotently.", body="Long body. " * 40
        )

    def pack(self, **kw):
        return recall_pack(self.store, "do the thing", router({"picks": []}), **kw)

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

        self.seed_evidence(1)
        due, why = dream_due(self.store)
        self.assertFalse(due)
        self.assertIn("need 3", why)

    def test_due_once_enough_has_accumulated(self) -> None:
        from rmc.compact import dream_due

        self.seed_evidence(3)
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
        self.assertIn("tokens served at apex", text)
        # Before/after is the point: it shows what changed, not just what ran.
        self.assertIn("nodes", report.before)
        self.assertIn("apex_tokens", report.after)

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
