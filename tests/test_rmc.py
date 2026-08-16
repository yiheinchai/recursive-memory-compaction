"""End-to-end tests, run with: python3 -m unittest discover -s tests

These lean on ``MockAdapter``'s simulated knowledge world: a task is solved iff
every ``@fact`` it requires appears in the lesson text supplied. That makes the
whole cycle — compress, fail, diagnose, descend, rescue — deterministic and
free, so the control flow is genuinely verified rather than asserted.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rmc import yamlish  # noqa: E402
from rmc.adapters.mock import MockAdapter, MockWorld  # noqa: E402
from rmc.compact import compress_node, due_nodes, repair  # noqa: E402
from rmc.node import Delta, Node  # noqa: E402
from rmc.recall import match_families, recall_pack, solve_with_descent  # noqa: E402
from rmc.redact import redact  # noqa: E402
from rmc.selection import Diagnosis, build_candidates, rank  # noqa: E402
from rmc.signals import SessionFacts, classify  # noqa: E402
from rmc.store import Episode, Store  # noqa: E402


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

    def add_episode(self, ident, family, prompt, *, outcome="success", served=()) -> Episode:
        ep = Episode(
            id=ident,
            family=family,
            prompt=prompt,
            outcome=outcome,
            confidence=0.9,
            served=list(served),
            accepted_summary="done correctly",
        )
        self.store.save_episode(ep)
        return ep


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
        )
        path = self.store.save_node(node)
        loaded = Node.from_markdown(path.read_text(), path)
        self.assertEqual(loaded.id, "n_abc")
        self.assertEqual(loaded.level, 2)
        self.assertEqual(loaded.derived_from, ["n_x"])
        self.assertEqual(loaded.dropped[0].kind, "parameter")
        self.assertEqual(loaded.dropped[0].holder, "n_x")
        self.assertIn("@backoff-constants", loaded.body)

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


class TestSignals(unittest.TestCase):
    def test_correction_reads_as_failure(self) -> None:
        facts = SessionFacts(
            user_messages=["add retry logic", "no, don't use a fixed delay"],
            assistant_messages=["done"],
            tool_calls=12,
        )
        outcome = classify(facts)
        self.assertEqual(outcome.label, "failure")
        self.assertGreaterEqual(outcome.confidence, 0.5)

    def test_approval_reads_as_success(self) -> None:
        facts = SessionFacts(
            user_messages=["add retry logic", "perfect, thanks"],
            assistant_messages=["done"],
            tool_calls=12,
        )
        self.assertEqual(classify(facts).label, "success")

    def test_small_session_is_unknown(self) -> None:
        facts = SessionFacts(user_messages=["hi"], assistant_messages=["hello"], tool_calls=1)
        self.assertEqual(classify(facts).label, "unknown")

    def test_red_then_green_is_not_a_failure(self) -> None:
        facts = SessionFacts(
            user_messages=["fix the build"],
            assistant_messages=["fixed"],
            tool_outputs=["2 failed", "12 passed"],
            tool_calls=15,
        )
        self.assertEqual(classify(facts).label, "success")


class TestSelection(StoreCase):
    def test_diagnosis_kind_drives_the_ranking(self) -> None:
        holder_a = self.add_node(id="n_a", family="retry", body="A", level=0)
        holder_b = self.add_node(id="n_b", family="retry", body="B", level=0)
        apex = self.add_node(
            id="n_apex",
            family="retry",
            body="Retry idempotent operations.",
            level=1,
            derived_from=["n_a", "n_b"],
            dropped=[
                Delta("worked example for the S3 client", "example", holder_a.id),
                Delta("backoff constants are 100ms, 400ms, 1.6s", "parameter", holder_b.id),
            ],
        )
        diag = Diagnosis(category="parameter", missing=["the backoff constants"], confidence=0.9)
        ranked = rank(
            build_candidates(apex, resolve=self.store.get, strategy="delta-patch"),
            diag=diag,
            task_sig={"retry", "backoff"},
            config=self.store.config,
        )
        self.assertEqual(ranked[0].kind, "delta")
        self.assertIn("backoff constants", ranked[0].text)

    def test_children_offered_when_manifest_is_empty(self) -> None:
        self.add_node(id="n_c", family="f", body="detail", level=0)
        apex = self.add_node(id="n_p", family="f", body="abstract", level=1, derived_from=["n_c"])
        cands = build_candidates(apex, resolve=self.store.get, strategy="delta-patch")
        self.assertTrue(any(c.kind == "node" and c.node.id == "n_c" for c in cands))

    def test_cost_breaks_ties_toward_cheaper(self) -> None:
        apex = self.add_node(
            id="n_p2",
            family="f",
            body="abstract",
            level=1,
            dropped=[
                Delta("short claim", "parameter", None),
                Delta("a very much longer claim " * 40, "parameter", None),
            ],
        )
        diag = Diagnosis(category="parameter", missing=["claim"], confidence=0.5)
        ranked = rank(
            build_candidates(apex, resolve=self.store.get, strategy="delta-patch"),
            diag=diag,
            task_sig=set(),
            config=self.store.config,
        )
        self.assertEqual(ranked[0].text, "short claim")


class TestRecall(StoreCase):
    def test_family_matching_and_pack_budget(self) -> None:
        self.add_node(
            id="n_r",
            family="retry",
            title="Retry",
            body="Retry idempotent HTTP requests with jittered exponential backoff.",
            level=2,
            tags=["retry", "http"],
        )
        self.add_node(
            id="n_g",
            family="graphql",
            title="GraphQL",
            body="Batch GraphQL queries through the dataloader.",
            level=1,
            tags=["graphql"],
        )
        matches = match_families(self.store, "the http client needs retry with backoff")
        self.assertTrue(matches)
        self.assertEqual(matches[0][0], "retry")

        pack = recall_pack(self.store, "the http client needs retry with backoff")
        self.assertIn("jittered exponential backoff", pack.text)
        self.assertIn("n_r", pack.served)

    def test_unrelated_prompt_matches_nothing(self) -> None:
        self.add_node(id="n_r2", family="retry", body="Retry idempotent HTTP requests.", tags=["retry"])
        self.assertEqual(match_families(self.store, "what colour should the logo be"), [])


class TestCompaction(StoreCase):
    """The full cycle against the simulated knowledge world."""

    def build_family(self) -> Node:
        # Deliberately verbose, the way a freshly-minted L0 lesson actually is,
        # with the wordiest content in the edge case the compressor will drop.
        # A terse fixture would fail the reduction gate before any replay runs.
        body = (
            "When calling flaky remote services, follow these rules carefully.\n"
            "- Retry only idempotent operations; a non-idempotent write needs a "
            "dedupe key established before the first attempt. @idempotent\n"
            "- Use jittered exponential backoff rather than a fixed delay, so that "
            "retries from many clients do not synchronise. @backoff\n"
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

    def test_compression_accepted_when_regression_set_still_passes(self) -> None:
        node = self.build_family()
        adapter = MockAdapter(world=self.world())
        result = compress_node(self.store, adapter, node)

        self.assertTrue(result.accepted, result.reason)
        self.assertLess(result.after_tokens, result.before_tokens)
        # The mock drops the trailing @s3-body line, which neither episode needs.
        self.assertTrue(any("@s3-body" in d.claim for d in result.dropped))
        self.assertEqual(result.pass_rate, 1.0)

        new = result.new_node
        self.assertEqual(new.level, 1)
        self.assertEqual(new.derived_from, ["n_base"])
        self.assertEqual(self.store.get("n_base").compressed_into, new.id)
        self.assertEqual(self.store.apex("retry").id, new.id)

    def test_compression_rejected_when_it_drops_a_needed_fact(self) -> None:
        node = self.build_family()
        # A world where every episode needs the fact the compressor wants to cut.
        world = MockWorld({"e1": {"idempotent", "s3-body"}, "e2": {"idempotent", "s3-body"}})
        adapter = MockAdapter(world=world)
        result = compress_node(self.store, adapter, node)

        self.assertFalse(result.accepted)
        self.assertEqual(result.pass_rate, 0.0)
        self.assertIsNone(self.store.get("n_base").compressed_into)
        # The rejection must leave hints behind so the next attempt converges.
        self.assertTrue(self.store.get("n_base").preserve)

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
    """Compress away a fact, then prove descent finds it again."""

    def test_delta_patch_rescues_a_dropped_fact(self) -> None:
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
            dropped=[Delta("S3 returns 200 with error bodies. @s3-body", "parameter", base.id)],
        )
        base.compressed_into = apex.id
        self.store.save_node(base)
        self.store.invalidate()

        world = MockWorld({"t_s3": {"idempotent", "s3-body"}})
        adapter = MockAdapter(world=world)

        def verify(run, pack):
            ok, missing = world.solves("t_s3", pack)
            return ok, f"missing {sorted(missing)}"

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
        # The apex alone must have failed first, or the test proves nothing.
        self.assertGreaterEqual(len(result.attempts), 2)
        self.assertFalse(result.attempts[0].ok)

    def test_escalates_to_level_zero_when_no_delta_helps(self) -> None:
        base = self.add_node(
            id="n_e0",
            family="f",
            body="Full lesson. @a @b",
            level=0,
        )
        apex = self.add_node(
            id="n_e1",
            family="f",
            body="Short lesson. @a",
            level=1,
            derived_from=[base.id],
            dropped=[],  # nothing declared: descent has to fall back
        )
        base.compressed_into = apex.id
        self.store.save_node(base)
        self.store.invalidate()

        world = MockWorld({"t": {"a", "b"}})
        adapter = MockAdapter(world=world)

        def verify(run, pack):
            ok, missing = world.solves("t", pack)
            return ok, f"missing {sorted(missing)}"

        result = solve_with_descent(
            self.store,
            adapter=adapter,
            task_id="t",
            task="do the thing",
            family="f",
            verify=verify,
        )
        self.assertTrue(result.ok)
        self.assertIn("@b", result.final_pack)


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


class TestObserve(StoreCase):
    def test_success_updates_stats_and_files_an_episode(self) -> None:
        from rmc.reflect import observe

        node = self.add_node(id="n_o", family="retry", body="Retry stuff.", level=1)
        facts = SessionFacts(
            user_messages=["add retry logic", "perfect, that works"],
            assistant_messages=["added retry with backoff"],
            tool_calls=14,
        )
        result = observe(self.store, facts, session_id="s1", served=[node.id])

        self.assertEqual(result.outcome.label, "success")
        self.assertEqual(self.store.get("n_o").stats.successes, 1)
        self.assertIsNotNone(result.episode)
        self.assertEqual(result.episode.outcome, "success")

    def test_failure_matches_the_correction_against_the_manifest(self) -> None:
        from rmc.reflect import observe

        node = self.add_node(
            id="n_o2",
            family="retry",
            body="Retry stuff.",
            level=1,
            dropped=[Delta("the backoff must be jittered, not fixed", "parameter", None)],
        )
        facts = SessionFacts(
            user_messages=["add retry", "no, don't use a fixed backoff delay, jitter it"],
            assistant_messages=["added retry"],
            tool_calls=12,
        )
        result = observe(self.store, facts, session_id="s2", served=[node.id])

        self.assertEqual(result.outcome.label, "failure")
        self.assertTrue(result.rescues, "correction should have matched a dropped claim")
        self.assertIn("jitter", result.rescues[0][1])

    def test_corrected_but_successful_session_counts_against_the_lesson(self) -> None:
        """A session the human had to steer is a success for the episode but a
        failure for whichever lesson should have prevented the mistake."""
        from rmc.reflect import observe

        node = self.add_node(
            id="n_o4",
            family="deploy",
            body="Deploy with kubectl apply.",
            level=1,
            dropped=[Delta("use the argo rollouts plugin, not raw kubectl", "procedure-step", None)],
        )
        facts = SessionFacts(
            user_messages=[
                "deploy staging",
                "no, don't use raw kubectl, use the argo rollouts plugin",
                "perfect, that works",
            ],
            assistant_messages=["deployed"],
            tool_outputs=["12 passed"],
            tool_calls=14,
        )
        result = observe(self.store, facts, session_id="s4", served=[node.id])

        self.assertEqual(result.outcome.label, "success")  # session ended well
        reloaded = self.store.get("n_o4")
        self.assertEqual(reloaded.stats.failures, 1)  # but the lesson did not hold
        self.assertEqual(reloaded.stats.successes, 0)
        self.assertTrue(result.rescues)
        self.assertEqual(result.episode.outcome, "success")

    def test_exact_threshold_is_not_lost_to_float_error(self) -> None:
        facts = SessionFacts(
            user_messages=["do it", "no, that's wrong", "perfect, that works"],
            assistant_messages=["done"],
            tool_outputs=["12 passed"],
            tool_calls=4,
        )
        # -0.65 (correction) + 0.6 (approval) + 0.35 (tests) == exactly 0.30
        self.assertEqual(classify(facts).label, "success")

    def test_low_confidence_changes_nothing(self) -> None:
        from rmc.reflect import observe

        node = self.add_node(id="n_o3", family="f", body="x", level=0)
        facts = SessionFacts(user_messages=["hi"], assistant_messages=["hello"], tool_calls=1)
        observe(self.store, facts, session_id="s3", served=[node.id])
        self.assertEqual(self.store.get("n_o3").stats.attempts, 0)


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
