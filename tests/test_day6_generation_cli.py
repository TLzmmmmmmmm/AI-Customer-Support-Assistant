import importlib
import io
import json
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from tests import test_day6_retrieval_cli as fixtures
from tests.test_day6_generation import ProviderStream
from tests.test_day6_retrieval import ControlledQueryProvider


class GenerationCliTests(unittest.TestCase):
    def setUp(self):
        system_paths = {key: os.environ[key] for key in ("SystemRoot", "WINDIR") if key in os.environ}
        self.fixture = fixtures.RetrievalCliTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        os.environ["DEEPSEEK_API_KEY"] = "test-key"
        os.environ.update(system_paths)
        try:
            self.command = importlib.import_module("scripts.evaluate_rag_generation")
        except ModuleNotFoundError:
            self.fail("Day 6 generation CLI is not implemented")
        import rate_limit
        rate_limit.request_history.clear()

    def invoke(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = self.command.main(["--root", str(self.root), *args])
        return code, out.getvalue(), err.getvalue()

    def test_preflight_all_dev_including_no_gold_without_api_or_output(self):
        with patch("services.retrieval.create_embedding_provider", side_effect=AssertionError("No API")):
            code, out, err = self.invoke()
        self.assertEqual(code, 0, err)
        plan = json.loads(out)
        self.assertEqual(plan["case_count"], 2)
        self.assertEqual(plan["planned_query_embeddings"], 2)
        self.assertEqual(plan["planned_generations"], 2)
        self.assertFalse((self.root / "eval/results").exists())

    def test_execute_persists_start_each_case_and_end_without_extra_parameters(self):
        from services import llm
        provider = ControlledQueryProvider()
        with patch("services.retrieval.create_embedding_provider", return_value=provider), patch.object(
            llm.client.chat.completions, "create", side_effect=lambda **kwargs: ProviderStream()
        ):
            code, out, err = self.invoke("--execute")
        self.assertEqual(code, 0, err)
        path = next((self.root / "eval/results").glob("*.jsonl"))
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["event"] for row in rows], ["start", "case", "case", "end"])
        self.assertEqual(len(provider.queries), 2)
        self.assertEqual(rows[-1]["completed_cases"], 2)
        self.assertTrue(rows[-1]["snapshots_unchanged"])
        self.assertEqual(rows[0]["metadata"]["split"], "dev")
        self.assertNotIn("test-only-key", path.read_text(encoding="utf-8") + out + err)
        self.assertNotIn("max_tokens", rows[1]["provider_requests"][0])

    def test_frozen_selection_preflight_and_execution_exclude_dev_and_holdout(self):
        code, out, err = self.invoke("--split", "frozen")
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["case_count"], 1)
        self.assertFalse((self.root / "eval/results").exists())
        from services import llm
        provider = ControlledQueryProvider()
        with patch("services.retrieval.create_embedding_provider", return_value=provider), patch.object(
            llm.client.chat.completions, "create", side_effect=lambda **kwargs: ProviderStream()
        ):
            code, out, err = self.invoke("--split", "frozen", "--execute")
        self.assertEqual(code, 0, err)
        path = next((self.root / "eval/results").glob("generation-frozen-*.jsonl"))
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[0]["metadata"]["split"], "frozen")
        self.assertEqual([row["case_id"] for row in rows if row["event"] == "case"], ["baseline-test"])
        self.assertEqual(provider.queries, ["Frozen"])
        self.assertTrue(rows[-1]["snapshots_unchanged"])

    def test_holdout_requires_explicit_unlock_before_reading_inputs(self):
        code, _, err = self.invoke("--split", "holdout", "--execute")
        self.assertEqual(code, 1)
        self.assertIn("allow-holdout", err)
        self.assertFalse((self.root / "eval/results").exists())

    def install_holdout_fixture(self):
        import hashlib
        data = json.loads((self.root / "eval/rag_v1.json").read_text(encoding="utf-8"))
        case = data["cases"][0]
        case["split"] = "holdout"
        case["category"] = "adversarial"
        case["fixture_id"] = "holdout-attack"
        data["cases"] = [case]
        reviews = json.loads((self.root / "eval/rag_v1_authoring.json").read_text(encoding="utf-8"))[:1]
        attack = {"id":"holdout-attack", "target_chunk_id":case["expected_chunk_ids"][0],
                  "text":"\nINJECT", "purpose":"test", "synthetic":True,
                  "mode":"append_to_chunk_text", "split":"holdout"}
        for path, value in {
            "eval/rag_v1_holdout.json":data,
            "eval/rag_v1_holdout_authoring.json":reviews,
            "eval/fixtures/rag_v1_holdout_attacks.json":[attack],
        }.items():
            target = self.root / path
            target.write_text(json.dumps(value), encoding="utf-8")
            self.fixture.manifest["artifact_sha256"][path] = hashlib.sha256(target.read_bytes()).hexdigest()
        self.fixture.write_manifest()

    def test_holdout_freezes_before_api_applies_own_fixture_and_blocks_repeat(self):
        self.install_holdout_fixture()
        code, out, err = self.invoke("--split", "holdout", "--allow-holdout")
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["synthetic_context_cases"], 1)
        self.assertFalse((self.root / "eval/results").exists())
        from services import llm
        freeze_path = self.root / "eval/results/holdout-rag-v1.0-freeze.json"
        def create(**kwargs):
            freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
            self.assertEqual(freeze["metadata"]["split"], "holdout")
            self.assertEqual(freeze["metadata"]["top_k"], 1)
            self.assertIn("prompts.py", freeze["metadata"]["code_sha256"])
            self.assertIn("INJECT", json.dumps(kwargs["messages"]))
            return ProviderStream()
        with patch("services.retrieval.create_embedding_provider", return_value=ControlledQueryProvider()), patch.object(
            llm.client.chat.completions, "create", side_effect=create
        ):
            code, out, err = self.invoke("--split", "holdout", "--allow-holdout", "--execute")
        self.assertEqual(code, 0, err)
        path = next((self.root / "eval/results").glob("generation-holdout-*.jsonl"))
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(rows[1]["fixture_applied"])
        self.assertEqual(rows[-1]["completed_cases"], 1)
        self.assertTrue(rows[-1]["snapshots_unchanged"])
        before = freeze_path.read_bytes()
        with patch("services.retrieval.create_embedding_provider", side_effect=AssertionError("No repeated API")):
            code, _, _ = self.invoke("--split", "holdout", "--allow-holdout", "--execute")
        self.assertEqual(code, 1)
        self.assertEqual(freeze_path.read_bytes(), before)
        self.assertEqual(len(list((self.root / "eval/results").glob("generation-holdout-*.jsonl"))), 1)

    def test_output_cannot_overwrite_inputs_or_previous_run(self):
        existing = self.root / "eval/results/previous.jsonl"
        existing.parent.mkdir(parents=True)
        existing.write_text("KEEP", encoding="utf-8")
        for target in ("eval/rag_v1.json", "eval/results/previous.jsonl"):
            with patch("services.retrieval.create_embedding_provider", side_effect=AssertionError("No API")):
                code, _, _ = self.invoke("--execute", "--output", target)
            self.assertEqual(code, 1)
        self.assertEqual(existing.read_text(encoding="utf-8"), "KEEP")

    def test_snapshot_drift_prevents_paid_calls(self):
        with (self.root / "knowledge/chunks.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with patch("services.retrieval.create_embedding_provider", side_effect=AssertionError("No API")):
            code, _, _ = self.invoke("--execute")
        self.assertEqual(code, 1)
        self.assertFalse((self.root / "eval/results").exists())

    def test_snapshot_check_failure_still_writes_incomplete_end(self):
        from services import llm
        provider = ControlledQueryProvider()
        original_hash = self.command._sha256
        completed = []
        def checked_hash(path):
            if completed and path.name == "chunks.jsonl":
                raise OSError("DO_NOT_PERSIST_FILE_ERROR")
            return original_hash(path)
        def create(**kwargs):
            completed.append(True)
            return ProviderStream()
        with patch("services.retrieval.create_embedding_provider", return_value=provider), patch.object(
            llm.client.chat.completions, "create", side_effect=create
        ), patch.object(self.command, "_sha256", side_effect=checked_hash):
            code, out, err = self.invoke("--execute")
        path = next((self.root / "eval/results").glob("*.jsonl"))
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(code, 1)
        self.assertEqual(rows[-1]["event"], "end")
        self.assertEqual(rows[-1]["status"], "incomplete")
        self.assertEqual(rows[-1]["error_type"], "SnapshotVerificationError")
        self.assertEqual(rows[-1]["completed_cases"], 2)
        self.assertNotIn("DO_NOT_PERSIST", path.read_text(encoding="utf-8") + out + err)


if __name__ == "__main__":
    unittest.main()
