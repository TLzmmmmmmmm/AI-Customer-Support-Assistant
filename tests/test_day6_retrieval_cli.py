import hashlib
import importlib
import io
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from knowledge_pipeline.retrieval.models import VectorRecord
from tests.test_day6_retrieval import ControlledQueryProvider, evaluation_case
from tests.test_rag_dataset import sample_review
from tests.test_retrieval_evaluation import chunk


class RetrievalCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {
            "EMBEDDING_DIMENSIONS": "2", "DASHSCOPE_API_KEY": "test-only-key",
            "DASHSCOPE_WORKSPACE_ID": "test-only-workspace", "RETRIEVAL_TOP_K": "1",
        }, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        source = chunk("alpha")
        known = evaluation_case("dev-known", "alpha 参数", [source])
        unknown = evaluation_case("dev-unknown", "price?", [], expected_behavior="abstain")
        frozen = evaluation_case("baseline-test", "Frozen", [source], split="frozen")
        record = VectorRecord.model_validate({
            **source.model_dump(mode="json", exclude={"parent_document_hash"}),
            "embedding_provider": "dashscope", "embedding_model": "qwen3.7-text-embedding",
            "embedding_dimensions": 2, "embedding_text_type": "document", "embedding": [1.0, 0.0],
        })
        reviews = []
        for case in (known, unknown, frozen):
            review = sample_review(case.id, evidence=False)
            if case.expected_chunk_ids:
                review["evidence"] = [{"chunk_id": source.chunk_id, "quote": "产品说明"}]
            else:
                review["unavailable_reason"] = "No price in curated source."
            reviews.append(review)
        artifacts = {
            "eval/rag_v1.json": {"schema_version": "1.0", "dataset_version": "rag-v1.0",
                                 "cases": [c.model_dump() for c in (known, unknown, frozen)]},
            "eval/rag_v1_authoring.json": reviews,
            "eval/fixtures/rag_v1_attacks.json": [],
        }
        protected = {
            "knowledge/documents.jsonl": {"document_id": source.parent_document_id, "text": source.text},
            "knowledge/chunks.jsonl": source.model_dump(mode="json"),
            "knowledge/vector_records.jsonl": record.model_dump(mode="json"),
        }
        self.manifest = dict(schema_version="1.0", dataset_version="rag-v1.0",
                             artifact_sha256={}, protected_sha256={}, system_rules={})
        for path, value in {**artifacts, **protected}.items():
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
            group = "artifact_sha256" if path in artifacts else "protected_sha256"
            self.manifest[group][path] = hashlib.sha256(target.read_bytes()).hexdigest()
        self.write_manifest()
        # If a dev run accidentally reads holdout it must fail, rather than pass unnoticed.
        (self.root / "eval/rag_v1_holdout.json").write_text("NOT JSON", encoding="utf-8")

    def write_manifest(self):
        (self.root / "eval/rag_v1_manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def invoke(self, *args):
        try:
            command = importlib.import_module("scripts.evaluate_rag_retrieval")
        except ModuleNotFoundError:
            self.fail("Day 6 retrieval CLI is not implemented")
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = command.main(["--root", str(self.root), *args])
        return code, out.getvalue(), err.getvalue()

    def test_preflight_defaults_to_dev_without_provider_or_output(self):
        with patch("services.retrieval.create_embedding_provider", side_effect=AssertionError("API must not be built")):
            code, out, err = self.invoke()
        self.assertEqual(code, 0, err)
        plan = json.loads(out)
        self.assertEqual(plan["mode"], "preflight")
        self.assertEqual(plan["selected_cases"], 2)
        self.assertEqual(plan["query_embedding_calls"], 1)
        self.assertEqual(plan["excluded_no_gold_cases"], 1)
        self.assertEqual(plan["top_k"], 1)
        self.assertFalse((self.root / "eval/results").exists())

    def test_preflight_uses_explicit_manifest_instead_of_day6_default(self):
        custom_review = self.root / "eval/rag_v1_1_regression_authoring.json"
        custom_review.write_bytes((self.root / "eval/rag_v1_authoring.json").read_bytes())
        alternate_manifest = json.loads(json.dumps(self.manifest))
        alternate_manifest["artifact_sha256"]["eval/rag_v1_1_regression_authoring.json"] = hashlib.sha256(
            custom_review.read_bytes()
        ).hexdigest()
        alternate_manifest["split_artifacts"] = {
            "dev": {
                "dataset": "eval/rag_v1.json",
                "review": "eval/rag_v1_1_regression_authoring.json",
                "fixtures": "eval/fixtures/rag_v1_attacks.json",
            }
        }
        alternate = self.root / "eval/rag_v1_1_regression_manifest.json"
        alternate.write_text(json.dumps(alternate_manifest), encoding="utf-8")
        (self.root / "eval/rag_v1_authoring.json").write_text("NOT JSON", encoding="utf-8")
        default = self.root / "eval/rag_v1_manifest.json"
        default.write_text(json.dumps({**self.manifest, "schema_version": "invalid"}), encoding="utf-8")
        with patch("services.retrieval.create_embedding_provider", side_effect=AssertionError("API must not be built")):
            code, out, err = self.invoke("--manifest", "eval/rag_v1_1_regression_manifest.json")
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["manifest"], "eval/rag_v1_1_regression_manifest.json")

    def test_preflight_accepts_v1_1_candidate_manifest(self):
        dataset_path = self.root / "eval/rag_v1.json"
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        dataset["dataset_version"] = "rag-v1.1"
        dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
        candidate = json.loads(json.dumps(self.manifest))
        candidate["dataset_version"] = "rag-v1.1"
        candidate["candidate_snapshot_version"] = "rag-v1.1"
        candidate["artifact_sha256"]["eval/rag_v1.json"] = hashlib.sha256(
            dataset_path.read_bytes()
        ).hexdigest()
        candidate_path = self.root / "eval/rag_v1_1_candidate_manifest.json"
        candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
        with patch("services.retrieval.create_embedding_provider", side_effect=AssertionError("No API")):
            code, out, err = self.invoke("--manifest", "eval/rag_v1_1_candidate_manifest.json")
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["manifest"], "eval/rag_v1_1_candidate_manifest.json")

    def test_execute_uses_real_service_and_writes_only_controlled_results(self):
        provider = ControlledQueryProvider()
        with patch("services.retrieval.create_embedding_provider", return_value=provider):
            code, out, err = self.invoke("--execute", "--top-k", "2")
        self.assertEqual(code, 0, err)
        files = list((self.root / "eval/results").glob("*.json"))
        self.assertEqual(len(files), 1)
        report = json.loads(files[0].read_text(encoding="utf-8"))
        self.assertEqual(provider.queries, ["alpha 参数"])
        self.assertEqual(report["top_k"], 2)
        self.assertEqual(report["metadata"]["split"], "dev")
        self.assertEqual(report["metadata"]["embedding_dimensions"], 2)
        self.assertEqual(report["cases"][0]["hits"][0]["match_origin"], "exact_entity")
        self.assertGreaterEqual(report["cases"][0]["retrieval_latency_seconds"], 0)
        self.assertEqual(report["summary"]["excluded_no_gold_cases"], 1)
        self.assertNotIn("test-only-key", files[0].read_text(encoding="utf-8") + out + err)

    def test_holdout_requires_explicit_unlock_even_for_preflight(self):
        code, out, err = self.invoke("--split", "holdout")
        self.assertEqual(code, 1)
        self.assertIn("allow-holdout", err)
        self.assertFalse((self.root / "eval/results").exists())

    def test_output_cannot_overwrite_input_or_existing_result(self):
        existing = self.root / "eval/results/existing.json"
        existing.parent.mkdir(parents=True)
        existing.write_text("KEEP", encoding="utf-8")
        for output in ("eval/rag_v1.json", "eval/results/existing.json"):
            with self.subTest(output=output), patch("services.retrieval.create_embedding_provider", side_effect=AssertionError("No API")):
                code, _, _ = self.invoke("--execute", "--output", output)
                self.assertEqual(code, 1)
        self.assertEqual(existing.read_text(encoding="utf-8"), "KEEP")

    def test_snapshot_drift_stops_before_paid_api(self):
        with (self.root / "knowledge/chunks.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with patch("services.retrieval.create_embedding_provider", side_effect=AssertionError("No API")):
            code, _, err = self.invoke("--execute")
        self.assertEqual(code, 1)
        self.assertIn("snapshot", err)

    def test_invalid_top_k_is_rejected_before_paid_api(self):
        code, _, err = self.invoke("--execute", "--top-k", "0")
        self.assertEqual(code, 1)
        self.assertIn("top_k", err)

    def test_failed_api_run_is_saved_as_incomplete_and_returns_nonzero(self):
        path = self.root / "eval/rag_v1.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["cases"][0]["question"] = "provider failure"
        path.write_text(json.dumps(data), encoding="utf-8")
        self.manifest["artifact_sha256"]["eval/rag_v1.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_manifest()
        with patch("services.retrieval.create_embedding_provider", return_value=ControlledQueryProvider()):
            code, out, err = self.invoke("--execute")
        self.assertEqual(code, 1)
        report_file = next((self.root / "eval/results").glob("*.json"))
        report = json.loads(report_file.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["summary"]["error_cases"], 1)
        self.assertIsNone(report["summary"]["hit_at_k"])
        self.assertNotIn("DO_NOT_PERSIST", report_file.read_text(encoding="utf-8") + out + err)

    def test_malformed_response_preserves_successes_before_and_after_error(self):
        path = self.root / "eval/rag_v1.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        known = data["cases"][0]
        data["cases"] = [{**known, "id": f"dev-{i}", "question": f"alpha 参数 {i}"} for i in range(3)]
        review_path = self.root / "eval/rag_v1_authoring.json"
        review = json.loads(review_path.read_text(encoding="utf-8"))[0]
        reviews = [{**review, "case_id": f"dev-{i}"} for i in range(3)]
        for target, value in ((path, data), (review_path, reviews)):
            target.write_text(json.dumps(value), encoding="utf-8")
            self.manifest["artifact_sha256"][target.relative_to(self.root).as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
        self.write_manifest()
        valid = SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0, 0.0])])
        with patch("knowledge_pipeline.retrieval.embedding.OpenAI") as client:
            client.return_value.embeddings.create.side_effect = [valid, SimpleNamespace(data=None), valid]
            code, out, err = self.invoke("--execute")
            self.assertEqual(client.return_value.embeddings.create.call_count, 3, err)
        self.assertEqual(code, 1, err)
        report_file = next((self.root / "eval/results").glob("*.json"))
        report = json.loads(report_file.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual([row["status"] for row in report["cases"]], ["evaluated", "error", "evaluated"])
        self.assertEqual(report["cases"][1]["error_type"], "TypeError")
        self.assertIsNone(report["cases"][1]["metrics"]["hit_at_k"])
        self.assertEqual(report["summary"]["scored_cases"], 2)
        self.assertEqual(report["summary"]["error_cases"], 1)
        self.assertEqual(report["summary"]["hit_at_k"], 1.0)
        self.assertEqual(report["summary"]["recall_at_k"], 1.0)
        self.assertNotIn("not iterable", report_file.read_text(encoding="utf-8") + out + err)


if __name__ == "__main__":
    unittest.main()
