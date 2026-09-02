import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from knowledge_pipeline.chunking import compute_chunk_content_hash
from knowledge_pipeline.models import KnowledgeChunk
from knowledge_pipeline.retrieval.evaluation import (
    evaluate_retrieval,
    load_evaluation_suite,
    validate_suite_against_chunks,
)
from knowledge_pipeline.retrieval.models import (
    EmbeddingBatch,
    RetrievalEvaluationCase,
    RetrievalEvaluationSuite,
    RetrievalResult,
    VectorRecord,
)
from knowledge_pipeline.retrieval.records import (
    load_chunks,
    serialize_vector_records,
)


ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "eval" / "retrieval_v1_1.json"
HISTORICAL_EVAL_PATH = ROOT / "eval" / "retrieval_v1.json"
HISTORICAL_RESULT_PATH = ROOT / "eval" / "retrieval_v1_results.json"
CHUNKS_PATH = ROOT / "knowledge" / "chunks.jsonl"


def chunk(entity_id: str) -> KnowledgeChunk:
    section = f"产品 {entity_id}"
    text = f"# {entity_id}\n\n产品说明"
    metadata = {
        "product_id": entity_id,
        "slug": entity_id,
        "category_id": "two-way-radio",
        "category_name": "对讲机通信",
    }
    return KnowledgeChunk.model_validate({
        "schema_version": "1.0",
        "chunk_id": f"product:{entity_id}:content",
        "parent_document_id": f"product:{entity_id}",
        "parent_document_hash": "d" * 64,
        "type": "product",
        "section": section,
        "text": text,
        "language": "zh-CN",
        "content_hash": compute_chunk_content_hash(
            "product", section, text, "zh-CN", metadata
        ),
        "source_url": f"https://example.com/{entity_id}/",
        "source_files": [f"src/content/products/{entity_id}.json"],
        "metadata": metadata,
    })


def result(source: KnowledgeChunk, rank: int, *, entity: bool = False):
    return RetrievalResult.model_validate({
        "rank": rank,
        "score": 1.0 / rank,
        "match_origin": "exact_entity" if entity else "dense",
        "matched_entity_ids": (
            [source.parent_document_id] if entity else []
        ),
        "chunk_id": source.chunk_id,
        "parent_document_id": source.parent_document_id,
        "type": source.type,
        "section": source.section,
        "text": source.text,
        "content_hash": source.content_hash,
        "metadata": source.metadata,
        "source_url": source.source_url,
        "source_files": source.source_files,
    })


def case(
    case_id: str,
    query: str,
    expected: list[KnowledgeChunk],
    *,
    entities: list[str] | None = None,
) -> RetrievalEvaluationCase:
    return RetrievalEvaluationCase.model_validate({
        "id": case_id,
        "source_case_id": case_id,
        "category": "test",
        "query": query,
        "expected_chunk_ids": [item.chunk_id for item in expected],
        "expected_parent_document_ids": list(dict.fromkeys(
            item.parent_document_id for item in expected
        )),
        "expected_entity_ids": entities or [],
        "match_requirement": "all" if len(expected) > 1 else "any",
        "relevance_groups": [],
    })


class ScriptedRetriever:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query

    def retrieve(self, query, top_k=None):
        return self.results_by_query[query][:top_k]


class FakeProvider:
    def __init__(self):
        self.query_calls = []

    def embed_query(self, text):
        self.query_calls.append(text)
        return EmbeddingBatch(vectors=([1.0, 0.0],), input_tokens=3)


class RetrievalEvaluationFixtureTests(unittest.TestCase):
    def test_historical_v1_suite_and_result_are_preserved(self):
        self.assertTrue(HISTORICAL_EVAL_PATH.is_file())
        self.assertTrue(HISTORICAL_RESULT_PATH.is_file())

    def test_suite_contains_exactly_the_approved_baseline_cases(self):
        suite = load_evaluation_suite(EVAL_PATH)

        self.assertEqual(len(suite.cases), 18)
        self.assertEqual(
            {item.source_case_id for item in suite.cases},
            {f"baseline-{index:03d}" for index in range(1, 18)}
            | {"baseline-019"},
        )

    def test_every_expected_chunk_and_parent_exists(self):
        validate_suite_against_chunks(
            load_evaluation_suite(EVAL_PATH),
            load_chunks(CHUNKS_PATH),
        )

    def test_suite_pins_exact_chunk_artifact_bytes(self):
        suite = load_evaluation_suite(EVAL_PATH)
        digest = hashlib.sha256(CHUNKS_PATH.read_bytes()).hexdigest()

        self.assertEqual(suite.chunk_snapshot_sha256, digest)

    def test_catalog_overview_case_uses_the_authoritative_summary_chunk(self):
        suite = load_evaluation_suite(EVAL_PATH)
        catalog_case = next(
            item for item in suite.cases
            if item.source_case_id == "baseline-001"
        )

        self.assertEqual(
            catalog_case.expected_chunk_ids,
            ["catalog:products:overview"],
        )
        self.assertEqual(
            catalog_case.expected_parent_document_ids,
            ["catalog:products"],
        )
        self.assertEqual(catalog_case.match_requirement, "any")
        self.assertEqual(catalog_case.relevance_groups, [])


class RetrievalEvaluationMetricTests(unittest.TestCase):
    def test_metrics_include_hit_mrr_recall_and_entity_accuracy(self):
        alpha, beta, noise = chunk("alpha"), chunk("beta"), chunk("noise")
        cases = [
            case(
                "case-1",
                "first",
                [alpha],
                entities=[alpha.parent_document_id],
            ),
            case(
                "case-2",
                "second",
                [beta],
                entities=[beta.parent_document_id],
            ),
        ]
        suite = RetrievalEvaluationSuite.model_validate({
            "schema_version": "1.0",
            "suite_id": "fixture",
            "chunk_snapshot_sha256": "e" * 64,
            "default_top_k": 5,
            "cases": [item.model_dump(mode="json") for item in cases],
        })
        retriever = ScriptedRetriever({
            "first": [result(alpha, 1, entity=True), result(noise, 2)],
            "second": [result(noise, 1), result(beta, 2, entity=True)],
        })

        evaluated = evaluate_retrieval(
            suite,
            retriever,
            [alpha, beta, noise],
            top_k=5,
        )

        self.assertEqual(evaluated.metrics.hit_at_1, 0.5)
        self.assertEqual(evaluated.metrics.hit_at_5, 1.0)
        self.assertEqual(evaluated.metrics.mean_reciprocal_rank, 0.75)
        self.assertEqual(evaluated.metrics.expected_chunk_recall_at_5, 1.0)
        self.assertEqual(evaluated.metrics.expected_parent_recall_at_5, 1.0)
        self.assertEqual(evaluated.metrics.entity_accuracy, 1.0)

    def test_important_failures_are_classified(self):
        alpha, beta, noise = chunk("alpha"), chunk("beta"), chunk("noise")
        cases = [
            case("missing", "missing", [alpha]),
            case("multi", "multi", [alpha, beta]),
            case(
                "entity",
                "entity",
                [alpha],
                entities=[alpha.parent_document_id],
            ),
            case("late", "late", [alpha]),
        ]
        suite = RetrievalEvaluationSuite.model_validate({
            "schema_version": "1.0",
            "suite_id": "failing-fixture",
            "chunk_snapshot_sha256": "f" * 64,
            "default_top_k": 5,
            "cases": [item.model_dump(mode="json") for item in cases],
        })
        retriever = ScriptedRetriever({
            "missing": [result(noise, 1)],
            "multi": [result(alpha, 1)],
            "entity": [result(alpha, 1)],
            "late": [
                result(noise, 1),
                result(noise, 2),
                result(noise, 3),
                result(alpha, 4),
            ],
        })

        failures = evaluate_retrieval(
            suite,
            retriever,
            [alpha, beta, noise],
        ).failures

        self.assertEqual(
            {failure.kind for failure in failures},
            {
                "missing_top_k",
                "incomplete_multi_source",
                "missed_entity",
                "late_rank",
            },
        )


class RetrievalEvaluationCliTests(unittest.TestCase):
    def test_cli_defaults_to_latest_v1_1_suite_and_result(self):
        from scripts import evaluate_retrieval

        args = evaluate_retrieval.parse_args([])

        self.assertEqual(args.suite.name, "retrieval_v1_1.json")
        self.assertEqual(args.output.name, "retrieval_v1_1_results.json")

    def test_plan_only_estimates_queries_without_provider_or_vectors(self):
        from scripts import evaluate_retrieval

        stdout = io.StringIO()
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            missing_vectors = Path(directory) / "missing.jsonl"
            with patch.dict(os.environ, {}, clear=True):
                with patch.object(
                    evaluate_retrieval,
                    "create_embedding_provider",
                ) as factory:
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        code = evaluate_retrieval.main([
                            "--suite", str(EVAL_PATH),
                            "--chunks", str(CHUNKS_PATH),
                            "--vectors", str(missing_vectors),
                        ])

        self.assertEqual(code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertIn("Mode: plan only", stdout.getvalue())
        self.assertIn("Queries: 18", stdout.getvalue())
        self.assertIn("Estimated query cost (CNY):", stdout.getvalue())
        factory.assert_not_called()

    def test_execute_evaluates_and_writes_deterministic_result(self):
        from scripts import evaluate_retrieval

        provider = FakeProvider()
        stdout = io.StringIO()
        stderr = io.StringIO()
        environment = {
            "EMBEDDING_DIMENSIONS": "2",
            "DASHSCOPE_API_KEY": "test-key",
            "DASHSCOPE_WORKSPACE_ID": "test-workspace",
        }
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = chunk("alpha")
            chunks_path = directory / "chunks.jsonl"
            chunks_path.write_text(
                json.dumps(
                    source.model_dump(mode="json"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            suite = RetrievalEvaluationSuite.model_validate({
                "schema_version": "1.0",
                "suite_id": "cli-fixture",
                "chunk_snapshot_sha256": hashlib.sha256(
                    chunks_path.read_bytes()
                ).hexdigest(),
                "default_top_k": 5,
                "cases": [case("cli-case", "Alpha", [source]).model_dump(
                    mode="json"
                )],
            })
            suite_path = directory / "suite.json"
            suite_path.write_text(
                json.dumps(suite.model_dump(mode="json")),
                encoding="utf-8",
            )
            vector = VectorRecord.model_validate({
                "schema_version": "1.0",
                **source.model_dump(
                    mode="json", exclude={"parent_document_hash"}
                ),
                "embedding_provider": "dashscope",
                "embedding_model": "qwen3.7-text-embedding",
                "embedding_dimensions": 2,
                "embedding_text_type": "document",
                "embedding": [1.0, 0.0],
            })
            vectors_path = directory / "vectors.jsonl"
            vectors_path.write_bytes(serialize_vector_records([vector]))
            output_path = directory / "result.json"

            with patch.dict(os.environ, environment, clear=True):
                with patch.object(
                    evaluate_retrieval,
                    "create_embedding_provider",
                    return_value=provider,
                ):
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        code = evaluate_retrieval.main([
                            "--suite", str(suite_path),
                            "--chunks", str(chunks_path),
                            "--vectors", str(vectors_path),
                            "--output", str(output_path),
                            "--execute",
                        ])
            result_bytes = output_path.read_bytes()

        self.assertEqual(code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(provider.query_calls, ["Alpha"])
        self.assertIn("Hit@5: 1.000000", stdout.getvalue())
        self.assertTrue(result_bytes.endswith(b"\n"))
        self.assertNotIn(b"embedding", result_bytes)


if __name__ == "__main__":
    unittest.main()
