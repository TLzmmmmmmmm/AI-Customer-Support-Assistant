import io
import json
import unittest
from contextlib import redirect_stdout

from evaluation.dataset import EvaluationCase
from evaluation.retrieval import score_hits, run_retrieval_evaluation
from knowledge_pipeline.retrieval import ExactEntityResolver, NumpyExactVectorIndex, Retriever
from knowledge_pipeline.retrieval.models import EmbeddingAPIError, EmbeddingBatch, VectorRecord
from tests.test_retrieval_evaluation import chunk, result


def evaluation_case(identity, question, expected, **changes):
    values = dict(id=identity, category="product_spec", split="dev", question=question,
                  expected_behavior="answer", expected_answer="Curated test fact",
                  expected_document_ids=list(dict.fromkeys(c.parent_document_id for c in expected)),
                  expected_chunk_ids=[c.chunk_id for c in expected], expected_system_rule_ids=[])
    values.update(changes)
    return EvaluationCase.model_validate(values)


class ControlledQueryProvider:
    """Only the external embedding boundary is replaced in integration tests."""

    def __init__(self):
        self.queries = []

    def embed_query(self, query):
        self.queries.append(query)
        if query == "provider failure":
            raise EmbeddingAPIError("DO_NOT_PERSIST_SECRET_OR_PROVIDER_BODY")
        return EmbeddingBatch(vectors=([1.0, 0.0],), input_tokens=1)


def real_retriever(provider):
    records = []
    for name, embedding in (("alpha", [0.0, 1.0]), ("noise", [1.0, 0.0]),
                            ("beta", [-1.0, 0.0])):
        source = chunk(name)
        records.append(VectorRecord.model_validate({
            **source.model_dump(mode="json", exclude={"parent_document_hash"}),
            "embedding_provider": "dashscope", "embedding_model": "qwen3.7-text-embedding",
            "embedding_dimensions": 2, "embedding_text_type": "document", "embedding": embedding,
        }))
    return Retriever(embedding_provider=provider, vector_index=NumpyExactVectorIndex(records),
                     entity_resolver=ExactEntityResolver.from_records(records), default_top_k=5)


class SimpleRetrievalMetricTests(unittest.TestCase):
    def test_partial_multifact_recall_is_not_full_recall(self):
        alpha, beta, noise = chunk("alpha"), chunk("beta"), chunk("noise")
        metrics = score_hits([alpha.chunk_id, beta.chunk_id],
                             [result(noise, 1), result(alpha, 2)], top_k=2)
        self.assertEqual(metrics, {"hit_at_k": 1, "recall_at_k": 0.5, "first_relevant_rank": 2})

    def test_top_k_boundary_excludes_later_relevant_hit(self):
        alpha, noise = chunk("alpha"), chunk("noise")
        self.assertEqual(score_hits([alpha.chunk_id], [result(noise, 1), result(alpha, 2)], top_k=1),
                         {"hit_at_k": 0, "recall_at_k": 0.0, "first_relevant_rank": None})

    def test_empty_gold_is_not_zero_or_perfect(self):
        self.assertEqual(score_hits([], [result(chunk("alpha"), 1)], top_k=5),
                         {"hit_at_k": None, "recall_at_k": None, "first_relevant_rank": None})

    def test_empty_results_for_nonempty_gold_are_a_real_miss(self):
        self.assertEqual(score_hits(["product:alpha:content"], [], top_k=5),
                         {"hit_at_k": 0, "recall_at_k": 0.0, "first_relevant_rank": None})

    def test_all_gold_found_and_low_cosine_is_not_filtered(self):
        alpha, beta = chunk("alpha"), chunk("beta")
        hits = [result(alpha, 1).model_copy(update={"score": -0.25}), result(beta, 2)]
        self.assertEqual(score_hits([alpha.chunk_id, beta.chunk_id], hits, top_k=5),
                         {"hit_at_k": 1, "recall_at_k": 1.0, "first_relevant_rank": 1})

    def test_rejects_bad_k_duplicate_hits_and_inconsistent_ranks(self):
        alpha = chunk("alpha")
        for k in (0, -1, True, 1.5):
            with self.subTest(k=k), self.assertRaises(ValueError):
                score_hits([alpha.chunk_id], [], top_k=k)
        for hits in ([result(alpha, 2)], [result(alpha, 1), result(alpha, 2)]):
            with self.subTest(hits=hits), self.assertRaises(ValueError):
                score_hits([alpha.chunk_id], hits, top_k=5)


class RetrievalRunnerTests(unittest.TestCase):
    def test_real_retriever_preserves_entity_order_and_provenance(self):
        provider = ControlledQueryProvider()
        cases = [evaluation_case("dev-1", "alpha 参数", [chunk("alpha")])]
        output = io.StringIO()
        with redirect_stdout(output):
            report = run_retrieval_evaluation(cases, real_retriever(provider), top_k=2,
                                              metadata={"dataset_version": "test"})
        row = report["cases"][0]
        self.assertEqual(provider.queries, ["alpha 参数"])
        self.assertEqual(row["retrieved_chunk_ids"], ["product:alpha:content", "product:noise:content"])
        self.assertEqual(row["hits"][0]["match_origin"], "exact_entity")
        self.assertEqual(row["hits"][0]["score"], 0.0)
        self.assertEqual(row["hits"][1]["score"], 1.0)
        self.assertEqual(row["hits"][0]["parent_document_id"], "product:alpha")
        self.assertEqual(row["hits"][0]["source_url"], "https://example.com/alpha/")
        self.assertEqual(row["hits"][0]["text"], "# alpha\n\n产品说明")
        self.assertEqual(row["metrics"]["first_relevant_rank"], 1)
        self.assertEqual(output.getvalue(), "")

    def test_skips_no_gold_queries_and_excludes_them_from_denominator(self):
        provider = ControlledQueryProvider()
        cases = [evaluation_case("known", "alpha", [chunk("alpha")]),
                 evaluation_case("unknown", "price?", [], category="unknown", expected_behavior="abstain"),
                 evaluation_case("system", "brand?", [], category="company",
                                 expected_system_rule_ids=["company_identity_v1"])]
        report = run_retrieval_evaluation(cases, real_retriever(provider), top_k=1, metadata={})
        self.assertEqual(provider.queries, ["alpha"])
        self.assertEqual(report["summary"]["scored_cases"], 1)
        self.assertEqual(report["summary"]["excluded_no_gold_cases"], 2)
        self.assertEqual(report["summary"]["hit_at_k"], 1.0)
        self.assertEqual(report["cases"][1]["status"], "excluded_no_gold")
        self.assertFalse(report["cases"][1]["retrieval_executed"])

    def test_failure_is_explicit_without_provider_body_and_later_cases_continue(self):
        provider = ControlledQueryProvider()
        cases = [evaluation_case("fail", "provider failure", [chunk("alpha")]),
                 evaluation_case("good", "alpha", [chunk("alpha")])]
        report = run_retrieval_evaluation(cases, real_retriever(provider), top_k=1, metadata={})
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["summary"]["error_cases"], 1)
        self.assertEqual(report["summary"]["scored_cases"], 1)
        self.assertEqual(report["cases"][0]["error_type"], "EmbeddingAPIError")
        self.assertIsNone(report["cases"][0]["metrics"]["hit_at_k"])
        self.assertNotIn("DO_NOT_PERSIST", json.dumps(report))
        self.assertEqual(report["cases"][1]["status"], "evaluated")

    def test_macro_average_and_rank_distribution_are_hand_calculated(self):
        provider = ControlledQueryProvider()
        cases = [evaluation_case("partial", "alpha", [chunk("alpha"), chunk("beta")]),
                 evaluation_case("miss", "unrelated question", [chunk("beta")])]
        report = run_retrieval_evaluation(cases, real_retriever(provider), top_k=1, metadata={})
        self.assertEqual(report["summary"]["hit_at_k"], 0.5)
        self.assertEqual(report["summary"]["recall_at_k"], 0.25)
        self.assertEqual(report["summary"]["first_relevant_rank_counts"], {"1": 1, "not_found": 1})

    def test_fixture_is_recorded_but_never_applied_during_retrieval(self):
        provider = ControlledQueryProvider()
        case = evaluation_case("attack", "alpha", [chunk("alpha")], category="adversarial", fixture_id="attack-1")
        report = run_retrieval_evaluation([case], real_retriever(provider), top_k=1, metadata={})
        row = report["cases"][0]
        self.assertEqual(row["fixture_id"], "attack-1")
        self.assertFalse(row["fixture_applied"])
        self.assertEqual(row["hits"][0]["text"], "# alpha\n\n产品说明")

    def test_all_excluded_has_no_numeric_quality_claim(self):
        case = evaluation_case("unknown", "price?", [], expected_behavior="abstain")
        report = run_retrieval_evaluation([case], real_retriever(ControlledQueryProvider()), top_k=5, metadata={})
        self.assertIsNone(report["summary"]["hit_at_k"])
        self.assertIsNone(report["summary"]["recall_at_k"])


if __name__ == "__main__":
    unittest.main()
