import importlib
import json
import os
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from agent import SAFE_AGENT_ANSWER
from tests.test_day6_retrieval import ControlledQueryProvider, evaluation_case, real_retriever
from tests.test_retrieval_evaluation import chunk, result


class ProviderCompletion:
    def __init__(self, text="已确认。", finish="stop"):
        self.model = "test-model"
        self.usage = None
        self.choices = [SimpleNamespace(
            finish_reason=finish,
            message=SimpleNamespace(content=text, tool_calls=None),
        )]


class GenerationTests(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
        try:
            self.module = importlib.import_module("evaluation.generation")
        except ModuleNotFoundError:
            self.fail("Day 6 generation evaluator is not implemented")
        import rate_limit
        rate_limit.request_history.clear()

    def test_fixture_copy_keeps_clean_provenance_and_marks_absent_target(self):
        clean = result(chunk("alpha"), 1)
        fixture = {"id": "attack", "target_chunk_id": clean.chunk_id, "text": "\nINJECT"}
        before = clean.model_dump()
        effective, applied = self.module.apply_fixture([clean], fixture)
        self.assertTrue(applied)
        self.assertEqual(clean.model_dump(), before)
        self.assertEqual(effective[0].text, clean.text + "\nINJECT")
        self.assertEqual(effective[0].content_hash, clean.content_hash)
        self.assertEqual(effective[0].source_url, clean.source_url)
        self.assertFalse(self.module.apply_fixture([clean], {**fixture, "target_chunk_id": "absent"})[1])

    def run_cases(self, cases, completions, fixtures=()):
        from services import llm
        rows, received = [], []
        provider = ControlledQueryProvider()
        retriever = real_retriever(provider)
        iterator = iter(completions)

        def create(**kwargs):
            received.append(deepcopy(kwargs))
            completion = next(iterator)
            if isinstance(completion, Exception):
                raise completion
            return completion

        with patch.object(llm.client.chat.completions, "create", side_effect=create):
            summary = self.module.run_generation_cases(
                cases, fixtures, rows.append, retriever_factory=lambda: retriever,
                request_interval=0,
            )
        return rows, summary, received, provider

    def test_real_route_unknown_case_is_not_skipped_and_inputs_are_unmodified(self):
        case = evaluation_case("dev-unknown", "Cannot know price?", [], expected_behavior="abstain")
        rows, summary, received, provider = self.run_cases(
            [case],
            [ProviderCompletion("无法确认价格。")],
        )
        row = rows[0]
        self.assertEqual(summary["completed_cases"], 1)
        self.assertEqual(
            row["answer"],
            "无法确认价格。",
        )
        self.assertEqual(row["event_types"], ["delta", "citations", "done"])
        self.assertEqual(row["http_status"], 200)
        self.assertEqual(provider.queries, [case.question])
        self.assertTrue(row["clean_hits"])
        self.assertEqual(row["provider_requests"][0], received[0])
        self.assertNotIn("max_tokens", received[0])
        self.assertNotIn("stream_options", received[0])
        self.assertFalse(received[0]["stream"])
        self.assertEqual(received[0]["extra_body"], {"thinking": {"type": "disabled"}})
        payload = json.loads(received[0]["messages"][-1]["content"].split("BEGIN_RAG_DATA\n", 1)[1].rsplit("\nEND_RAG_DATA", 1)[0])
        self.assertEqual(payload["user_question"], case.question)
        self.assertEqual(payload["retrieved_context"], row["context"])
        self.assertNotIn("expected_answer", json.dumps(received[0]))
        self.assertEqual(row["review_status"], "pending_review")
        self.assertGreaterEqual(row["retrieval_latency_seconds"], 0)
        self.assertGreaterEqual(row["llm_ttft_seconds"], 0)
        self.assertEqual(row["generation_latency_seconds"], row["llm_ttft_seconds"])
        self.assertEqual(row["llm_total_latency_seconds"], row["generation_latency_seconds"])
        self.assertEqual(row["llm_streaming_latency_seconds"], 0.0)
        self.assertGreaterEqual(row["total_request_latency_seconds"], row["generation_latency_seconds"])
        self.assertEqual(row["latency_seconds"], row["total_request_latency_seconds"])
        self.assertEqual(row["retrieved_chunk_count"], len(row["clean_hits"]))
        self.assertGreater(row["retrieved_context_characters"], 0)
        self.assertGreaterEqual(row["provider_input_characters"], row["retrieved_context_characters"])

    def test_injection_uses_real_retrieval_and_records_actual_context(self):
        case = evaluation_case("dev-attack", "alpha 参数", [chunk("alpha")], fixture_id="attack")
        fixture = {"id": "attack", "target_chunk_id": "product:alpha:content", "text": "\nINJECT"}
        rows, _, received, provider = self.run_cases(
            [case],
            [ProviderCompletion()],
            [fixture],
        )
        row = rows[0]
        self.assertEqual(provider.queries, [case.question])
        self.assertTrue(row["fixture_applied"])
        self.assertEqual(row["evaluation_mode"], "synthetic_context")
        self.assertNotIn("INJECT", row["clean_hits"][0]["text"])
        self.assertIn("INJECT", row["context"][0]["text"])
        self.assertIn("INJECT", json.dumps(received[0]["messages"]))
        self.assertEqual(row["clean_hits"][0]["content_hash"], row["effective_hits"][0]["content_hash"])

    def test_length_finish_is_incomplete_even_with_done_event(self):
        case = evaluation_case("dev-cut", "alpha 参数", [chunk("alpha")])
        rows, summary, _, _ = self.run_cases(
            [case],
            [ProviderCompletion("部分回答", "length")],
        )
        self.assertEqual(rows[0]["answer"], SAFE_AGENT_ANSWER)
        self.assertEqual(rows[0]["status"], "incomplete")
        self.assertEqual(rows[0]["error_type"], "GenerationNotComplete")
        self.assertEqual(summary["completed_cases"], 0)

    def test_completion_timeout_is_safe_http_error_without_partial_answer(self):
        import httpx2 as httpx
        from openai import APITimeoutError
        case = evaluation_case("dev-timeout", "alpha 参数", [chunk("alpha")])
        error = APITimeoutError(request=httpx.Request("POST", "https://example.invalid/SECRET"))
        rows, summary, _, _ = self.run_cases([case], [error])
        self.assertEqual(rows[0]["answer"], "")
        self.assertEqual(rows[0]["event_types"], [])
        self.assertEqual(rows[0]["http_status"], 504)
        self.assertEqual(rows[0]["error_type"], "APITimeoutError")
        self.assertNotIn("SECRET", json.dumps(rows))
        self.assertEqual(summary["incomplete_cases"], 1)

    def test_missing_fixture_target_not_claimed_exercised(self):
        case = evaluation_case("dev-attack", "alpha 参数", [chunk("alpha")], fixture_id="attack")
        fixture = {"id": "attack", "target_chunk_id": "absent", "text": "INJECT"}
        rows, _, _, _ = self.run_cases([case], [ProviderCompletion()], [fixture])
        self.assertFalse(rows[0]["fixture_applied"])
        self.assertEqual(rows[0]["fixture_status"], "not_exercised")

    def test_completed_case_is_emitted_before_next_provider_call(self):
        from services import llm
        provider = ControlledQueryProvider()
        cases = [evaluation_case(f"dev-{i}", f"alpha 参数{i}", [chunk("alpha")]) for i in range(2)]
        rows = []
        calls = []
        def create(**kwargs):
            calls.append(len(rows))
            return ProviderCompletion()
        with patch.object(llm.client.chat.completions, "create", side_effect=create):
            self.module.run_generation_cases(cases, [], rows.append,
                retriever_factory=lambda: real_retriever(provider), request_interval=0)
        self.assertEqual(calls, [0, 1])
        self.assertEqual(len(rows), 2)

    def test_holdout_is_rejected_before_constructing_provider(self):
        case = evaluation_case("holdout-1", "alpha 参数", [chunk("alpha")], split="holdout")
        with self.assertRaises(ValueError):
            self.module.run_generation_cases([case], [], lambda row: None,
                retriever_factory=lambda: self.fail("Must not build provider"), request_interval=0)

    def test_frozen_case_runs_through_real_route_without_changing_question(self):
        case = evaluation_case("baseline-test", "alpha 参数", [chunk("alpha")], split="frozen")
        rows, summary, received, provider = self.run_cases(
            [case],
            [ProviderCompletion("产品说明")],
        )
        self.assertEqual(summary["completed_cases"], 1)
        self.assertEqual(rows[0]["split"], "frozen")
        self.assertEqual(rows[0]["query"], "alpha 参数")
        self.assertEqual(
            rows[0]["answer"],
            "产品说明",
        )
        self.assertEqual(provider.queries, ["alpha 参数"])

    def test_mixed_splits_are_rejected_before_provider(self):
        cases = [evaluation_case("dev-one", "alpha", [chunk("alpha")]),
                 evaluation_case("baseline-one", "alpha", [chunk("alpha")], split="frozen")]
        with self.assertRaises(ValueError):
            self.module.run_generation_cases(cases, [], lambda row: None,
                retriever_factory=lambda: self.fail("Must not build provider"))

    def test_keyboard_interrupt_emits_current_attempt_before_propagating(self):
        from fastapi.testclient import TestClient
        rows = []
        case = evaluation_case("dev-interrupted", "alpha 参数", [chunk("alpha")])
        with patch.object(TestClient, "post", side_effect=KeyboardInterrupt()), self.assertRaises(KeyboardInterrupt):
            self.module.run_generation_cases([case], [], rows.append,
                retriever_factory=lambda: real_retriever(ControlledQueryProvider()), request_interval=0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["case_id"], "dev-interrupted")
        self.assertEqual(rows[0]["status"], "incomplete")
        self.assertEqual(rows[0]["error_type"], "KeyboardInterrupt")


if __name__ == "__main__":
    unittest.main()
