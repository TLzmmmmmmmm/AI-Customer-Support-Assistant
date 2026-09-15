import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

from trace_models import (
    bind_request_state,
    initialize_request_trace,
    record_model_response,
    record_model_usage,
    request_trace_fields,
    reset_request_state,
)


class RequestTraceContextTests(unittest.TestCase):
    def test_shared_usage_accumulator_matches_completion_accumulation(self):
        state = SimpleNamespace()
        initialize_request_trace(state)
        token = bind_request_state(state)
        try:
            record_model_usage(SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=2,
                prompt_cache_hit_tokens=7,
                prompt_cache_miss_tokens=3,
            ))
            record_model_usage(SimpleNamespace(
                prompt_tokens=8,
                completion_tokens=1,
                prompt_cache_hit_tokens=5,
                prompt_cache_miss_tokens=3,
            ))
        finally:
            reset_request_state(token)

        fields = request_trace_fields(state)
        self.assertEqual(fields["input_tokens"], 18)
        self.assertEqual(fields["output_tokens"], 3)
        self.assertEqual(fields["prompt_cache_hit_tokens"], 12)
        self.assertEqual(fields["prompt_cache_miss_tokens"], 6)

    def test_invalid_usage_permanently_nulls_totals(self):
        state = SimpleNamespace()
        initialize_request_trace(state)
        valid = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=2,
            prompt_cache_hit_tokens=7,
            prompt_cache_miss_tokens=3,
        )
        token = bind_request_state(state)
        try:
            record_model_usage(valid)
            record_model_usage(None)
            record_model_usage(valid)
        finally:
            reset_request_state(token)

        fields = request_trace_fields(state)
        self.assertIsNone(fields["input_tokens"])
        self.assertIsNone(fields["output_tokens"])
        self.assertIsNone(fields["prompt_cache_hit_tokens"])
        self.assertIsNone(fields["prompt_cache_miss_tokens"])

    def test_record_model_response_delegates_its_usage(self):
        from unittest.mock import patch

        usage = object()
        with patch("trace_models.record_model_usage") as record:
            record_model_response(SimpleNamespace(usage=usage))

        record.assert_called_once_with(usage)

    def test_cache_usage_is_aggregated_across_model_responses(self):
        state = SimpleNamespace()
        initialize_request_trace(state)
        token = bind_request_state(state)
        try:
            record_model_response(SimpleNamespace(usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=2,
                prompt_cache_hit_tokens=7,
                prompt_cache_miss_tokens=3,
            )))
            record_model_response(SimpleNamespace(usage=SimpleNamespace(
                prompt_tokens=8,
                completion_tokens=1,
                prompt_cache_hit_tokens=5,
                prompt_cache_miss_tokens=3,
            )))
        finally:
            reset_request_state(token)

        fields = request_trace_fields(state)
        self.assertEqual(fields["prompt_cache_hit_tokens"], 12)
        self.assertEqual(fields["prompt_cache_miss_tokens"], 6)

    def test_invalid_cache_usage_nulls_only_cache_totals(self):
        invalid_usage = (
            {"prompt_cache_hit_tokens": None, "prompt_cache_miss_tokens": 10},
            {"prompt_cache_hit_tokens": True, "prompt_cache_miss_tokens": 9},
            {"prompt_cache_hit_tokens": -1, "prompt_cache_miss_tokens": 11},
            {"prompt_cache_hit_tokens": 4, "prompt_cache_miss_tokens": 5},
        )

        for cache_fields in invalid_usage:
            with self.subTest(cache_fields=cache_fields):
                state = SimpleNamespace()
                initialize_request_trace(state)
                token = bind_request_state(state)
                try:
                    record_model_response(SimpleNamespace(usage=SimpleNamespace(
                        prompt_tokens=10,
                        completion_tokens=2,
                        **cache_fields,
                    )))
                finally:
                    reset_request_state(token)

                fields = request_trace_fields(state)
                self.assertEqual(fields["input_tokens"], 10)
                self.assertEqual(fields["output_tokens"], 2)
                self.assertIsNone(fields["prompt_cache_hit_tokens"])
                self.assertIsNone(fields["prompt_cache_miss_tokens"])

    def test_concurrent_request_metrics_remain_isolated(self):
        barrier = Barrier(2)

        def collect(input_tokens, output_tokens):
            state = SimpleNamespace()
            initialize_request_trace(state)
            token = bind_request_state(state)
            try:
                barrier.wait()
                record_model_response(SimpleNamespace(usage=SimpleNamespace(
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                )))
                barrier.wait()
                return request_trace_fields(state)
            finally:
                reset_request_state(token)

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(collect, 11, 3)
            second = pool.submit(collect, 29, 7)

        self.assertEqual(first.result()["input_tokens"], 11)
        self.assertEqual(first.result()["output_tokens"], 3)
        self.assertEqual(second.result()["input_tokens"], 29)
        self.assertEqual(second.result()["output_tokens"], 7)


if __name__ == "__main__":
    unittest.main()
