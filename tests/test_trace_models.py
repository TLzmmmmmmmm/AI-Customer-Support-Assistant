import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

from trace_models import (
    bind_request_state,
    initialize_request_trace,
    record_model_response,
    request_trace_fields,
    reset_request_state,
)


class RequestTraceContextTests(unittest.TestCase):
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
