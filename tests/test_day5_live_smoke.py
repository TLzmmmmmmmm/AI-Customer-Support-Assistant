"""Regression tests for the bounded live-smoke harness."""

import inspect
import unittest
from types import SimpleNamespace

from scripts import day5_live_smoke


class LiveSmokeBudgetTests(unittest.TestCase):
    def test_prompt_byte_budget_is_explicitly_configurable(self):
        parameter = inspect.signature(day5_live_smoke.run).parameters.get(
            "max_prompt_utf8_bytes"
        )

        self.assertIsNotNone(parameter)
        self.assertEqual(parameter.default, 30_000)

    def test_completion_observation_reads_non_streaming_response(self):
        completion = SimpleNamespace(
            model="test-model",
            usage=SimpleNamespace(
                model_dump=lambda mode: {"total_tokens": 12},
            ),
            choices=[SimpleNamespace(finish_reason="stop")],
        )

        observation = day5_live_smoke._completion_observation(completion)

        self.assertEqual(
            observation,
            {
                "provider_model": "test-model",
                "generation_usage": {"total_tokens": 12},
                "finish_reason": "stop",
            },
        )


if __name__ == "__main__":
    unittest.main()
