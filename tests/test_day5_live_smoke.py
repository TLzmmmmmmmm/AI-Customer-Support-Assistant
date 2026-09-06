"""Regression tests for the bounded live-smoke harness."""

import inspect
import unittest

from scripts.day5_live_smoke import run


class LiveSmokeBudgetTests(unittest.TestCase):
    def test_prompt_byte_budget_is_explicitly_configurable(self):
        parameter = inspect.signature(run).parameters.get(
            "max_prompt_utf8_bytes"
        )

        self.assertIsNotNone(parameter)
        self.assertEqual(parameter.default, 30_000)


if __name__ == "__main__":
    unittest.main()
