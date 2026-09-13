import unittest

from pydantic import ValidationError


class AgentModelTests(unittest.TestCase):
    def test_tool_schemas_have_exact_names_arguments_and_no_extra_properties(self):
        from agent.models import llm_tool_schemas

        schemas = llm_tool_schemas()
        by_name = {
            item["function"]["name"]: item["function"]["parameters"]
            for item in schemas
        }

        self.assertEqual(
            set(by_name),
            {
                "search_products",
                "get_product_details",
                "get_contact_info",
            },
        )
        self.assertEqual(by_name["search_products"]["required"], ["query"])
        self.assertEqual(
            by_name["get_product_details"]["required"],
            ["product_id"],
        )
        self.assertEqual(by_name["get_contact_info"]["properties"], {})
        self.assertTrue(
            all(
                parameters["additionalProperties"] is False
                for parameters in by_name.values()
            )
        )

    def test_tool_schema_results_are_fresh_nested_dictionaries(self):
        from agent.models import llm_tool_schemas

        first = llm_tool_schemas()
        first[0]["function"]["parameters"]["properties"].clear()

        second = llm_tool_schemas()

        self.assertEqual(
            set(second[0]["function"]["parameters"]["properties"]),
            {"query"},
        )

    def test_argument_models_reuse_day_one_limits_and_reject_invalid_values(self):
        from agent.models import ProductDetailsArguments, SearchProductsArguments

        self.assertEqual(
            SearchProductsArguments(query="  酒店对讲机  ").query,
            "酒店对讲机",
        )
        self.assertEqual(
            ProductDetailsArguments(product_id=" LY198 ").product_id,
            "LY198",
        )
        cases = (
            (SearchProductsArguments, {"query": "x" * 4001}),
            (ProductDetailsArguments, {"product_id": "x" * 129}),
            (ProductDetailsArguments, {"product_id": "LY198", "extra": 1}),
            (SearchProductsArguments, {"query": 123}),
        )
        for model, payload in cases:
            with self.subTest(model=model.__name__, payload=payload):
                with self.assertRaises(ValidationError):
                    model.model_validate(payload)

    def test_deadline_uses_injected_monotonic_clock(self):
        from agent.models import AgentDeadline, AgentDeadlineExceeded

        times = iter([10.0, 129.9, 130.0])
        deadline = AgentDeadline.start(120.0, clock=lambda: next(times))

        deadline.ensure_active()
        with self.assertRaises(AgentDeadlineExceeded):
            deadline.ensure_active()

    def test_deadline_rejects_non_positive_timeout(self):
        from agent.models import AgentDeadline

        for timeout in (0.0, -1.0):
            with self.subTest(timeout=timeout):
                with self.assertRaises(ValueError):
                    AgentDeadline.start(timeout, clock=lambda: 10.0)


if __name__ == "__main__":
    unittest.main()
