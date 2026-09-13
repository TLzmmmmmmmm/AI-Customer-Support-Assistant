import json
import unittest
from types import MappingProxyType

from agent.models import AgentToolCall
from support_tools import (
    ContactInfoResult,
    ProductSearchItem,
    ProductSearchResult,
    ToolError,
    ToolErrorCode,
)


class AgentExecutorTests(unittest.TestCase):
    def test_success_preserves_structured_sources_across_cache_reuse(self):
        from agent.executor import ToolExecutor
        from knowledge_pipeline.models import SourceRef

        source = SourceRef(
            title="LY198 产品详情",
            url="https://example.com/products/ly198/",
        )

        def search_products(query):
            return ProductSearchResult(products=[ProductSearchItem(
                product_id="ly198",
                name="LY198",
                category_id="two-way-radio",
                category_name="对讲机",
                relevant_content="候选产品",
                sources=[source],
            )])

        executor = ToolExecutor(MappingProxyType({
            "search_products": search_products,
        }))
        cache = {}

        first = executor.execute_named(
            "search_products", {"query": "对讲机"}, cache
        )
        reused = executor.execute_named(
            "search_products", {"query": "对讲机"}, cache
        )

        self.assertEqual(first.sources, (source,))
        self.assertEqual(reused.sources, (source,))
        self.assertTrue(reused.reused)

    def test_failed_observation_has_no_sources(self):
        from agent.executor import ToolExecutor

        result = ToolExecutor(MappingProxyType({})).execute_named(
            "get_contact_info", {}, {}
        )

        self.assertEqual(result.sources, ())

    def test_named_and_native_calls_share_validation_execution_and_cache(self):
        from agent.executor import ToolExecutor

        calls = []

        def search_products(query):
            calls.append(query)
            return ProductSearchResult(products=[])

        executor = ToolExecutor(MappingProxyType({
            "search_products": search_products,
        }))
        cache = {}

        named = executor.execute_named(
            "search_products",
            {"query": " 酒店 "},
            cache,
        )
        native = executor.execute(
            AgentToolCall(
                "call-1",
                "search_products",
                '{"query":"酒店"}',
            ),
            cache,
        )

        self.assertEqual(calls, ["酒店"])
        self.assertTrue(named.success)
        self.assertFalse(named.reused)
        self.assertTrue(native.success)
        self.assertTrue(native.reused)
        self.assertEqual(named.content, native.content)
        self.assertEqual(named.tool_name, "search_products")
        self.assertIsNone(named.error_code)

    def test_named_invalid_arguments_do_not_execute_handler(self):
        from agent.executor import ToolExecutor

        calls = []
        executor = ToolExecutor(MappingProxyType({
            "search_products": lambda query: calls.append(query),
        }))

        observation = executor.execute_named(
            "search_products",
            {"query": "", "extra": True},
            {},
        )

        self.assertFalse(observation.success)
        self.assertEqual(observation.tool_name, "search_products")
        self.assertEqual(observation.error_code, "INVALID_ARGUMENT")
        self.assertEqual(calls, [])

    def test_observation_metadata_distinguishes_safe_domain_and_system_errors(self):
        from agent.executor import ToolExecutor

        def missing(product_id):
            raise ToolError(
                code=ToolErrorCode.PRODUCT_NOT_FOUND,
                message="No product matches.",
                tool_name="get_product_details",
            )

        missing_result = ToolExecutor(MappingProxyType({
            "get_product_details": missing,
        })).execute_named(
            "get_product_details",
            {"product_id": "UNKNOWN"},
            {},
        )
        unknown_result = ToolExecutor(MappingProxyType({})).execute_named(
            "private_tool",
            {},
            {},
        )

        self.assertEqual(missing_result.tool_name, "get_product_details")
        self.assertEqual(missing_result.error_code, "PRODUCT_NOT_FOUND")
        self.assertEqual(unknown_result.tool_name, "tool_executor")
        self.assertEqual(unknown_result.error_code, "INVALID_ARGUMENT")
        self.assertNotIn("private_tool", unknown_result.content)

    def test_success_serializes_existing_result_and_reuses_request_cache(self):
        from agent.executor import ToolExecutor

        calls = []

        def search_products(query):
            calls.append(query)
            return ProductSearchResult(products=[])

        executor = ToolExecutor(MappingProxyType({
            "search_products": search_products,
        }))
        cache = {}

        first = executor.execute(
            AgentToolCall("call-1", "search_products", '{"query":" 酒店 "}'),
            cache,
        )
        second = executor.execute(
            AgentToolCall("call-2", "search_products", '{"query":"酒店"}'),
            cache,
        )

        self.assertEqual(calls, ["酒店"])
        self.assertTrue(first.success)
        self.assertFalse(first.reused)
        self.assertTrue(second.reused)
        self.assertEqual(first.content, second.content)
        self.assertEqual(
            json.loads(first.content),
            {"ok": True, "result": {"products": []}},
        )

    def test_day_one_error_is_safe_observation_and_is_not_cached(self):
        from agent.executor import ToolExecutor

        calls = []

        def details(product_id):
            calls.append(product_id)
            raise ToolError(
                code=ToolErrorCode.PRODUCT_NOT_FOUND,
                message="No product matches the supplied canonical identifier.",
                tool_name="get_product_details",
            )

        executor = ToolExecutor(MappingProxyType({
            "get_product_details": details,
        }))
        cache = {}

        result = executor.execute(
            AgentToolCall(
                "call-1",
                "get_product_details",
                '{"product_id":"LY999"}',
            ),
            cache,
        )

        self.assertFalse(result.success)
        self.assertFalse(result.reused)
        self.assertIsNone(result.cache_key)
        self.assertEqual(cache, {})
        self.assertEqual(calls, ["LY999"])
        self.assertEqual(
            json.loads(result.content),
            {
                "ok": False,
                "error": {
                    "code": "PRODUCT_NOT_FOUND",
                    "message": (
                        "No product matches the supplied canonical identifier."
                    ),
                    "tool_name": "get_product_details",
                },
            },
        )

    def test_failure_followed_by_corrected_arguments_executes_again(self):
        from agent.executor import ToolExecutor

        calls = []

        def details(product_id):
            calls.append(product_id)
            if product_id == "UNKNOWN":
                raise ToolError(
                    code=ToolErrorCode.PRODUCT_NOT_FOUND,
                    message="Product not found.",
                    tool_name="get_product_details",
                )
            return ProductSearchResult(products=[])

        executor = ToolExecutor(MappingProxyType({
            "get_product_details": details,
        }))
        cache = {}

        executor.execute(
            AgentToolCall(
                "call-1",
                "get_product_details",
                '{"product_id":"UNKNOWN"}',
            ),
            cache,
        )
        result = executor.execute(
            AgentToolCall(
                "call-2",
                "get_product_details",
                '{"product_id":"LY198"}',
            ),
            cache,
        )

        self.assertEqual(calls, ["UNKNOWN", "LY198"])
        self.assertFalse(result.success)
        self.assertEqual(
            json.loads(result.content)["error"]["code"],
            "TOOL_EXECUTION_ERROR",
        )

    def test_same_tool_with_different_arguments_executes_both(self):
        from agent.executor import ToolExecutor

        calls = []

        def search_products(query):
            calls.append(query)
            return ProductSearchResult(products=[])

        executor = ToolExecutor(MappingProxyType({
            "search_products": search_products,
        }))
        cache = {}

        first = executor.execute(
            AgentToolCall("call-1", "search_products", '{"query":"酒店"}'),
            cache,
        )
        second = executor.execute(
            AgentToolCall("call-2", "search_products", '{"query":"学校"}'),
            cache,
        )

        self.assertEqual(calls, ["酒店", "学校"])
        self.assertTrue(first.success)
        self.assertTrue(second.success)
        self.assertFalse(second.reused)

    def test_unknown_tool_is_redacted_and_never_executes_registry(self):
        from agent.executor import ToolExecutor

        calls = []
        executor = ToolExecutor(MappingProxyType({
            "search_products": lambda query: calls.append(query),
        }))

        result = executor.execute(
            AgentToolCall("call-1", "run_private_command", "{}"),
            {},
        )
        payload = json.loads(result.content)

        self.assertEqual(calls, [])
        self.assertEqual(payload["error"]["code"], "INVALID_ARGUMENT")
        self.assertEqual(payload["error"]["tool_name"], "tool_executor")
        self.assertNotIn("run_private_command", result.content)

    def test_invalid_argument_shapes_do_not_execute_tool(self):
        from agent.executor import ToolExecutor

        calls = []

        def search_products(query):
            calls.append(query)
            return ProductSearchResult(products=[])

        executor = ToolExecutor(MappingProxyType({
            "search_products": search_products,
        }))
        invalid_arguments = (
            "not-json",
            "[]",
            '"hotel"',
            "null",
            "{}",
            '{"query":"hotel","extra":true}',
            '{"query":123}',
        )

        for index, arguments in enumerate(invalid_arguments):
            with self.subTest(arguments=arguments):
                result = executor.execute(
                    AgentToolCall(
                        f"call-{index}",
                        "search_products",
                        arguments,
                    ),
                    {},
                )
                self.assertEqual(
                    json.loads(result.content)["error"]["code"],
                    "INVALID_ARGUMENT",
                )

        self.assertEqual(calls, [])

    def test_registry_spec_mismatch_is_invalid_argument(self):
        from agent.executor import ToolExecutor

        executor = ToolExecutor(MappingProxyType({}))

        result = executor.execute(
            AgentToolCall("call-1", "get_contact_info", "{}"),
            {},
        )

        self.assertEqual(
            json.loads(result.content)["error"],
            {
                "code": "INVALID_ARGUMENT",
                "message": "The proposed tool arguments are invalid.",
                "tool_name": "tool_executor",
            },
        )

    def test_wrong_result_model_becomes_tool_execution_error(self):
        from agent.executor import ToolExecutor

        def search_products(query):
            return ContactInfoResult.model_validate({
                "company_name": "company",
                "duty_phone": "123",
                "email": "service@example.com",
                "sources": [{"title": "contact", "url": "https://example.com"}],
            })

        executor = ToolExecutor(MappingProxyType({
            "search_products": search_products,
        }))

        result = executor.execute(
            AgentToolCall("call-1", "search_products", '{"query":"酒店"}'),
            {},
        )

        self.assertEqual(
            json.loads(result.content)["error"]["code"],
            "TOOL_EXECUTION_ERROR",
        )

    def test_unexpected_callable_error_is_redacted(self):
        from agent.executor import ToolExecutor

        def broken(query):
            raise RuntimeError("private path and traceback")

        executor = ToolExecutor(MappingProxyType({"search_products": broken}))

        result = executor.execute(
            AgentToolCall("call-1", "search_products", '{"query":"酒店"}'),
            {},
        )

        self.assertEqual(
            json.loads(result.content)["error"]["code"],
            "TOOL_EXECUTION_ERROR",
        )
        self.assertNotIn("private", result.content)
        self.assertNotIn("traceback", result.content)


if __name__ == "__main__":
    unittest.main()
