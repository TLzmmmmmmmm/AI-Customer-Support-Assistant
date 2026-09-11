import importlib
import unittest

from routing import Route
from trace_models import ToolTrace


class RejectAttributeError(Exception):
    rejected_field = ""

    def __setattr__(self, name, value):
        if name == self.rejected_field:
            raise AttributeError(name)
        super().__setattr__(name, value)


class RejectRouteWithAttributeError(RejectAttributeError):
    rejected_field = "route"


class RejectToolCallsWithAttributeError(RejectAttributeError):
    rejected_field = "tool_calls"


class RejectToolCallsWithTypeError(Exception):
    def __setattr__(self, name, value):
        if name == "tool_calls":
            raise TypeError(name)
        super().__setattr__(name, value)


class RoutingErrorTests(unittest.TestCase):
    def _set_error_context(self):
        module = importlib.import_module("routing.errors")
        helper = getattr(module, "set_error_context", None)
        self.assertIsNotNone(helper)
        return helper

    def test_writes_route_tool_calls_and_retrieved_chunk_ids(self):
        error = RuntimeError("failed")
        traces = (ToolTrace("get_contact_info", True),)

        self._set_error_context()(
            error,
            route=Route.CONTACT,
            tool_calls=traces,
            retrieved_chunk_ids=("solution:one:content",),
        )

        self.assertEqual(error.route, Route.CONTACT)
        self.assertEqual(error.tool_calls, traces)
        self.assertEqual(
            error.retrieved_chunk_ids,
            ("solution:one:content",),
        )

    def test_overwrites_existing_context_values(self):
        error = RuntimeError("failed")
        error.route = Route.FALLBACK
        error.tool_calls = (ToolTrace("old", False),)
        error.retrieved_chunk_ids = ("old",)

        self._set_error_context()(error, route=Route.DIRECT)

        self.assertEqual(error.route, Route.DIRECT)
        self.assertEqual(error.tool_calls, ())
        self.assertEqual(error.retrieved_chunk_ids, ())

    def test_repeated_invocation_uses_latest_values(self):
        error = RuntimeError("failed")
        helper = self._set_error_context()
        helper(error, route=Route.DIRECT)
        traces = (ToolTrace("search_products", True),)

        helper(
            error,
            route=Route.PRODUCT_SEARCH,
            tool_calls=traces,
            retrieved_chunk_ids=("product:one:content",),
        )

        self.assertEqual(error.route, Route.PRODUCT_SEARCH)
        self.assertEqual(error.tool_calls, traces)
        self.assertEqual(
            error.retrieved_chunk_ids,
            ("product:one:content",),
        )

    def test_rejected_middle_write_does_not_block_later_field(self):
        error = RejectToolCallsWithAttributeError("failed")

        self._set_error_context()(
            error,
            route=Route.KNOWLEDGE,
            tool_calls=(ToolTrace("ignored", False),),
            retrieved_chunk_ids=("solution:one:content",),
        )

        self.assertEqual(error.route, Route.KNOWLEDGE)
        self.assertFalse(hasattr(error, "tool_calls"))
        self.assertEqual(
            error.retrieved_chunk_ids,
            ("solution:one:content",),
        )

    def test_attribute_error_is_ignored_for_one_field(self):
        error = RejectRouteWithAttributeError("failed")
        traces = (ToolTrace("get_contact_info", True),)

        self._set_error_context()(
            error,
            route=Route.CONTACT,
            tool_calls=traces,
        )

        self.assertFalse(hasattr(error, "route"))
        self.assertEqual(error.tool_calls, traces)
        self.assertEqual(error.retrieved_chunk_ids, ())

    def test_type_error_is_ignored_for_one_field(self):
        error = RejectToolCallsWithTypeError("failed")

        self._set_error_context()(
            error,
            route=Route.KNOWLEDGE,
            retrieved_chunk_ids=("solution:one:content",),
        )

        self.assertEqual(error.route, Route.KNOWLEDGE)
        self.assertFalse(hasattr(error, "tool_calls"))
        self.assertEqual(
            error.retrieved_chunk_ids,
            ("solution:one:content",),
        )


if __name__ == "__main__":
    unittest.main()
