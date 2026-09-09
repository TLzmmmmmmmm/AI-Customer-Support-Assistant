import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from app_logging import log_request
import error_handling
from routing import Route, RouteTrace
from trace_models import FailureLayer, ToolTrace


class ErrorHandlingPrivacyTests(unittest.TestCase):
    def test_retrieval_error_redacts_provenance_and_logs_only_error_type(self):
        request = SimpleNamespace(state=SimpleNamespace(
            request_id="request-1",
            started_at=1.0,
        ))
        exception = HTTPException(
            status_code=503,
            detail={
                "code": "retrieval_unavailable",
                "message": "服务暂时不可用，请稍后再试。",
                "internal_error": "EmbeddingAPIError",
                "query": "private question",
                "chunk_id": "product:hp780:specifications",
                "parent_document_id": "product:hp780",
                "score": 0.91,
                "source_url": "https://example.com/hp780/",
            },
        )

        with patch.object(error_handling, "log_request") as logged:
            response = asyncio.run(
                error_handling.http_exception_handler(request, exception)
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers["X-Request-ID"], "request-1")
        self.assertEqual(
            json.loads(response.body),
            {
                "error": {
                    "code": "retrieval_unavailable",
                    "message": "服务暂时不可用，请稍后再试。",
                    "request_id": "request-1",
                }
            },
        )
        logged.assert_called_once_with(
            request_id="request-1",
            http_status=503,
            outcome="retrieval_unavailable",
            started_at=1.0,
            error="EmbeddingAPIError",
            trace=None,
            failure_layer=None,
        )

    def test_http_error_logs_only_allowlisted_trace_fields(self):
        trace = RouteTrace(
            route=Route.PRODUCT_SEARCH,
            tool_calls=(ToolTrace(
                name="search_products",
                success=False,
                error_code="TOOL_EXECUTION_ERROR",
            ),),
            tool_call_count=1,
            retrieved_chunk_ids=("product:hp780:content",),
            failure_layer=FailureLayer.TOOL_EXECUTION,
        )
        request = SimpleNamespace(state=SimpleNamespace(
            request_id="request-2",
            started_at=1.0,
            route_trace=trace,
            failure_layer=None,
        ))
        exception = HTTPException(
            status_code=503,
            detail={
                "code": "provider_unavailable",
                "message": "safe message",
                "internal_error": "APIConnectionError",
                "query": "private question",
                "phone": "123456",
            },
        )

        with patch.object(error_handling, "log_request") as logged:
            asyncio.run(error_handling.http_exception_handler(request, exception))

        logged.assert_called_once_with(
            request_id="request-2",
            http_status=503,
            outcome="provider_unavailable",
            started_at=1.0,
            error="APIConnectionError",
            trace=trace,
            failure_layer=None,
        )

    def test_request_log_contains_bounded_trace_without_private_payloads(self):
        trace = RouteTrace(
            route=Route.KNOWLEDGE,
            tool_calls=(ToolTrace(
                name="get_contact_info",
                success=True,
            ),),
            tool_call_count=1,
            retrieved_chunk_ids=("solution:hotel:content",),
        )
        private_values = (
            "private user question",
            "private chunk text",
            "secret@example.com",
            "13800000000",
            "private address",
        )

        with (
            patch("app_logging.time.monotonic", return_value=2.0),
            self.assertLogs("ai_customer_support", level="INFO") as logs,
        ):
            log_request(
                request_id="request-3",
                http_status=200,
                outcome="success",
                started_at=1.0,
                trace=trace,
            )

        message = logs.records[0].getMessage()
        self.assertIn("route=knowledge", message)
        self.assertIn('tool_calls=[{"name":"get_contact_info"', message)
        self.assertIn("tool_call_count=1", message)
        self.assertIn('retrieved_chunk_ids=["solution:hotel:content"]', message)
        self.assertIn("failure_layer=-", message)
        for value in private_values:
            self.assertNotIn(value, message)


if __name__ == "__main__":
    unittest.main()
