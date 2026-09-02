import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

import error_handling


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
        )


if __name__ == "__main__":
    unittest.main()
