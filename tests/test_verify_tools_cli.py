import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch


class RecordingRetriever:
    def __init__(self):
        self.queries: list[str] = []

    def retrieve(
        self,
        query,
        top_k=None,
        *,
        allowed_types=None,
        unique_parent_documents=False,
    ):
        self.queries.append(query)
        return []


class VerifyToolsCliTests(unittest.TestCase):
    def test_product_and_contact_commands_are_offline_and_print_json(self):
        from scripts import verify_tools

        for argv, expected_field in (
            (["product", "LY198"], "product_id"),
            (["contact"], "company_name"),
        ):
            stdout = io.StringIO()
            stderr = io.StringIO()
            with self.subTest(argv=argv):
                with patch.object(verify_tools, "build_retriever") as retriever:
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        code = verify_tools.main(argv)
                payload = json.loads(stdout.getvalue())
                self.assertEqual(code, 0)
                self.assertEqual(stderr.getvalue(), "")
                self.assertIn(expected_field, payload)
                retriever.assert_not_called()

    def test_search_uses_the_existing_retriever_without_query_rewriting(self):
        from scripts import verify_tools

        retriever = RecordingRetriever()
        stdout = io.StringIO()
        with patch.object(
            verify_tools,
            "build_retriever",
            return_value=retriever,
        ):
            with redirect_stdout(stdout):
                code = verify_tools.main(["search", "适合酒店使用的对讲机"])

        self.assertEqual(code, 0)
        self.assertEqual(retriever.queries, ["适合酒店使用的对讲机"])
        self.assertEqual(json.loads(stdout.getvalue()), {"products": []})

    def test_tool_error_is_safe_structured_json_on_stderr(self):
        from scripts import verify_tools

        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = verify_tools.main(["product", "missing-secret-model"])

        payload = json.loads(stderr.getvalue())
        self.assertEqual(code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(payload["code"], "PRODUCT_NOT_FOUND")
        self.assertEqual(payload["tool_name"], "get_product_details")
        self.assertNotIn("missing-secret-model", payload["message"])


if __name__ == "__main__":
    unittest.main()
