import unittest
from types import SimpleNamespace
from unittest.mock import patch

from services import llm


PROVIDER_MESSAGES = [
    {"role": "system", "content": "应用程序控制的系统规则"},
    {"role": "user", "content": "介绍 HP780。"},
    {"role": "assistant", "content": "HP780 是一款对讲机。"},
    {
        "role": "user",
        "content": (
            "BEGIN_RAG_DATA\n"
            '{"retrieved_context":[],"user_question":"参数？"}\n'
            "END_RAG_DATA"
        ),
    },
]


class LlmProviderMessageTests(unittest.TestCase):
    def _assert_request(self, create) -> None:
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["model"], llm.DEEPSEEK_MODEL)
        self.assertEqual(kwargs["messages"], PROVIDER_MESSAGES)
        self.assertTrue(kwargs["stream"])
        self.assertEqual(
            kwargs["extra_body"],
            {"thinking": {"type": "disabled"}},
        )

    def test_open_chat_stream_sends_provider_messages_unchanged(self):
        sentinel = object()
        with patch.object(
            llm.client.chat.completions,
            "create",
            return_value=sentinel,
        ) as create:
            try:
                result = llm.open_chat_stream(PROVIDER_MESSAGES)
            except (AttributeError, TypeError):
                self.fail("open_chat_stream does not accept provider messages")

        self.assertIs(result, sentinel)
        self._assert_request(create)

    def test_stream_chat_retains_provider_messages_and_content_streaming(self):
        stream = [
            SimpleNamespace(choices=[]),
            SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content="第一段"),
            )]),
            SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content="第二段"),
            )]),
        ]
        with patch.object(
            llm.client.chat.completions,
            "create",
            return_value=stream,
        ) as create:
            try:
                content = list(llm.stream_chat(PROVIDER_MESSAGES))
            except (AttributeError, TypeError):
                self.fail("stream_chat does not accept provider messages")

        self.assertEqual(content, ["第一段", "第二段"])
        self._assert_request(create)


if __name__ == "__main__":
    unittest.main()
