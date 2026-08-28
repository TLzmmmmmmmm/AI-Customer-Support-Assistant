import os
import unittest

from pydantic import ValidationError


os.environ["DEEPSEEK_API_KEY"] = "test-placeholder-key"

from models import ChatRequest


class ChatRequestValidationTests(unittest.TestCase):
    def test_trims_content_before_applying_length_limits(self):
        payload = ChatRequest.model_validate({
            "messages": [
                {
                    "role": "user",
                    "content": f"  {'x' * 4000}\n",
                },
            ],
        })

        self.assertEqual(payload.messages[0].content, "x" * 4000)

    def test_rejects_whitespace_only_content(self):
        with self.assertRaises(ValidationError):
            ChatRequest.model_validate({
                "messages": [
                    {
                        "role": "user",
                        "content": " \n\t ",
                    },
                ],
            })

    def test_accepts_valid_user_terminated_role_sequences(self):
        valid_sequences = [
            [
                {"role": "user", "content": "first"},
            ],
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "follow-up"},
            ],
        ]

        for messages in valid_sequences:
            with self.subTest(messages=messages):
                payload = ChatRequest.model_validate({"messages": messages})
                self.assertEqual(len(payload.messages), len(messages))

    def test_rejects_an_assistant_as_the_first_message(self):
        with self.assertRaises(ValidationError):
            ChatRequest.model_validate({
                "messages": [
                    {"role": "assistant", "content": "answer"},
                    {"role": "user", "content": "question"},
                ],
            })

    def test_rejects_non_alternating_roles(self):
        invalid_sequences = [
            [
                {"role": "user", "content": "first"},
                {"role": "user", "content": "second"},
            ],
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "assistant", "content": "another answer"},
                {"role": "user", "content": "follow-up"},
            ],
        ]

        for messages in invalid_sequences:
            with self.subTest(messages=messages):
                with self.assertRaises(ValidationError):
                    ChatRequest.model_validate({"messages": messages})

    def test_rejects_an_assistant_as_the_last_message(self):
        with self.assertRaises(ValidationError):
            ChatRequest.model_validate({
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ],
            })


if __name__ == "__main__":
    unittest.main()
