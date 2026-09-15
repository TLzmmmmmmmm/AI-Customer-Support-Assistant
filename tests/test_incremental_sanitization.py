import unittest


CASES = (
    ("plain-zh", "  普通回答。  ", "普通回答。"),
    ("plain-en-unicode-space", "\u2003English answer.\u00a0", "English answer."),
    (
        "plain-url",
        "详情 https://example.com/a?x=1 后续",
        "详情  后续",
    ),
    ("angle-url", "前<https://example.com/a>后", "前后"),
    (
        "markdown-link",
        "查看[产品页]( https://example.com/a )。",
        "查看产品页。",
    ),
    (
        "url-in-markdown-label",
        "[访问 https://inside.example](https://target.example) 后",
        "访问  后",
    ),
    (
        "invalid-markdown-scheme",
        "[标签](ftp://example.com) 后",
        "[标签](ftp://example.com) 后",
    ),
    (
        "invalid-angle-url",
        "<https://a.example path>尾",
        "< path>尾",
    ),
    (
        "english-reference-heading",
        "正文\n## References:\nhttps://x\n尾",
        "正文",
    ),
    (
        "chinese-reference-heading",
        "正文\n  参考资料：  \n后",
        "正文",
    ),
    (
        "reference-heading-space-before-colon",
        "正文\nReferences :\n后",
        "正文",
    ),
    (
        "heading-like-prose",
        "正文\nReferences are useful\n尾",
        "正文\nReferences are useful\n尾",
    ),
    ("fully-sanitized", "References:\nhttps://x", ""),
    ("incomplete-scheme", "尾 http", "尾 http"),
    ("scheme-without-body", "尾 http://", "尾 http://"),
    ("complete-url-at-eof", "尾 http://x", "尾"),
    ("incomplete-markdown", "[x](https://a", "[x]("),
    ("complete-markdown", "[x](https://a)", "x"),
    ("url-body-adjacent-text", "前www.example.com后", "前"),
)


class BufferedSanitizationCompatibilityTests(unittest.TestCase):
    def test_expected_outputs_are_frozen_independently(self):
        from citation import sanitize_generated_answer

        for name, raw, expected in CASES:
            with self.subTest(name=name):
                self.assertEqual(sanitize_generated_answer(raw), expected)
if __name__ == "__main__":
    unittest.main()
