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


class IncrementalCitationMarkerFilterTests(unittest.TestCase):
    def test_removes_complete_markers_across_every_chunk_boundary(self):
        from citation import IncrementalCitationMarkerFilter

        marker = "【C_0123456789abcdef】"
        raw = f"前文{marker}后文"
        for split_at in range(len(raw) + 1):
            with self.subTest(split_at=split_at):
                marker_filter = IncrementalCitationMarkerFilter()
                visible = [
                    *marker_filter.feed(raw[:split_at]),
                    *marker_filter.feed(raw[split_at:]),
                    marker_filter.finish(),
                ]
                self.assertEqual("".join(visible), "前文后文")

    def test_preserves_incomplete_or_invalid_markers(self):
        from citation import IncrementalCitationMarkerFilter

        for raw in ("正文【C_0123", "正文【C_0123456789abcdeg】结尾"):
            with self.subTest(raw=raw):
                marker_filter = IncrementalCitationMarkerFilter()
                visible = [
                    *marker_filter.feed(raw),
                    marker_filter.finish(),
                ]
                self.assertEqual("".join(visible), raw)
if __name__ == "__main__":
    unittest.main()
