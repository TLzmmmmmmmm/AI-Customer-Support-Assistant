import random
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


def _stream(raw: str, cut_points: tuple[int, ...]):
    from citation import IncrementalAnswerSanitizer

    sanitizer = IncrementalAnswerSanitizer()
    emitted: list[str] = []
    start = 0
    snapshots: list[str] = []
    for end in (*cut_points, len(raw)):
        emitted.extend(sanitizer.feed(raw[start:end]))
        snapshots.append("".join(emitted))
        start = end
    emitted.append(sanitizer.finish())
    return "".join(emitted), snapshots


class BufferedSanitizationCompatibilityTests(unittest.TestCase):
    def test_expected_outputs_are_frozen_independently(self):
        from citation import sanitize_generated_answer

        for name, raw, expected in CASES:
            with self.subTest(name=name):
                self.assertEqual(sanitize_generated_answer(raw), expected)


class IncrementalSanitizationTests(unittest.TestCase):
    def _assert_partition(self, raw: str, expected: str, cuts: tuple[int, ...]):
        actual, snapshots = _stream(raw, cuts)

        self.assertEqual(actual, expected)
        for snapshot in snapshots:
            self.assertTrue(
                expected.startswith(snapshot),
                msg=f"emitted non-prefix {snapshot!r} for {raw!r} at {cuts!r}",
            )

    def test_every_single_split_matches_buffered_contract_and_is_prefix_safe(self):
        for name, raw, expected in CASES:
            for cut in range(len(raw) + 1):
                with self.subTest(name=name, cut=cut):
                    self._assert_partition(raw, expected, (cut,))

    def test_seeded_random_multi_splits_match_for_every_case(self):
        randomizer = random.Random(20260914)
        for name, raw, expected in CASES:
            for partition in range(50):
                cut_count = randomizer.randint(0, len(raw))
                cuts = tuple(sorted(randomizer.sample(range(len(raw)), cut_count)))
                with self.subTest(name=name, partition=partition, cuts=cuts):
                    self._assert_partition(raw, expected, cuts)

    def test_feed_returns_only_non_empty_deltas(self):
        from citation import IncrementalAnswerSanitizer

        sanitizer = IncrementalAnswerSanitizer()
        self.assertEqual(sanitizer.feed("   https://example.com"), ())
        self.assertEqual(sanitizer.finish(), "")

    def test_fully_sanitized_answer_uses_the_existing_fallback_contract(self):
        from citation import IncrementalAnswerSanitizer, render_answer

        sanitizer = IncrementalAnswerSanitizer()
        self.assertEqual(sanitizer.feed("References:\nhttps://example.com"), ())
        self.assertEqual(sanitizer.finish(), "")

        rendered = render_answer(
            "References:\nhttps://example.com",
            (),
            fallback="Safe fallback",
            language_hint="question",
        )
        self.assertEqual(rendered.text, "Safe fallback")
        self.assertEqual(rendered.sources, ())
        self.assertTrue(rendered.used_fallback)


if __name__ == "__main__":
    unittest.main()
