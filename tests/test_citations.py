import unittest

from knowledge_pipeline.models import SourceRef


class CitationCollectorTests(unittest.TestCase):
    def test_three_chunks_from_same_page_collapse_to_one_source(self):
        from citation import collect_sources

        sources = [
            SourceRef(title="Page", url="https://example.com/page")
            for _ in range(3)
        ]

        result = collect_sources(sources)

        self.assertEqual(result.sources, (sources[0],))
        self.assertEqual(result.deduplicated_count, 2)

    def test_multi_tool_duplicates_produce_stable_a_b_c_order(self):
        from citation import collect_sources

        source_a = SourceRef(title="A", url="https://example.com/a")
        source_b = SourceRef(title="B", url="https://example.com/b")
        source_c = SourceRef(title="C", url="https://example.com/c")

        result = collect_sources([source_a, source_b, source_b, source_c])

        self.assertEqual(result.sources, (source_a, source_b, source_c))
    def test_deduplicates_normalized_urls_in_first_seen_order(self):
        from citation import collect_sources

        first = SourceRef(
            title="First title",
            url="HTTPS://Example.COM/products/ly198 ",
        )
        duplicate = SourceRef(
            title="Second title",
            url="https://example.com/products/ly198",
        )
        last = SourceRef(
            title="Contact",
            url="https://example.com/contact?channel=phone#office",
        )

        result = collect_sources([first, duplicate, last])

        self.assertEqual(result.sources, (first, last))
        self.assertEqual(result.input_count, 3)
        self.assertEqual(result.invalid_source_count, 0)
        self.assertEqual(result.deduplicated_count, 1)

    def test_does_not_assume_distinct_trailing_slash_paths_are_equivalent(self):
        from citation import collect_sources

        without_slash = SourceRef(
            title="Without slash",
            url="https://example.com/products/ly198",
        )
        with_slash = SourceRef(
            title="With slash",
            url="https://example.com/products/ly198/",
        )

        result = collect_sources([without_slash, with_slash])

        self.assertEqual(result.sources, (without_slash, with_slash))
        self.assertEqual(result.deduplicated_count, 0)

    def test_drops_invalid_sources_without_losing_valid_source(self):
        from citation import collect_sources

        valid = SourceRef(title="LY198", url="https://example.com/ly198")
        result = collect_sources([
            {"title": "", "url": "https://fake.example"},
            {"title": "Fake", "url": "javascript:alert(1)"},
            valid,
        ])

        self.assertEqual(result.sources, (valid,))
        self.assertEqual(result.input_count, 3)
        self.assertEqual(result.invalid_source_count, 2)
        self.assertEqual(result.deduplicated_count, 0)


class GeneratedAnswerGuardTests(unittest.TestCase):
    def test_removes_common_urls_and_keeps_markdown_label(self):
        from citation import sanitize_generated_answer

        answer = (
            "请查看 [产品详情](https://fake.example/ly198)，或访问 "
            "www.fake.example/help 和 <https://fake.example/contact>。"
        )

        self.assertEqual(
            sanitize_generated_answer(answer),
            "请查看 产品详情，或访问  和 。",
        )

    def test_removes_standalone_model_generated_reference_section(self):
        from citation import sanitize_generated_answer

        answer = (
            "LY198 的输出功率为 5W。\n\n"
            "### 参考资料：\n"
            "Fake：https://fake.example/ly198\n"
            "Other: https://fake.example/other"
        )

        self.assertEqual(
            sanitize_generated_answer(answer),
            "LY198 的输出功率为 5W。",
        )


class CitationFormatterTests(unittest.TestCase):
    def test_formats_chinese_and_english_headings_with_original_titles(self):
        from citation import format_citations

        sources = (SourceRef(
            title="LY198 产品详情",
            url="https://example.com/ly198",
        ),)

        self.assertEqual(
            format_citations(sources, language_text="产品回答"),
            "参考资料：\nLY198 产品详情：https://example.com/ly198",
        )
        self.assertEqual(
            format_citations(sources, language_text="Product answer"),
            "References:\nLY198 产品详情: https://example.com/ly198",
        )

    def test_empty_sources_format_to_empty_text(self):
        from citation import format_citations

        self.assertEqual(format_citations((), language_text="answer"), "")

    def test_ambiguous_answer_language_uses_latest_question_hint(self):
        from citation import format_citations

        sources = (SourceRef(
            title="LY198",
            url="https://example.com/ly198",
        ),)

        self.assertEqual(
            format_citations(
                sources,
                language_text="LY198 5W",
                language_hint="LY198 的功率是多少？",
            ),
            "参考资料：\nLY198：https://example.com/ly198",
        )


class CitationRenderingTests(unittest.TestCase):
    def test_all_invalid_sources_keep_answer_without_empty_reference_section(self):
        from citation import render_answer

        result = render_answer(
            "Supported answer",
            (
                {"title": "", "url": "https://fake.example"},
                {"title": "Fake", "url": "not-a-url"},
            ),
            fallback="Safe fallback",
            language_hint="question",
        )

        self.assertEqual(result.text, "Supported answer")
        self.assertEqual(result.sources, ())
        self.assertEqual(result.invalid_source_count, 2)
        self.assertNotIn("References:", result.text)

    def test_trusted_url_in_model_body_is_removed_then_backend_appends_it(self):
        from citation import render_answer

        source = SourceRef(
            title="Product",
            url="https://trusted.example/product",
        )

        result = render_answer(
            "Read https://trusted.example/product",
            (source,),
            fallback="Safe fallback",
            language_hint="question",
        )

        self.assertEqual(
            result.text,
            "Read\n\nReferences:\n"
            "Product: https://trusted.example/product",
        )
        self.assertEqual(result.text.count(source.url), 1)

    def test_renders_clean_answer_and_trusted_references(self):
        from citation import render_answer

        trusted = SourceRef(
            title="LY198 产品详情",
            url="https://trusted.example/ly198",
        )

        result = render_answer(
            "详情见 https://fake.example/ly198",
            [trusted],
            fallback="安全回答",
            language_hint="LY198 功率是多少？",
        )

        self.assertEqual(
            result.text,
            "详情见\n\n参考资料：\n"
            "LY198 产品详情：https://trusted.example/ly198",
        )
        self.assertEqual(result.sources, (trusted,))
        self.assertEqual(result.invalid_source_count, 0)
        self.assertEqual(result.deduplicated_count, 0)
        self.assertTrue(result.answer_sanitized)
        self.assertFalse(result.used_fallback)

    def test_uses_fallback_only_when_sanitized_answer_is_empty(self):
        from citation import render_answer

        result = render_answer(
            "References:\nhttps://fake.example",
            (),
            fallback="Safe fallback",
            language_hint="question",
        )

        self.assertEqual(result.text, "Safe fallback")
        self.assertTrue(result.answer_sanitized)
        self.assertTrue(result.used_fallback)


if __name__ == "__main__":
    unittest.main()
