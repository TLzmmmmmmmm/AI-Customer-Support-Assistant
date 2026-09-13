import unittest

from agent import (
    AgentDeadlineExceeded,
    AgentResult,
    AgentToolCall,
    AgentTurn,
    ToolObservation,
)
from agent.runtime import (
    AGENT_TOOL_POLICY,
    assistant_tool_message,
    attach_agent_error_metadata,
    build_agent_messages,
    ensure_agent_active,
    finalize_agent_result,
    invalid_tool_call_traces,
    record_agent_observation,
    tool_call_batch_rejection,
)
from knowledge_pipeline.models import SourceRef
from trace_models import FailureLayer, ToolTrace


class ExpiredDeadline:
    def __init__(self, error):
        self.error = error

    def ensure_active(self):
        raise self.error


class AgentRuntimeTests(unittest.TestCase):
    def test_agent_policy_requires_current_details_evidence_for_product_facts(self):
        self.assertIn(
            "successful get_product_details observation",
            AGENT_TOOL_POLICY,
        )
        self.assertIn("previous assistant text", AGENT_TOOL_POLICY)
        self.assertIn("model memory", AGENT_TOOL_POLICY)
        self.assertIn("general knowledge", AGENT_TOOL_POLICY)
        self.assertIn("Multiple products explicitly requested", AGENT_TOOL_POLICY)
        self.assertIn("genuinely ambiguous singular reference", AGENT_TOOL_POLICY)

    def test_build_agent_messages_copies_input_and_inserts_policy(self):
        original = [
            {"role": "system", "content": "trust boundary"},
            {"role": "user", "content": "问题"},
        ]

        result = build_agent_messages(original)
        result[-1]["content"] = "changed"

        self.assertEqual(original[-1]["content"], "问题")
        self.assertEqual(result[:2], [
            {"role": "system", "content": "trust boundary"},
            {"role": "system", "content": AGENT_TOOL_POLICY},
        ])

    def test_assistant_tool_message_preserves_proposal(self):
        turn = AgentTurn(
            content="internal action",
            tool_calls=(AgentToolCall(
                id="call-1",
                name="search_products",
                arguments='{"query":"酒店"}',
            ),),
            finish_reason="tool_calls",
        )

        self.assertEqual(assistant_tool_message(turn), {
            "role": "assistant",
            "content": "internal action",
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "search_products",
                    "arguments": '{"query":"酒店"}',
                },
            }],
        })

    def test_assistant_tool_message_serializes_all_proposals_in_order(self):
        turn = AgentTurn(
            content=None,
            tool_calls=(
                AgentToolCall("call-1", "get_product_details", '{"product_id":"ly198"}'),
                AgentToolCall("call-2", "get_contact_info", "{}"),
            ),
            finish_reason="tool_calls",
        )

        message = assistant_tool_message(turn)

        self.assertEqual(
            [item["id"] for item in message["tool_calls"]],
            ["call-1", "call-2"],
        )
        self.assertEqual(
            [item["function"]["name"] for item in message["tool_calls"]],
            ["get_product_details", "get_contact_info"],
        )
        self.assertEqual(
            [item["function"]["arguments"] for item in message["tool_calls"]],
            ['{"product_id":"ly198"}', "{}"],
        )

    def test_tool_call_batch_rejection_checks_unique_ids_and_whole_budget(self):
        one = (AgentToolCall("call-1", "get_contact_info", "{}"),)
        three = (
            *one,
            AgentToolCall("call-2", "get_contact_info", "{}"),
            AgentToolCall("call-3", "get_contact_info", "{}"),
        )
        duplicate = (
            AgentToolCall("call-1", "get_contact_info", "{}"),
            AgentToolCall("call-1", "get_product_details", "{}"),
        )

        self.assertIsNone(tool_call_batch_rejection(
            one, processed_calls=0, max_tool_calls=3,
        ))
        self.assertIsNone(tool_call_batch_rejection(
            three, processed_calls=0, max_tool_calls=3,
        ))
        self.assertIn("duplicate", tool_call_batch_rejection(
            duplicate, processed_calls=0, max_tool_calls=3,
        ).lower())
        self.assertEqual(
            tool_call_batch_rejection(
                three[:2], processed_calls=2, max_tool_calls=3,
            ),
            "Tool-call batch size 2 exceeds remaining budget 1.",
        )

    def test_invalid_proposal_traces_preserve_order_and_hide_unknown_name(self):
        calls = (
            AgentToolCall("call-1", "get_contact_info", "{}"),
            AgentToolCall("call-2", "private_tool", "{}"),
        )

        self.assertEqual(invalid_tool_call_traces(calls), (
            ToolTrace(
                name="get_contact_info",
                success=False,
                error_code="INVALID_ARGUMENT",
            ),
            ToolTrace(
                name="tool_executor",
                success=False,
                error_code="INVALID_ARGUMENT",
            ),
        ))

    def test_successful_observation_recovers_only_current_agent_failures(self):
        source_a = SourceRef(title="A", url="https://example.com/a")
        source_b = SourceRef(title="B", url="https://example.com/b")
        observation = ToolObservation(
            content='{"ok":true}',
            success=True,
            reused=True,
            cache_key="key",
            tool_name="search_products",
            sources=(source_a, source_b),
        )
        existing_trace = ToolTrace(
            name="search_products",
            success=False,
            error_code="INVALID_ARGUMENT",
        )

        traces, sources, pending = record_agent_observation(
            observation,
            tool_traces=(existing_trace,),
            successful_sources=(source_b,),
            pending_failures=(
                (FailureLayer.TOOL_SELECTION, "tool_executor"),
                (FailureLayer.ARGUMENT_GENERATION, "search_products"),
                (FailureLayer.TOOL_EXECUTION, "search_products"),
                (FailureLayer.TOOL_EXECUTION, "get_contact_info"),
            ),
        )

        self.assertEqual(traces, (
            existing_trace,
            ToolTrace(
                name="search_products",
                success=True,
                reused=True,
            ),
        ))
        self.assertEqual(sources, (source_b, source_a, source_b))
        self.assertEqual(pending, (
            (FailureLayer.TOOL_EXECUTION, "get_contact_info"),
        ))

    def test_failed_observation_appends_only_classified_failure(self):
        execution_failure = ToolObservation(
            content='{"ok":false}',
            success=False,
            reused=False,
            cache_key=None,
            tool_name="get_contact_info",
            error_code="TOOL_EXECUTION_ERROR",
        )
        domain_outcome = ToolObservation(
            content='{"ok":false}',
            success=False,
            reused=False,
            cache_key=None,
            tool_name="get_product_details",
            error_code="PRODUCT_NOT_FOUND",
        )

        traces, sources, pending = record_agent_observation(
            execution_failure,
            tool_traces=(),
            successful_sources=(),
            pending_failures=(),
        )
        traces, sources, pending = record_agent_observation(
            domain_outcome,
            tool_traces=traces,
            successful_sources=sources,
            pending_failures=pending,
        )

        self.assertEqual([trace.error_code for trace in traces], [
            "TOOL_EXECUTION_ERROR",
            "PRODUCT_NOT_FOUND",
        ])
        self.assertEqual(sources, ())
        self.assertEqual(pending, (
            (FailureLayer.TOOL_EXECUTION, "get_contact_info"),
        ))

    def test_finalize_agent_result_uses_explicit_then_first_pending_failure(self):
        trace = ToolTrace("get_contact_info", False, "INVALID_ARGUMENT")
        source = SourceRef(title="A", url="https://example.com/a")
        inputs = {
            "tool_traces": (trace,),
            "successful_sources": (source,),
            "pending_failures": (
                (FailureLayer.TOOL_SELECTION, "tool_executor"),
                (FailureLayer.TOOL_EXECUTION, "get_contact_info"),
            ),
        }

        pending_result = finalize_agent_result("answer", **inputs)
        explicit_result = finalize_agent_result(
            "safe",
            failure=FailureLayer.GENERATION,
            **inputs,
        )

        self.assertEqual(pending_result, AgentResult(
            answer="answer",
            tool_calls=(trace,),
            failure_layer=FailureLayer.TOOL_SELECTION,
            sources=(source,),
        ))
        self.assertEqual(
            explicit_result.failure_layer,
            FailureLayer.GENERATION,
        )
        self.assertEqual(explicit_result.sources, (source,))

    def test_deadline_failure_receives_current_agent_metadata(self):
        expected = AgentDeadlineExceeded("expired")
        traces = (ToolTrace("get_contact_info", True),)

        with self.assertRaises(AgentDeadlineExceeded) as caught:
            ensure_agent_active(
                ExpiredDeadline(expected),
                layer=FailureLayer.GENERATION,
                tool_traces=traces,
            )

        self.assertIs(caught.exception, expected)
        self.assertEqual(
            caught.exception.failure_layer,
            FailureLayer.GENERATION,
        )
        self.assertEqual(caught.exception.tool_calls, traces)

    def test_provider_error_metadata_preserves_original_exception(self):
        expected = RuntimeError("provider unavailable")
        traces = (ToolTrace("search_products", True),)

        attach_agent_error_metadata(
            expected,
            layer=FailureLayer.TOOL_SELECTION,
            tool_traces=traces,
        )

        self.assertEqual(
            expected.failure_layer,
            FailureLayer.TOOL_SELECTION,
        )
        self.assertEqual(expected.tool_calls, traces)


if __name__ == "__main__":
    unittest.main()
