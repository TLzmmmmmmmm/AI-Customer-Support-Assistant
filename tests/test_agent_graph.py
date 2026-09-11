import json
import unittest
from collections import deque
from contextlib import nullcontext
from copy import deepcopy
from types import MappingProxyType, SimpleNamespace
from unittest.mock import patch

from agent import (
    AGENT_TOOL_POLICY,
    AgentDeadlineExceeded,
    AgentLoop,
    SAFE_AGENT_ANSWER,
    ToolExecutor,
    ToolObservation,
    normalize_agent_turn,
)
from agent_graph import (
    AgentGraphNodes,
    GraphRouteOrchestrator,
    build_agent_graph,
)
from knowledge_pipeline.models import (
    SourceRef,
    TechnicalParameterGroup,
    TechnicalParameterItem,
)
from knowledge_pipeline.retrieval.models import RetrievalResult
from models import ChatMessage
from prompts import build_direct_messages
from routing import (
    SAFE_FALLBACK_ANSWER,
    FailureLayer,
    HybridRouter,
    Route,
    RouteDecision,
    RouteExecutionResult,
    RouteOrchestrator,
    RoutingResult,
    ToolTrace,
)
from support_tools import (
    ContactInfoResult,
    ProductDetailsResult,
    ProductSearchItem,
    ProductSearchResult,
    ToolError,
    ToolErrorCode,
)


class RecordingRouter:
    def __init__(self, decision, visited, failure=None):
        self.decision = decision
        self.visited = visited
        self.failure = failure

    def route(self, messages, *, deadline):
        self.visited.append("route")
        return RoutingResult(self.decision, self.failure)


class RecordingRetriever:
    def __init__(self, results=(), *, visited=None, error=None):
        self.results = list(results)
        self.visited = visited
        self.error = error
        self.queries = []

    def resolve_entities(self, query):
        self.queries.append(query)
        return []

    def retrieve(self, query):
        if self.visited is not None:
            self.visited.append("retrieve")
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return list(self.results)


class RecordingCompletion:
    def __init__(self, responses, *, visited=None, label=None):
        self.responses = deque(responses)
        self.calls = []
        self.visited = visited
        self.label = label

    def __call__(self, messages, *, tools=None):
        if self.visited is not None and self.label is not None:
            self.visited.append(self.label)
        self.calls.append((deepcopy(messages), deepcopy(tools)))
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response


class KeywordRecordingCompletion:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append((deepcopy(messages), deepcopy(kwargs)))
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response


class RecordingDeadline:
    def __init__(self, fail_on=None, *, error=None):
        self.checks = 0
        self.fail_on = fail_on
        self.error = error or AgentDeadlineExceeded("expired")

    def ensure_active(self):
        self.checks += 1
        if self.checks == self.fail_on:
            raise self.error


class RecordingGraphExecutor:
    def __init__(self, visited, observations=()):
        self.visited = visited
        self.observations = deque(observations)

    def execute(self, call, cache):
        self.visited.append("execute_tool")
        if self.observations:
            return self.observations.popleft()
        return ToolObservation(
            content='{"ok":true}',
            success=True,
            reused=False,
            cache_key="key",
            tool_name=call.name,
        )

    def execute_named(self, name, arguments, cache):
        self.visited.append("deterministic_tool")
        return ToolObservation(
            content='{"ok":true}',
            success=True,
            reused=False,
            cache_key="key",
            tool_name=name,
        )


def _completion(content):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="stop",
        message=SimpleNamespace(content=content, tool_calls=None),
    )])


def _tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _tool_completion(call_id, name, arguments, *, content=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="tool_calls",
        message=SimpleNamespace(
            content=content,
            tool_calls=[_tool_call(call_id, name, arguments)],
        ),
    )])


def _multiple_tool_completion(*calls):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="tool_calls",
        message=SimpleNamespace(content=None, tool_calls=list(calls)),
    )])


def _initial_state(*, deadline=None, question="测试问题"):
    message = ChatMessage(role="user", content=question)
    return {
        "messages": (message,),
        "last_user_message": message,
        "deadline": deadline or RecordingDeadline(),
        "retrieval_hits": (),
        "tool_call": None,
        "tool_result": None,
        "answer": None,
        "sources": (),
        "error": None,
    }


def _retrieval_result(source, *, chunk_id="solution:test:content"):
    return RetrievalResult.model_validate({
        "rank": 1,
        "score": 0.9,
        "match_origin": "dense",
        "matched_entity_ids": [],
        "chunk_id": chunk_id,
        "parent_document_id": chunk_id.rsplit(":", 1)[0],
        "type": "solution",
        "section": "方案介绍",
        "text": "# 测试方案\n\n方案内容。",
        "content_hash": "a" * 64,
        "metadata": {
            "solution_id": "test",
            "slug": "test",
        },
        "source_url": source.url,
        "source_files": ["src/content/solutions/test.md"],
        "sources": [source],
    })


def _nodes(
    visited,
    *,
    decision=None,
    router=None,
    executor=None,
    retriever=None,
    complete_chat=None,
    routing_failure=None,
):
    decision = decision or RouteDecision(Route.DIRECT)
    if router is None:
        router = RecordingRouter(
            decision,
            visited,
            routing_failure,
        )
    if complete_chat is None:
        if decision.agentic and decision.route != Route.FALLBACK:
            generation_label = "agent_step"
        elif decision.route == Route.KNOWLEDGE:
            generation_label = "rag_generate"
        elif decision.route in {
            Route.PRODUCT_SEARCH,
            Route.EXACT_PRODUCT,
            Route.CONTACT,
        }:
            generation_label = "deterministic_generate"
        else:
            generation_label = "direct"
        complete_chat = RecordingCompletion(
            [_completion("generated answer")],
            visited=visited,
            label=generation_label,
        )
    return AgentGraphNodes(
        router=router,
        executor=executor or RecordingGraphExecutor(visited),
        retriever=retriever or RecordingRetriever(visited=visited),
        complete_chat=complete_chat,
    )


class AgentGraphTests(unittest.TestCase):
    def test_rag_path_retrieves_before_generation_and_finalization(self):
        visited = []
        graph = build_agent_graph(_nodes(
            visited,
            decision=RouteDecision(Route.KNOWLEDGE),
        ))

        result = graph.invoke(_initial_state())

        self.assertEqual(
            visited,
            ["route", "retrieve", "rag_generate"],
        )
        self.assertIsInstance(result["result"], RouteExecutionResult)

    def test_rag_path_stores_real_hits_and_overwrites_sources(self):
        visited = []
        source = SourceRef(
            title="测试方案",
            url="https://example.com/solutions/test/",
        )
        stale = SourceRef(
            title="旧资料",
            url="https://example.com/stale/",
        )
        hit = _retrieval_result(source)
        deadline = RecordingDeadline()
        graph = build_agent_graph(_nodes(
            visited,
            decision=RouteDecision(Route.KNOWLEDGE),
            retriever=RecordingRetriever([hit], visited=visited),
        ))
        state = _initial_state(deadline=deadline, question="最新问题")
        state["sources"] = (stale,)

        result = graph.invoke(state)

        self.assertEqual(result["retrieval_hits"], (hit,))
        self.assertIsInstance(result["retrieval_hits"], tuple)
        self.assertEqual(result["sources"], (source,))
        self.assertEqual(deadline.checks, 4)
        self.assertEqual(
            visited,
            ["route", "retrieve", "rag_generate"],
        )
        self.assertEqual(
            result["result"].trace.retrieved_chunk_ids,
            ("solution:test:content",),
        )
        self.assertEqual(result["result"].sources, (source,))

    def test_rag_path_zero_hits_overwrites_state_with_empty_tuples(self):
        visited = []
        stale = SourceRef(
            title="旧资料",
            url="https://example.com/stale/",
        )
        graph = build_agent_graph(_nodes(
            visited,
            decision=RouteDecision(Route.KNOWLEDGE),
            retriever=RecordingRetriever(visited=visited),
        ))
        state = _initial_state()
        state["retrieval_hits"] = (_retrieval_result(stale),)
        state["sources"] = (stale,)

        result = graph.invoke(state)

        self.assertEqual(result["retrieval_hits"], ())
        self.assertEqual(result["sources"], ())
        self.assertEqual(result["result"].trace.retrieved_chunk_ids, ())
        self.assertEqual(result["result"].sources, ())

    def test_rag_path_adds_raw_retrieval_exception_metadata(self):
        visited = []
        expected = RuntimeError("retrieval unavailable")
        graph = build_agent_graph(_nodes(
            visited,
            decision=RouteDecision(Route.KNOWLEDGE),
            retriever=RecordingRetriever(visited=visited, error=expected),
        ))

        with self.assertRaises(RuntimeError) as caught:
            graph.invoke(_initial_state())

        self.assertIs(caught.exception, expected)
        self.assertEqual(
            caught.exception.failure_layer,
            FailureLayer.RETRIEVAL,
        )
        self.assertEqual(caught.exception.route, Route.KNOWLEDGE)
        self.assertEqual(caught.exception.tool_calls, ())
        self.assertEqual(caught.exception.retrieved_chunk_ids, ())

    def test_route_exception_adds_only_raw_routing_failure(self):
        expected = RuntimeError("routing unavailable")

        class FailingRouter:
            def route(self, messages, *, deadline):
                raise expected

        graph = build_agent_graph(_nodes([], router=FailingRouter()))

        with self.assertRaises(RuntimeError) as caught:
            graph.invoke(_initial_state())

        self.assertIs(caught.exception, expected)
        self.assertEqual(
            caught.exception.failure_layer,
            FailureLayer.ROUTING,
        )
        self.assertFalse(hasattr(caught.exception, "route"))
        self.assertFalse(hasattr(caught.exception, "tool_calls"))
        self.assertFalse(hasattr(caught.exception, "retrieved_chunk_ids"))

    def test_retrieval_deadlines_use_empty_raw_context(self):
        source = SourceRef(title="资料", url="https://example.com/source")
        hit = _retrieval_result(source)
        for fail_on in (1, 2):
            with self.subTest(fail_on=fail_on):
                expected = AgentDeadlineExceeded(f"deadline {fail_on}")
                graph = build_agent_graph(_nodes(
                    [],
                    decision=RouteDecision(Route.KNOWLEDGE),
                    retriever=RecordingRetriever([hit]),
                ))

                with self.assertRaises(AgentDeadlineExceeded) as caught:
                    graph.invoke(_initial_state(deadline=RecordingDeadline(
                        fail_on,
                        error=expected,
                    )))

                self.assertIs(caught.exception, expected)
                self.assertEqual(
                    caught.exception.failure_layer,
                    FailureLayer.RETRIEVAL,
                )
                self.assertEqual(caught.exception.route, Route.KNOWLEDGE)
                self.assertEqual(caught.exception.tool_calls, ())
                self.assertEqual(caught.exception.retrieved_chunk_ids, ())

    def test_knowledge_prompt_failures_use_empty_retrieval_context(self):
        source = SourceRef(title="资料", url="https://example.com/source")
        hit = _retrieval_result(source)
        for agentic in (False, True):
            with self.subTest(agentic=agentic):
                expected = RuntimeError("context build failed")
                graph = build_agent_graph(_nodes(
                    [],
                    decision=RouteDecision(
                        Route.KNOWLEDGE,
                        agentic=agentic,
                    ),
                    retriever=RecordingRetriever([hit]),
                ))

                with patch(
                    "agent_graph.nodes.build_retrieved_context",
                    side_effect=expected,
                ):
                    with self.assertRaises(RuntimeError) as caught:
                        graph.invoke(_initial_state())

                self.assertIs(caught.exception, expected)
                self.assertEqual(
                    caught.exception.failure_layer,
                    FailureLayer.RETRIEVAL,
                )
                self.assertEqual(caught.exception.route, Route.KNOWLEDGE)
                self.assertEqual(caught.exception.tool_calls, ())
                self.assertEqual(caught.exception.retrieved_chunk_ids, ())

    def test_direct_path_skips_retrieval_and_tools(self):
        visited = []
        graph = build_agent_graph(_nodes(
            visited,
            decision=RouteDecision(Route.DIRECT),
        ))

        result = graph.invoke(_initial_state())

        self.assertEqual(visited, ["route", "direct"])
        self.assertEqual(result["answer"], "generated answer")
        self.assertIsNone(result["generation_failure"])
        self.assertEqual(result["result"].trace.route, Route.DIRECT)
        self.assertEqual(result["result"].sources, ())

    def test_fallback_path_goes_directly_to_finalization(self):
        visited = []
        graph = build_agent_graph(_nodes(
            visited,
            decision=RouteDecision(Route.FALLBACK),
        ))

        result = graph.invoke(_initial_state())

        self.assertEqual(visited, ["route"])
        self.assertEqual(result["answer"], SAFE_FALLBACK_ANSWER)
        self.assertNotIn("generation_failure", result)
        self.assertEqual(result["result"].answer, SAFE_FALLBACK_ANSWER)

    def test_agent_path_executes_one_tool_then_finalizes_agentic_result(self):
        visited = []
        observation = ToolObservation(
            content='{"ok":true}',
            success=True,
            reused=False,
            cache_key="get_contact_info:{}",
            tool_name="get_contact_info",
        )
        complete = RecordingCompletion(
            [
                _tool_completion("call-1", "get_contact_info", "{}"),
                _completion("agent answer"),
            ],
            visited=visited,
            label="agent_step",
        )
        deadline = RecordingDeadline()

        graph = build_agent_graph(_nodes(
            visited,
            decision=RouteDecision(Route.KNOWLEDGE, agentic=True),
            executor=RecordingGraphExecutor(visited, [observation]),
            complete_chat=complete,
        ))

        result = graph.invoke(
            _initial_state(deadline=deadline),
            config={"recursion_limit": 10},
        )

        self.assertEqual(
            visited,
            [
                "route",
                "retrieve",
                "agent_step",
                "execute_tool",
                "agent_step",
            ],
        )
        self.assertEqual(result["answer"], "agent answer")
        self.assertEqual(result["tool_result"], observation)
        self.assertEqual(result["agent_processed_calls"], 1)
        self.assertEqual(result["agent_tool_traces"], (
            ToolTrace("get_contact_info", True),
        ))
        self.assertEqual(result["result"].answer, "agent answer")
        self.assertEqual(result["result"].trace.route, Route.KNOWLEDGE)
        self.assertEqual(result["result"].trace.tool_call_count, 1)
        self.assertEqual(deadline.checks, 8)
        self.assertIsNotNone(complete.calls[0][1])
        self.assertIsNotNone(complete.calls[1][1])
        self.assertIn("BEGIN_RAG_DATA", complete.calls[0][0][-1]["content"])
        self.assertEqual(
            complete.calls[0][0][1],
            {"role": "system", "content": AGENT_TOOL_POLICY},
        )

    def test_non_agentic_routes_follow_production_mapping(self):
        cases = (
            (
                Route.PRODUCT_SEARCH,
                ["route", "deterministic_tool", "deterministic_generate"],
            ),
            (
                Route.EXACT_PRODUCT,
                ["route", "deterministic_tool", "deterministic_generate"],
            ),
            (
                Route.CONTACT,
                ["route", "deterministic_tool", "deterministic_generate"],
            ),
            (Route.KNOWLEDGE, ["route", "retrieve", "rag_generate"]),
            (Route.DIRECT, ["route", "direct"]),
            (Route.FALLBACK, ["route"]),
        )

        for route, expected in cases:
            with self.subTest(route=route):
                visited = []
                graph = build_agent_graph(_nodes(
                    visited,
                    decision=RouteDecision(route, agentic=False),
                ))

                result = graph.invoke(_initial_state())

                self.assertEqual(visited, expected)
                self.assertIsInstance(result["result"], RouteExecutionResult)

    def test_deterministic_routes_store_real_observations_and_sources(self):
        source = SourceRef(
            title="权威资料",
            url="https://example.com/source/",
        )
        calls = []

        def search_products(query):
            calls.append(("search_products", query))
            return ProductSearchResult(products=[ProductSearchItem(
                product_id="ly198",
                name="LY198",
                category_id="two-way-radio",
                category_name="对讲机",
                relevant_content="候选产品",
                sources=[source],
            )])

        def get_product_details(product_id):
            calls.append(("get_product_details", product_id))
            return ProductDetailsResult(
                product_id=product_id,
                name="LY198",
                category_id="two-way-radio",
                category_name="对讲机",
                key_features=["清晰通话"],
                product_features="适用于日常通信。",
                technical_parameters=[TechnicalParameterGroup(
                    group="基本参数",
                    items=[TechnicalParameterItem(name="功率", value="2W")],
                )],
                sources=[source],
            )

        def get_contact_info():
            calls.append(("get_contact_info", None))
            return ContactInfoResult(
                company_name="测试公司",
                duty_phone="4000000000",
                email="support@example.com",
                sources=[source],
            )

        executor = ToolExecutor(MappingProxyType({
            "search_products": search_products,
            "get_product_details": get_product_details,
            "get_contact_info": get_contact_info,
        }))
        cases = (
            (
                RouteDecision(Route.PRODUCT_SEARCH),
                "推荐几款对讲机",
                ("search_products", "推荐几款对讲机"),
            ),
            (
                RouteDecision(Route.EXACT_PRODUCT, product_id="ly198"),
                "这个型号的功率是多少？",
                ("get_product_details", "ly198"),
            ),
            (
                RouteDecision(Route.CONTACT),
                "联系方式是什么？",
                ("get_contact_info", None),
            ),
        )

        for decision, question, expected_call in cases:
            with self.subTest(route=decision.route):
                visited = []
                deadline = RecordingDeadline()
                complete = RecordingCompletion([_completion("工具回答")])
                graph = build_agent_graph(_nodes(
                    visited,
                    decision=decision,
                    executor=executor,
                    complete_chat=complete,
                ))

                result = graph.invoke(_initial_state(
                    deadline=deadline,
                    question=question,
                ))

                self.assertEqual(calls[-1], expected_call)
                self.assertIsInstance(result["tool_result"], ToolObservation)
                self.assertTrue(result["tool_result"].success)
                self.assertIsNone(result["tool_failure"])
                self.assertEqual(result["sources"], (source,))
                self.assertEqual(result["answer"], "工具回答")
                self.assertIsNone(result["generation_failure"])
                self.assertEqual(deadline.checks, 4)
                self.assertEqual(len(complete.calls), 1)
                provider_messages, tools = complete.calls[0]
                self.assertIsNone(tools)
                self.assertEqual(provider_messages[-1]["role"], "user")
                serialized = provider_messages[-1]["content"].split(
                    "BEGIN_TOOL_DATA\n",
                    1,
                )[1].split("\nEND_TOOL_DATA", 1)[0]
                self.assertEqual(
                    json.loads(serialized),
                    {
                        "route": decision.route.value,
                        "tool_observation": json.loads(
                            result["tool_result"].content,
                        ),
                        "user_question": question,
                    },
                )
                execution = result["result"]
                self.assertIsInstance(execution, RouteExecutionResult)
                self.assertEqual(execution.answer, "工具回答")
                self.assertEqual(execution.trace.route, decision.route)
                self.assertEqual(
                    execution.trace.tool_calls,
                    (ToolTrace(
                        name=result["tool_result"].tool_name,
                        success=True,
                        reused=False,
                    ),),
                )
                self.assertEqual(execution.trace.tool_call_count, 1)
                self.assertEqual(execution.trace.retrieved_chunk_ids, ())
                self.assertIsNone(execution.trace.failure_layer)
                self.assertEqual(execution.sources, (source,))
                self.assertEqual(visited, ["route"])

    def test_failed_deterministic_observation_clears_sources(self):
        stale_source = SourceRef(
            title="旧资料",
            url="https://example.com/stale/",
        )

        def missing(product_id):
            raise ToolError(
                code=ToolErrorCode.PRODUCT_NOT_FOUND,
                message="No product matches.",
                tool_name="get_product_details",
            )

        visited = []
        graph = build_agent_graph(_nodes(
            visited,
            decision=RouteDecision(
                Route.EXACT_PRODUCT,
                product_id="missing",
            ),
            executor=ToolExecutor(MappingProxyType({
                "get_product_details": missing,
            })),
        ))
        state = _initial_state()
        state["sources"] = (stale_source,)

        result = graph.invoke(state)

        self.assertFalse(result["tool_result"].success)
        self.assertEqual(result["tool_result"].error_code, "PRODUCT_NOT_FOUND")
        self.assertIsNone(result["tool_failure"])
        self.assertEqual(result["sources"], ())
        self.assertEqual(result["result"].trace.tool_call_count, 1)
        self.assertIsNone(result["result"].trace.failure_layer)
        self.assertEqual(result["result"].sources, ())

    def test_failed_deterministic_execution_writes_classified_tool_failure(self):
        def broken():
            raise RuntimeError("private")

        invalid = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="length",
            message=SimpleNamespace(content="partial", tool_calls=None),
        )])
        graph = build_agent_graph(_nodes(
            [],
            decision=RouteDecision(Route.CONTACT),
            executor=ToolExecutor(MappingProxyType({
                "get_contact_info": broken,
            })),
            complete_chat=RecordingCompletion([invalid]),
        ))

        result = graph.invoke(_initial_state())

        self.assertEqual(result["tool_failure"], FailureLayer.TOOL_EXECUTION)
        self.assertEqual(
            result["generation_failure"],
            FailureLayer.GENERATION,
        )
        self.assertEqual(result["sources"], ())
        self.assertEqual(result["result"].answer, SAFE_AGENT_ANSWER)
        self.assertEqual(
            result["result"].trace.failure_layer,
            FailureLayer.TOOL_EXECUTION,
        )
        self.assertEqual(result["result"].sources, ())

    def test_thrown_deterministic_execution_adds_raw_metadata(self):
        expected = RuntimeError("executor boundary failed")

        class FailingExecutor:
            def execute_named(self, name, arguments, cache):
                raise expected

        graph = build_agent_graph(_nodes(
            [],
            decision=RouteDecision(Route.CONTACT),
            executor=FailingExecutor(),
        ))

        with self.assertRaises(RuntimeError) as caught:
            graph.invoke(_initial_state())

        self.assertIs(caught.exception, expected)
        self.assertEqual(
            caught.exception.failure_layer,
            FailureLayer.TOOL_EXECUTION,
        )
        self.assertEqual(caught.exception.route, Route.CONTACT)
        self.assertEqual(caught.exception.tool_calls, ())
        self.assertEqual(caught.exception.retrieved_chunk_ids, ())

    def test_deterministic_generation_exception_boundaries_add_raw_context(self):
        cases = (
            ("pre_deadline", 3),
            ("provider", None),
            ("post_deadline", 4),
            ("normalization", None),
        )
        for boundary, fail_on in cases:
            with self.subTest(boundary=boundary):
                expected = (
                    AgentDeadlineExceeded(boundary)
                    if fail_on is not None
                    else RuntimeError(boundary)
                )
                complete = RecordingCompletion([
                    expected if boundary == "provider" else _completion("answer"),
                ])
                graph = build_agent_graph(_nodes(
                    [],
                    decision=RouteDecision(Route.CONTACT),
                    complete_chat=complete,
                ))
                state = _initial_state(deadline=RecordingDeadline(
                    fail_on,
                    error=expected,
                ))

                normalization = (
                    patch(
                        "routing.generation.normalize_agent_turn",
                        side_effect=expected,
                    )
                    if boundary == "normalization"
                    else nullcontext()
                )
                with normalization:
                    with self.assertRaises(type(expected)) as caught:
                        graph.invoke(state)

                self.assertIs(caught.exception, expected)
                self.assertEqual(
                    caught.exception.failure_layer,
                    FailureLayer.GENERATION,
                )
                self.assertEqual(caught.exception.route, Route.CONTACT)
                self.assertEqual(caught.exception.tool_calls, (
                    ToolTrace("get_contact_info", True),
                ))
                self.assertEqual(caught.exception.retrieved_chunk_ids, ())

    def test_knowledge_generation_exception_uses_ordered_retrieval_ids(self):
        expected = RuntimeError("provider failed")
        first = SourceRef(title="A", url="https://example.com/a")
        second = SourceRef(title="B", url="https://example.com/b")
        graph = build_agent_graph(_nodes(
            [],
            decision=RouteDecision(Route.KNOWLEDGE),
            retriever=RecordingRetriever([
                _retrieval_result(first, chunk_id="solution:first:content"),
                _retrieval_result(second, chunk_id="solution:second:content"),
            ]),
            complete_chat=RecordingCompletion([expected]),
        ))

        with self.assertRaises(RuntimeError) as caught:
            graph.invoke(_initial_state())

        self.assertIs(caught.exception, expected)
        self.assertEqual(
            caught.exception.failure_layer,
            FailureLayer.GENERATION,
        )
        self.assertEqual(caught.exception.route, Route.KNOWLEDGE)
        self.assertEqual(caught.exception.tool_calls, ())
        self.assertEqual(caught.exception.retrieved_chunk_ids, (
            "solution:first:content",
            "solution:second:content",
        ))

    def test_direct_generation_exception_adds_empty_raw_context(self):
        expected = RuntimeError("provider failed")
        graph = build_agent_graph(_nodes(
            [],
            decision=RouteDecision(Route.DIRECT),
            complete_chat=RecordingCompletion([expected]),
        ))

        with self.assertRaises(RuntimeError) as caught:
            graph.invoke(_initial_state())

        self.assertIs(caught.exception, expected)
        self.assertEqual(
            caught.exception.failure_layer,
            FailureLayer.GENERATION,
        )
        self.assertEqual(caught.exception.route, Route.DIRECT)
        self.assertEqual(caught.exception.tool_calls, ())
        self.assertEqual(caught.exception.retrieved_chunk_ids, ())

    def test_direct_and_tool_prompt_exceptions_remain_unannotated(self):
        cases = (
            (RouteDecision(Route.DIRECT), "build_direct_messages"),
            (RouteDecision(Route.CONTACT), "build_tool_messages"),
        )
        for decision, builder in cases:
            with self.subTest(builder=builder):
                expected = RuntimeError(f"{builder} failed")
                graph = build_agent_graph(_nodes([], decision=decision))

                with patch(
                    f"agent_graph.nodes.{builder}",
                    side_effect=expected,
                ):
                    with self.assertRaises(RuntimeError) as caught:
                        graph.invoke(_initial_state())

                self.assertIs(caught.exception, expected)
                self.assertFalse(hasattr(caught.exception, "failure_layer"))
                self.assertFalse(hasattr(caught.exception, "route"))
                self.assertFalse(hasattr(caught.exception, "tool_calls"))
                self.assertFalse(hasattr(
                    caught.exception,
                    "retrieved_chunk_ids",
                ))

    def test_rag_generation_uses_retrieved_context(self):
        source = SourceRef(
            title="测试方案",
            url="https://example.com/solutions/test/",
        )
        hit = _retrieval_result(source)
        complete = RecordingCompletion([_completion("检索回答")])
        graph = build_agent_graph(_nodes(
            [],
            decision=RouteDecision(Route.KNOWLEDGE),
            retriever=RecordingRetriever([hit]),
            complete_chat=complete,
        ))

        result = graph.invoke(_initial_state(question="方案是什么？"))

        self.assertEqual(result["answer"], "检索回答")
        self.assertIsNone(result["generation_failure"])
        provider_messages, tools = complete.calls[0]
        self.assertIsNone(tools)
        self.assertIn('"retrieved_context":[{', provider_messages[-1]["content"])
        self.assertIn('"section":"方案介绍"', provider_messages[-1]["content"])
        self.assertIn('"text":"# 测试方案\\n\\n方案内容。"', provider_messages[-1]["content"])
        self.assertIn('"type":"solution"', provider_messages[-1]["content"])
        self.assertIn('"user_question":"方案是什么？"', provider_messages[-1]["content"])
        self.assertEqual(result["result"].answer, "检索回答")
        self.assertEqual(result["result"].trace.route, Route.KNOWLEDGE)
        self.assertEqual(
            result["result"].trace.retrieved_chunk_ids,
            ("solution:test:content",),
        )
        self.assertEqual(result["result"].sources, (source,))

    def test_rag_finalization_preserves_hit_and_source_order_and_duplicates(self):
        first = SourceRef(title="A", url="https://example.com/a")
        second = SourceRef(title="B", url="https://example.com/b")
        hits = (
            _retrieval_result(first, chunk_id="solution:first:content"),
            _retrieval_result(second, chunk_id="solution:second:content"),
            _retrieval_result(first, chunk_id="solution:third:content"),
        )
        graph = build_agent_graph(_nodes(
            [],
            decision=RouteDecision(Route.KNOWLEDGE),
            retriever=RecordingRetriever(hits),
        ))

        result = graph.invoke(_initial_state())

        self.assertEqual(
            result["result"].trace.retrieved_chunk_ids,
            (
                "solution:first:content",
                "solution:second:content",
                "solution:third:content",
            ),
        )
        self.assertEqual(result["result"].sources, (first, second, first))

    def test_zero_hit_rag_still_generates(self):
        complete = RecordingCompletion([_completion("无检索结果回答")])
        graph = build_agent_graph(_nodes(
            [],
            decision=RouteDecision(Route.KNOWLEDGE),
            complete_chat=complete,
        ))

        result = graph.invoke(_initial_state())

        self.assertEqual(result["answer"], "无检索结果回答")
        self.assertIsNone(result["generation_failure"])
        self.assertEqual(len(complete.calls), 1)
        self.assertIn(
            '"retrieved_context":[]',
            complete.calls[0][0][-1]["content"],
        )

    def test_direct_generation_preserves_message_history(self):
        complete = RecordingCompletion([_completion("直接回答")])
        first = ChatMessage(role="user", content="第一个问题")
        prior = ChatMessage(role="assistant", content="先前回答")
        current = ChatMessage(role="user", content="继续说明")
        state = _initial_state()
        state["messages"] = (first, prior, current)
        state["last_user_message"] = current
        graph = build_agent_graph(_nodes(
            [],
            decision=RouteDecision(Route.DIRECT),
            complete_chat=complete,
        ))

        result = graph.invoke(state)

        self.assertEqual(result["answer"], "直接回答")
        self.assertIsNone(result["generation_failure"])
        provider_messages, tools = complete.calls[0]
        self.assertIsNone(tools)
        self.assertEqual(
            provider_messages[1:],
            [
                {"role": "user", "content": "第一个问题"},
                {"role": "assistant", "content": "先前回答"},
                {"role": "user", "content": "继续说明"},
            ],
        )
        self.assertEqual(result["result"].answer, "直接回答")
        self.assertEqual(result["result"].trace.route, Route.DIRECT)
        self.assertEqual(result["result"].trace.tool_calls, ())
        self.assertEqual(result["result"].trace.tool_call_count, 0)
        self.assertEqual(result["result"].trace.retrieved_chunk_ids, ())
        self.assertIsNone(result["result"].trace.failure_layer)
        self.assertEqual(result["result"].sources, ())

    def test_invalid_direct_completion_stores_safe_generation_failure(self):
        invalid = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="length",
            message=SimpleNamespace(content="partial", tool_calls=None),
        )])
        graph = build_agent_graph(_nodes(
            [],
            decision=RouteDecision(Route.DIRECT),
            complete_chat=RecordingCompletion([invalid]),
        ))

        result = graph.invoke(_initial_state())

        self.assertEqual(result["answer"], SAFE_AGENT_ANSWER)
        self.assertEqual(
            result["generation_failure"],
            FailureLayer.GENERATION,
        )
        self.assertEqual(result["result"].answer, SAFE_AGENT_ANSWER)
        self.assertEqual(
            result["result"].trace.failure_layer,
            FailureLayer.GENERATION,
        )
        self.assertEqual(result["result"].sources, ())

    def test_invalid_rag_completion_suppresses_retrieval_sources(self):
        source = SourceRef(title="资料", url="https://example.com/source")
        invalid = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="length",
            message=SimpleNamespace(content="partial", tool_calls=None),
        )])
        graph = build_agent_graph(_nodes(
            [],
            decision=RouteDecision(Route.KNOWLEDGE),
            retriever=RecordingRetriever([_retrieval_result(source)]),
            complete_chat=RecordingCompletion([invalid]),
        ))

        result = graph.invoke(_initial_state())

        self.assertEqual(result["sources"], (source,))
        self.assertEqual(result["result"].answer, SAFE_AGENT_ANSWER)
        self.assertEqual(result["result"].sources, ())

    def test_fallback_does_not_call_generation(self):
        complete = RecordingCompletion([_completion("must not be used")])
        graph = build_agent_graph(_nodes(
            [],
            decision=RouteDecision(Route.FALLBACK),
            complete_chat=complete,
        ))

        result = graph.invoke(_initial_state())

        self.assertEqual(result["answer"], SAFE_FALLBACK_ANSWER)
        self.assertEqual(complete.calls, [])
        self.assertNotIn("generation_failure", result)
        self.assertEqual(result["result"].trace.route, Route.FALLBACK)
        self.assertEqual(result["result"].trace.tool_calls, ())
        self.assertEqual(result["result"].trace.tool_call_count, 0)
        self.assertEqual(result["result"].trace.retrieved_chunk_ids, ())
        self.assertIsNone(result["result"].trace.failure_layer)
        self.assertEqual(result["result"].sources, ())

    def test_route_node_stores_routing_failure_for_fallback_finalization(self):
        graph = build_agent_graph(_nodes(
            [],
            decision=RouteDecision(Route.FALLBACK),
            routing_failure=FailureLayer.ROUTING,
        ))

        result = graph.invoke(_initial_state())

        self.assertEqual(result["routing_failure"], FailureLayer.ROUTING)
        self.assertEqual(
            result["result"].trace.failure_layer,
            FailureLayer.ROUTING,
        )

    def test_agentic_non_knowledge_non_fallback_routes_use_agent(self):
        for route in (
            Route.PRODUCT_SEARCH,
            Route.EXACT_PRODUCT,
            Route.CONTACT,
            Route.DIRECT,
        ):
            with self.subTest(route=route):
                visited = []
                complete = RecordingCompletion(
                    [_completion("agent answer")],
                    visited=visited,
                    label="agent_step",
                )
                graph = build_agent_graph(_nodes(
                    visited,
                    decision=RouteDecision(route, agentic=True),
                    complete_chat=complete,
                ))

                result = graph.invoke(_initial_state())

                self.assertEqual(
                    visited,
                    ["route", "agent_step"],
                )
                self.assertEqual(len(complete.calls), 1)
                self.assertNotIn("generation_failure", result)
                self.assertEqual(result["result"].trace.route, route)

    def test_agentic_knowledge_retrieves_before_agent_step(self):
        visited = []
        complete = RecordingCompletion(
            [_completion("知识回答")],
            visited=visited,
            label="agent_step",
        )
        graph = build_agent_graph(_nodes(
            visited,
            decision=RouteDecision(Route.KNOWLEDGE, agentic=True),
            complete_chat=complete,
        ))

        result = graph.invoke(_initial_state())

        self.assertEqual(visited, ["route", "retrieve", "agent_step"])
        self.assertIn("BEGIN_RAG_DATA", complete.calls[0][0][-1]["content"])
        self.assertEqual(result["result"].trace.route, Route.KNOWLEDGE)

    def test_fallback_wins_over_agentic_flag(self):
        visited = []
        complete = RecordingCompletion([_completion("must not be used")])
        graph = build_agent_graph(_nodes(
            visited,
            decision=RouteDecision(Route.FALLBACK, agentic=True),
            complete_chat=complete,
        ))

        result = graph.invoke(_initial_state())

        self.assertEqual(visited, ["route"])
        self.assertEqual(complete.calls, [])
        self.assertEqual(result["result"].answer, SAFE_FALLBACK_ANSWER)

    def test_agent_provider_history_matches_raw_agent_loop(self):
        messages = (
            ChatMessage(role="user", content="第一个问题"),
            ChatMessage(role="assistant", content="先前回答"),
            ChatMessage(role="user", content="怎么联系？"),
        )
        observation = ToolObservation(
            content='{"ok":true}',
            success=True,
            reused=False,
            cache_key="key",
            tool_name="get_contact_info",
        )
        raw_complete = RecordingCompletion([
            _tool_completion("call-1", "get_contact_info", "{}"),
            _completion("联系我们。"),
        ])
        graph_complete = RecordingCompletion([
            _tool_completion("call-1", "get_contact_info", "{}"),
            _completion("联系我们。"),
        ])
        raw = AgentLoop(
            executor=RecordingGraphExecutor([], [observation]),
            complete_chat=raw_complete,
        ).run(
            build_direct_messages(messages),
            deadline=RecordingDeadline(),
        )
        state = _initial_state()
        state["messages"] = messages
        state["last_user_message"] = messages[-1]
        graph = build_agent_graph(_nodes(
            [],
            decision=RouteDecision(Route.CONTACT, agentic=True),
            executor=RecordingGraphExecutor([], [observation]),
            complete_chat=graph_complete,
        ))

        graph_state = graph.invoke(state)

        self.assertEqual(graph_complete.calls, raw_complete.calls)
        self.assertEqual(graph_state["result"].answer, raw.answer)
        self.assertEqual(graph_state["result"].trace.tool_calls, raw.tool_calls)
        self.assertEqual(
            graph_state["result"].trace.failure_layer,
            raw.failure_layer,
        )
        self.assertEqual(graph_state["result"].sources, raw.sources)

    def test_agent_cache_hits_consume_three_slots_then_disable_tools(self):
        source = SourceRef(title="联系资料", url="https://example.com/contact")
        calls = []

        def get_contact_info():
            calls.append("called")
            return ContactInfoResult(
                company_name="测试公司",
                duty_phone="4000000000",
                email="support@example.com",
                sources=[source],
            )

        complete = KeywordRecordingCompletion([
            _tool_completion(f"call-{index}", "get_contact_info", "{}")
            for index in range(1, 4)
        ] + [_completion("最终回答")])
        graph = build_agent_graph(_nodes(
            [],
            decision=RouteDecision(Route.CONTACT, agentic=True),
            executor=ToolExecutor(MappingProxyType({
                "get_contact_info": get_contact_info,
            })),
            complete_chat=complete,
        ))

        result = graph.invoke(
            _initial_state(),
            config={"recursion_limit": 12},
        )

        self.assertEqual(calls, ["called"])
        self.assertEqual(result["agent_processed_calls"], 3)
        self.assertEqual(
            [trace.reused for trace in result["agent_tool_traces"]],
            [False, True, True],
        )
        self.assertEqual(result["agent_successful_sources"], (
            source,
            source,
            source,
        ))
        self.assertEqual(len(result["agent_successful_observations"]), 1)
        self.assertTrue(all("tools" in call[1] for call in complete.calls[:3]))
        self.assertNotIn("tools", complete.calls[3][1])
        self.assertEqual(result["result"].sources, (source, source, source))

    def test_compiled_graph_does_not_share_tool_cache_between_requests(self):
        calls = []

        def get_contact_info():
            calls.append("called")
            return ContactInfoResult(
                company_name="测试公司",
                duty_phone="4000000000",
                email="support@example.com",
            )

        complete = KeywordRecordingCompletion([
            _tool_completion("call-1", "get_contact_info", "{}"),
            _completion("第一次回答"),
            _tool_completion("call-2", "get_contact_info", "{}"),
            _completion("第二次回答"),
        ])
        graph = build_agent_graph(_nodes(
            [],
            decision=RouteDecision(Route.CONTACT, agentic=True),
            executor=ToolExecutor(MappingProxyType({
                "get_contact_info": get_contact_info,
            })),
            complete_chat=complete,
        ))
        orchestrator = GraphRouteOrchestrator(graph)
        messages = [ChatMessage(role="user", content="怎么联系公司？")]

        first = orchestrator.run(
            messages,
            deadline=RecordingDeadline(),
        )
        second = orchestrator.run(
            messages,
            deadline=RecordingDeadline(),
        )

        self.assertEqual(calls, ["called", "called"])
        self.assertEqual(
            [trace.reused for trace in first.trace.tool_calls],
            [False],
        )
        self.assertEqual(
            [trace.reused for trace in second.trace.tool_calls],
            [False],
        )

    def test_disabled_tools_completion_cannot_execute_fourth_call(self):
        visited = []
        complete = KeywordRecordingCompletion([
            _tool_completion(
                f"call-{index}",
                "get_contact_info",
                "{}",
            )
            for index in range(1, 5)
        ])
        graph = build_agent_graph(_nodes(
            visited,
            decision=RouteDecision(Route.CONTACT, agentic=True),
            executor=RecordingGraphExecutor(visited),
            complete_chat=complete,
        ))

        result = graph.invoke(
            _initial_state(),
            config={"recursion_limit": 12},
        )

        self.assertEqual(visited.count("execute_tool"), 3)
        self.assertEqual(result["agent_processed_calls"], 3)
        self.assertEqual(len(result["agent_tool_traces"]), 4)
        self.assertFalse(result["agent_tool_traces"][-1].success)
        self.assertNotIn("tools", complete.calls[3][1])
        self.assertEqual(result["result"].answer, SAFE_AGENT_ANSWER)
        self.assertEqual(
            result["result"].trace.failure_layer,
            FailureLayer.GENERATION,
        )

    def test_successful_retry_clears_argument_failure(self):
        def search_products(query):
            return ProductSearchResult(products=[])

        complete = RecordingCompletion([
            _tool_completion("call-1", "search_products", '{"query":'),
            _tool_completion(
                "call-2",
                "search_products",
                '{"query":"酒店"}',
            ),
            _completion("候选回答"),
        ])
        graph = build_agent_graph(_nodes(
            [],
            decision=RouteDecision(Route.PRODUCT_SEARCH, agentic=True),
            executor=ToolExecutor(MappingProxyType({
                "search_products": search_products,
            })),
            complete_chat=complete,
        ))

        result = graph.invoke(_initial_state())

        self.assertEqual(
            [trace.error_code for trace in result["agent_tool_traces"]],
            ["INVALID_ARGUMENT", None],
        )
        self.assertEqual(result["agent_pending_failures"], ())
        self.assertIsNone(result["result"].trace.failure_layer)

    def test_unrecovered_tool_execution_failure_reaches_final_result(self):
        observation = ToolObservation(
            content='{"ok":false}',
            success=False,
            reused=False,
            cache_key=None,
            tool_name="get_contact_info",
            error_code="TOOL_EXECUTION_ERROR",
        )
        graph = build_agent_graph(_nodes(
            [],
            decision=RouteDecision(Route.CONTACT, agentic=True),
            executor=RecordingGraphExecutor([], [observation]),
            complete_chat=RecordingCompletion([
                _tool_completion("call-1", "get_contact_info", "{}"),
                _completion("暂时没有联系信息。"),
            ]),
        ))

        result = graph.invoke(_initial_state())

        self.assertEqual(result["agent_pending_failures"], (
            (FailureLayer.TOOL_EXECUTION, "get_contact_info"),
        ))
        self.assertEqual(
            result["result"].trace.failure_layer,
            FailureLayer.TOOL_EXECUTION,
        )
        self.assertEqual(result["result"].sources, ())

    def test_multiple_agent_tool_proposals_terminate_without_execution(self):
        visited = []
        complete = RecordingCompletion([_multiple_tool_completion(
            _tool_call("call-1", "get_contact_info", "{}"),
            _tool_call("call-2", "private_tool", "{}"),
        )])
        graph = build_agent_graph(_nodes(
            visited,
            decision=RouteDecision(Route.CONTACT, agentic=True),
            executor=RecordingGraphExecutor(visited),
            complete_chat=complete,
        ))

        result = graph.invoke(_initial_state())

        self.assertNotIn("execute_tool", visited)
        self.assertEqual(result["agent_processed_calls"], 0)
        self.assertEqual(
            [trace.name for trace in result["result"].trace.tool_calls],
            ["get_contact_info", "tool_executor"],
        )
        self.assertEqual(result["result"].answer, SAFE_AGENT_ANSWER)
        self.assertEqual(
            result["result"].trace.failure_layer,
            FailureLayer.TOOL_SELECTION,
        )

    def test_safe_agent_answer_suppresses_successful_tool_sources(self):
        source = SourceRef(title="资料", url="https://example.com/source")
        observation = ToolObservation(
            content='{"ok":true}',
            success=True,
            reused=False,
            cache_key="key",
            tool_name="get_contact_info",
            sources=(source,),
        )
        invalid = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="length",
            message=SimpleNamespace(content="partial", tool_calls=None),
        )])
        graph = build_agent_graph(_nodes(
            [],
            decision=RouteDecision(Route.CONTACT, agentic=True),
            executor=RecordingGraphExecutor([], [observation]),
            complete_chat=RecordingCompletion([
                _tool_completion("call-1", "get_contact_info", "{}"),
                invalid,
            ]),
        ))

        result = graph.invoke(_initial_state())

        self.assertEqual(result["agent_successful_sources"], (source,))
        self.assertEqual(result["result"].answer, SAFE_AGENT_ANSWER)
        self.assertEqual(result["result"].sources, ())

    def test_agent_provider_exception_keeps_current_trace_metadata(self):
        expected = RuntimeError("provider unavailable")
        source = SourceRef(title="资料", url="https://example.com/source")
        hit = _retrieval_result(source)
        observation = ToolObservation(
            content='{"ok":true}',
            success=True,
            reused=False,
            cache_key="key",
            tool_name="get_contact_info",
        )
        graph = build_agent_graph(_nodes(
            [],
            decision=RouteDecision(Route.KNOWLEDGE, agentic=True),
            retriever=RecordingRetriever([hit]),
            executor=RecordingGraphExecutor([], [observation]),
            complete_chat=RecordingCompletion([
                _tool_completion("call-1", "get_contact_info", "{}"),
                expected,
            ]),
        ))

        with self.assertRaises(RuntimeError) as caught:
            graph.invoke(_initial_state())

        self.assertIs(caught.exception, expected)
        self.assertEqual(
            caught.exception.failure_layer,
            FailureLayer.TOOL_SELECTION,
        )
        self.assertEqual(caught.exception.tool_calls, (
            ToolTrace("get_contact_info", True),
        ))
        self.assertEqual(caught.exception.route, Route.KNOWLEDGE)
        self.assertEqual(
            caught.exception.retrieved_chunk_ids,
            ("solution:test:content",),
        )

    def test_agent_deadline_after_provider_prevents_tool_execution(self):
        visited = []
        graph = build_agent_graph(_nodes(
            visited,
            decision=RouteDecision(Route.CONTACT, agentic=True),
            executor=RecordingGraphExecutor(visited),
            complete_chat=RecordingCompletion([
                _tool_completion("call-1", "get_contact_info", "{}"),
            ]),
        ))

        with self.assertRaises(AgentDeadlineExceeded) as caught:
            graph.invoke(_initial_state(
                deadline=RecordingDeadline(fail_on=2),
            ))

        self.assertNotIn("execute_tool", visited)
        self.assertEqual(
            caught.exception.failure_layer,
            FailureLayer.TOOL_SELECTION,
        )
        self.assertEqual(caught.exception.tool_calls, ())
        self.assertEqual(caught.exception.route, Route.CONTACT)
        self.assertEqual(caught.exception.retrieved_chunk_ids, ())

    def test_agent_deadline_before_tool_adds_raw_outer_context(self):
        expected = AgentDeadlineExceeded("before tool")
        graph = build_agent_graph(_nodes(
            [],
            decision=RouteDecision(Route.CONTACT, agentic=True),
            complete_chat=RecordingCompletion([
                _tool_completion("call-1", "get_contact_info", "{}"),
            ]),
        ))

        with self.assertRaises(AgentDeadlineExceeded) as caught:
            graph.invoke(_initial_state(deadline=RecordingDeadline(
                fail_on=3,
                error=expected,
            )))

        self.assertIs(caught.exception, expected)
        self.assertEqual(
            caught.exception.failure_layer,
            FailureLayer.TOOL_EXECUTION,
        )
        self.assertEqual(caught.exception.tool_calls, ())
        self.assertEqual(caught.exception.route, Route.CONTACT)
        self.assertEqual(caught.exception.retrieved_chunk_ids, ())

    def test_agent_deadline_after_tool_includes_new_tool_trace(self):
        observation = ToolObservation(
            content='{"ok":true}',
            success=True,
            reused=False,
            cache_key="key",
            tool_name="get_contact_info",
        )
        graph = build_agent_graph(_nodes(
            [],
            decision=RouteDecision(Route.CONTACT, agentic=True),
            executor=RecordingGraphExecutor([], [observation]),
            complete_chat=RecordingCompletion([
                _tool_completion("call-1", "get_contact_info", "{}"),
            ]),
        ))

        with self.assertRaises(AgentDeadlineExceeded) as caught:
            graph.invoke(_initial_state(
                deadline=RecordingDeadline(fail_on=4),
            ))

        self.assertEqual(
            caught.exception.failure_layer,
            FailureLayer.TOOL_EXECUTION,
        )
        self.assertEqual(caught.exception.tool_calls, (
            ToolTrace("get_contact_info", True),
        ))
        self.assertEqual(caught.exception.route, Route.CONTACT)
        self.assertEqual(caught.exception.retrieved_chunk_ids, ())

    def test_unexpected_agent_executor_exception_adds_only_raw_context(self):
        expected = RuntimeError("executor infrastructure failure")

        class FailingExecutor:
            def execute(self, call, cache):
                raise expected

            def execute_named(self, name, arguments, cache):
                raise AssertionError("not used")

        graph = build_agent_graph(_nodes(
            [],
            decision=RouteDecision(Route.CONTACT, agentic=True),
            executor=FailingExecutor(),
            complete_chat=RecordingCompletion([
                _tool_completion("call-1", "get_contact_info", "{}"),
            ]),
        ))

        with self.assertRaises(RuntimeError) as caught:
            graph.invoke(_initial_state())

        self.assertIs(caught.exception, expected)
        self.assertFalse(hasattr(caught.exception, "failure_layer"))
        self.assertEqual(caught.exception.route, Route.CONTACT)
        self.assertEqual(caught.exception.tool_calls, ())
        self.assertEqual(caught.exception.retrieved_chunk_ids, ())

    def test_agent_normalization_exception_does_not_recover_state_traces(self):
        expected = RuntimeError("normalization failed")
        source = SourceRef(title="资料", url="https://example.com/source")
        hit = _retrieval_result(source)
        proposal = _tool_completion("call-1", "get_contact_info", "{}")
        first_turn = normalize_agent_turn(proposal)
        self.assertIsNotNone(first_turn)
        graph = build_agent_graph(_nodes(
            [],
            decision=RouteDecision(Route.KNOWLEDGE, agentic=True),
            retriever=RecordingRetriever([hit]),
            executor=RecordingGraphExecutor([], [ToolObservation(
                content='{"ok":true}',
                success=True,
                reused=False,
                cache_key="key",
                tool_name="get_contact_info",
            )]),
            complete_chat=RecordingCompletion([
                proposal,
                _completion("unused"),
            ]),
        ))

        with patch(
            "agent_graph.nodes.normalize_agent_turn",
            side_effect=[first_turn, expected],
        ):
            with self.assertRaises(RuntimeError) as caught:
                graph.invoke(_initial_state())

        self.assertIs(caught.exception, expected)
        self.assertFalse(hasattr(caught.exception, "failure_layer"))
        self.assertEqual(caught.exception.route, Route.KNOWLEDGE)
        self.assertEqual(caught.exception.tool_calls, ())
        self.assertEqual(
            caught.exception.retrieved_chunk_ids,
            ("solution:test:content",),
        )

    def test_agent_message_build_exception_gets_only_raw_outer_context(self):
        expected = RuntimeError("agent message build failed")
        source = SourceRef(title="资料", url="https://example.com/source")
        hit = _retrieval_result(source)
        graph = build_agent_graph(_nodes(
            [],
            decision=RouteDecision(Route.KNOWLEDGE, agentic=True),
            retriever=RecordingRetriever([hit]),
        ))

        with patch(
            "agent_graph.nodes.build_agent_messages",
            side_effect=expected,
        ):
            with self.assertRaises(RuntimeError) as caught:
                graph.invoke(_initial_state())

        self.assertIs(caught.exception, expected)
        self.assertFalse(hasattr(caught.exception, "failure_layer"))
        self.assertEqual(caught.exception.route, Route.KNOWLEDGE)
        self.assertEqual(caught.exception.tool_calls, ())
        self.assertEqual(
            caught.exception.retrieved_chunk_ids,
            ("solution:test:content",),
        )

    def test_disabled_tools_provider_exception_preserves_generation_metadata(self):
        expected = RuntimeError("disabled-tools provider failed")
        complete = KeywordRecordingCompletion([
            *(
                _tool_completion(
                    f"call-{index}",
                    "get_contact_info",
                    "{}",
                )
                for index in range(1, 4)
            ),
            expected,
        ])
        graph = build_agent_graph(_nodes(
            [],
            decision=RouteDecision(Route.CONTACT, agentic=True),
            complete_chat=complete,
        ))

        with self.assertRaises(RuntimeError) as caught:
            graph.invoke(
                _initial_state(),
                config={"recursion_limit": 12},
            )

        self.assertIs(caught.exception, expected)
        self.assertEqual(
            caught.exception.failure_layer,
            FailureLayer.GENERATION,
        )
        self.assertEqual(len(caught.exception.tool_calls), 3)
        self.assertEqual(caught.exception.route, Route.CONTACT)
        self.assertEqual(caught.exception.retrieved_chunk_ids, ())
        self.assertNotIn("tools", complete.calls[-1][1])

    def test_direct_provider_exception_metadata_matches_raw_orchestrator(self):
        raw_error = RuntimeError("provider failed")
        graph_error = RuntimeError("provider failed")
        raw = RouteOrchestrator(
            router=RecordingRouter(RouteDecision(Route.DIRECT), []),
            retriever=RecordingRetriever(),
            executor=RecordingGraphExecutor([]),
            agent_loop=object(),
            complete_chat=RecordingCompletion([raw_error]),
        )
        graph = build_agent_graph(_nodes(
            [],
            decision=RouteDecision(Route.DIRECT),
            complete_chat=RecordingCompletion([graph_error]),
        ))

        with self.assertRaises(RuntimeError) as raw_caught:
            raw.run(
                _initial_state()["messages"],
                deadline=RecordingDeadline(),
            )
        with self.assertRaises(RuntimeError) as graph_caught:
            graph.invoke(_initial_state())

        self.assertIs(raw_caught.exception, raw_error)
        self.assertIs(graph_caught.exception, graph_error)
        raw_metadata = (
            type(raw_caught.exception),
            raw_caught.exception.failure_layer,
            raw_caught.exception.route,
            raw_caught.exception.tool_calls,
            raw_caught.exception.retrieved_chunk_ids,
        )
        graph_metadata = (
            type(graph_caught.exception),
            graph_caught.exception.failure_layer,
            graph_caught.exception.route,
            graph_caught.exception.tool_calls,
            graph_caught.exception.retrieved_chunk_ids,
        )
        self.assertEqual(graph_metadata, raw_metadata)

    def test_route_node_uses_real_hybrid_router_contract(self):
        visited = []
        retriever = RecordingRetriever()
        complete = RecordingCompletion([_completion("direct")])
        router = HybridRouter(
            retriever=retriever,
            complete_chat=complete,
        )
        deadline = RecordingDeadline()
        first = ChatMessage(role="user", content="先前的问题")
        answer = ChatMessage(role="assistant", content="先前的回答")
        question = ChatMessage(role="user", content="请说明相关情况")
        state = _initial_state(deadline=deadline)
        state["messages"] = (first, answer, question)
        state["last_user_message"] = question
        graph = build_agent_graph(_nodes(
            visited,
            router=router,
        ))

        result = graph.invoke(state)

        self.assertEqual(
            result["route_decision"],
            RouteDecision(Route.DIRECT),
        )
        self.assertIn("routing_failure", result)
        self.assertIsNone(result["routing_failure"])
        self.assertEqual(retriever.queries, ["请说明相关情况"])
        self.assertEqual(deadline.checks, 4)
        self.assertEqual(len(complete.calls), 1)
        provider_messages, tools = complete.calls[0]
        self.assertIsNone(tools)
        self.assertEqual(
            provider_messages[1:],
            [
                {"role": "user", "content": "先前的问题"},
                {"role": "assistant", "content": "先前的回答"},
                {"role": "user", "content": "请说明相关情况"},
            ],
        )
        self.assertEqual(visited, ["direct"])
        self.assertEqual(result["result"].trace.route, Route.DIRECT)


if __name__ == "__main__":
    unittest.main()
