import unittest
from types import MappingProxyType

from agent import AgentLoop
from agent_graph import AgentGraphNodes, GraphRouteOrchestrator, build_agent_graph
from evaluation.agent import AgentEvaluationRunner, RecordingToolExecutor
from knowledge_pipeline.models import (
    SourceRef,
    TechnicalParameterGroup,
    TechnicalParameterItem,
)
from knowledge_pipeline.retrieval.models import EntityMatch
from models import ChatMessage
from routing import (
    SAFE_FALLBACK_ANSWER,
    HybridRouter,
    Route,
    RouteDecision,
    RouteOrchestrator,
)
from support_tools import ContactInfoResult, ProductDetailsResult, ProductSearchResult
from tests.test_agent_graph import (
    RecordingCompletion,
    RecordingDeadline,
    RecordingRetriever,
    RecordingRouter,
    _completion,
    _retrieval_result,
    _tool_completion,
)


def _messages(question, history=()):
    return (
        *(ChatMessage(role=role, content=content) for role, content in history),
        ChatMessage(role="user", content=question),
    )


def _tool_registry():
    source = SourceRef(
        title="权威资料",
        url="https://example.com/source/",
    )

    def search_products(query):
        return ProductSearchResult(products=[])

    def get_product_details(product_id):
        return ProductDetailsResult(
            product_id=product_id,
            name=product_id,
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
        return ContactInfoResult(
            company_name="测试公司",
            duty_phone="4000000000",
            email="support@example.com",
            sources=[source],
        )

    return MappingProxyType({
        "search_products": search_products,
        "get_product_details": get_product_details,
        "get_contact_info": get_contact_info,
    })


def _responses(agent_actions, answer, *, fallback):
    if fallback:
        return []
    return [
        *(
            _tool_completion(
                f"call-{index}",
                action["name"],
                action["arguments"],
            )
            for index, action in enumerate(agent_actions, start=1)
        ),
        _completion(answer),
    ]


def _dependencies(case):
    executor = RecordingToolExecutor(_tool_registry())
    complete = RecordingCompletion(_responses(
        case["agent_actions"],
        case["answer"],
        fallback=case["route"] == Route.FALLBACK,
    ))
    router = RecordingRouter(
        RouteDecision(
            case["route"],
            agentic=case["agentic"],
            product_id=case["product_id"],
        ),
        [],
    )
    retriever = RecordingRetriever(
        [_retrieval_result(SourceRef(
            title="应急通信解决方案",
            url="https://example.com/solutions/emergency/",
        ))]
        if case["route"] == Route.KNOWLEDGE
        else ()
    )
    return executor, complete, router, retriever


def _raw_runner(case):
    executor, complete, router, retriever = _dependencies(case)
    orchestrator = RouteOrchestrator(
        router=router,
        retriever=retriever,
        executor=executor,
        agent_loop=AgentLoop(
            executor=executor,
            complete_chat=complete,
        ),
        complete_chat=complete,
    )
    return AgentEvaluationRunner(orchestrator, executor)


def _graph_runner(case):
    executor, complete, router, retriever = _dependencies(case)
    graph = build_agent_graph(AgentGraphNodes(
        router=router,
        executor=executor,
        retriever=retriever,
        complete_chat=complete,
    ))
    return AgentEvaluationRunner(GraphRouteOrchestrator(graph), executor)


PARITY_CASES = (
    {
        "name": "product_search_non_agentic",
        "route": Route.PRODUCT_SEARCH,
        "agentic": False,
        "product_id": None,
        "messages": _messages("推荐适合酒店的产品"),
        "agent_actions": (),
        "expected_calls": ({
            "name": "search_products",
            "arguments": {"query": "推荐适合酒店的产品"},
        },),
        "answer": "最终回答",
    },
    {
        "name": "product_search_agentic",
        "route": Route.PRODUCT_SEARCH,
        "agentic": True,
        "product_id": None,
        "messages": _messages("推荐几个产品，如果有多个，再比较参数选一个。"),
        "agent_actions": ({
            "name": "search_products",
            "arguments": '{"query":"推荐几个产品，如果有多个，再比较参数选一个。"}',
        },),
        "expected_calls": ({
            "name": "search_products",
            "arguments": {"query": "推荐几个产品，如果有多个，再比较参数选一个。"},
        },),
        "answer": "候选型号如下，请联系专业技术人员确认最终选型。",
    },
    {
        "name": "exact_product_non_agentic",
        "route": Route.EXACT_PRODUCT,
        "agentic": False,
        "product_id": "ly198",
        "messages": _messages("LY198 功率是多少？"),
        "agent_actions": (),
        "expected_calls": ({
            "name": "get_product_details",
            "arguments": {"product_id": "ly198"},
        },),
        "answer": "最终回答",
    },
    {
        "name": "exact_product_agentic_follow_up",
        "route": Route.EXACT_PRODUCT,
        "agentic": True,
        "product_id": None,
        "messages": _messages(
            "第二个功率呢？",
            (
                ("user", "推荐两款产品。"),
                ("assistant", "可以考虑 LY198 和 HP780。"),
            ),
        ),
        "agent_actions": ({
            "name": "get_product_details",
            "arguments": '{"product_id":"HP780"}',
        },),
        "expected_calls": ({
            "name": "get_product_details",
            "arguments": {"product_id": "HP780"},
        },),
        "answer": "HP780 的防护等级为 IP68。",
    },
    {
        "name": "contact_non_agentic",
        "route": Route.CONTACT,
        "agentic": False,
        "product_id": None,
        "messages": _messages("公司的电话是多少？"),
        "agent_actions": (),
        "expected_calls": ({"name": "get_contact_info", "arguments": {}},),
        "answer": "最终回答",
    },
    {
        "name": "contact_agentic_follow_up",
        "route": Route.CONTACT,
        "agentic": True,
        "product_id": None,
        "messages": _messages(
            "怎么联系？",
            (
                ("user", "第一个问题"),
                ("assistant", "先前回答"),
            ),
        ),
        "agent_actions": ({
            "name": "get_contact_info",
            "arguments": "{}",
        },),
        "expected_calls": ({"name": "get_contact_info", "arguments": {}},),
        "answer": "联系我们。",
    },
    {
        "name": "knowledge_non_agentic",
        "route": Route.KNOWLEDGE,
        "agentic": False,
        "product_id": None,
        "messages": _messages("你们有哪些解决方案？"),
        "agent_actions": (),
        "expected_calls": (),
        "answer": "解决方案回答",
    },
    {
        "name": "knowledge_agentic",
        "route": Route.KNOWLEDGE,
        "agentic": True,
        "product_id": None,
        "messages": _messages("介绍应急通信解决方案，另外怎么联系你们？"),
        "agent_actions": ({
            "name": "get_contact_info",
            "arguments": "{}",
        },),
        "expected_calls": ({"name": "get_contact_info", "arguments": {}},),
        "answer": "方案和联系方式",
    },
    {
        "name": "direct_single_turn",
        "route": Route.DIRECT,
        "agentic": False,
        "product_id": None,
        "messages": _messages("你好"),
        "agent_actions": (),
        "expected_calls": (),
        "answer": "您好，请问有什么可以帮您？",
    },
    {
        "name": "direct_follow_up",
        "route": Route.DIRECT,
        "agentic": False,
        "product_id": None,
        "messages": _messages(
            "继续说明",
            (
                ("user", "第一个问题"),
                ("assistant", "先前回答"),
            ),
        ),
        "agent_actions": (),
        "expected_calls": (),
        "answer": "直接回答",
    },
    {
        "name": "fallback_inventory",
        "route": Route.FALLBACK,
        "agentic": False,
        "product_id": "ly198",
        "messages": _messages("LY198 今天还有多少库存？"),
        "agent_actions": (),
        "expected_calls": (),
        "answer": SAFE_FALLBACK_ANSWER,
    },
    {
        "name": "fallback_wins_over_agentic",
        "route": Route.FALLBACK,
        "agentic": True,
        "product_id": None,
        "messages": _messages("LY198 今天还有多少库存？"),
        "agent_actions": (),
        "expected_calls": (),
        "answer": SAFE_FALLBACK_ANSWER,
    },
)


class OrchestratorParityTests(unittest.TestCase):
    def test_unique_contextual_exact_product_is_deterministic_in_raw_and_graph(self):
        prior_question = "LY198 的功率是多少？"
        contextual_messages = _messages(
            "那它支持什么频段？",
            (
                ("user", prior_question),
                ("assistant", "LY198 的输出功率是 ≤2W。"),
            ),
        )

        class ContextProductRetriever(RecordingRetriever):
            def resolve_entities(self, query):
                self.queries.append(query)
                if query == prior_question:
                    return [EntityMatch(
                        parent_document_id="product:ly198",
                        alias="LY198",
                        start=0,
                    )]
                return []

        def build_dependencies():
            executor = RecordingToolExecutor(_tool_registry())
            complete = RecordingCompletion([_completion("支持 400-480MHz。")])
            retriever = ContextProductRetriever()
            router = HybridRouter(
                retriever=retriever,
                complete_chat=complete,
            )
            return executor, complete, retriever, router

        raw_executor, raw_complete, raw_retriever, raw_router = build_dependencies()
        raw = AgentEvaluationRunner(RouteOrchestrator(
            router=raw_router,
            retriever=raw_retriever,
            executor=raw_executor,
            agent_loop=AgentLoop(
                executor=raw_executor,
                complete_chat=raw_complete,
            ),
            complete_chat=raw_complete,
        ), raw_executor).run(
            contextual_messages,
            deadline=RecordingDeadline(),
        )

        graph_executor, graph_complete, graph_retriever, graph_router = (
            build_dependencies()
        )
        graph = build_agent_graph(AgentGraphNodes(
            router=graph_router,
            executor=graph_executor,
            retriever=graph_retriever,
            complete_chat=graph_complete,
        ))
        graph_result = AgentEvaluationRunner(
            GraphRouteOrchestrator(graph),
            graph_executor,
        ).run(
            contextual_messages,
            deadline=RecordingDeadline(),
        )

        expected_calls = ({
            "name": "get_product_details",
            "arguments": {"product_id": "ly198"},
        },)
        self.assertEqual(raw["route"], Route.EXACT_PRODUCT)
        self.assertEqual(graph_result["route"], raw["route"])
        self.assertEqual(raw["tool_calls"], expected_calls)
        self.assertEqual(graph_result["tool_calls"], expected_calls)
        self.assertEqual(graph_result["final_answer"], raw["final_answer"])
        self.assertEqual(len(raw_complete.calls), 1)
        self.assertEqual(len(graph_complete.calls), 1)

    def test_raw_and_graph_preserve_observable_business_behavior(self):
        for case in PARITY_CASES:
            with self.subTest(case=case["name"]):
                raw = _raw_runner(case).run(
                    case["messages"],
                    deadline=RecordingDeadline(),
                )
                graph = _graph_runner(case).run(
                    case["messages"],
                    deadline=RecordingDeadline(),
                )

                self.assertEqual(raw["route"], case["route"])
                self.assertEqual(graph["route"], raw["route"])
                self.assertCountEqual(
                    raw["tool_calls"],
                    case["expected_calls"],
                )
                self.assertCountEqual(
                    graph["tool_calls"],
                    raw["tool_calls"],
                )
                self.assertEqual(raw["final_answer"], case["answer"])
                self.assertEqual(
                    graph["final_answer"],
                    raw["final_answer"],
                )


if __name__ == "__main__":
    unittest.main()
