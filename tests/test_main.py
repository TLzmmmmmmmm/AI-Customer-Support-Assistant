import asyncio
import unittest
from types import MappingProxyType, SimpleNamespace
from unittest.mock import patch

import main
from agent_graph import GraphRouteOrchestrator
from routes import chat


class ApplicationRoutingLifespanTests(unittest.TestCase):
    def test_builds_one_application_scoped_graph_orchestrator(self):
        retriever = object()
        tools = SimpleNamespace()
        registry = MappingProxyType({})
        router = object()
        compiled_graph = object()

        async def run_lifespan() -> None:
            async with main.app.router.lifespan_context(main.app):
                orchestrator = main.app.state.route_orchestrator
                self.assertIsInstance(orchestrator, GraphRouteOrchestrator)
                self.assertIs(orchestrator._compiled_graph, compiled_graph)
                self.assertIs(
                    main.app.state.route_orchestrator,
                    orchestrator,
                )
                first_request = SimpleNamespace(app=main.app)
                second_request = SimpleNamespace(app=main.app)
                self.assertIs(
                    chat.get_route_orchestrator(first_request),
                    orchestrator,
                )
                self.assertIs(
                    chat.get_route_orchestrator(second_request),
                    orchestrator,
                )

        with (
            patch.object(
                main,
                "build_retriever",
                return_value=retriever,
            ) as build_retriever,
            patch.object(
                main,
                "build_deterministic_tools",
                return_value=tools,
            ) as build_tools,
            patch.object(
                main,
                "build_tool_registry",
                return_value=registry,
            ) as build_registry,
            patch.object(
                main,
                "HybridRouter",
                return_value=router,
            ) as build_router,
            patch.object(
                main,
                "build_agent_graph",
                return_value=compiled_graph,
            ) as build_graph,
            patch.object(
                main,
                "RouteOrchestrator",
                side_effect=AssertionError("Raw orchestrator constructed"),
                create=True,
            ),
            patch.object(
                main,
                "AgentLoop",
                side_effect=AssertionError("Raw agent loop constructed"),
                create=True,
            ),
        ):
            asyncio.run(run_lifespan())

        build_retriever.assert_called_once_with()
        build_tools.assert_called_once_with(retriever=retriever)
        build_registry.assert_called_once_with(tools)
        build_router.assert_called_once_with(
            retriever=retriever,
            complete_chat=main.complete_chat,
        )
        build_graph.assert_called_once()
        nodes = build_graph.call_args.args[0]
        self.assertIs(nodes.router, router)
        self.assertIs(nodes.retriever, retriever)
        self.assertIs(nodes.executor._registry, registry)
        self.assertIs(nodes.complete_chat, main.complete_chat)
        self.assertFalse(hasattr(main.app.state, "route_orchestrator"))


if __name__ == "__main__":
    unittest.main()
