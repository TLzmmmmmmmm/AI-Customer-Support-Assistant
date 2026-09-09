import asyncio
import unittest
from types import MappingProxyType, SimpleNamespace
from unittest.mock import patch

import main
from routing import RouteOrchestrator


class ApplicationRoutingLifespanTests(unittest.TestCase):
    def test_builds_one_application_scoped_orchestrator_from_shared_components(self):
        retriever = object()
        tools = SimpleNamespace()
        registry = MappingProxyType({})

        async def run_lifespan() -> None:
            async with main.app.router.lifespan_context(main.app):
                orchestrator = main.app.state.route_orchestrator
                self.assertIsInstance(orchestrator, RouteOrchestrator)
                self.assertIs(orchestrator._retriever, retriever)
                self.assertIs(orchestrator._executor._registry, registry)
                self.assertIs(orchestrator._agent_loop._executor, orchestrator._executor)

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
        ):
            asyncio.run(run_lifespan())

        build_retriever.assert_called_once_with()
        build_tools.assert_called_once_with(retriever=retriever)
        build_registry.assert_called_once_with(tools)
        self.assertFalse(hasattr(main.app.state, "route_orchestrator"))


if __name__ == "__main__":
    unittest.main()
