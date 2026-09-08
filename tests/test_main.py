import asyncio
import unittest
from types import MappingProxyType, SimpleNamespace
from unittest.mock import patch

import main
from agent import AgentLoop


class ApplicationRetrievalLifespanTests(unittest.TestCase):
    def test_builds_agent_with_one_shared_application_scoped_retriever(self):
        retriever = object()
        tools = SimpleNamespace()
        registry = MappingProxyType({})

        async def run_lifespan() -> None:
            async with main.app.router.lifespan_context(main.app):
                self.assertIs(main.app.state.retriever, retriever)
                self.assertIsInstance(main.app.state.agent_loop, AgentLoop)

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
        self.assertFalse(hasattr(main.app.state, "retriever"))
        self.assertFalse(hasattr(main.app.state, "agent_loop"))


if __name__ == "__main__":
    unittest.main()
