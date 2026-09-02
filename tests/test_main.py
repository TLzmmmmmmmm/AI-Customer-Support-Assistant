import asyncio
import unittest
from unittest.mock import patch

import main


class ApplicationRetrievalLifespanTests(unittest.TestCase):
    def test_builds_one_application_scoped_retriever(self):
        builder = getattr(main, "build_retriever", None)
        self.assertIsNotNone(builder)
        sentinel = object()

        async def run_lifespan() -> None:
            async with main.app.router.lifespan_context(main.app):
                self.assertIs(main.app.state.retriever, sentinel)

        with patch.object(
            main,
            "build_retriever",
            return_value=sentinel,
        ) as build:
            asyncio.run(run_lifespan())

        build.assert_called_once_with()
        self.assertFalse(hasattr(main.app.state, "retriever"))


if __name__ == "__main__":
    unittest.main()
