import unittest

from agent import ToolObservation
from models import ChatMessage
from routing import Route, RouteDecision
from routing.deterministic import execute_deterministic_route


class RecordingDeadline:
    def __init__(self, events):
        self.events = events

    def ensure_active(self):
        self.events.append("deadline")


class RecordingExecutor:
    def __init__(self, observation, events):
        self.observation = observation
        self.events = events
        self.calls = []

    def execute_named(self, name, arguments, cache):
        self.events.append("execute")
        self.calls.append((name, arguments, cache))
        return self.observation


class FailingExecutor:
    def __init__(self, error, events):
        self.error = error
        self.events = events

    def execute_named(self, name, arguments, cache):
        self.events.append("execute")
        raise self.error


class DeterministicRoutingTests(unittest.TestCase):
    def test_routes_use_existing_names_arguments_and_deadline_order(self):
        cases = (
            (
                RouteDecision(Route.PRODUCT_SEARCH),
                "search_products",
                {"query": "推荐几款对讲机"},
            ),
            (
                RouteDecision(Route.EXACT_PRODUCT, product_id="ly198"),
                "get_product_details",
                {"product_id": "ly198"},
            ),
            (
                RouteDecision(Route.CONTACT),
                "get_contact_info",
                {},
            ),
        )
        messages = (ChatMessage(
            role="user",
            content="推荐几款对讲机",
        ),)

        caches = []
        for decision, expected_name, expected_arguments in cases:
            with self.subTest(route=decision.route):
                events = []
                observation = ToolObservation(
                    content='{"ok":true}',
                    success=True,
                    reused=False,
                    cache_key="key",
                    tool_name=expected_name,
                )
                executor = RecordingExecutor(observation, events)

                result = execute_deterministic_route(
                    decision,
                    messages,
                    deadline=RecordingDeadline(events),
                    executor=executor,
                )

                self.assertIs(result, observation)
                self.assertEqual(events, ["deadline", "execute", "deadline"])
                self.assertEqual(len(executor.calls), 1)
                name, arguments, cache = executor.calls[0]
                self.assertEqual(name, expected_name)
                self.assertEqual(arguments, expected_arguments)
                self.assertEqual(cache, {})
                caches.append(cache)

        self.assertIsNot(caches[0], caches[1])
        self.assertIsNot(caches[1], caches[2])

    def test_executor_exception_propagates_without_second_deadline_check(self):
        error = RuntimeError("executor failed")
        events = []

        with self.assertRaises(RuntimeError) as caught:
            execute_deterministic_route(
                RouteDecision(Route.CONTACT),
                (ChatMessage(role="user", content="联系方式"),),
                deadline=RecordingDeadline(events),
                executor=FailingExecutor(error, events),
            )

        self.assertIs(caught.exception, error)
        self.assertEqual(events, ["deadline", "execute"])


if __name__ == "__main__":
    unittest.main()
