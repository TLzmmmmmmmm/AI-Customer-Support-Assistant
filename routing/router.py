from __future__ import annotations

import re
from collections.abc import Sequence

from agent import AgentDeadline
from models import ChatMessage

from .models import FailureLayer, Route, RouteDecision, RoutingResult


ROUTER_SYSTEM_PROMPT = """
Classify the user's request by the primary execution capability required.
Return exactly one token from this list and nothing else:
product_search, exact_product, contact, knowledge, direct, fallback.

Meanings:
- product_search: product/model candidates, discovery, recommendation, or selection.
- exact_product: authoritative facts or details for a particular product.
- contact: official company phone, email, or contact channel.
- knowledge: company, category, solution, or support knowledge requiring retrieval.
- direct: conversation requiring neither retrieval nor a deterministic tool.
- fallback: a fact or capability the system cannot currently verify.

For mixed requests choose the primary route in this order:
knowledge, product_search, exact_product, contact.
Do not provide a reason, confidence, tool, arguments, or plan.
""".strip()

_DIRECT_UTTERANCES = frozenset({
    "你好", "您好", "谢谢", "感谢", "再见",
    "hello", "hi", "thanks", "thankyou",
})
_CONTACT = re.compile(r"电话|邮箱|电子邮件|联系方式|怎么联系|如何联系|contact|phone|e-?mail", re.I)
_PRODUCT_TERM = re.compile(r"产品|型号|对讲机|设备|候选|product|model|radio", re.I)
_PRODUCT_DISCOVERY = re.compile(r"推荐|候选|有哪些|哪几款|怎么选|如何选|选择|适合|recommend|candidate", re.I)
_KNOWLEDGE = re.compile(r"解决方案|通信方案|系统方案|售后服务|技术支持|支持服务|solution|support service", re.I)
_INVENTORY = re.compile(r"库存|存货|现货|有货|inventory|in stock|stock level|availability", re.I)
_CURRENT_STATE = re.compile(r"今天|现在|当前|实时|还有|多少|有没有|是否|查询|查一下|today|current|real.?time", re.I)
_CONTEXT_REFERENCE = re.compile(r"第二个|第[一二三四五六七八九十0-9]+个|刚才那个|它|这个产品|那个产品|the previous|that one", re.I)
_PRODUCT_COMPARISON = re.compile(r"相比|比较|对比|区别|差异|哪(?:个|款).*(?:好|优)|versus|\bvs\.?\b", re.I)
_OBSERVATION_DEPENDENT = re.compile(r"如果.+(?:多个|几款|有).+(?:再|然后|比较|选)|if .+ then", re.I)
_TRAILING_PUNCTUATION = re.compile(r"[\s，。！？、,.!?]+")


def _provider_messages(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        *(
            {"role": message.role, "content": message.content}
            for message in messages
        ),
    ]


def _normalize_route(completion) -> Route | None:
    try:
        choices = completion.choices
        if len(choices) != 1:
            return None
        choice = choices[0]
        if choice.finish_reason in {"length", "content_filter"}:
            return None
        message = choice.message
        if message.tool_calls:
            return None
        if not isinstance(message.content, str):
            return None
        return Route(message.content.strip())
    except (AttributeError, TypeError, ValueError):
        return None


def _historical_product_ids(messages, retriever) -> set[str]:
    product_ids: set[str] = set()
    for message in messages[:-1]:
        if message.role != "user":
            continue
        for match in retriever.resolve_entities(message.content):
            product_ids.add(
                match.parent_document_id.removeprefix("product:")
            )
    return product_ids


class HybridRouter:
    def __init__(self, *, retriever, complete_chat) -> None:
        self._retriever = retriever
        self._complete_chat = complete_chat

    def route(
        self,
        messages: Sequence[ChatMessage],
        *,
        deadline: AgentDeadline,
    ) -> RoutingResult:
        question = messages[-1].content
        matches = self._retriever.resolve_entities(question)
        product_id = None
        if len(matches) == 1:
            product_id = matches[0].parent_document_id.removeprefix("product:")

        normalized = _TRAILING_PUNCTUATION.sub("", question).casefold()
        contact = bool(_CONTACT.search(question))
        product_search = bool(
            _PRODUCT_TERM.search(question)
            and _PRODUCT_DISCOVERY.search(question)
        )
        knowledge = bool(_KNOWLEDGE.search(question))
        exact_product = bool(matches)
        unsupported = bool(
            _INVENTORY.search(question)
            and (matches or _CURRENT_STATE.search(question))
        )
        has_context_reference = bool(_CONTEXT_REFERENCE.search(question))
        contextual = bool(has_context_reference and not matches)
        comparative_reference = bool(
            matches
            and _CONTEXT_REFERENCE.search(question)
            and _PRODUCT_COMPARISON.search(question)
        )
        observation_dependent = bool(_OBSERVATION_DEPENDENT.search(question))

        capabilities = {
            route
            for enabled, route in (
                (knowledge, Route.KNOWLEDGE),
                (product_search, Route.PRODUCT_SEARCH),
                (exact_product, Route.EXACT_PRODUCT),
                (contact, Route.CONTACT),
            )
            if enabled
        }
        agentic = (
            len(capabilities) > 1
            or len(matches) > 1
            or comparative_reference
            or observation_dependent
            or has_context_reference
        )

        if normalized in _DIRECT_UTTERANCES and not capabilities:
            return RoutingResult(RouteDecision(route=Route.DIRECT))
        if unsupported:
            return RoutingResult(RouteDecision(
                route=Route.FALLBACK,
                product_id=product_id,
            ))

        if contextual and not capabilities:
            historical_product_ids = _historical_product_ids(
                messages,
                self._retriever,
            )
            if historical_product_ids:
                return RoutingResult(RouteDecision(
                    route=Route.EXACT_PRODUCT,
                    agentic=len(historical_product_ids) != 1,
                    product_id=(
                        next(iter(historical_product_ids))
                        if len(historical_product_ids) == 1
                        else None
                    ),
                ))

        for route in (
            Route.KNOWLEDGE,
            Route.PRODUCT_SEARCH,
            Route.EXACT_PRODUCT,
            Route.CONTACT,
        ):
            if route in capabilities:
                route_agentic = agentic
                if (
                    route == Route.EXACT_PRODUCT
                    and product_id is not None
                    and capabilities == {Route.EXACT_PRODUCT}
                    and not comparative_reference
                    and not observation_dependent
                ):
                    route_agentic = False
                return RoutingResult(RouteDecision(
                    route=route,
                    agentic=route_agentic,
                    product_id=product_id,
                ))

        deadline.ensure_active()
        completion = self._complete_chat(_provider_messages(messages))
        deadline.ensure_active()
        route = _normalize_route(completion)
        if route is None:
            return RoutingResult(
                RouteDecision(route=Route.FALLBACK),
                failure_layer=FailureLayer.ROUTING,
            )

        if route == Route.EXACT_PRODUCT and product_id is None:
            agentic = True
        return RoutingResult(RouteDecision(
            route=route,
            agentic=agentic,
            product_id=product_id,
        ))


__all__ = ["HybridRouter", "ROUTER_SYSTEM_PROMPT"]
