import json
import logging
import time

from routing import RouteTrace
from trace_models import FailureLayer


logger = logging.getLogger("ai_customer_support")

if not logger.handlers:
    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s level=%(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)
logger.propagate = False


def log_request(
    *,
    request_id: str,
    http_status: int,
    outcome: str,
    started_at: float,
    error: str | None = None,
    trace: RouteTrace | None = None,
    failure_layer: FailureLayer | None = None,
) -> None:
    latency_ms = (
        time.monotonic() - started_at
    ) * 1000

    tool_calls = [] if trace is None else [
        {
            "name": item.name,
            "success": item.success,
            "error_code": item.error_code,
            "reused": item.reused,
        }
        for item in trace.tool_calls[:3]
    ]
    retrieved_chunk_ids = (
        [] if trace is None else list(trace.retrieved_chunk_ids[:5])
    )
    route = "-" if trace is None else trace.route.value
    layer = failure_layer or (None if trace is None else trace.failure_layer)

    logger.info(
        "request_id=%s http_status=%s "
        "outcome=%s latency_ms=%.1f error=%s "
        "route=%s tool_calls=%s tool_call_count=%s "
        "retrieved_chunk_ids=%s failure_layer=%s",
        request_id,
        http_status,
        outcome,
        latency_ms,
        error or "-",
        route,
        json.dumps(tool_calls, ensure_ascii=True, separators=(",", ":")),
        0 if trace is None else trace.tool_call_count,
        json.dumps(
            retrieved_chunk_ids,
            ensure_ascii=True,
            separators=(",", ":"),
        ),
        "-" if layer is None else layer.value,
    )
