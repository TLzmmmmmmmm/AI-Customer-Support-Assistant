import json
import logging
import time

from routing import RouteTrace
from trace_models import FailureLayer, request_trace_fields


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
    failure_code: str | None = None,
    request_state: object | None = None,
    completed_at: float | None = None,
) -> None:
    try:
        finished_at = time.monotonic() if completed_at is None else completed_at
        total_latency_ms = (finished_at - started_at) * 1000
        telemetry = request_trace_fields(request_state)
        if trace is not None:
            telemetry = {
                name: getattr(trace, name)
                for name in telemetry
            }

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
        code = failure_code or (
            None if trace is None else trace.failure_code
        )

        logger.info(
            "request_id=%s http_status=%s "
            "outcome=%s latency_ms=%.1f total_latency_ms=%.1f error=%s "
            "route=%s router_type=%s tool_calls=%s tool_call_count=%s "
            "retrieved_chunk_ids=%s retrieval_used=%s retrieved_count=%s "
            "executed_tool_names=%s tool_execution_count=%s "
            "tool_execution_success=%s router_latency_ms=%s "
            "retrieval_latency_ms=%s tool_latency_ms=%s model_latency_ms=%s "
            "input_tokens=%s output_tokens=%s "
            "prompt_cache_hit_tokens=%s prompt_cache_miss_tokens=%s "
            "failure_layer=%s "
            "failure_code=%s citation_count=%s "
            "deduplicated_citation_count=%s invalid_source_count=%s "
            "citation_status=%s answer_sanitized=%s",
            request_id,
            http_status,
            outcome,
            total_latency_ms,
            total_latency_ms,
            error or "-",
            route,
            telemetry["router_type"] or "null",
            json.dumps(tool_calls, ensure_ascii=True, separators=(",", ":")),
            0 if trace is None else trace.tool_call_count,
            json.dumps(
                retrieved_chunk_ids,
                ensure_ascii=True,
                separators=(",", ":"),
            ),
            json.dumps(telemetry["retrieval_used"]),
            telemetry["retrieved_count"],
            json.dumps(
                telemetry["executed_tool_names"],
                ensure_ascii=True,
                separators=(",", ":"),
            ),
            telemetry["tool_execution_count"],
            json.dumps(telemetry["tool_execution_success"]),
            _format_latency(telemetry["router_latency_ms"]),
            _format_latency(telemetry["retrieval_latency_ms"]),
            _format_latency(telemetry["tool_latency_ms"]),
            _format_latency(telemetry["model_latency_ms"]),
            _format_optional(telemetry["input_tokens"]),
            _format_optional(telemetry["output_tokens"]),
            _format_optional(telemetry["prompt_cache_hit_tokens"]),
            _format_optional(telemetry["prompt_cache_miss_tokens"]),
            "-" if layer is None else layer.value,
            code or "null",
            0 if trace is None else trace.citation_count,
            0 if trace is None else trace.deduplicated_citation_count,
            0 if trace is None else trace.invalid_source_count,
            "none" if trace is None else trace.citation_status,
            False if trace is None else trace.answer_sanitized,
        )
    except Exception:
        pass


def _format_latency(value: object) -> str:
    return "null" if value is None else f"{float(value):.1f}"


def _format_optional(value: object) -> object:
    return "null" if value is None else value
