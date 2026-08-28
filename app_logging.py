import logging
import time


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
) -> None:
    latency_ms = (
        time.monotonic() - started_at
    ) * 1000

    logger.info(
        "request_id=%s http_status=%s "
        "outcome=%s latency_ms=%.1f error=%s",
        request_id,
        http_status,
        outcome,
        latency_ms,
        error or "-",
    )