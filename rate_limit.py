import math
import time

from collections import defaultdict, deque
from threading import Lock

from fastapi import HTTPException, Request

from config import (
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
)


request_history: dict[str, deque[float]] = defaultdict(deque)
rate_limit_lock = Lock()


def enforce_rate_limit(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS

    with rate_limit_lock:
        timestamps = request_history[client_ip]

        while timestamps and timestamps[0] <= cutoff:
            timestamps.popleft()

        if len(timestamps) >= RATE_LIMIT_REQUESTS:
            retry_after = math.ceil(
                timestamps[0]
                + RATE_LIMIT_WINDOW_SECONDS
                - now
            )

            raise HTTPException(
                status_code=429,
                detail="Too many requests",
                headers={
                    "Retry-After": str(retry_after),
                },
            )

        timestamps.append(now)