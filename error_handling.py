from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app_logging import log_request


def _trace_fields(request: Request) -> dict[str, object]:
    return {
        "trace": getattr(request.state, "route_trace", None),
        "failure_layer": getattr(request.state, "failure_layer", None),
    }


DEFAULT_MESSAGES = {
    400: "请求格式不正确，请检查后重试。",
    429: "请求过于频繁，请稍后再试。",
    500: "服务暂时出现异常，请稍后再试。",
    502: "服务暂时出现异常，请稍后再试。",
    503: "服务暂时不可用，请稍后再试。",
    504: "服务响应超时，请重新尝试。",
}

def error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    request_id: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    response_headers = dict(headers or {})
    response_headers["X-Request-ID"] = request_id

    response = JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
            }
        },
        headers=response_headers,
    )

    return response

async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    request_id = request.state.request_id
    started_at = request.state.started_at

    log_request(
        request_id=request_id,
        http_status=422,
        outcome="validation_error",
        started_at=started_at,
        error=type(exc).__name__,
        **_trace_fields(request),
    )

    return error_response(
        status_code=422,
        code="validation_error",
        message="提交内容格式不正确或超过限制，请调整后重试。",
        request_id=request_id,
    )

async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    request_id = request.state.request_id
    started_at = request.state.started_at

    if isinstance(exc.detail, dict):
        code = str(
            exc.detail.get("code", "http_error")
        )

        message = str(
            exc.detail.get(
                "message",
                DEFAULT_MESSAGES.get(
                    exc.status_code,
                    "请求处理失败，请稍后再试。",
                ),
            )
        )

        internal_error = exc.detail.get(
            "internal_error"
        )

    else:
        code = "http_error"

        message = DEFAULT_MESSAGES.get(
            exc.status_code,
            "请求处理失败，请稍后再试。",
        )

        internal_error = None

    log_request(
        request_id=request_id,
        http_status=exc.status_code,
        outcome=code,
        started_at=started_at,
        error=(
            str(internal_error)
            if internal_error
            else None
        ),
        **_trace_fields(request),
    )

    return error_response(
        status_code=exc.status_code,
        code=code,
        message=message,
        request_id=request_id,
        headers=exc.headers,
    )

async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    request_id = request.state.request_id
    started_at = request.state.started_at

    log_request(
        request_id=request_id,
        http_status=500,
        outcome="internal_error",
        started_at=started_at,
        error=type(exc).__name__,
        **_trace_fields(request),
    )

    return error_response(
        status_code=500,
        code="internal_error",
        message="服务暂时出现异常，请稍后再试。",
        request_id=request_id,
    )
