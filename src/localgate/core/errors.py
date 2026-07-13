"""Error types and OpenAI-shaped error envelopes.

Clients point the OpenAI SDK at localgate, and that SDK parses failures out of
``{"error": {"message", "type", "code"}}``. FastAPI's default ``{"detail": ...}``
envelope would surface to those clients as an unhelpful generic error, so every
error localgate returns is rendered in the OpenAI shape instead (see
``install_exception_handlers``).
"""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class LocalgateError(Exception):
    """Base class for errors that carry an HTTP status and an OpenAI error type."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_type: str = "internal_error"

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class AuthenticationError(LocalgateError):
    status_code = status.HTTP_401_UNAUTHORIZED
    error_type = "authentication_error"


class RateLimitError(LocalgateError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    error_type = "rate_limit_error"


class InvalidRequestError(LocalgateError):
    status_code = status.HTTP_400_BAD_REQUEST
    error_type = "invalid_request_error"


class BackendError(LocalgateError):
    """The inference backend was unreachable or returned a failure."""

    status_code = status.HTTP_502_BAD_GATEWAY
    error_type = "backend_error"


class ConfigurationError(LocalgateError):
    """localgate itself is misconfigured — raised at startup, not per-request."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_type = "configuration_error"


def error_body(message: str, error_type: str, code: str | None = None) -> dict[str, Any]:
    return {"error": {"message": message, "type": error_type, "code": code}}


def describe_backend_failure(exc: Exception, backend_url: str, backend_type: str) -> str:
    """Turns a raw transport exception into something the operator can act on.

    A bare ``ConnectionRefused`` tells a user nothing about what to do next. The
    three failures that account for almost every real report — backend not
    running, model not pulled, backend erroring — each get a specific remedy.
    """
    if isinstance(exc, httpx.ConnectError):
        hint = (
            "Is Ollama running? Start it with `ollama serve`."
            if backend_type == "ollama"
            else f"Is the {backend_type} server running and listening on that address?"
        )
        return f"Could not reach the inference backend at {backend_url}. {hint}"

    if isinstance(exc, httpx.TimeoutException):
        return (
            f"The inference backend at {backend_url} did not respond in time. "
            "Large models on CPU can exceed the default timeout — raise "
            "LOCALGATE_BACKEND_TIMEOUT (seconds) if this is expected."
        )

    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        body = exc.response.text[:300]
        if code == 404:
            pull = (
                "Pull it with `ollama pull <model>`."
                if backend_type == "ollama"
                else "Check that the model name matches one the backend has loaded."
            )
            return (
                f"The backend returned 404 — the model is probably not available. "
                f"{pull} Response: {body}"
            )
        return f"The backend returned HTTP {code}: {body}"

    return f"Unexpected error calling the backend: {type(exc).__name__}: {exc}"


def install_exception_handlers(app: FastAPI) -> None:
    """Renders every error path in the OpenAI envelope so SDK clients can read it."""

    @app.exception_handler(LocalgateError)
    async def _localgate_error(_: Request, exc: LocalgateError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(exc.message, exc.error_type, exc.code),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        type_by_status = {
            401: "authentication_error",
            403: "permission_error",
            404: "not_found_error",
            429: "rate_limit_error",
            502: "backend_error",
        }
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(
                str(exc.detail),
                type_by_status.get(exc.status_code, "invalid_request_error"),
            ),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        message = first.get("msg", "Invalid request")
        return JSONResponse(
            status_code=422,  # starlette is mid-rename between ..._ENTITY and ..._CONTENT
            content=error_body(
                f"{location}: {message}" if location else message,
                "invalid_request_error",
                code="validation_failed",
            ),
        )
