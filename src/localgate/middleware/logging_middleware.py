"""Request context: correlation IDs, access logs, and request metrics.

This is the only middleware localgate installs. Authentication, rate limiting and
token accounting are FastAPI dependencies instead, because each needs the resolved
API key and the parsed body — things a middleware would have to re-derive by hand.
See docs/decisions/0001-dependencies-over-middleware.md.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from localgate.core import metrics
from localgate.core.logging import bind_request_id, get_logger

logger = get_logger("localgate.request")

REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Tag every request with an id, log its outcome, and record its metrics."""

    def __init__(self, app: ASGIApp, metrics_enabled: bool = True) -> None:
        super().__init__(app)
        self.metrics_enabled = metrics_enabled

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Honour an inbound id if there is one, so a request traced through a proxy
        # or another service keeps a single identity across all of their logs.
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:16]
        bind_request_id(request_id)
        request.state.request_id = request_id

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # The exception handlers will turn this into a 5xx response, but without
            # this branch the access log and the metric would never record that the
            # request happened at all.
            elapsed = time.perf_counter() - started
            self._record(request, 500, elapsed)
            logger.exception(
                "request_failed",
                method=request.method,
                path=request.url.path,
                duration_ms=round(elapsed * 1000, 2),
            )
            raise

        elapsed = time.perf_counter() - started
        self._record(request, response.status_code, elapsed)
        response.headers[REQUEST_ID_HEADER] = request_id

        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(elapsed * 1000, 2),
        )
        return response

    def _record(self, request: Request, status_code: int, elapsed: float) -> None:
        if not self.metrics_enabled:
            return
        path = route_template(request)
        metrics.requests_total.labels(
            method=request.method, path=path, status=str(status_code)
        ).inc()
        metrics.request_duration_seconds.labels(method=request.method, path=path).observe(elapsed)


UNMATCHED = "<unmatched>"


def route_template(request: Request) -> str:
    """The low-cardinality label for a request's path.

    Two failure modes to avoid, both of which end with Prometheus running out of memory:

    * **Concrete ids.** Labelling with the raw path mints a new time series per session
      id. So path parameters are folded back into their placeholders:
      ``/v1/conversations/abc123`` becomes ``/v1/conversations/{session_id}``.

    * **Unmatched paths.** A scanner probing ``/wp-admin``, ``/.env``, ``/phpmyadmin``
      would otherwise create a series per URL it tries. Anything that matched no route
      is labelled ``<unmatched>``.

    Reconstructing from the concrete path rather than reading ``route.path`` is
    deliberate: FastAPI reports an included router's route with its *un-prefixed* path
    (``/keys``, not ``/admin/keys``), which would silently merge admin metrics with any
    same-named public route.
    """
    if request.scope.get("route") is None:
        return UNMATCHED

    params = request.scope.get("path_params") or {}
    if not params:
        return request.url.path

    placeholders = {str(value): "{" + name + "}" for name, value in params.items()}
    return "/".join(placeholders.get(segment, segment) for segment in request.url.path.split("/"))
