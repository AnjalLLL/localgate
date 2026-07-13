"""``GET /health``, ``GET /health/live``, and ``GET /metrics``.

Health is split in two because orchestrators ask two different questions and
conflating them causes outages:

* **Liveness** (``/health/live``) — "is this process wedged, should you kill it?"
  It must not depend on anything external. A liveness probe that fails because
  *Ollama* is down would have Kubernetes restart a perfectly healthy gateway, over
  and over, fixing nothing.
* **Readiness** (``/health``) — "can this instance serve traffic right now?" This
  one *does* check the backend and the database, because a gateway that can reach
  neither has nothing useful to offer a caller.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request, Response
from sqlalchemy import text

from localgate import __version__
from localgate.core import metrics

router = APIRouter(tags=["operations"])

_started_at = time.monotonic()


@router.get("/health/live")
async def liveness() -> dict[str, Any]:
    """Process-local liveness. Deliberately checks nothing external."""
    return {"status": "alive", "version": __version__, "uptime_seconds": _uptime()}


@router.get("/health")
async def health(request: Request, response: Response) -> dict[str, Any]:
    """Full readiness: backend reachability, database connectivity, memory config.

    Returns 503 when a hard dependency is down, so a load balancer can act on the
    status code alone without parsing the body.
    """
    app = request.app
    settings = app.state.settings

    backend_ok = await _check_backend(app)
    database_ok, database_error = await _check_database(app)

    metrics.backend_up.set(1 if backend_ok else 0)

    ready = backend_ok and database_ok
    if not ready:
        response.status_code = 503

    return {
        "status": "ok" if ready else "degraded",
        "version": __version__,
        "uptime_seconds": _uptime(),
        "backend": {
            "type": settings.backend_type,
            "url": settings.backend_url,
            "reachable": backend_ok,
        },
        "database": {
            "dialect": settings.database_url.split("://", 1)[0],
            "connected": database_ok,
            "error": database_error,
        },
        "memory": {
            "enabled": settings.memory_enabled,
            "embedding_model": settings.embedding_model,
        },
        "cache": app.state.cache.stats() if settings.cache_enabled else {"enabled": False},
        "warnings": _warnings(settings),
    }


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics(request: Request) -> Response:
    """Prometheus scrape endpoint.

    Unauthenticated, like almost every /metrics in the ecosystem — Prometheus has
    no good way to hold a bearer token per target, and the endpoint exposes counts
    and latencies, never prompts or key material. If the gateway is exposed to the
    internet, keep this path off the public listener at the reverse proxy.
    """
    if not request.app.state.settings.metrics_enabled:
        return Response(status_code=404)
    return Response(content=metrics.render(), media_type=metrics.CONTENT_TYPE)


def _uptime() -> float:
    return round(time.monotonic() - _started_at, 1)


async def _check_backend(app: Any) -> bool:
    try:
        return bool(await app.state.backend.health())
    except Exception:  # noqa: BLE001 — a health check that raises has answered "no"
        return False


async def _check_database(app: Any) -> tuple[bool, str | None]:
    try:
        async with app.state.session_factory() as session:
            await session.execute(text("SELECT 1"))
        return True, None
    except Exception as exc:  # noqa: BLE001 — see above
        return False, f"{type(exc).__name__}: {exc}"


def _warnings(settings: Any) -> list[str]:
    """Configuration that is legal but probably not what the operator wants."""
    warnings: list[str] = []
    if settings.uses_insecure_admin_key:
        warnings.append(
            "The admin key is still the default placeholder. Anyone who can reach "
            "/admin can mint API keys. Set LOCALGATE_ADMIN_KEY."
        )
    if settings.memory_enabled and settings.database_url.startswith("sqlite"):
        warnings.append(
            "Memory is enabled on SQLite, where similarity search scans every chunk "
            "in the session. Fine for local use; move to Postgres before this grows."
        )
    return warnings
