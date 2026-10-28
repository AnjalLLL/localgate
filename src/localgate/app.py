"""FastAPI application factory — wires config, database, backend, memory and routes.

Everything expensive or fallible (opening the database, creating HTTP clients) happens
in the lifespan rather than at import. That keeps ``create_app`` cheap enough for a
test to call once per test, and it means an unreachable database surfaces as a
startup failure instead of an import error with an unreadable traceback.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

from localgate import __version__
from localgate.backends import available_backends
from localgate.config import Settings, load_settings
from localgate.core.backend_pool import BackendPool
from localgate.core.cache import PromptCache
from localgate.core.db_config_store import (
    _default_config_path,
    is_database_established,
    load_database_url,
)
from localgate.core.errors import ConfigurationError, error_body, install_exception_handlers
from localgate.core.logging import configure_logging, get_logger
from localgate.core.model_info import ModelInfoCache
from localgate.core.netinfo import is_loopback_host
from localgate.core.rate_limiter import RateLimiter
from localgate.db.engine import init_models, make_engine, make_session_factory
from localgate.middleware import RequestContextMiddleware

DESCRIPTION = """
An OpenAI-compatible gateway for local LLMs.

Point any OpenAI SDK at this server's `/v1` and authenticate with a localgate API key.
The gateway adds key management, per-key rate limiting, token accounting, and
RAG-backed memory that extends a small model's effective context well past its native
window.

**Authentication.** Client routes (`/v1/*`) take `Authorization: Bearer <api-key>`.
Admin routes (`/admin/*`) take `X-Admin-Key: <admin-key>`.
"""


DASHBOARD_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; media-src 'self' blob:; frame-ancestors 'none'; base-uri 'none'; "
    "form-action 'self'"
)


#: Paths that only `admin_allowed_networks` may reach.
_ADMIN_PREFIXES = ("/admin", "/dashboard", "/docs", "/redoc", "/openapi.json", "/metrics")
#: Files under /dashboard that the public landing and connect pages load.
PUBLIC_ASSETS = frozenset(
    {
        "home.css",
        "home.js",
        "connect.js",
        "font-inter.woff2",
        "font-serif.woff2",
        "font-serif-italic.woff2",
        "font-mono.woff2",
    }
)


def _is_admin_surface(path: str) -> bool:
    if path.startswith("/dashboard/") and path.rsplit("/", 1)[-1] in PUBLIC_ASSETS:
        return False
    return any(path == p or path.startswith(p + "/") for p in _ADMIN_PREFIXES) or (
        path.startswith("/dashboard")
    )


def _untrusted_response(path: str) -> Response:
    message = (
        "The dashboard and admin API are only available from trusted networks "
        "(LOCALGATE_ADMIN_ALLOWED_NETWORKS). The /v1 API works from here with an API key."
    )
    if path.startswith("/dashboard"):
        return HTMLResponse(
            f"<!doctype html><title>localgate</title><p>{message}</p>", status_code=403
        )
    return JSONResponse(
        error_body(message, "permission_error", "untrusted_network"), status_code=403
    )


def resolve_database_url(settings: Settings, config_path: Path | None = None) -> str:
    """A database established through the admin UI outranks ``LOCALGATE_DATABASE_URL``.

    The stored URL was connection-tested at the moment it was saved (see
    ``api/config.py``); the one in ``.env`` is just a string someone typed. Preferring
    the tested one is what makes a database "established" rather than merely
    "configured".
    """
    stored = load_database_url(config_path)
    return stored or settings.database_url


def create_app(
    settings: Settings | None = None, database_config_path: Path | None = None
) -> FastAPI:
    """Build the application.

    ``database_config_path`` lets tests point at an isolated (or absent) config file so
    a test run can never pick up the developer's real established database from the
    working directory.
    """
    settings = settings or load_settings()
    config_path = database_config_path or _default_config_path()
    active_database_url = resolve_database_url(settings, config_path)

    configure_logging(settings.log_level, settings.log_format)
    logger = get_logger("localgate")

    if settings.backend_type not in available_backends():
        raise ConfigurationError(
            f"LOCALGATE_BACKEND_TYPE is {settings.backend_type!r}, which is not installed. "
            f"Available backends: {', '.join(available_backends())}."
        )

    for spec in settings.backends:
        if spec.type not in available_backends():
            raise ConfigurationError(
                f"LOCALGATE_BACKENDS: {spec.name!r} has type {spec.type!r}, which is not "
                f"installed. Available backends: {', '.join(available_backends())}."
            )

    if not is_loopback_host(settings.host) and settings.uses_insecure_admin_key:
        raise ConfigurationError(
            f"Refusing to listen on {settings.host} with the default admin key. "
            "This would expose your gateway to the network with a key that is printed in the docs. "
            "Either set LOCALGATE_ADMIN_KEY to a real value (generate one with "
            "`openssl rand -hex 32`), or bind to 127.0.0.1 (the default) for local-only use."
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # One engine, one session factory. Every table — keys, usage, conversations,
        # memory — sits behind it, so establishing a database moves *all* of
        # localgate's data rather than an arbitrary subset of it.
        engine = make_engine(active_database_url)
        await init_models(engine)

        app.state.session_factory = make_session_factory(engine)
        # Every inference backend, each with its own fair queue. The primary also
        # does gateway-internal work (memory embeddings, summaries); with one
        # backend configured it is the only one.
        app.state.pool = BackendPool.from_settings(settings)
        app.state.backend = app.state.pool.primary.backend
        app.state.scheduler = app.state.pool.primary.scheduler
        app.state.rate_limiter = RateLimiter()
        app.state.model_info = ModelInfoCache()
        app.state.cache = PromptCache(
            max_entries=settings.cache_max_entries, ttl_seconds=settings.cache_ttl_seconds
        )

        logger.info(
            "localgate_started",
            version=__version__,
            backend=settings.backend_type,
            backend_url=settings.backend_url,
            database=active_database_url.split("://", 1)[0],
            database_established=is_database_established(config_path),
            memory_enabled=settings.memory_enabled,
            cache_enabled=settings.cache_enabled,
        )
        if settings.uses_insecure_admin_key:
            logger.warning(
                "insecure_admin_key",
                message=(
                    "LOCALGATE_ADMIN_KEY is still the default placeholder. Anyone who can "
                    "reach /admin can mint API keys. Set a real key before exposing this "
                    "gateway to anything."
                ),
            )

        yield

        # Uvicorn stops accepting connections and drains in-flight requests before
        # running this, so nothing is still using these when they close. That is what
        # makes SIGTERM graceful rather than abrupt.
        await app.state.pool.aclose()
        await engine.dispose()
        logger.info("localgate_stopped")

    app = FastAPI(
        title="localgate",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.state.settings = settings
    # Expose the resolved URL separately rather than mutating the settings object,
    # so settings stays immutable and /admin/config reads from this field.
    app.state.active_database_url = active_database_url
    app.state.database_config_path = config_path

    app.add_middleware(RequestContextMiddleware, metrics_enabled=settings.metrics_enabled)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    install_exception_handlers(app)

    @app.middleware("http")
    async def trusted_networks_only(request: Any, call_next: Any) -> Any:
        """The dashboard, /admin, the API docs and /metrics answer only clients on
        `admin_allowed_networks` (this machine, LANs, Tailscale by default). The
        /v1 API — what other devices are given keys for — is unaffected."""
        path = request.url.path
        if _is_admin_surface(path):
            client = request.client.host if request.client else None
            if not settings.is_trusted_address(client):
                return _untrusted_response(path)
        return await call_next(request)

    @app.middleware("http")
    async def security_headers(request: Any, call_next: Any) -> Any:
        """Strict headers on the dashboard and landing page: scripts and styles only from this
        origin (no inline script), no framing, no referrer leakage."""
        response = await call_next(request)
        if request.url.path in ("/", "/connect") or request.url.path.startswith("/dashboard"):
            response.headers.setdefault("Content-Security-Policy", DASHBOARD_CSP)
            response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response

    from localgate.api import (
        admin_ops,
        audio,
        chat,
        completions,
        config,
        conversations,
        embeddings,
        export,
        health,
        keys,
        models,
        share,
        usage,
    )
    from localgate.dashboard.routes import mount_connect, mount_dashboard, mount_home

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(audio.router)
    app.include_router(completions.router)
    app.include_router(embeddings.router)
    app.include_router(models.router)
    app.include_router(conversations.router)
    app.include_router(usage.self_router)
    app.include_router(keys.router, prefix="/admin")
    app.include_router(usage.router, prefix="/admin")
    app.include_router(config.router, prefix="/admin")
    app.include_router(export.router, prefix="/admin")
    app.include_router(admin_ops.session_router, prefix="/admin")
    app.include_router(admin_ops.router, prefix="/admin")
    app.include_router(share.router, prefix="/admin")

    mount_dashboard(app)
    mount_home(app)
    mount_connect(app)

    return app
