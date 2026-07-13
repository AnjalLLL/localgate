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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from localgate import __version__
from localgate.backends import available_backends, get_backend
from localgate.config import Settings
from localgate.core.cache import PromptCache
from localgate.core.db_config_store import (
    DEFAULT_CONFIG_PATH,
    is_database_established,
    load_database_url,
)
from localgate.core.errors import ConfigurationError, install_exception_handlers
from localgate.core.logging import configure_logging, get_logger
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


def resolve_database_url(settings: Settings, config_path: Path | None = None) -> str:
    """A database established through the admin UI outranks ``LOCALGATE_DATABASE_URL``.

    The stored URL was connection-tested at the moment it was saved (see
    ``api/config.py``); the one in ``.env`` is just a string someone typed. Preferring
    the tested one is what makes a database "established" rather than merely
    "configured".
    """
    stored = load_database_url(config_path or DEFAULT_CONFIG_PATH)
    return stored or settings.database_url


def create_app(
    settings: Settings | None = None, database_config_path: Path | None = None
) -> FastAPI:
    """Build the application.

    ``database_config_path`` lets tests point at an isolated (or absent) config file so
    a test run can never pick up the developer's real established database from the
    working directory.
    """
    settings = settings or Settings()
    config_path = database_config_path or DEFAULT_CONFIG_PATH
    active_database_url = resolve_database_url(settings, config_path)

    configure_logging(settings.log_level, settings.log_format)
    logger = get_logger("localgate")

    if settings.backend_type not in available_backends():
        raise ConfigurationError(
            f"LOCALGATE_BACKEND_TYPE is {settings.backend_type!r}, which is not installed. "
            f"Available backends: {', '.join(available_backends())}."
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # One engine, one session factory. Every table — keys, usage, conversations,
        # memory — sits behind it, so establishing a database moves *all* of
        # localgate's data rather than an arbitrary subset of it.
        engine = make_engine(active_database_url)
        await init_models(engine)

        app.state.session_factory = make_session_factory(engine)
        app.state.backend = get_backend(
            settings.backend_type,
            settings.backend_url,
            timeout=settings.backend_timeout,
            api_key=settings.backend_api_key,
        )
        app.state.rate_limiter = RateLimiter()
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
        await app.state.backend.aclose()
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
    # Reflect where the data actually lives, so /admin/config can't report the .env
    # value while the gateway is really writing somewhere else.
    app.state.settings.database_url = active_database_url
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

    from localgate.api import (
        chat,
        completions,
        config,
        conversations,
        embeddings,
        export,
        health,
        keys,
        models,
        usage,
    )
    from localgate.dashboard.routes import mount_dashboard

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(completions.router)
    app.include_router(embeddings.router)
    app.include_router(models.router)
    app.include_router(conversations.router)
    app.include_router(keys.router, prefix="/admin")
    app.include_router(usage.router, prefix="/admin")
    app.include_router(config.router, prefix="/admin")
    app.include_router(export.router, prefix="/admin")

    mount_dashboard(app)

    return app
