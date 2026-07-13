"""FastAPI app factory — wires config, DB, backend, memory, and routers together."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from localgate.backends import get_backend
from localgate.config import Settings
from localgate.core.db_config_store import is_database_established, load_database_url
from localgate.core.rate_limiter import RateLimiter
from localgate.db.engine import init_models, make_engine, make_session_factory

DASHBOARD_DIR = Path(__file__).parent / "dashboard" / "static"
logger = logging.getLogger("localgate")


def _resolve_database_url(settings: Settings, config_path: Path | None = None) -> str:
    """A database saved through the admin UI (and connection-tested at save time)
    always wins over LOCALGATE_DATABASE_URL from .env — that's what makes it
    "established" rather than just configured.
    """
    from localgate.core.db_config_store import DEFAULT_CONFIG_PATH

    stored_url = load_database_url(config_path or DEFAULT_CONFIG_PATH)
    if stored_url:
        logger.info("Using established database from localgate.config.json")
        return stored_url
    logger.info("No established database found — using LOCALGATE_DATABASE_URL from .env")
    return settings.database_url


def create_app(settings: Settings | None = None, database_config_path: Path | None = None) -> FastAPI:
    """`database_config_path` lets tests point at an isolated (or nonexistent) config
    file so they never silently pick up a real established database from the
    developer's working directory."""
    settings = settings or Settings()
    active_database_url = _resolve_database_url(settings, database_config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup — every table (API keys, usage, conversations, memory chunks)
        # lives in this one engine/session factory, so once a database is
        # established, ALL of localgate's data goes there, not just some of it.
        engine = make_engine(active_database_url)
        await init_models(engine)
        app.state.session_factory = make_session_factory(engine)
        app.state.backend = get_backend(settings.backend_type, settings.backend_url)
        app.state.rate_limiter = RateLimiter()

        logger.info(f"localgate started. Database established: {is_database_established()}")

        yield

        # Shutdown
        aclose = getattr(app.state.backend, "aclose", None)
        if aclose:
            await aclose()
        await engine.dispose()

    app = FastAPI(title="localgate", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.settings.database_url = active_database_url  # reflect reality in /admin/config too

    from localgate.api import chat, config, conversations, keys, usage

    app.include_router(chat.router)
    app.include_router(conversations.router)
    app.include_router(keys.router, prefix="/admin")
    app.include_router(usage.router, prefix="/admin")
    app.include_router(config.router, prefix="/admin")

    @app.get("/health")
    async def health() -> dict:
        backend_ok = await app.state.backend.health()
        return {"status": "ok", "backend_reachable": backend_ok}

    # Serve the simple dashboard UI at /dashboard (index.html at /dashboard/)
    if DASHBOARD_DIR.exists():
        app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")

    return app
