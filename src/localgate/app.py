"""FastAPI app factory — wires config, backend, and routers together."""
from fastapi import FastAPI

from localgate.backends import get_backend
from localgate.config import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="localgate", version="0.1.0")

    app.state.settings = settings
    app.state.backend = get_backend(settings.backend_type, settings.backend_url)

    from localgate.api import chat
    app.include_router(chat.router)

    # from localgate.api import models, keys, usage
    # app.include_router(models.router)
    # app.include_router(keys.router, prefix="/admin")
    # app.include_router(usage.router, prefix="/admin")

    @app.get("/health")
    async def health() -> dict:
        backend_ok = await app.state.backend.health()
        return {"status": "ok", "backend_reachable": backend_ok}

    @app.on_event("shutdown")
    async def shutdown() -> None:
        aclose = getattr(app.state.backend, "aclose", None)
        if aclose:
            await aclose()

    return app
