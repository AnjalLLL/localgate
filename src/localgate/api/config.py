"""GET/PUT /admin/config — view current config, and establish a new database.

Database and backend URLs are NOT changed live in the running process:
swapping the database connection live would require re-creating the engine
and re-running migrations mid-flight, which is out of scope for a simple
admin panel. Establishing a new database here tests the connection, then
persists it to localgate.config.json (see core/db_config_store.py) — a
restart is required to actually start using it.
"""
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from localgate.api.deps import require_admin
from localgate.core.db_config_store import is_database_established, save_database_url
from localgate.db.engine import make_engine

router = APIRouter(dependencies=[Depends(require_admin)])


def _redact_database_url(url: str) -> str:
    """Hides credentials in connection strings like postgresql://user:pass@host/db."""
    if "@" not in url:
        return url
    scheme_and_creds, rest = url.rsplit("@", 1)
    scheme = scheme_and_creds.split("://", 1)[0]
    return f"{scheme}://***:***@{rest}"


@router.get("/config")
async def get_config(request: Request):
    settings = request.app.state.settings
    return {
        "backend_type": settings.backend_type,
        "backend_url": settings.backend_url,
        "default_model": settings.default_model,
        "database_url": _redact_database_url(settings.database_url),
        "database_established": is_database_established(),
        "memory_enabled": settings.memory_enabled,
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "max_retrieved_chunks": settings.max_retrieved_chunks,
    }


class DatabaseUrlUpdate(BaseModel):
    database_url: str


async def _test_connection(url: str) -> None:
    """Raises HTTPException with a clear message if the URL doesn't actually connect."""
    engine = make_engine(url)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not connect to that database: {e.__class__.__name__}: {e}",
        ) from e
    finally:
        await engine.dispose()


@router.put("/config/database-url")
async def update_database_url(body: DatabaseUrlUpdate):
    url = body.database_url.strip()
    if "://" not in url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Doesn't look like a valid database URL (expected e.g. "
            "'postgresql+asyncpg://user:pass@host/db' or 'sqlite+aiosqlite:///./local.db').",
        )

    # Prove it's reachable BEFORE persisting it — this is what makes it "established"
    # rather than just a string someone typed into a form.
    await _test_connection(url)

    save_database_url(url)
    return {
        "established": True,
        "database_url": _redact_database_url(url),
        "restart_required": True,
        "message": "Connection verified and saved to localgate.config.json. "
        "Restart the server to start using it — all data (keys, usage, chat "
        "history, memory) will then be stored there.",
    }
