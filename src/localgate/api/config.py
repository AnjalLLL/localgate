"""``/admin/config`` — inspect the running config, and establish a database.

The database URL is **not** swapped live. Doing that would mean rebuilding the
engine and re-running migrations underneath in-flight requests, and there is no
version of that which is safe in a admin panel. Instead, establishing a database
here proves the connection works, persists it to ``localgate.config.json``, and
tells the operator to restart — at which point *all* of localgate's data (keys,
usage, history, memory) is served from it.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from localgate.api.deps import require_admin
from localgate.backends import available_backends
from localgate.core.db_config_store import is_database_established, save_database_url
from localgate.core.errors import InvalidRequestError
from localgate.db.engine import make_engine

router = APIRouter(tags=["admin"], dependencies=[Depends(require_admin)])


def redact_database_url(url: str) -> str:
    """Hide credentials in a connection string before it is displayed or logged."""
    if "@" not in url:
        return url
    scheme_and_creds, rest = url.rsplit("@", 1)
    scheme = scheme_and_creds.split("://", 1)[0]
    return f"{scheme}://***:***@{rest}"


@router.get("/config")
async def get_config(request: Request) -> dict:
    settings = request.app.state.settings
    config_path = request.app.state.database_config_path

    return {
        "environment": settings.environment,
        "backend": {
            "type": settings.backend_type,
            "url": settings.backend_url,
            "timeout": settings.backend_timeout,
            "default_model": settings.default_model,
            "model_aliases": settings.model_aliases,
            "available_types": available_backends(),
        },
        "database": {
            "url": redact_database_url(settings.database_url),
            "established": is_database_established(config_path),
        },
        "memory": {
            "enabled": settings.memory_enabled,
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "max_retrieved_chunks": settings.max_retrieved_chunks,
            "min_score": settings.memory_min_score,
            "summarize_after_messages": settings.summarize_after_messages,
        },
        "cache": {
            "enabled": settings.cache_enabled,
            **(request.app.state.cache.stats() if settings.cache_enabled else {}),
        },
        "security": {"using_default_admin_key": settings.uses_insecure_admin_key},
    }


class DatabaseUrlUpdate(BaseModel):
    database_url: str


async def _test_connection(url: str) -> None:
    """Raise a 400 with an actionable message if the URL does not actually connect.

    The ``except`` clauses here are broader than they look like they should be, and
    deliberately so. A missing driver raises ``ModuleNotFoundError`` and a bad query
    param raises ``TypeError`` — neither is a ``SQLAlchemyError``, so a narrow
    handler would let both escape as a raw 500 with a traceback, which is precisely
    the failure this endpoint exists to prevent. Every one of these is a mistake a
    person makes while pasting a connection string, and each gets told what to fix.
    """
    engine = None
    try:
        engine = make_engine(url)
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

    except ModuleNotFoundError as exc:
        hint = ""
        if url.startswith(("postgresql://", "postgres://")):
            hint = (
                " That is a plain 'postgresql://' URL, which selects a synchronous driver. "
                "localgate is async — use 'postgresql+asyncpg://...' (the same URL with "
                "'+asyncpg' added after 'postgresql')."
            )
        raise InvalidRequestError(f"Missing database driver ({exc}).{hint}") from exc

    except TypeError as exc:
        hint = ""
        if "sslmode" in str(exc):
            hint = (
                " Your URL has '?sslmode=require', which is the psycopg spelling. asyncpg "
                "calls the same thing 'ssl=' — replace 'sslmode=' with 'ssl=' and retry. "
                "(Neon's copy-paste string uses the psycopg spelling, so this catches most "
                "people once.)"
            )
        raise InvalidRequestError(f"Connection argument error: {exc}.{hint}") from exc

    except SQLAlchemyError as exc:
        raise InvalidRequestError(
            f"Could not connect to that database: {type(exc).__name__}: {exc}"
        ) from exc

    except Exception as exc:  # noqa: BLE001 — see docstring
        raise InvalidRequestError(
            f"Unexpected error testing that connection: {type(exc).__name__}: {exc}"
        ) from exc

    finally:
        if engine is not None:
            await engine.dispose()


@router.put("/config/database-url")
async def update_database_url(body: DatabaseUrlUpdate, request: Request) -> dict:
    url = body.database_url.strip()
    if "://" not in url:
        raise InvalidRequestError(
            "That does not look like a database URL. Expected something like "
            "'postgresql+asyncpg://user:pass@host/db' or 'sqlite+aiosqlite:///./local.db'."
        )

    # Prove it is reachable BEFORE persisting it. That is the whole difference
    # between a database that is "established" and a string someone typed into a form.
    await _test_connection(url)

    save_database_url(url, request.app.state.database_config_path)
    return {
        "established": True,
        "database_url": redact_database_url(url),
        "restart_required": True,
        "message": (
            "Connection verified and saved to localgate.config.json. Restart the server to "
            "start using it — API keys, usage, chat history and memory will all be stored there."
        ),
    }
