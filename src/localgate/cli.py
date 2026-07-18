"""The ``localgate`` command line.

The CLI talks to the database and the backend directly rather than to a running
server. That is deliberate: ``localgate keys create`` has to work *before* you have
a key, and ``localgate db upgrade`` has to work when the server won't start because
the schema is out of date. A CLI that could only drive a healthy server would be
useless in exactly the situations you reach for it.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Coroutine
from pathlib import Path
from typing import Any, TypeVar

import typer
import uvicorn
from sqlalchemy.exc import OperationalError

from localgate import __version__
from localgate.agent.loop import AgentTurnLimitExceeded
from localgate.agent.memory import AgentMemory, get_or_create_local_agent_key_id, project_session_id
from localgate.agent.repl import run_repl, run_single_shot
from localgate.app import resolve_database_url
from localgate.backends import available_backends, get_backend
from localgate.config import Settings
from localgate.db.engine import current_revision, init_models, make_engine, make_session_factory
from localgate.db.repositories.keys import APIKeyRepository
from localgate.db.repositories.usage import UsageRepository

app = typer.Typer(
    help="localgate — a local-first API gateway for open-source LLMs.",
    no_args_is_help=True,
)
keys_app = typer.Typer(help="Create, inspect and revoke API keys.", no_args_is_help=True)
db_app = typer.Typer(help="Initialize and migrate the database.", no_args_is_help=True)
app.add_typer(keys_app, name="keys")
app.add_typer(db_app, name="db")

T = TypeVar("T")

err = typer.style


def _settings() -> Settings:
    """Load settings, turning a config error into a readable message rather than a traceback."""
    try:
        return Settings()
    except Exception as exc:  # noqa: BLE001 — pydantic raises ValidationError; any of it is fatal
        typer.secho(f"Configuration error:\n{exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


def _run(coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


async def _with_session(settings: Settings, fn: Callable[[Any], Awaitable[T]]) -> T:
    """Open the same database the server would use, run ``fn``, and clean up.

    A missing table means the database has never been migrated — by far the most
    likely reason a command fails on a fresh install. That deserves a sentence, not
    a SQLAlchemy traceback.
    """
    engine = make_engine(resolve_database_url(settings))
    try:
        async with make_session_factory(engine)() as session:
            return await fn(session)
    except OperationalError as exc:
        if "no such table" in str(exc) or "does not exist" in str(exc):
            typer.secho(
                "This database has no localgate schema yet. Run: localgate db upgrade",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1) from exc
        typer.secho(f"Database error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    finally:
        await engine.dispose()


# --------------------------------------------------------------------------- server


@app.command()
def serve(
    host: str | None = typer.Option(None, help="Interface to bind. Defaults to LOCALGATE_HOST."),
    port: int | None = typer.Option(None, help="Port to bind. Defaults to LOCALGATE_PORT."),
    reload: bool = typer.Option(False, help="Reload on code changes (development only)."),
    workers: int = typer.Option(
        1,
        help=(
            "Worker processes. Note that rate limits and the prompt cache are "
            "per-process, so N workers means N independent limiters."
        ),
    ),
) -> None:
    """Start the gateway."""
    settings = _settings()
    uvicorn.run(
        "localgate.app:create_app",
        factory=True,
        host=host or settings.host,
        port=port or settings.port,
        reload=reload,
        workers=None if reload else workers,
        log_config=None,  # localgate configures structlog itself; don't fight over it
    )


@app.command()
def health() -> None:
    """Check that the backend and the database are actually reachable."""
    settings = _settings()
    exit_code = 0

    backend = get_backend(
        settings.backend_type,
        settings.backend_url,
        timeout=settings.backend_timeout,
        api_key=settings.backend_api_key,
    )

    async def check() -> tuple[bool, list[str], str | None, str | None]:
        try:
            ok = await backend.health()
            models = await backend.list_models() if ok else []
        except Exception:  # noqa: BLE001 — a health check that raises has answered "no"
            ok, models = False, []

        engine = make_engine(resolve_database_url(settings))
        try:
            revision = await current_revision(engine)
            db_error = None
        except Exception as exc:  # noqa: BLE001
            revision, db_error = None, f"{type(exc).__name__}: {exc}"
        finally:
            await engine.dispose()
            await backend.aclose()
        return ok, models, db_error, revision

    backend_ok, models, db_error, revision = _run(check())

    if backend_ok:
        typer.secho(
            f"✓ backend  {settings.backend_type} at {settings.backend_url} "
            f"({len(models)} model{'s' if len(models) != 1 else ''})",
            fg=typer.colors.GREEN,
        )
    else:
        typer.secho(
            f"✗ backend  {settings.backend_type} at {settings.backend_url} is unreachable",
            fg=typer.colors.RED,
        )
        exit_code = 1

    dialect = resolve_database_url(settings).split("://", 1)[0]
    if db_error is not None:
        typer.secho(f"✗ database {db_error}", fg=typer.colors.RED)
        exit_code = 1
    elif revision is None:
        # Connected but never migrated — the server would fix this itself on startup,
        # but saying so beats reporting a healthy database with no tables in it.
        typer.secho(
            f"! database {dialect} — connected, but not migrated. Run: localgate db upgrade",
            fg=typer.colors.YELLOW,
        )
    else:
        typer.secho(
            f"✓ database {dialect} — connected (migration {revision})", fg=typer.colors.GREEN
        )

    if settings.uses_insecure_admin_key:
        typer.secho(
            "! admin key is still the default placeholder — set LOCALGATE_ADMIN_KEY",
            fg=typer.colors.YELLOW,
        )

    raise typer.Exit(code=exit_code)


@app.command()
def backends() -> None:
    """List the installed backends, including any provided by plugins."""
    for name in available_backends():
        typer.echo(name)


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


# ---------------------------------------------------------------------------- agent


@app.command()
def code(
    task: str | None = typer.Argument(
        None, help="What to do, e.g. 'add a health check to app.py'. Omit for a REPL."
    ),
    directory: Path = typer.Option(
        Path.cwd(), "--dir", "-C", help="Project root the agent may read and write within."
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Defaults to LOCALGATE_DEFAULT_MODEL."
    ),
    auto_approve: bool = typer.Option(
        False,
        "--auto-approve",
        help="Write files without asking first. Use with a clean git tree.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Skip the dirty-working-tree warning and proceed anyway."
    ),
    auto_commit: bool = typer.Option(
        False,
        "--auto-commit",
        help="Commit whatever the agent wrote after each turn, tagged 'localgate-agent:'.",
    ),
    no_memory: bool = typer.Option(
        False,
        "--no-memory",
        help="Skip RAG memory for this run, even if LOCALGATE_MEMORY_ENABLED is on.",
    ),
) -> None:
    """Run a coding agent against the backend, editing files under DIRECTORY.

    With TASK, runs one task and exits. Without it, starts an interactive
    session — keep talking about the same project without re-invoking the
    command each time. Talks to the inference backend directly (like
    `localgate health`), not to a running gateway — no API key needed for local use.

    Conversation history and recalled context are stored per project (see
    `.localgate/session_id`), the same memory layer the HTTP API uses — so
    re-running this in a project you worked on before picks up where you left off.
    """
    settings = _settings()
    root = directory.resolve()
    if not root.is_dir():
        typer.secho(f"Not a directory: {root}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    resolved_model = settings.resolve_model(model)
    backend = get_backend(
        settings.backend_type,
        settings.backend_url,
        timeout=settings.backend_timeout,
        api_key=settings.backend_api_key,
    )
    engine = make_engine(resolve_database_url(settings))

    async def go() -> str | None:
        try:
            memory = None
            if settings.memory_enabled and not no_memory:
                await init_models(engine)
                session_factory = make_session_factory(engine)
                async with session_factory() as db_session:
                    api_key_id = await get_or_create_local_agent_key_id(db_session, settings)
                memory = AgentMemory(
                    session_factory, backend, settings, project_session_id(root), api_key_id
                )

            if task is None:
                await run_repl(
                    backend,
                    resolved_model,
                    root,
                    auto_approve=auto_approve,
                    force=force,
                    auto_commit=auto_commit,
                    memory=memory,
                )
                return None
            return await run_single_shot(
                backend,
                resolved_model,
                root,
                task,
                auto_approve=auto_approve,
                force=force,
                auto_commit=auto_commit,
                memory=memory,
            )
        finally:
            await backend.aclose()
            await engine.dispose()

    try:
        _run(go())
    except AgentTurnLimitExceeded as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        typer.secho("\nCancelled.", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=130) from None


# ----------------------------------------------------------------------------- keys


@keys_app.command("create")
def keys_create(
    name: str = typer.Option(..., "--name", "-n", help="A label, e.g. the app that will use it."),
    rate_limit: int | None = typer.Option(
        None,
        "--rate-limit",
        help="Requests per minute. Defaults to LOCALGATE_DEFAULT_RATE_LIMIT_PER_MIN.",
    ),
) -> None:
    """Create an API key and print it. This is the only time it is ever shown."""
    settings = _settings()
    limit = rate_limit or settings.default_rate_limit_per_min

    async def create(session: Any) -> tuple[str, str]:
        key, raw = await APIKeyRepository(session).create(name, limit)
        return key.id, raw

    async def migrate_then_create() -> tuple[str, str]:
        # A first-run `keys create` shouldn't fail just because nobody has run
        # `db upgrade` yet — creating your first key is how you start.
        engine = make_engine(resolve_database_url(settings))
        try:
            await init_models(engine)
        finally:
            await engine.dispose()
        return await _with_session(settings, create)

    key_id, raw_key = _run(migrate_then_create())

    typer.secho(f"\n  {raw_key}\n", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  id           {key_id}")
    typer.echo(f"  name         {name}")
    typer.echo(f"  rate limit   {limit}/min")
    typer.secho(
        "\n  Store it now — only its hash is kept, so it cannot be shown again.\n",
        fg=typer.colors.YELLOW,
    )


@keys_app.command("list")
def keys_list(
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """List every key, active and revoked."""
    settings = _settings()

    async def fetch(session: Any) -> list[dict]:
        return [
            {
                "id": key.id,
                "name": key.name,
                "prefix": key.key_prefix,
                "revoked": key.revoked,
                "rate_limit_per_min": key.rate_limit_per_min,
                "last_used_at": key.last_used_at.isoformat() if key.last_used_at else None,
            }
            for key in await APIKeyRepository(session).list_all()
        ]

    rows = _run(_with_session(settings, fetch))

    if as_json:
        typer.echo(json.dumps(rows, indent=2))
        return

    if not rows:
        typer.echo("No API keys yet. Create one with: localgate keys create --name my-app")
        return

    typer.echo(f"{'ID':<38} {'NAME':<20} {'PREFIX':<13} {'LIMIT':<7} STATUS")
    for row in rows:
        status = (
            typer.style("revoked", fg=typer.colors.RED)
            if row["revoked"]
            else typer.style("active", fg=typer.colors.GREEN)
        )
        typer.echo(
            f"{row['id']:<38} {row['name'][:19]:<20} {row['prefix']:<13} "
            f"{str(row['rate_limit_per_min']) + '/min':<7} {status}"
        )


@keys_app.command("revoke")
def keys_revoke(key_id: str = typer.Argument(..., help="The key's id (from `keys list`).")) -> None:
    """Revoke a key. Its usage history is kept."""
    settings = _settings()

    async def revoke(session: Any) -> bool:
        return await APIKeyRepository(session).revoke(key_id)

    if not _run(_with_session(settings, revoke)):
        typer.secho(f"No API key with id {key_id!r}.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.secho(f"Revoked {key_id}.", fg=typer.colors.GREEN)


@keys_app.command("usage")
def keys_usage(key_id: str = typer.Argument(..., help="The key's id (from `keys list`).")) -> None:
    """Show token usage for one key."""
    settings = _settings()

    async def fetch(session: Any) -> dict:
        return await UsageRepository(session).summary_for_key(key_id)

    summary = _run(_with_session(settings, fetch))
    typer.echo(json.dumps(summary, indent=2))


# ------------------------------------------------------------------------------- db


@db_app.command("init")
def db_init() -> None:
    """Create the schema in a fresh database (an alias for `db upgrade`)."""
    db_upgrade()


@db_app.command("upgrade")
def db_upgrade() -> None:
    """Apply any pending migrations."""
    settings = _settings()
    url = resolve_database_url(settings)

    async def upgrade() -> str | None:
        engine = make_engine(url)
        try:
            await init_models(engine)
            return await current_revision(engine)
        finally:
            await engine.dispose()

    try:
        revision = _run(upgrade())
    except Exception as exc:  # noqa: BLE001 — the point is a readable message, not a traceback
        typer.secho(f"Migration failed: {type(exc).__name__}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.secho(f"Database is up to date (revision {revision}).", fg=typer.colors.GREEN)


@db_app.command("current")
def db_current() -> None:
    """Print the migration revision the database is currently at."""
    settings = _settings()

    async def revision() -> str | None:
        engine = make_engine(resolve_database_url(settings))
        try:
            return await current_revision(engine)
        finally:
            await engine.dispose()

    current = _run(revision())
    if current is None:
        typer.secho(
            "This database has never been migrated. Run: localgate db upgrade",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=1)
    typer.echo(current)


if __name__ == "__main__":
    app()
