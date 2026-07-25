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
import os
import secrets
import stat
from collections.abc import Awaitable, Callable, Coroutine
from pathlib import Path
from typing import Any, TypeVar

import httpx
import typer
import uvicorn
from sqlalchemy.exc import OperationalError

from localgate import __version__, paths
from localgate.agent.loop import AgentTurnLimitExceeded
from localgate.agent.mcp import McpRegistry, load_mcp_servers
from localgate.agent.memory import AgentMemory, get_or_create_local_agent_key_id, project_session_id
from localgate.agent.repl import describe_backend_error, run_repl, run_single_shot
from localgate.agent.theme import THEMES
from localgate.agent.userconfig import UserConfig
from localgate.agent.websearch import make_search_fn
from localgate.app import resolve_database_url
from localgate.backends import available_backends, get_backend
from localgate.config import INSECURE_ADMIN_KEY, Settings, load_settings
from localgate.core.db_config_store import (
    _default_config_path,
    redact_database_url,
    save_database_url,
)
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

#: `localgate code`'s exit codes — distinct on purpose, so a script can tell
#: "the agent gave up" apart from "the backend was unreachable" apart from
#: "a write was withheld", rather than lumping every failure into a bare 1.
EXIT_MAX_TURNS_EXCEEDED = 3
EXIT_WRITE_DECLINED = 4
EXIT_BACKEND_REJECTED = 5
EXIT_BACKEND_UNREACHABLE = 6

T = TypeVar("T")

err = typer.style


def _version_callback(show: bool) -> None:
    if show:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(  # noqa: ARG001 — read only by _version_callback's side effect
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the installed version and exit.",
    ),
) -> None:
    """localgate — a local-first API gateway for open-source LLMs."""


def _settings() -> Settings:
    """Load settings, turning a config error into a readable message rather than a traceback."""
    try:
        return load_settings()
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
    proxy_headers: bool = typer.Option(
        False,
        "--proxy-headers/--no-proxy-headers",
        help="Trust X-Forwarded-For/X-Forwarded-Proto headers from a reverse proxy.",
    ),
    forwarded_allow_ips: str | None = typer.Option(
        None,
        "--forwarded-allow-ips",
        help=(
            "Comma-separated IPs (or '*') that may set forwarded headers. "
            "Requires --proxy-headers. Defaults to FORWARDED_ALLOW_IPS env var."
        ),
    ),
) -> None:
    """Start the gateway."""
    settings = _settings()
    uvicorn_kwargs: dict = dict(
        factory=True,
        host=host or settings.host,
        port=port or settings.port,
        reload=reload,
        workers=None if reload else workers,
        log_config=None,  # localgate configures structlog itself; don't fight over it
    )
    if proxy_headers:
        uvicorn_kwargs["proxy_headers"] = True
        if forwarded_allow_ips is not None:
            uvicorn_kwargs["forwarded_allow_ips"] = forwarded_allow_ips
    uvicorn.run("localgate.app:create_app", **uvicorn_kwargs)


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


# ----------------------------------------------------------------------------- init


@app.command()
def init(
    force: bool = typer.Option(False, "--force", help="Regenerate the admin key even if one exists."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview what would be written without touching the filesystem."),
) -> None:
    """Set up localgate for the first time: create config dirs, generate a secure admin key,
    and run database migrations.

    Safe to re-run — skips steps that are already done unless --force is given.
    """
    config_dir = paths.ensure_config_dir()
    data_dir = paths.ensure_data_dir()
    env_file = paths.user_env_file()

    if dry_run:
        typer.echo(f"[dry-run] config dir : {config_dir}")
        typer.echo(f"[dry-run] data dir   : {data_dir}")
        typer.echo(f"[dry-run] env file   : {env_file}")
        typer.echo("[dry-run] Would write admin key to env file (0600).")
        return

    # Generate a new admin key only when the env file is absent or --force.
    wrote_key = False
    if force or not env_file.exists():
        admin_key = secrets.token_urlsafe(32)
        # Include bare minimum so the file is immediately usable.
        content = (
            f"LOCALGATE_ADMIN_KEY={admin_key}\n"
            "# Add other LOCALGATE_* vars here — see localgate --help for options.\n"
        )
        paths.write_secret_text(env_file, content)
        wrote_key = True

    # Run migrations against the default database.
    settings = _settings()
    db_url = resolve_database_url(settings)

    async def migrate() -> str | None:
        engine = make_engine(db_url)
        try:
            await init_models(engine)
            return await current_revision(engine)
        finally:
            await engine.dispose()

    try:
        revision = _run(migrate())
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"Migration failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.secho("localgate init complete.", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  config dir  {config_dir}")
    typer.echo(f"  data dir    {data_dir}")
    if wrote_key:
        typer.secho(f"  env file    {env_file}  (admin key written — 0600)", fg=typer.colors.GREEN)
        typer.secho(
            "  Run `localgate serve` to start. The admin key is in the env file above.",
            fg=typer.colors.YELLOW,
        )
    else:
        typer.echo(f"  env file    {env_file}  (already exists — skipped; use --force to regenerate)")
    typer.echo(f"  database    {redact_database_url(db_url)}  (revision {revision})")


# ---------------------------------------------------------------------------- doctor


@app.command()
def doctor() -> None:
    """Diagnose the localgate installation: config paths, env files loaded, database URL source,
    secret file permissions, and backend/embedding model reachability.
    """
    ok = True

    def _check(label: str, value: str, warn: bool = False) -> None:
        color = typer.colors.YELLOW if warn else typer.colors.GREEN
        marker = "!" if warn else "✓"
        typer.secho(f"{marker} {label:<30} {value}", fg=color)

    def _fail(label: str, value: str) -> None:
        nonlocal ok
        ok = False
        typer.secho(f"✗ {label:<30} {value}", fg=typer.colors.RED)

    typer.secho("--- paths ---", bold=True)
    config_dir = paths.config_dir()
    data_dir = paths.data_dir()
    typer.echo(f"  config dir   {config_dir}  ({paths.describe_mode(config_dir) if config_dir.exists() else 'missing'})")
    typer.echo(f"  data dir     {data_dir}  ({paths.describe_mode(data_dir) if data_dir.exists() else 'missing'})")

    env_file = paths.user_env_file()
    env_mode = paths.describe_mode(env_file)
    if not env_file.exists():
        _fail("user env file", f"{env_file} (missing — run `localgate init`)")
    elif env_mode != "0600":
        _check("user env file", f"{env_file}  perms {env_mode}", warn=True)
    else:
        _check("user env file", f"{env_file}  (0600)")

    db_config = _default_config_path()
    if db_config.exists():
        _check("db config file", f"{db_config}  ({paths.describe_mode(db_config)})")
    else:
        typer.echo(f"  db config    {db_config}  (not set — using env/defaults)")

    legacy = Path("localgate.config.json")
    if legacy.exists():
        _check("legacy config", f"{legacy} exists in CWD — run `localgate init` to migrate", warn=True)

    typer.secho("\n--- settings ---", bold=True)
    try:
        settings = load_settings()
    except Exception as exc:  # noqa: BLE001
        _fail("settings", f"failed to load: {exc}")
        raise typer.Exit(code=1) from exc

    from localgate.core.db_config_store import load_database_url  # noqa: PLC0415

    stored_url = load_database_url()
    if stored_url:
        db_url = stored_url
        db_source = "db config file (highest precedence)"
    elif os.environ.get("LOCALGATE_DATABASE_URL"):
        db_url = settings.database_url
        db_source = "LOCALGATE_DATABASE_URL env var"
    else:
        db_url = settings.database_url
        db_source = "default (data_dir)"
    typer.echo(f"  database URL  {redact_database_url(db_url)}")
    typer.echo(f"  source        {db_source}")
    typer.echo(f"  backend       {settings.backend_type} at {settings.backend_url}")

    if settings.uses_insecure_admin_key:
        _check("admin key", "placeholder — run `localgate init` to generate a real one", warn=True)
    else:
        _check("admin key", "set (non-default)")

    typer.secho("\n--- database ---", bold=True)

    async def check_db() -> tuple[str | None, str | None]:
        engine = make_engine(db_url)
        try:
            revision = await current_revision(engine)
            return revision, None
        except Exception as exc:  # noqa: BLE001
            return None, str(exc)
        finally:
            await engine.dispose()

    revision, db_err = _run(check_db())
    if db_err:
        _fail("database", db_err)
    elif revision is None:
        _check("database", "connected but not migrated — run `localgate db upgrade`", warn=True)
    else:
        _check("database", f"connected (revision {revision})")

    raise typer.Exit(code=0 if ok else 1)


# ----------------------------------------------------------------------------- login/whoami

_GATEWAY_URL_FILE = "gateway_url"
_GATEWAY_KEY_FILE = "gateway_key"


def _read_gateway_creds() -> tuple[str, str] | None:
    """Read stored gateway URL and key, or return None if not logged in."""
    url_file = paths.config_dir() / _GATEWAY_URL_FILE
    key_file = paths.config_dir() / _GATEWAY_KEY_FILE
    if not url_file.exists() or not key_file.exists():
        return None
    url = url_file.read_text(encoding="utf-8").strip()
    key = key_file.read_text(encoding="utf-8").strip()
    if not url or not key:
        return None
    return url, key


@app.command()
def login(
    url: str = typer.Option(..., "--url", help="Base URL of the localgate gateway, e.g. https://gw.example.com"),
    api_key: str = typer.Option(..., "--api-key", help="An API key (not the admin key) minted on that gateway."),
) -> None:
    """Save credentials for a remote localgate gateway.

    Verifies the key is valid by hitting /v1/usage (pure DB, no backend needed),
    and checks the gateway version via /health/live.
    After login, `localgate code --remote` will route to this gateway.
    """
    base = url.rstrip("/")

    if api_key == INSECURE_ADMIN_KEY or api_key.startswith("X-Admin-Key"):
        typer.secho("This looks like an admin key. Use a regular API key here.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    async def verify() -> None:
        async with httpx.AsyncClient(base_url=base, timeout=10.0) as client:
            # Check the gateway is reachable and get version info.
            try:
                live = await client.get("/health/live")
                remote_version = live.json().get("version", "unknown") if live.status_code == 200 else "unknown"
            except httpx.HTTPError:
                remote_version = "unknown"

            # Probe with the API key — /v1/usage never calls the backend.
            resp = await client.get("/v1/usage", headers={"Authorization": f"Bearer {api_key}"})
            if resp.status_code == 401:
                typer.secho("Invalid or revoked API key.", fg=typer.colors.RED, err=True)
                raise typer.Exit(code=1)
            resp.raise_for_status()

            from localgate import __version__ as local_version  # noqa: PLC0415
            if remote_version != "unknown" and remote_version.split(".")[0] != local_version.split(".")[0]:
                typer.secho(
                    f"Warning: local version {local_version} and gateway version {remote_version} "
                    "have different major versions — some features may not work.",
                    fg=typer.colors.YELLOW,
                )

    try:
        _run(verify())
    except httpx.HTTPError as exc:
        typer.secho(f"Could not reach gateway: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=6) from exc

    url_file = paths.config_dir() / _GATEWAY_URL_FILE
    key_file = paths.config_dir() / _GATEWAY_KEY_FILE
    paths.ensure_config_dir()
    paths.write_secret_text(url_file, base + "\n")
    paths.write_secret_text(key_file, api_key + "\n")

    typer.secho(f"Logged in to {base}", fg=typer.colors.GREEN)
    typer.echo("Run `localgate whoami` to verify, or `localgate code --remote` to use the gateway.")


@app.command()
def whoami() -> None:
    """Show usage stats for the currently logged-in gateway key."""
    creds = _read_gateway_creds()
    if creds is None:
        typer.secho(
            "Not logged in. Run `localgate login --url <url> --api-key <key>` first.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    base, api_key = creds

    async def fetch() -> dict:
        async with httpx.AsyncClient(base_url=base, timeout=10.0) as client:
            resp = await client.get("/v1/usage", headers={"Authorization": f"Bearer {api_key}"})
            resp.raise_for_status()
            return resp.json()

    try:
        data = _run(fetch())
    except httpx.HTTPError as exc:
        typer.secho(f"Could not reach gateway: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=6) from exc

    typer.echo(f"Gateway: {base}")
    typer.echo(json.dumps(data, indent=2))


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
    auto_approve: bool | None = typer.Option(
        None,
        "--auto-approve/--no-auto-approve",
        help="Write files without asking first. Defaults to the config file, else off.",
    ),
    plan: bool = typer.Option(
        False,
        "--plan",
        help="Start in plan mode: writes are queued and reviewed as one batch, "
        "not applied as they happen. Toggle live with shift+tab or /mode.",
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
    allow_delegation: bool = typer.Option(
        False,
        "--allow-delegation",
        help="Let the agent hand off self-contained sub-tasks to a nested, "
        "read-only-by-default sub-agent (depth 1, no recursion). Off by default — "
        "test against your model before relying on it for small local models.",
    ),
    no_mcp: bool = typer.Option(
        False,
        "--no-mcp",
        help="Skip connecting to configured MCP servers "
        "(~/.config/localgate/mcp_servers.json) for this run.",
    ),
    theme: str | None = typer.Option(
        None,
        "--theme",
        help=f"One of {', '.join(THEMES)}. Persists to the config file when set.",
    ),
    no_color: bool = typer.Option(
        False, "--no-color", help="Disable colored output. Also respects the NO_COLOR env var."
    ),
    max_turns: int | None = typer.Option(
        None,
        "--max-turns",
        help="Tool-call turns before giving up on a task. Defaults to the config file, else 20.",
    ),
    remote: bool | None = typer.Option(
        None,
        "--remote/--local",
        help=(
            "Route inference through a remote localgate gateway (set via `localgate login`). "
            "Defaults to remote when login credentials are present. "
            "--remote implies --no-memory (the gateway handles memory server-side)."
        ),
    ),
) -> None:
    """Run a coding agent against the backend, editing files under DIRECTORY.

    With TASK, runs one task and exits. Without it, starts an interactive
    session — keep talking about the same project without re-invoking the
    command each time. Talks to the inference backend directly (like
    `localgate health`), not to a running gateway — no API key needed for local use.

    When `localgate login` credentials are present, --remote routes inference through
    the gateway instead. File-editing tools always run locally.

    Conversation history and recalled context are stored per project (see
    `.localgate/session_id`), the same memory layer the HTTP API uses — so
    re-running this in a project you worked on before picks up where you left off.

    Preferences (theme, default model, auto-approve, max-turns) persist across
    invocations in `~/.config/localgate/config.toml` — see `/config` in the REPL.
    Precedence: flags > `LOCALGATE_*` env vars > that config file > built-in defaults.

    Exit codes (with TASK, i.e. non-interactive use): 0 success, 2 bad usage
    (e.g. --dir isn't a directory), 3 the agent hit --max-turns without
    finishing, 4 a write was declined, 5 the backend rejected the request
    (e.g. the model doesn't support tool calling), 6 the backend was
    unreachable, 130 interrupted (Ctrl+C).
    """
    settings = _settings()
    root = directory.resolve()
    if not root.is_dir():
        typer.secho(f"Not a directory: {root}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    user_config = UserConfig.load()
    if theme is not None:
        if theme not in THEMES:
            typer.secho(
                f"Invalid theme {theme!r} — choose from {', '.join(THEMES)}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)
        user_config = user_config.with_override(theme=theme)
        user_config.save()

    resolved_theme = theme or user_config.theme
    resolved_no_color = no_color or bool(os.environ.get("NO_COLOR"))
    resolved_auto_approve = auto_approve if auto_approve is not None else user_config.auto_approve
    resolved_max_turns = max_turns if max_turns is not None else user_config.max_turns

    # LOCALGATE_DEFAULT_MODEL (an env var) outranks the config file's default_model,
    # matching the flags > env > config file > built-in-default precedence order.
    env_model = os.environ.get("LOCALGATE_DEFAULT_MODEL")
    model_candidate = model or env_model or user_config.default_model
    resolved_model = settings.resolve_model(model_candidate)

    search_fn = None
    if settings.search_provider:
        try:
            search_fn = make_search_fn(
                settings.search_provider,
                api_key=settings.search_api_key,
                base_url=settings.search_base_url,
            )
        except ValueError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from exc

    # --- remote vs local mode ---
    gateway_creds = _read_gateway_creds()
    use_remote = remote if remote is not None else (gateway_creds is not None)
    effective_no_memory = no_memory

    if use_remote:
        if gateway_creds is None:
            typer.secho(
                "No gateway credentials found. Run `localgate login --url <url> --api-key <key>` first.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)
        gw_url, gw_key = gateway_creds
        backend = get_backend("openai_compat", gw_url, api_key=gw_key, timeout=settings.backend_timeout)
        # Remote mode disables local memory: the gateway handles server-side memory,
        # and without a stable X-Session-Id the server-side retrieval is useless anyway.
        effective_no_memory = True
        typer.secho(f"Using remote gateway: {gw_url}", fg=typer.colors.CYAN, err=True)
    else:
        backend = get_backend(
            settings.backend_type,
            settings.backend_url,
            timeout=settings.backend_timeout,
            api_key=settings.backend_api_key,
        )

    engine = make_engine(resolve_database_url(settings))

    async def go() -> bool:
        """Returns whether a write was declined during a single-shot run — the
        REPL path always returns False, since an interactive session has no
        single script-visible outcome to report.
        """
        mcp_registry = McpRegistry()
        try:
            if not no_mcp:
                configured_servers = load_mcp_servers()
                if configured_servers:

                    def _warn_mcp_error(name: str, exc: Exception) -> None:
                        typer.secho(
                            f"MCP server {name!r} failed to connect: {exc}",
                            fg=typer.colors.YELLOW,
                            err=True,
                        )

                    await mcp_registry.connect_all(configured_servers, on_error=_warn_mcp_error)

            memory = None
            if settings.memory_enabled and not effective_no_memory:
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
                    auto_approve=resolved_auto_approve,
                    force=force,
                    auto_commit=auto_commit,
                    memory=memory,
                    theme=resolved_theme,
                    no_color=resolved_no_color,
                    max_turns=resolved_max_turns,
                    user_config=user_config,
                    plan_mode=plan,
                    allow_delegation=allow_delegation,
                    search_fn=search_fn,
                    mcp_registry=mcp_registry,
                )
                return False
            result = await run_single_shot(
                backend,
                resolved_model,
                root,
                task,
                auto_approve=resolved_auto_approve,
                force=force,
                auto_commit=auto_commit,
                memory=memory,
                plan_mode=plan,
                allow_delegation=allow_delegation,
                search_fn=search_fn,
                mcp_registry=mcp_registry,
                theme=resolved_theme,
                no_color=resolved_no_color,
                max_turns=resolved_max_turns,
            )
            return result.declined_a_write
        finally:
            await mcp_registry.aclose()
            await backend.aclose()
            await engine.dispose()

    try:
        declined_a_write = _run(go())
    except AgentTurnLimitExceeded as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=EXIT_MAX_TURNS_EXCEEDED) from exc
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            retry_after = exc.response.headers.get("Retry-After", "unknown")
            typer.secho(
                f"Rate limit exceeded (429). Retry after {retry_after}s. "
                "Raise the key's limit with `localgate keys update <id> --rate-limit N`.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=EXIT_BACKEND_REJECTED) from exc
        typer.secho(
            f"Backend rejected the request — {describe_backend_error(exc)}",
            fg=typer.colors.RED,
            err=True,
        )
        if exc.response.status_code == 400:
            typer.secho(
                "This usually means the current model doesn't support tool calling — "
                "try --model <tool-capable-model>, e.g. --model qwen2.5-coder:7b.",
                fg=typer.colors.YELLOW,
                err=True,
            )
        raise typer.Exit(code=EXIT_BACKEND_REJECTED) from exc
    except httpx.HTTPError as exc:
        typer.secho(f"Couldn't reach the backend: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=EXIT_BACKEND_UNREACHABLE) from exc
    except KeyboardInterrupt:
        typer.secho("\nCancelled.", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=130) from None

    if declined_a_write:
        raise typer.Exit(code=EXIT_WRITE_DECLINED)


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


@keys_app.command("update")
def keys_update(
    key_id: str = typer.Argument(..., help="The key's id (from `keys list`)."),
    rate_limit: int = typer.Option(
        ..., "--rate-limit", help="New requests-per-minute limit for this key."
    ),
) -> None:
    """Update a key's rate limit."""
    settings = _settings()

    async def update(session: Any) -> bool:
        return await APIKeyRepository(session).set_rate_limit(key_id, rate_limit)

    if not _run(_with_session(settings, update)):
        typer.secho(f"No API key with id {key_id!r}.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.secho(f"Updated {key_id}: rate limit is now {rate_limit}/min.", fg=typer.colors.GREEN)


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


@db_app.command("set-url")
def db_set_url(
    database_url: str = typer.Argument(..., help="SQLAlchemy async database URL, e.g. postgresql+asyncpg://user:pass@host/db"),
) -> None:
    """Test a database URL and save it as the active database.

    Reuses the same connection test as the admin dashboard endpoint. After this, all
    localgate commands use this database. Restart the server to apply if it is running.
    """
    url = database_url.strip()
    if "://" not in url:
        typer.secho(
            "That does not look like a database URL. Expected something like "
            "'postgresql+asyncpg://user:pass@host/db' or 'sqlite+aiosqlite:///./local.db'.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    from sqlalchemy import text  # noqa: PLC0415
    from sqlalchemy.exc import SQLAlchemyError  # noqa: PLC0415

    async def test_and_save() -> None:
        engine = make_engine(url)
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        except ModuleNotFoundError as exc:
            typer.secho(f"Missing database driver: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
        except SQLAlchemyError as exc:
            typer.secho(f"Could not connect: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
        finally:
            await engine.dispose()
        save_database_url(url)

    _run(test_and_save())
    typer.secho(
        f"Connected and saved. All localgate commands will now use: {redact_database_url(url)}",
        fg=typer.colors.GREEN,
    )
    typer.echo("Restart the server if it is running: `localgate serve`")


# ----------------------------------------------------------------------------- deploy


@app.command()
def deploy(
    domain: str = typer.Option(..., "--domain", help="Public domain name, e.g. gw.example.com"),
    target: str = typer.Option(
        "compose",
        "--target",
        help="Deployment type: 'compose' (Docker Compose + Caddy) or 'systemd' (systemd unit + system Caddy).",
    ),
    expose_admin: bool = typer.Option(
        False,
        "--expose-admin",
        help="Also expose /admin* and /dashboard* routes (add an IP allowlist in the Caddyfile).",
    ),
    output_dir: Path = typer.Option(
        Path("."),
        "--output-dir",
        "-o",
        help="Directory to write generated files into.",
    ),
) -> None:
    """Generate deployment files (Caddyfile + compose or systemd) for self-hosting with automatic HTTPS.

    Caddy handles ACME certificate issuance automatically. Only /v1/* and /health* are
    exposed by default; pass --expose-admin to also expose /admin* and /dashboard*.

    After generation, follow the printed instructions to deploy and then run:
      localgate login --url https://<domain> --api-key <key>
    """
    if target not in ("compose", "systemd"):
        typer.secho(f"Unknown target {target!r} — choose 'compose' or 'systemd'.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate a fresh admin key for this deployment.
    deploy_admin_key = secrets.token_urlsafe(32)

    # --- Caddyfile ---
    admin_routes = ""
    if expose_admin:
        admin_routes = """
    # Admin and dashboard — restrict to trusted IPs or remove this block and use SSH tunnelling.
    @admin {
        path /admin* /dashboard* /metrics /docs /redoc
    }
    # Uncomment and fill in your IP:
    # @allowed_admin { remote_ip <your-ip> }
    # handle @admin {
    #     reverse_proxy localhost:8000
    # }
    handle @admin {
        respond "Forbidden" 403
    }
"""
    caddyfile_content = f"""{domain} {{
    # Expose only the client-facing API and health by default.
    @public {{
        path /v1/* /health*
    }}
    handle @public {{
        reverse_proxy localhost:8000
    }}
{admin_routes}
    # Block everything else.
    handle {{
        respond "Not found" 404
    }}
}}
"""

    # --- .env file for deployment ---
    env_content = (
        f"LOCALGATE_ADMIN_KEY={deploy_admin_key}\n"
        "LOCALGATE_ENVIRONMENT=production\n"
        "LOCALGATE_HOST=127.0.0.1\n"
        f"LOCALGATE_DATABASE_URL=sqlite+aiosqlite:////data/localgate.db\n"
    )

    if target == "compose":
        compose_content = f"""services:
  localgate:
    image: ghcr.io/anjallll/localgate:latest
    restart: unless-stopped
    env_file: localgate.env
    volumes:
      - localgate_data:/data
    ports:
      - "127.0.0.1:8000:8000"

  caddy:
    image: caddy:2-alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy_data:/data
      - caddy_config:/config

volumes:
  localgate_data:
  caddy_data:
  caddy_config:
"""
        (output_dir / "docker-compose.yml").write_text(compose_content)
        typer.echo(f"  wrote  {output_dir / 'docker-compose.yml'}")

        paths.write_secret_text(output_dir / "localgate.env", env_content)
        typer.echo(f"  wrote  {output_dir / 'localgate.env'}  (0600 — contains admin key)")

    else:  # systemd
        unit_content = f"""[Unit]
Description=localgate API gateway
After=network.target

[Service]
Type=simple
User=localgate
EnvironmentFile=/etc/localgate/localgate.env
ExecStart=/usr/local/bin/localgate serve
Restart=on-failure
RuntimeDirectory=localgate
StateDirectory=localgate

[Install]
WantedBy=multi-user.target
"""
        (output_dir / "localgate.service").write_text(unit_content)
        typer.echo(f"  wrote  {output_dir / 'localgate.service'}")

        env_path = output_dir / "localgate.env"
        paths.write_secret_text(env_path, env_content)
        typer.echo(f"  wrote  {env_path}  (0600 — contains admin key)")

    (output_dir / "Caddyfile").write_text(caddyfile_content)
    typer.echo(f"  wrote  {output_dir / 'Caddyfile'}")

    typer.secho("\nAdmin key (save this — it is not stored anywhere else):", bold=True)
    typer.secho(f"  {deploy_admin_key}", fg=typer.colors.GREEN, bold=True)

    typer.echo(f"\nDomain:  https://{domain}")
    if target == "compose":
        typer.echo("Next steps:")
        typer.echo("  1. Copy files to your server and run: docker compose up -d")
        typer.echo("  2. Mint an API key: localgate keys create --name me")
        typer.echo("  3. On your local machine: localgate login --url https://" + domain + " --api-key <key>")
    else:
        typer.echo("Next steps:")
        typer.echo("  1. Install files: sudo cp localgate.service /etc/systemd/system/")
        typer.echo("  2. sudo mkdir -p /etc/localgate && sudo cp localgate.env /etc/localgate/ && sudo chmod 0600 /etc/localgate/localgate.env")
        typer.echo("  3. sudo systemctl enable --now localgate && sudo caddy reload --config Caddyfile")
        typer.echo("  4. Mint a key and login as above.")


if __name__ == "__main__":
    app()
