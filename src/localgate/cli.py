"""The ``localgate`` command line.

The CLI talks to the database and the backend directly rather than to a running
server. That is deliberate: ``localgate keys create`` has to work *before* you have
a key, and ``localgate db upgrade`` has to work when the server won't start because
the schema is out of date. A CLI that could only drive a healthy server would be
useless in exactly the situations you reach for it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
import subprocess
import sys
from collections.abc import Awaitable, Callable, Coroutine
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TypeVar

import httpx
import typer
import uvicorn
from sqlalchemy.exc import OperationalError

from localgate import __version__, gateway_client, paths
from localgate.app import resolve_database_url
from localgate.backends import available_backends, get_backend
from localgate.config import INSECURE_ADMIN_KEY, Settings, env_file_chain, load_settings
from localgate.core.db_config_store import (
    _default_config_path,
    redact_database_url,
    save_database_url,
)
from localgate.core.key_policy import QUOTA_PERIODS, as_utc, is_expired
from localgate.core.netinfo import is_loopback_host, share_urls
from localgate.core.passwords import password_problem
from localgate.db.engine import current_revision, init_models, make_engine, make_session_factory
from localgate.db.repositories.keys import APIKeyRepository
from localgate.db.repositories.usage import UsageRepository
from localgate.db.repositories.users import UserRepository, normalize_email

app = typer.Typer(
    help="localgate — a local-first API gateway for open-source LLMs.",
    no_args_is_help=True,
)
keys_app = typer.Typer(help="Create, inspect and revoke API keys.", no_args_is_help=True)
db_app = typer.Typer(help="Initialize and migrate the database.", no_args_is_help=True)
service_app = typer.Typer(
    help="Run the gateway in the background: start at login, restart if it stops.",
    no_args_is_help=True,
)
users_app = typer.Typer(
    help="Dashboard accounts: sign in with email and password.", no_args_is_help=True
)
app.add_typer(keys_app, name="keys")
app.add_typer(db_app, name="db")
app.add_typer(users_app, name="users")
app.add_typer(service_app, name="service")
bench_app = typer.Typer(help="Measure how reliably models drive the agent.", no_args_is_help=True)
app.add_typer(bench_app, name="bench")
voice_app = typer.Typer(
    help="Local voice: Piper text-to-speech and Whisper speech-to-text.", no_args_is_help=True
)
app.add_typer(voice_app, name="voice")

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
    ctx: typer.Context,
    version: bool = typer.Option(  # noqa: ARG001 — read only by _version_callback's side effect
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the installed version and exit.",
    ),
) -> None:
    """localgate — a local-first API gateway for open-source LLMs."""
    # Commands are for people at a terminal: show warnings, not migration chatter.
    # `serve` sets up its own server logging; LOCALGATE_LOG_LEVEL=DEBUG shows all.
    if ctx.invoked_subcommand != "serve" and os.environ.get("LOCALGATE_LOG_LEVEL") != "DEBUG":
        from localgate.core.logging import configure_logging  # noqa: PLC0415

        configure_logging("WARNING")


def _settings() -> Settings:
    """Load settings, turning a config error into a readable message rather than a traceback."""
    try:
        return load_settings()
    except Exception as exc:  # noqa: BLE001 — pydantic raises ValidationError; any of it is fatal
        typer.secho(f"Configuration error:\n{exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


# --------------------------------------------------------------------- admin key

_ADMIN_KEY_VAR = "LOCALGATE_ADMIN_KEY"


def _admin_key_source() -> str | None:
    """Where LOCALGATE_ADMIN_KEY is set, highest precedence first, or ``None``."""
    if os.environ.get(_ADMIN_KEY_VAR):
        return "the LOCALGATE_ADMIN_KEY environment variable"
    # `env_file_chain` lists files lowest precedence first; the last one wins.
    for env_file in reversed(env_file_chain()):
        try:
            lines = env_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        if any(line.strip().startswith(f"{_ADMIN_KEY_VAR}=") for line in lines):
            return str(env_file.resolve())
    return None


def _write_admin_key(admin_key: str) -> Path:
    """Store ``admin_key`` in the per-user env file (owner-only), keeping its other lines."""
    env_file = paths.user_env_file()
    try:
        kept = [
            line
            for line in env_file.read_text(encoding="utf-8").splitlines()
            if not line.strip().startswith(f"{_ADMIN_KEY_VAR}=")
        ]
    except OSError:
        kept = ["# Add other LOCALGATE_* vars here — see localgate --help for options."]
    paths.write_secret_text(env_file, "\n".join([f"{_ADMIN_KEY_VAR}={admin_key}", *kept]) + "\n")
    return env_file


def _ensure_admin_key() -> str | None:
    """First run of a pip-installed gateway: nothing sets an admin key, so the
    public placeholder would guard /admin. Generate a real one instead and save
    it where `load_settings` finds it. Returns the new key, or ``None`` when one
    was already configured (even if it is the placeholder — that was a choice)."""
    if _admin_key_source() is not None:
        return None
    admin_key = secrets.token_urlsafe(32)
    _write_admin_key(admin_key)
    return admin_key


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
    share: bool = typer.Option(
        False,
        "--share",
        help=(
            "Let other devices use this machine's models: listen on the network and "
            "print the addresses to give them. The dashboard stays on trusted networks."
        ),
    ),
) -> None:
    """Start the gateway."""
    new_admin_key = _ensure_admin_key()
    settings = _settings()
    bind_host = "0.0.0.0" if share else (host or settings.host)  # noqa: S104
    bind_port = port or settings.port
    if not is_loopback_host(bind_host) and settings.uses_insecure_admin_key:
        typer.secho(
            f"Refusing to listen on {bind_host} with the placeholder admin key — anyone on "
            "the network could manage this gateway. Remove LOCALGATE_ADMIN_KEY from your "
            ".env to get a generated one, or set a real key.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    # 0.0.0.0 isn't an address a browser can open; the owner uses this machine's.
    local_host = "127.0.0.1" if bind_host in ("0.0.0.0", "::") else bind_host  # noqa: S104
    dashboard = f"http://{local_host}:{bind_port}/dashboard/"
    account = _first_account_setup(settings)
    if account is not None:
        typer.secho(f"  Sign in at {dashboard} as {account}.", fg=typer.colors.GREEN)
        if new_admin_key is not None:
            typer.echo(
                f"  An admin key for scripts was saved to {paths.user_env_file()} "
                "(`localgate admin-key` shows it)."
            )
        typer.echo()
    elif new_admin_key is not None:
        typer.secho("First run: generated an admin key for this gateway.", fg=typer.colors.GREEN)
        typer.echo(f"  admin key   {new_admin_key}")
        typer.echo(f"  saved to    {paths.user_env_file()}")
        typer.echo(f"  sign in at  {dashboard}   (show the key again: localgate admin-key)")
        typer.echo("  Or create an email/password login: localgate users create")
        typer.echo()
    if not is_loopback_host(bind_host):
        _print_share_box(bind_port, settings)
    common: dict = {
        "host": bind_host,
        "port": bind_port,
        "log_config": None,  # localgate configures structlog itself
    }
    if proxy_headers:
        common["proxy_headers"] = True
        if forwarded_allow_ips is not None:
            common["forwarded_allow_ips"] = forwarded_allow_ips

    if reload or workers <= 1:
        # Single-process: pass the app object directly so uvicorn runs in-process
        # rather than forking a supervisor that exits immediately after handing off.
        from localgate.app import create_app  # noqa: PLC0415

        # The app needs the port it's actually on (for the LAN announcement), and
        # --share turns that announcement on.
        app_settings = settings.model_copy(
            update={"port": bind_port, "discovery_enabled": settings.discovery_enabled or share}
        )
        uvicorn.run(create_app(app_settings), reload=reload, **common)
    else:
        # Multi-worker: must use a string import path + factory so each worker
        # forks and calls create_app() independently.
        uvicorn.run("localgate.app:create_app", factory=True, workers=workers, **common)


def _print_share_box(port: int, settings: Settings) -> None:
    """Where other devices reach this gateway, and how to hand them access."""
    urls = share_urls(port, settings.public_url)
    typer.secho("Sharing this machine's models", fg=typer.colors.GREEN, bold=True)
    if not urls:
        typer.echo("  No network address found — is this machine on a network?")
    for entry in urls:
        typer.echo(f"  {entry['label']:<14} {entry['url']}/v1")
    typer.echo(
        f"  Give each device its own key: http://127.0.0.1:{port}/dashboard/#/share "
        "(or `localgate keys create --name <device>`)."
    )
    if not settings.public_url:
        typer.secho(
            "  This is plain HTTP — fine on your own network or Tailscale. For the internet, "
            "put HTTPS in front (docs/sharing.md: Tailscale, Cloudflare Tunnel, Caddy).",
            fg=typer.colors.YELLOW,
        )
    typer.echo()


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
    force: bool = typer.Option(
        False,
        "--force",
        help="Regenerate the admin key even if one exists.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview what would be written without touching the filesystem.",
    ),
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

    # Generate a new admin key only when none is configured yet, or with --force.
    new_admin_key: str | None = None
    if force:
        new_admin_key = secrets.token_urlsafe(32)
        _write_admin_key(new_admin_key)
    else:
        new_admin_key = _ensure_admin_key()

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
    if new_admin_key is not None:
        typer.secho(f"  env file    {env_file}  (admin key written — 0600)", fg=typer.colors.GREEN)
        typer.echo(f"  admin key   {new_admin_key}")
        typer.secho(
            "  Run `localgate serve` and sign in to the dashboard with it "
            "(`localgate admin-key` shows it again).",
            fg=typer.colors.YELLOW,
        )
    else:
        typer.echo(
            f"  admin key   already set in {_admin_key_source()} "
            "(`localgate admin-key` shows it; --force regenerates)"
        )
    typer.echo(f"  database    {redact_database_url(db_url)}  (revision {revision})")


@app.command("admin-key")
def admin_key(
    rotate: bool = typer.Option(
        False, "--rotate", help="Replace it with a new random key (signs out dashboard sessions)."
    ),
) -> None:
    """Show this gateway's admin key, used to sign in to the dashboard and the /admin API.

    It's created on the first `localgate serve` (or `localgate init`) and kept in
    the per-user env file, unless LOCALGATE_ADMIN_KEY is set elsewhere.
    """
    if rotate:
        source = _admin_key_source()
        if source is not None and source != str(paths.user_env_file().resolve()):
            typer.secho(
                f"The admin key comes from {source}; change it there instead.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)
        new_key = secrets.token_urlsafe(32)
        env_file = _write_admin_key(new_key)
        typer.secho(f"New admin key saved to {env_file}", fg=typer.colors.GREEN)
        typer.echo(new_key)
        typer.secho(
            "Restart `localgate serve` to use it. Keys created with reveal enabled can no "
            "longer be revealed (they still work).",
            fg=typer.colors.YELLOW,
        )
        return

    source = _admin_key_source()
    if source is None:
        typer.secho(
            "No admin key yet — `localgate serve` or `localgate init` creates one.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=1)
    settings = _settings()
    typer.echo(settings.admin_key)
    typer.secho(f"from {source}", fg=typer.colors.BRIGHT_BLACK, err=True)
    if settings.uses_insecure_admin_key:
        typer.secho(
            "This is the public placeholder — anyone can use it. Set a real key there, "
            "or remove the line and run `localgate admin-key --rotate`.",
            fg=typer.colors.YELLOW,
            err=True,
        )


# ------------------------------------------------------------------------ users


async def _migrate(settings: Settings) -> None:
    engine = make_engine(resolve_database_url(settings))
    try:
        await init_models(engine)
    finally:
        await engine.dispose()


def _prompt_password() -> str:
    while True:
        password = typer.prompt("  Password", hide_input=True, confirmation_prompt="  Repeat it")
        problem = password_problem(password)
        if problem is None:
            return str(password)
        typer.secho(f"  {problem}", fg=typer.colors.RED)


def _prompt_email(prompt: str = "  Email") -> str:
    while True:
        email = str(typer.prompt(prompt, default="", show_default=False)).strip()
        if not email or ("@" in email and "." in email.rsplit("@", 1)[-1]):
            return email
        typer.secho("  That doesn't look like an email address.", fg=typer.colors.RED)


def _first_account_setup(settings: Settings) -> str | None:
    """First `localgate serve` with no dashboard account: create one right here,
    in the terminal of whoever runs the server — the only person who should be
    able to claim it. Returns the new account's email, or ``None`` when one
    already exists, the setup was skipped, or there is no terminal to ask in."""

    async def existing_accounts() -> int:
        await _migrate(settings)
        return await _with_session(settings, lambda s: UserRepository(s).count())

    try:
        if _run(existing_accounts()) > 0:
            return None
    except Exception:  # noqa: BLE001 — the server reports database problems itself
        return None
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        typer.secho(
            "No dashboard account yet — create one with: localgate users create",
            fg=typer.colors.YELLOW,
        )
        return None
    typer.secho("Create your dashboard login", fg=typer.colors.GREEN, bold=True)
    typer.echo("  Used to sign in at the dashboard. Leave the email blank to skip.")
    email = _prompt_email()
    if not email:
        return None
    password = _prompt_password()
    _run(_with_session(settings, lambda s: UserRepository(s).create(email, password)))
    return normalize_email(email)


def _audit_cli(settings: Settings, action: str, target: str | None, **details: Any) -> None:
    """Record an admin action taken from this terminal (actor "cli")."""
    from localgate.db.repositories.audit import AuditRepository  # noqa: PLC0415

    async def write(session: Any) -> None:
        await AuditRepository(session).record("cli", action, target, details or None)

    with contextlib.suppress(Exception):  # auditing must never break the action
        _run(_with_session(settings, write))


@users_app.command("create")
def users_create(
    email: str | None = typer.Option(None, "--email", help="Sign-in email."),
) -> None:
    """Create a dashboard account. The password is asked for, never passed as a flag."""
    settings = _settings()
    email = email or _prompt_email("  Email")
    if not email:
        raise typer.Exit(code=2)
    password = _prompt_password()

    async def create() -> bool:
        await _migrate(settings)

        async def go(session: Any) -> bool:
            repo = UserRepository(session)
            if await repo.get(email) is not None:
                return False
            await repo.create(email, password)
            return True

        return await _with_session(settings, go)

    if not _run(create()):
        typer.secho(f"An account for {normalize_email(email)} already exists.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    _audit_cli(settings, "user.created", normalize_email(email))
    typer.secho(
        f"Created {normalize_email(email)} — sign in at /dashboard/.", fg=typer.colors.GREEN
    )


@users_app.command("list")
def users_list() -> None:
    """List dashboard accounts."""
    settings = _settings()
    users = _run(_with_session(settings, lambda s: UserRepository(s).list_all()))
    if not users:
        typer.echo("No accounts yet. Create one: localgate users create")
        return
    for user in users:
        last = user.last_login_at.strftime("%Y-%m-%d %H:%M") if user.last_login_at else "never"
        typer.echo(f"  {user.email:<40} last sign-in {last}")


@users_app.command("passwd")
def users_passwd(email: str = typer.Argument(..., help="The account's email.")) -> None:
    """Set a new password for an account."""
    settings = _settings()
    password = _prompt_password()

    async def change(session: Any) -> bool:
        repo = UserRepository(session)
        user = await repo.get(email)
        if user is None:
            return False
        await repo.set_password(user, password)
        return True

    if not _run(_with_session(settings, change)):
        typer.secho(f"No account for {normalize_email(email)}.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    _audit_cli(settings, "user.password_changed", normalize_email(email))
    typer.secho("Password changed.", fg=typer.colors.GREEN)


@users_app.command("delete")
def users_delete(email: str = typer.Argument(..., help="The account's email.")) -> None:
    """Delete an account. Its open dashboard sessions end when they expire (12 h),
    or immediately with `localgate admin-key --rotate`."""
    settings = _settings()

    async def remove(session: Any) -> bool:
        repo = UserRepository(session)
        user = await repo.get(email)
        if user is None:
            return False
        await repo.delete(user)
        return True

    if not _run(_with_session(settings, remove)):
        typer.secho(f"No account for {normalize_email(email)}.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    _audit_cli(settings, "user.deleted", normalize_email(email))
    typer.secho(f"Deleted {normalize_email(email)}.", fg=typer.colors.GREEN)


# ---------------------------------------------------------------------- service


def _service_plan(serve_args: list[str]) -> Any:
    from localgate import service  # noqa: PLC0415

    try:
        return service.plan(serve_args)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


def _run_steps(steps: list[list[str]], *, check: bool = True) -> bool:
    for step in steps:
        try:
            done = subprocess.run(step, check=False, capture_output=True, text=True)  # noqa: S603
        except FileNotFoundError:
            typer.secho(f"  {step[0]} isn't available on this machine.", fg=typer.colors.RED)
            return False
        if done.returncode != 0 and check:
            typer.secho(
                f"  `{' '.join(step)}` failed: {(done.stderr or done.stdout).strip()}",
                fg=typer.colors.RED,
            )
            return False
    return True


@service_app.command("install")
def service_install(
    share: bool = typer.Option(
        False, "--share", help="Serve other devices too (see `serve --share`)."
    ),
    host: str | None = typer.Option(None, "--host", help="Interface to bind."),
    port: int | None = typer.Option(None, "--port", help="Port to bind."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be installed."),
) -> None:
    """Start `localgate serve` at login and keep it running."""
    serve_args = (
        (["--share"] if share else [])
        + (["--host", host] if host else [])
        + (["--port", str(port)] if port else [])
    )
    plan = _service_plan(serve_args)
    if dry_run:
        typer.secho(f"{plan.platform}" + (f" — {plan.path}" if plan.path else ""), bold=True)
        typer.echo(plan.content)
        for step in plan.install:
            typer.echo("$ " + " ".join(step))
        return
    # The service runs without a terminal, so it can't ask for anything: make sure
    # there's an admin key and a migrated database before it starts.
    _ensure_admin_key()
    paths.log_dir().mkdir(parents=True, exist_ok=True)
    if plan.path is not None:
        plan.path.parent.mkdir(parents=True, exist_ok=True)
        _run_steps(plan.uninstall, check=False)  # replace an older install cleanly
        plan.path.write_text(plan.content, encoding="utf-8")
    if not _run_steps(plan.install):
        raise typer.Exit(code=1)
    typer.secho(f"Installed: {plan.platform}.", fg=typer.colors.GREEN)
    typer.echo("  localgate starts now and at every login. `localgate service status` checks it.")
    if sys.platform.startswith("linux"):
        typer.echo("  To keep it running while logged out: loginctl enable-linger $USER")


@service_app.command("uninstall")
def service_uninstall() -> None:
    """Stop the background gateway and remove it from login."""
    plan = _service_plan([])
    _run_steps(plan.uninstall, check=False)
    if plan.path is not None and plan.path.exists():
        plan.path.unlink()
    typer.secho("Removed the localgate service.", fg=typer.colors.GREEN)


@service_app.command("status")
def service_status() -> None:
    """Whether the background gateway is installed and running."""
    plan = _service_plan([])
    if plan.path is not None and not plan.path.exists():
        typer.echo("Not installed. Install it with: localgate service install")
        raise typer.Exit(code=1)
    result = subprocess.run(plan.status, check=False, capture_output=True, text=True)  # noqa: S603
    typer.echo((result.stdout or result.stderr).strip() or "(no output)")
    raise typer.Exit(code=0 if result.returncode == 0 else 1)


@service_app.command("logs")
def service_logs() -> None:
    """The background gateway's recent log output."""
    plan = _service_plan([])
    result = subprocess.run(plan.logs, check=False, capture_output=True, text=True)  # noqa: S603
    typer.echo((result.stdout or result.stderr).strip() or "(no log output yet)")


@app.command()
def audit(
    limit: int = typer.Option(30, "--limit", "-n", help="How many recent actions to show."),
) -> None:
    """Recent admin actions: keys created, changed, revealed or revoked; sign-ins; accounts."""
    from localgate.db.repositories.audit import AuditRepository  # noqa: PLC0415

    settings = _settings()
    events = _run(_with_session(settings, lambda s: AuditRepository(s).recent(limit)))
    if not events:
        typer.echo("No admin actions recorded yet.")
        return
    for event in events:
        when = as_utc(event.at).strftime("%Y-%m-%d %H:%M") if event.at else "?"
        target = f" {event.target}" if event.target else ""
        typer.echo(f"  {when}  {event.actor:<24} {event.action}{target}")


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
    cfg_mode = paths.describe_mode(config_dir) if config_dir.exists() else "missing"
    dat_mode = paths.describe_mode(data_dir) if data_dir.exists() else "missing"
    typer.echo(f"  config dir   {config_dir}  ({cfg_mode})")
    typer.echo(f"  data dir     {data_dir}  ({dat_mode})")

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
        _check(
            "legacy config",
            f"{legacy} exists in CWD — run `localgate init` to migrate",
            warn=True,
        )

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

    typer.secho("\n--- network exposure ---", bold=True)
    if is_loopback_host(settings.host):
        _check("listens on", f"{settings.host} (this machine only; `serve --share` to share)")
    else:
        _check("listens on", f"{settings.host} (reachable from the network)", warn=True)
    everyone = "0.0.0.0/0" in settings.admin_allowed_networks
    _check(
        "dashboard, /admin, /docs",
        "anyone (LOCALGATE_ADMIN_ALLOWED_NETWORKS=*)"
        if everyone
        else "trusted networks only: " + ", ".join(settings.admin_allowed_networks),
        warn=everyone,
    )
    _check(
        "public URL",
        settings.public_url or "not set (devices use this machine's address; plain HTTP)",
        warn=bool(settings.public_url and settings.public_url.startswith("http://")),
    )
    _check(
        "inference queue",
        f"{settings.max_concurrent_requests} at once, wait up to "
        f"{settings.queue_timeout:.0f} s, at most {settings.max_queue} waiting",
    )

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


def _read_gateway_creds() -> tuple[str, str] | None:
    """The saved gateway URL and key, or None when this machine isn't connected."""
    return gateway_client.read_creds()


def _connect_gateway(url: str, api_key: str) -> gateway_client.Gateway:
    """Verify and save a gateway connection, or exit with a message that says why not."""
    try:
        gateway = _run(gateway_client.verify(url, api_key))
    except gateway_client.GatewayError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    gateway_client.save_creds(gateway.url, gateway.api_key)
    if gateway.version.split(".")[0] == "0":
        # 0.x and 1.x speak the same API; a pre-1.0 gateway only lacks the model
        # details and limits a 1.x client can show.
        typer.secho(
            f"Note: the gateway runs localgate {gateway.version}. Upgrading it to 1.x "
            "gives `localgate code --remote` model details and context sizing.",
            fg=typer.colors.YELLOW,
        )
    return gateway


@app.command()
def login(
    url: str = typer.Option(
        ...,
        "--url",
        help="The gateway's URL (http://192.168.1.20:8000), or a name from `localgate discover`.",
    ),
    api_key: str = typer.Option(
        ...,
        "--api-key",
        help="An API key (not the admin key) minted on that gateway.",
    ),
) -> None:
    """Connect this machine to a localgate gateway running somewhere else.

    Only needed for a *remote* gateway: `localgate code` and `localgate serve` on
    this machine need no login. The API key comes from whoever runs the gateway —
    on that machine, `localgate keys create --name <you>` or Dashboard → API keys.

    Verifies the key is valid by hitting /v1/usage (pure DB, no backend needed),
    and checks the gateway version via /health/live.
    After login, `localgate code --remote` will route to this gateway.
    """
    if api_key == INSECURE_ADMIN_KEY:
        typer.secho(
            "This looks like an admin key. Use the API key you were given.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    gateway = _connect_gateway(_resolve_gateway_url(url), api_key)
    who = f" as {gateway.key_name}" if gateway.key_name else ""
    typer.secho(f"Connected to {gateway.url}{who}.", fg=typer.colors.GREEN)
    typer.echo("`localgate code` now uses this gateway; `localgate whoami` shows your limits.")


@app.command()
def discover(
    timeout: float = typer.Option(3.0, "--timeout", help="Seconds to listen for gateways."),
) -> None:
    """List localgate gateways on this network (ones started with `serve --share`)."""
    from localgate.core.discovery import DiscoveryUnavailable, discover  # noqa: PLC0415

    try:
        found = _run(discover(timeout))
    except DiscoveryUnavailable as exc:
        typer.secho(str(exc), fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=2) from exc
    if not found:
        typer.echo("No gateways found. On the GPU machine, run: localgate serve --share")
        raise typer.Exit(code=1)
    for gateway in found:
        typer.echo(f"  {gateway.name:<24} {gateway.url:<28} v{gateway.version}")
    typer.echo(f"\nConnect: localgate login --url {found[0].name} --api-key <key>")


def _resolve_gateway_url(url: str) -> str:
    """A URL as given, or a gateway name found on the local network."""
    if "://" in url or any(ch in url for ch in ".:") or url == "localhost":
        try:
            return gateway_client.normalize_url(url)
        except gateway_client.GatewayError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from exc
    from localgate.core.discovery import DiscoveryUnavailable, discover  # noqa: PLC0415

    try:
        found = {g.name: g.url for g in _run(discover(3.0))}
    except DiscoveryUnavailable as exc:
        typer.secho(f"{url!r} isn't a URL. {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    if url not in found:
        typer.secho(
            f"No gateway called {url!r} on this network. `localgate discover` lists them; "
            "or pass a URL like http://192.168.1.20:8000.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    typer.echo(f"Found {url} at {found[url]}")
    return found[url]


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

    typer.echo(f"Gateway   {base}")
    if data.get("name"):
        typer.echo(f"Key       {data['name']}")
    typer.echo(
        f"Usage     {data.get('request_count', 0):,} requests, "
        f"{data.get('total_tokens', 0):,} tokens"
    )
    limits = data.get("limits") or {}
    models = limits.get("allowed_models")
    typer.echo(f"Models    {', '.join(models) if models else 'any'}")
    quota = limits.get("token_quota")
    if quota:
        used = data.get("tokens_this_period") or 0
        typer.echo(
            f"Quota     {used:,} / {quota:,} tokens this {limits.get('quota_period', 'month')}"
        )
    if limits.get("max_tokens"):
        typer.echo(f"Replies   up to {limits['max_tokens']:,} tokens")
    if limits.get("expires_at"):
        typer.echo(f"Expires   {limits['expires_at'][:10]}")
    if limits.get("store_conversations") is False:
        typer.echo("History   not stored on the gateway")


@app.command()
def logout() -> None:
    """Disconnect from the gateway; `localgate code` goes back to this machine's models."""
    if gateway_client.forget_creds():
        typer.secho(
            "Disconnected. `localgate code` now uses this machine's models.", fg=typer.colors.GREEN
        )
    else:
        typer.echo("Not connected to a gateway.")


# ------------------------------------------------------- first-run model setup


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _local_backend_up(settings: Settings) -> bool:
    """Whether this machine's model server (Ollama by default) answers."""

    async def check() -> bool:
        backend = get_backend(settings.backend_type, settings.backend_url, timeout=3.0)
        try:
            return bool(await asyncio.wait_for(backend.health(), 3.0))
        except Exception:  # noqa: BLE001 — any failure means "not running"
            return False
        finally:
            await backend.aclose()

    return _run(check())


def _choose_where_models_run(settings: Settings) -> tuple[str, str] | None:
    """First start with no model server here: run models locally, or use a
    gateway someone shared? Returns the gateway's (url, key), or None for local."""
    typer.secho("Where should the models run?", bold=True)
    typer.echo(
        f"  1. On this machine — {settings.backend_type} at {settings.backend_url} isn't "
        "answering yet (start it, e.g. `ollama serve`)"
    )
    typer.echo("  2. On a shared gateway — someone gave you an address and an API key")
    choice = typer.prompt("  Choose", default="2" if settings.backend_type == "ollama" else "1")
    if choice.strip() != "2":
        typer.echo()
        return None
    return _prompt_gateway()


def _prompt_gateway() -> tuple[str, str] | None:
    """Ask for a gateway's address and API key, check them, and save them."""
    suggestions = _nearby_gateways()
    if suggestions:
        typer.echo("  Gateways on this network:")
        for number, gateway in enumerate(suggestions, start=1):
            typer.echo(f"    {number}. {gateway.name}  {gateway.url}")
    for _ in range(3):
        answer = typer.prompt(
            "  Gateway address (e.g. 192.168.1.20)" + (" or number" if suggestions else "")
        )
        if answer.strip().isdigit() and 1 <= int(answer) <= len(suggestions):
            answer = suggestions[int(answer) - 1].url
        api_key = typer.prompt("  API key", hide_input=True)
        try:
            url = gateway_client.normalize_url(answer)
            gateway = _run(gateway_client.verify(url, api_key))
        except gateway_client.GatewayError as exc:
            typer.secho(f"  {exc}", fg=typer.colors.RED)
            continue
        gateway_client.save_creds(gateway.url, gateway.api_key)
        who = f" as {gateway.key_name}" if gateway.key_name else ""
        typer.secho(
            f"  Connected to {gateway.url}{who}. Saved for next time.", fg=typer.colors.GREEN
        )
        typer.echo("  (`localgate logout` or /disconnect goes back to this machine's models.)\n")
        return gateway.url, gateway.api_key
    typer.secho("  Not connected — continuing with this machine's models.", fg=typer.colors.YELLOW)
    return None


def _nearby_gateways() -> list[Any]:
    """Gateways announced on this network, when discovery is installed (quick look)."""
    from localgate.core.discovery import DiscoveryUnavailable, discover  # noqa: PLC0415

    try:
        return _run(discover(1.5))
    except (DiscoveryUnavailable, OSError):
        return []


async def _remote_model(backend: Any, wanted: str) -> str:
    """The model to use on a gateway: ``wanted`` if the key may use it, else the
    first available one, preferring models that can call tools."""
    try:
        available = await backend.list_models()
    except httpx.HTTPError:
        return wanted
    if not available:
        return wanted
    matched = gateway_client.match_model(wanted, available)
    if matched is not None:
        return matched
    for model in available:
        if await backend.check_tool_support(model):
            chosen = model
            break
    else:
        chosen = available[0]
    typer.secho(
        f"{wanted} isn't available on the gateway for this key — using {chosen}.",
        fg=typer.colors.YELLOW,
        err=True,
    )
    return str(chosen)


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
        help="One of dark, light, none. Persists to the config file when set.",
    ),
    no_color: bool = typer.Option(
        False, "--no-color", help="Disable colored output. Also respects the NO_COLOR env var."
    ),
    max_turns: int | None = typer.Option(
        None,
        "--max-turns",
        help="Tool-call turns before giving up on a task. Defaults to the config file, else 20.",
    ),
    voice: bool = typer.Option(
        False,
        "--voice",
        help="Read the agent's replies aloud with Piper (local). Toggle with /voice; "
        "dictate with /listen. Needs: pip install 'localgate[voice]'.",
    ),
    no_commands: bool = typer.Option(
        False,
        "--no-commands",
        help="Don't offer the sandboxed run_command tool, even if a sandbox is available.",
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
    from rich import console as rich_console  # noqa: PLC0415

    from localgate.agent.loop import AgentToolUseRequired, AgentTurnLimitExceeded  # noqa: PLC0415
    from localgate.agent.mcp import McpRegistry, load_mcp_servers  # noqa: PLC0415
    from localgate.agent.memory import (  # noqa: PLC0415
        AgentMemory,
        get_or_create_local_agent_key_id,
        project_session_id,
    )
    from localgate.agent.repl import (  # noqa: PLC0415
        describe_backend_error,
        run_repl,
        run_single_shot,
    )
    from localgate.agent.roles import load_roles  # noqa: PLC0415
    from localgate.agent.sandbox import detect_sandbox  # noqa: PLC0415
    from localgate.agent.theme import THEMES  # noqa: PLC0415
    from localgate.agent.userconfig import UserConfig  # noqa: PLC0415
    from localgate.agent.websearch import make_search_fn  # noqa: PLC0415
    from localgate.core.logging import configure_logging  # noqa: PLC0415
    from localgate.voice.terminal import VoiceIO  # noqa: PLC0415
    from localgate.web.search import chain_from_settings  # noqa: PLC0415

    settings = _settings()
    # The REPL is a UI, not a server log: only warnings unless DEBUG was asked for.
    configure_logging("DEBUG" if settings.log_level == "DEBUG" else "WARNING", settings.log_format)
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

    try:
        search_fn: Any = make_search_fn("", chain=chain_from_settings(settings))
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    # --- remote vs local mode ---
    gateway_creds = _read_gateway_creds()
    if gateway_creds is None and remote is not False and task is None and _interactive():
        if remote is True:
            gateway_creds = _prompt_gateway()
        elif not _local_backend_up(settings):
            gateway_creds = _choose_where_models_run(settings)
    use_remote = remote if remote is not None else (gateway_creds is not None)
    effective_no_memory = no_memory

    if use_remote:
        if gateway_creds is None:
            typer.secho(
                "No gateway credentials found."
                " Run `localgate login --url <url> --api-key <key>` first.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)
        gw_url, gw_key = gateway_creds
        backend = get_backend(
            "localgate",
            gw_url,
            api_key=gw_key,
            timeout=settings.backend_timeout,
        )
        # Remote mode disables local memory: the gateway handles server-side memory,
        # and without a stable X-Session-Id the server-side retrieval is useless anyway.
        effective_no_memory = True
        typer.secho(
            f"Using remote gateway: {gw_url}",
            fg=typer.colors.CYAN,
            err=True,
        )
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
        nonlocal resolved_model
        mcp_registry = McpRegistry()
        if use_remote:
            resolved_model = await _remote_model(backend, resolved_model)
        try:
            supports_tools = await backend.check_tool_support(resolved_model)
            native_tools = True
            if supports_tools is False and backend.accepts_options:
                # Ollama can grammar-constrain replies, so a model without native
                # tools still gets a working (if slower) JSON tool protocol.
                native_tools = False
                if task is not None:  # the REPL shows this in its startup banner
                    typer.secho(
                        f"Model {resolved_model!r} does not advertise tool-calling support; "
                        "using JSON tool mode. A tool-capable model (e.g. qwen2.5-coder) "
                        "is faster and more reliable.",
                        fg=typer.colors.YELLOW,
                        err=True,
                    )
            elif supports_tools is False:
                message = (
                    f"Model {resolved_model!r} does not advertise tool-calling support; "
                    "select a tool-capable model with --model."
                )
                if task is not None:
                    raise AgentToolUseRequired(message)
                typer.secho(
                    message + " Use /model inside the REPL to switch models.",
                    fg=typer.colors.YELLOW,
                    err=True,
                )
            sandbox = (
                None
                if no_commands
                else detect_sandbox(settings.agent_sandbox, settings.agent_sandbox_image)
            )
            voice_io = (
                VoiceIO(settings, rich_console.Console(stderr=True), speak=voice)
                if settings.voice_enabled
                else None
            )
            agent_roles = load_roles(paths.config_dir() / "agents", root / ".localgate" / "agents")
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
                    settings=settings,
                    session_id=project_session_id(root) if memory else None,
                    sandbox=sandbox,
                    agent_roles=agent_roles,
                    native_tools=native_tools,
                    voice=voice_io,
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
                settings=settings,
                session_id=project_session_id(root) if memory else None,
                sandbox=sandbox,
                agent_roles=agent_roles,
                native_tools=native_tools,
                voice=voice_io,
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
    except AgentToolUseRequired as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=EXIT_BACKEND_REJECTED) from exc
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


@bench_app.command("agent")
def bench_agent(
    models: list[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to test; repeat for several. Defaults to LOCALGATE_DEFAULT_MODEL.",
    ),
    task_filter: str | None = typer.Option(
        None, "--task", help="Run only the task (or category) with this name."
    ),
    json_out: Path | None = typer.Option(None, "--json", help="Also write results as JSON here."),
) -> None:
    """Run the agent benchmark: small throwaway projects, checked on disk.

    Writes happen only in temporary directories and are auto-approved there.
    Use it before and after changing prompts, tools or models.
    """
    from localgate.agent.bench import TASKS, TaskResult, run_bench  # noqa: PLC0415

    settings = _settings()
    targets = list(models) if models else [settings.resolve_model(None)]
    backend = get_backend(
        settings.backend_type,
        settings.backend_url,
        timeout=settings.backend_timeout,
        api_key=settings.backend_api_key,
    )
    count = sum(1 for t in TASKS if not task_filter or task_filter in (t.name, t.category))
    if count == 0:
        typer.secho(f"No task named {task_filter!r}.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    typer.echo(f"{count} task(s) × {len(targets)} model(s)")

    def show(model: str, result: TaskResult) -> None:
        mark = (
            typer.style("PASS", fg=typer.colors.GREEN)
            if result.passed
            else typer.style("FAIL", fg=typer.colors.RED)
        )
        detail = f"  {result.error[:80]}" if result.error else ""
        typer.echo(
            f"  {mark} {model:<24} {result.task:<20} {result.seconds:>6.1f}s "
            f"tools={result.tool_calls} failed={result.failed_tool_calls}{detail}"
        )

    async def go() -> list[Any]:
        try:
            return await run_bench(
                backend, targets, settings=settings, task_filter=task_filter, on_result=show
            )
        finally:
            await backend.aclose()

    try:
        reports = _run(go())
    except httpx.HTTPError as exc:
        typer.secho(f"Couldn't reach the backend: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=EXIT_BACKEND_UNREACHABLE) from exc

    typer.echo("")
    for report in reports:
        typer.echo(f"{report.model}: {report.passed}/{len(report.results)} ({report.rate:.0%})")
    if json_out is not None:
        json_out.write_text(json.dumps([r.to_dict() for r in reports], indent=2), encoding="utf-8")
        typer.echo(f"wrote {json_out}")


@voice_app.command("status")
def voice_status() -> None:
    """What voice support is installed, and which voices are downloaded."""
    from localgate.voice import INSTALL_HINT, stt  # noqa: PLC0415
    from localgate.voice.tts import engine_name, installed_voices, voices_dir  # noqa: PLC0415

    settings = _settings()
    directory = voices_dir(settings)
    engine = engine_name()
    typer.echo(f"text-to-speech  {engine or 'not installed'} (Piper)")
    typer.echo(f"speech-to-text  {'faster-whisper' if stt.available() else 'not installed'}")
    typer.echo(f"default voice   {settings.voice_tts_voice}")
    typer.echo(f"whisper model   {settings.voice_stt_model} ({settings.voice_stt_device})")
    typer.echo(f"voices folder   {directory}")
    voices = installed_voices(directory)
    typer.echo(f"installed       {', '.join(voices) if voices else '(none yet)'}")
    if engine is None or not stt.available():
        typer.secho(INSTALL_HINT, fg=typer.colors.YELLOW)


@voice_app.command("download")
def voice_download(
    name: str = typer.Argument(
        None, help="Voice such as en_US-lessac-medium. Defaults to LOCALGATE_VOICE_TTS_VOICE."
    ),
) -> None:
    """Download a Piper voice (list them at huggingface.co/rhasspy/piper-voices)."""
    from localgate.voice import VoiceUnavailable  # noqa: PLC0415
    from localgate.voice.tts import ensure_voice, voices_dir  # noqa: PLC0415

    settings = _settings()
    voice = name or settings.voice_tts_voice
    try:
        model = ensure_voice(voice, voices_dir(settings), download=True)
    except VoiceUnavailable as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"ready: {model}")


@voice_app.command("say")
def voice_say(
    text: str = typer.Argument(..., help="What to say."),
    voice: str | None = typer.Option(None, "--voice", "-v", help="Piper voice name."),
    out: Path | None = typer.Option(
        None, "--out", "-o", help="Write a WAV file instead of playing."
    ),
) -> None:
    """Speak text with Piper (or save it as a WAV file)."""
    from localgate.voice import VoiceUnavailable  # noqa: PLC0415
    from localgate.voice.audio import play_wav  # noqa: PLC0415
    from localgate.voice.tts import PiperTTS, voices_dir  # noqa: PLC0415

    settings = _settings()
    engine = PiperTTS(
        voices_dir(settings), settings.voice_tts_voice, download=settings.voice_auto_download
    )
    try:
        audio = engine.synthesize_sync(text, voice)
        if out is not None:
            out.write_bytes(audio)
            typer.echo(f"wrote {out}")
        else:
            play_wav(audio)
    except (VoiceUnavailable, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@voice_app.command("listen")
def voice_listen(
    file: Path | None = typer.Argument(
        None, help="Transcribe this audio file instead of the microphone."
    ),
    language: str | None = typer.Option(None, "--language", "-l", help="e.g. en, ne, hi."),
) -> None:
    """Transcribe speech with Whisper: from the microphone (press Enter to stop) or a file."""
    from localgate.voice import VoiceUnavailable  # noqa: PLC0415
    from localgate.voice.stt import WhisperSTT  # noqa: PLC0415

    settings = _settings()
    engine = WhisperSTT(settings.voice_stt_model, settings.voice_stt_device)
    try:
        if file is not None:
            result = engine.transcribe_sync(file.read_bytes(), language)
        else:
            from localgate.voice.audio import record_until_enter  # noqa: PLC0415

            typer.secho("● listening — press Enter to stop", fg=typer.colors.CYAN, err=True)
            samples = record_until_enter()
            result = engine.transcribe_samples(samples, language)
    except (VoiceUnavailable, ValueError, OSError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(result.text)


# ------------------------------------------------------------------ key limits

_DURATION_UNITS = {"h": 3600, "d": 86400, "w": 7 * 86400}


def _parse_expires(text: str) -> datetime | None:
    """`30d`, `12h`, `2w`, `2026-12-31`, or `never` (no expiry)."""
    value = text.strip().lower()
    if value in ("never", "none", ""):
        return None
    if value[-1:] in _DURATION_UNITS and value[:-1].isdigit():
        return datetime.now(timezone.utc) + timedelta(
            seconds=int(value[:-1]) * _DURATION_UNITS[value[-1]]
        )
    try:
        moment = datetime.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(
            "use a duration like 30d / 12h / 2w, a date like 2026-12-31, or 'never'"
        ) from exc
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _parse_models(text: str) -> list[str] | None:
    """Comma-separated model names; `any` (or empty) means no restriction."""
    names = [name.strip() for name in text.split(",") if name.strip()]
    return None if not names or names == ["any"] else names


def _limit_columns(
    *,
    models: str | None,
    max_tokens: int | None,
    quota: int | None,
    quota_period: str | None,
    expires: str | None,
    max_concurrent: int | None,
    store_conversations: bool | None,
) -> dict[str, Any]:
    """Only the limits that were given. For numbers, 0 removes the limit."""
    columns: dict[str, Any] = {}
    if models is not None:
        columns["allowed_models"] = _parse_models(models)
    for column, value in (
        ("max_tokens", max_tokens),
        ("token_quota", quota),
        ("max_concurrent", max_concurrent),
    ):
        if value is not None:
            columns[column] = value or None
    if quota_period is not None:
        if quota_period not in QUOTA_PERIODS:
            raise typer.BadParameter("--quota-period must be 'day' or 'month'")
        columns["quota_period"] = quota_period
    if expires is not None:
        columns["expires_at"] = _parse_expires(expires)
    if store_conversations is not None:
        columns["store_conversations"] = store_conversations
    return columns


def _describe_limits(key: Any) -> list[str]:
    parts = []
    if key.allowed_models:
        parts.append("models " + ",".join(key.allowed_models))
    if key.max_tokens:
        parts.append(f"max {key.max_tokens} tokens/reply")
    if key.token_quota:
        parts.append(f"{key.token_quota:,} tokens/{key.quota_period or 'month'}")
    if key.expires_at:
        parts.append(f"expires {as_utc(key.expires_at):%Y-%m-%d}")
    if key.max_concurrent:
        parts.append(f"{key.max_concurrent} at once")
    if key.store_conversations is False:
        parts.append("conversations not stored")
    return parts


@keys_app.command("create")
def keys_create(
    name: str = typer.Option(..., "--name", "-n", help="A label, e.g. the app that will use it."),
    rate_limit: int | None = typer.Option(
        None,
        "--rate-limit",
        help="Requests per minute. Defaults to LOCALGATE_DEFAULT_RATE_LIMIT_PER_MIN.",
    ),
    models: str | None = typer.Option(
        None, "--models", help="Only these models (comma-separated). Default: any."
    ),
    max_tokens: int | None = typer.Option(None, "--max-tokens", help="Cap on reply length."),
    quota: int | None = typer.Option(None, "--quota", help="Tokens allowed per quota period."),
    quota_period: str | None = typer.Option(None, "--quota-period", help="'day' or 'month'."),
    expires: str | None = typer.Option(
        None, "--expires", help="When the key stops working: 30d, 12h, 2w or 2026-12-31."
    ),
    max_concurrent: int | None = typer.Option(
        None, "--max-concurrent", help="Requests this key may have running at once."
    ),
    store_conversations: bool = typer.Option(
        True,
        "--store-conversations/--no-store-conversations",
        help="Keep this key's conversations for memory (the owner can read them).",
    ),
) -> None:
    """Create an API key and print it. This is the only time it is ever shown."""
    settings = _settings()
    limit = rate_limit or settings.default_rate_limit_per_min
    policy = _limit_columns(
        models=models,
        max_tokens=max_tokens,
        quota=quota,
        quota_period=quota_period,
        expires=expires,
        max_concurrent=max_concurrent,
        store_conversations=None if store_conversations else False,
    )

    async def create(session: Any) -> tuple[Any, str]:
        vault_key = settings.admin_key if settings.api_key_reveal_enabled else None
        return await APIKeyRepository(session).create(
            name, limit, admin_key=vault_key, policy=policy
        )

    async def migrate_then_create() -> tuple[Any, str]:
        # A first-run `keys create` shouldn't fail just because nobody has run
        # `db upgrade` yet — creating your first key is how you start.
        await _migrate(settings)
        return await _with_session(settings, create)

    key, raw_key = _run(migrate_then_create())

    typer.secho(f"\n  {raw_key}\n", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  id           {key.id}")
    typer.echo(f"  name         {name}")
    typer.echo(f"  rate limit   {limit}/min")
    for part in _describe_limits(key):
        typer.echo(f"  limit        {part}")
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
                "limits": _describe_limits(key),
                "expired": is_expired(key),
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
        if row["revoked"]:
            status = typer.style("revoked", fg=typer.colors.RED)
        elif row["expired"]:
            status = typer.style("expired", fg=typer.colors.YELLOW)
        else:
            status = typer.style("active", fg=typer.colors.GREEN)
        typer.echo(
            f"{row['id']:<38} {row['name'][:19]:<20} {row['prefix']:<13} "
            f"{str(row['rate_limit_per_min']) + '/min':<7} {status}"
        )
        if row["limits"]:
            typer.secho(f"{'':<38} " + " · ".join(row["limits"]), fg=typer.colors.BRIGHT_BLACK)


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
    rate_limit: int | None = typer.Option(
        None, "--rate-limit", help="New requests-per-minute limit for this key."
    ),
    models: str | None = typer.Option(
        None, "--models", help="Only these models (comma-separated); 'any' removes the limit."
    ),
    max_tokens: int | None = typer.Option(None, "--max-tokens", help="Reply cap; 0 removes it."),
    quota: int | None = typer.Option(None, "--quota", help="Tokens per period; 0 removes it."),
    quota_period: str | None = typer.Option(None, "--quota-period", help="'day' or 'month'."),
    expires: str | None = typer.Option(
        None, "--expires", help="30d, 12h, 2w, 2026-12-31, or 'never'."
    ),
    max_concurrent: int | None = typer.Option(
        None, "--max-concurrent", help="Requests at once; 0 removes the limit."
    ),
    store_conversations: bool | None = typer.Option(
        None, "--store-conversations/--no-store-conversations", help="Keep conversations."
    ),
) -> None:
    """Change a key's rate limit or other limits without reissuing it."""
    settings = _settings()
    columns = _limit_columns(
        models=models,
        max_tokens=max_tokens,
        quota=quota,
        quota_period=quota_period,
        expires=expires,
        max_concurrent=max_concurrent,
        store_conversations=store_conversations,
    )
    if rate_limit is not None:
        columns["rate_limit_per_min"] = rate_limit
    if not columns:
        typer.secho("Nothing to change — pass at least one option.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=2)

    async def update(session: Any) -> Any:
        repo = APIKeyRepository(session)
        if not await repo.update_policy(key_id, columns):
            return None
        key = await repo.get(key_id)
        if key is not None:
            await session.refresh(key)
        return key

    key = _run(_with_session(settings, update))
    if key is None:
        typer.secho(f"No API key with id {key_id!r}.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    summary = [f"{key.rate_limit_per_min}/min", *_describe_limits(key)]
    typer.secho(f"Updated {key_id}: " + " · ".join(summary), fg=typer.colors.GREEN)


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
    database_url: str = typer.Argument(
        ..., help="SQLAlchemy async database URL, e.g. postgresql+asyncpg://user:pass@host/db"
    ),
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
    domain: str = typer.Option(
        ...,
        "--domain",
        help="Public domain name, e.g. gw.example.com",
    ),
    target: str = typer.Option(
        "compose",
        "--target",
        help=(
            "Deployment type: 'compose' (Docker Compose + Caddy)"
            " or 'systemd' (systemd unit + system Caddy)."
        ),
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
    """Generate deployment files for self-hosting with automatic HTTPS.

    Caddy handles ACME certificate issuance automatically.
    Only /v1/* and /health* are exposed by default;
    pass --expose-admin to also expose /admin* and /dashboard*.

    After generation, follow the printed instructions to deploy and then run:
      localgate login --url https://<domain> --api-key <key>
    """
    if target not in ("compose", "systemd"):
        typer.secho(
            f"Unknown target {target!r} — choose 'compose' or 'systemd'.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate a fresh admin key for this deployment.
    deploy_admin_key = secrets.token_urlsafe(32)

    # Caddy runs next to the gateway: in its own container for compose (so the
    # gateway is `localgate:8000` on the compose network), on the host for systemd.
    upstream = "localgate:8000" if target == "compose" else "localhost:8000"

    # --- Caddyfile ---
    admin_routes = ""
    if expose_admin:
        admin_routes = f"""
    # Admin and dashboard — restrict to trusted IPs or remove this block and use SSH tunnelling.
    @admin {{
        path /admin* /dashboard* /metrics /docs /redoc
    }}
    # Uncomment and fill in your IP:
    # @allowed_admin {{ remote_ip <your-ip> }}
    # handle @admin {{
    #     reverse_proxy {upstream}
    # }}
    handle @admin {{
        respond "Forbidden" 403
    }}
"""
    caddyfile_content = f"""{domain} {{
    # Expose only the client-facing API and health by default.
    @public {{
        path /v1/* /health*
    }}
    handle @public {{
        reverse_proxy {upstream}
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
        # Inside a container 127.0.0.1 is the container itself, unreachable from Caddy.
        f"LOCALGATE_HOST={'0.0.0.0' if target == 'compose' else '127.0.0.1'}\n"  # noqa: S104
        f"LOCALGATE_DATABASE_URL=sqlite+aiosqlite:////data/localgate.db\n"
    )

    if target == "compose":
        compose_content = """services:
  localgate:
    image: ghcr.io/anjallll/localgate:latest
    restart: unless-stopped
    env_file: localgate.env
    # Behind Caddy every request comes from Caddy's address; trusting its
    # X-Forwarded-For is what lets the gateway see — and restrict — real clients.
    command: ["localgate", "serve", "--proxy-headers", "--forwarded-allow-ips", "*"]
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
        unit_content = """[Unit]
Description=localgate API gateway
After=network.target

[Service]
Type=simple
User=localgate
EnvironmentFile=/etc/localgate/localgate.env
ExecStart=/usr/local/bin/localgate serve --proxy-headers --forwarded-allow-ips 127.0.0.1
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
        typer.echo(
            f"  3. On your local machine: localgate login --url https://{domain} --api-key <key>"
        )
    else:
        typer.echo("Next steps:")
        typer.echo("  1. Install files: sudo cp localgate.service /etc/systemd/system/")
        typer.echo(
            "  2. sudo mkdir -p /etc/localgate"
            " && sudo cp localgate.env /etc/localgate/"
            " && sudo chmod 0600 /etc/localgate/localgate.env"
        )
        typer.echo(
            "  3. sudo systemctl enable --now localgate && sudo caddy reload --config Caddyfile"
        )
        typer.echo("  4. Mint a key and login as above.")


if __name__ == "__main__":
    app()
