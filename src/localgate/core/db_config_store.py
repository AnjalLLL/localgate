"""Persists the "established" database connection to a small JSON config file.

This is checked on every startup, taking priority over LOCALGATE_DATABASE_URL
from .env: once a database has been set up and verified through the admin UI,
it becomes the source of truth for where ALL data (API keys, usage records,
conversation history, memory chunks — every table in db/models.py) gets
stored, via the single shared engine created in app.py's lifespan.

We only ever write to this file after actually testing the connection
(see api/config.py) — so "the config file has a database_url" is meant to
imply "this database was reachable at the time it was saved," not just
"someone typed a string into a form."
"""

import json
import logging
from pathlib import Path
from typing import TypedDict

from localgate import paths

_logger = logging.getLogger(__name__)

#: The legacy CWD-relative path used before Stage A'.
_LEGACY_CONFIG_PATH = Path("localgate.config.json")


def _default_config_path() -> Path:
    """The per-user database config file, resolved at call time."""
    return paths.config_dir() / "database.json"


# For backwards compat with callers that imported the old constant: resolve lazily
# via a property-like function. Legacy CWD path is checked for migration in
# load_database_config().
DEFAULT_CONFIG_PATH = _LEGACY_CONFIG_PATH  # kept only for callers that haven't migrated yet


class DatabaseConfig(TypedDict, total=False):
    database_url: str


def redact_database_url(url: str) -> str:
    """Hide credentials in a connection string before it is displayed or logged.

    Lives here rather than in ``api/config.py`` because the CLI (``doctor``) and the
    Alembic environment both need it, and neither should import the FastAPI layer
    to get at a pure string function.
    """
    if "@" not in url:
        return url
    scheme_and_creds, rest = url.rsplit("@", 1)
    scheme = scheme_and_creds.split("://", 1)[0]
    return f"{scheme}://***:***@{rest}"


def load_database_config(path: Path | None = None) -> DatabaseConfig:
    resolved = path if path is not None else _default_config_path()
    if not resolved.exists():
        # Fall back to the legacy CWD-relative path once, with a warning.
        if resolved != _LEGACY_CONFIG_PATH and _LEGACY_CONFIG_PATH.exists():
            _logger.warning(
                "Found legacy localgate.config.json in the current directory. "
                "Run `localgate init` to migrate to %s",
                resolved,
            )
            try:
                return json.loads(_LEGACY_CONFIG_PATH.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
        return {}
    try:
        return json.loads(resolved.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def load_database_url(path: Path | None = None) -> str | None:
    return load_database_config(path).get("database_url")


def save_database_url(url: str, path: Path | None = None) -> None:
    resolved = path if path is not None else _default_config_path()
    config = load_database_config(resolved)
    config["database_url"] = url
    paths.write_secret_text(resolved, json.dumps(config, indent=2) + "\n")


def is_database_established(path: Path | None = None) -> bool:
    return load_database_url(path) is not None
