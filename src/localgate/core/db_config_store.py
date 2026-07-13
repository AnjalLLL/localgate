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
from pathlib import Path
from typing import TypedDict

DEFAULT_CONFIG_PATH = Path("localgate.config.json")


class DatabaseConfig(TypedDict, total=False):
    database_url: str


def load_database_config(path: Path = DEFAULT_CONFIG_PATH) -> DatabaseConfig:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def load_database_url(path: Path = DEFAULT_CONFIG_PATH) -> str | None:
    return load_database_config(path).get("database_url")


def save_database_url(url: str, path: Path = DEFAULT_CONFIG_PATH) -> None:
    config = load_database_config(path)
    config["database_url"] = url
    path.write_text(json.dumps(config, indent=2) + "\n")


def is_database_established(path: Path = DEFAULT_CONFIG_PATH) -> bool:
    return load_database_url(path) is not None
