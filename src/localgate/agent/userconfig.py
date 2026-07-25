"""Persisted preferences for `localgate code`: `~/.config/localgate/config.toml`.

Deliberately separate from `localgate.config.Settings` — that class is env-var
driven server configuration, validated once at gateway startup. This is a
handful of CLI-only preferences (`theme`, `default_model`, `auto_approve`,
`max_turns`) a human sets interactively via `/theme` and `/config` and expects
to persist between invocations, the way git/gh/npm config files work.

Precedence when the CLI resolves an effective value: flags > env vars >
this file > built-in defaults — see `cli.py`'s `code` command, which layers
this on top of `Settings` rather than replacing it.

The file only ever holds flat scalars (str/bool/int), so rather than pull in a
TOML-writer dependency, this reads/writes the small subset of TOML that's
actually needed: `key = value` lines, `#` comments, blank lines. It stays
readable by any real TOML parser, just doesn't need to *be* one.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Literal

from localgate import paths

Theme = Literal["dark", "light", "none"]


def config_path() -> Path:
    """Where these preferences live.

    A function, not a module constant: a path computed at import time is frozen
    before anything can redirect it, which both defeats test isolation and
    ignores ``LOCALGATE_CONFIG_DIR``. See ``localgate.paths``, the single place
    allowed to resolve a home-relative path.
    """
    return paths.config_dir() / "config.toml"


@dataclass(frozen=True)
class UserConfig:
    theme: Theme = "dark"
    default_model: str | None = None
    auto_approve: bool = False
    max_turns: int = 20

    @classmethod
    def load(cls, path: Path | None = None) -> UserConfig:
        """`path` defaults to :func:`config_path`, resolved at call time."""
        path = path if path is not None else config_path()
        if not path.is_file():
            return cls()
        try:
            raw = _parse(path.read_text(encoding="utf-8"))
        except OSError:
            return cls()
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in raw.items() if k in known}
        try:
            return cls(**values)  # type: ignore[arg-type]
        except TypeError:
            # A hand-edited file with a stray key/wrong type shouldn't crash the
            # CLI — fall back to defaults for anything that doesn't fit.
            return cls()

    def save(self, path: Path | None = None) -> None:
        path = path if path is not None else config_path()
        path.parent.mkdir(mode=paths.DIR_MODE, parents=True, exist_ok=True)
        path.write_text(_dump(self), encoding="utf-8")

    def with_override(self, **changes: object) -> UserConfig:
        return replace(self, **changes)  # type: ignore[arg-type]


def _dump(config: UserConfig) -> str:
    lines = ["# localgate code — persisted preferences. Edit by hand or via /config, /theme."]
    for f in fields(config):
        value = getattr(config, f.name)
        lines.append(f"{f.name} = {_toml_literal(value)}")
    return "\n".join(lines) + "\n"


def _toml_literal(value: object) -> str:
    if value is None:
        return '""'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _parse(text: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, raw_value = line.partition("=")
        key = key.strip()
        raw_value = raw_value.strip()
        result[key] = _parse_literal(raw_value)
    return result


def _parse_literal(raw: str) -> object:
    if raw == "true":
        return True
    if raw == "false":
        return False
    if len(raw) >= 2 and raw[0] == raw[-1] == '"':
        inner = raw[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        return inner or None
    try:
        return int(raw)
    except ValueError:
        return raw
