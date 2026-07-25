"""Where localgate keeps per-user config and data — the *only* module in this
package allowed to compute a home-relative path.

That rule is enforced by a test (``tests/unit/test_paths.py``) that greps the
source tree for ``Path.home()``/``expanduser`` outside this file. It exists
because config paths resolved ad-hoc in several modules is how a test suite ends
up writing to the developer's real home directory — which has happened here
before.

Resolution order, for both directories:

1. ``LOCALGATE_CONFIG_DIR`` / ``LOCALGATE_DATA_DIR`` — an explicit override.
   This is what the test suite sets, and it's the documented escape hatch for
   anyone packaging localgate into an unusual environment.
2. ``XDG_CONFIG_HOME`` / ``XDG_DATA_HOME`` — the freedesktop convention.
3. ``~/.config/localgate`` / ``~/.local/share/localgate``.

Step 3 uses the XDG-style paths on *every* platform, including macOS, rather
than ``~/Library/Application Support``. That matches what `gh`, `aws`, and `uv`
do, and it keeps the ``~/.config/localgate`` locations that already shipped
(the coding agent's ``config.toml`` and ``mcp_servers.json``) working unchanged.

**Reading a path never creates it.** ``config_dir()``/``data_dir()`` are pure;
only ``ensure_config_dir()``/``ensure_data_dir()`` touch the filesystem. Without
that split, merely importing ``localgate.config`` would create directories on
the machine of anyone who so much as ran ``localgate --help``.
"""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "localgate"

ENV_CONFIG_DIR = "LOCALGATE_CONFIG_DIR"
ENV_DATA_DIR = "LOCALGATE_DATA_DIR"

#: Mode for directories we create: owner-only, since they hold credentials.
DIR_MODE = 0o700
#: Mode for files holding secrets (admin key, database URLs with passwords).
SECRET_FILE_MODE = 0o600


def _home() -> Path:
    """The single funnel to the real home directory.

    Everything that could fall back to ``$HOME`` goes through here so the test
    suite can monkeypatch this one function to raise — turning "a test wrote to
    the developer's real config" from a silent corruption into a loud failure.
    """
    return Path.home()


def _resolve(env_override: str, xdg_var: str, *fallback: str) -> Path:
    override = os.environ.get(env_override)
    if override:
        # `expanduser` is a no-op unless the value starts with `~`, so this does
        # not reach the real home for the absolute paths tests supply.
        return Path(override).expanduser()
    xdg = os.environ.get(xdg_var)
    if xdg:
        return Path(xdg).expanduser() / APP_NAME
    return _home().joinpath(*fallback, APP_NAME)


def config_dir() -> Path:
    """Where user config lives. Pure — does not create anything."""
    return _resolve(ENV_CONFIG_DIR, "XDG_CONFIG_HOME", ".config")


def data_dir() -> Path:
    """Where user data (the SQLite database) lives. Pure — creates nothing."""
    return _resolve(ENV_DATA_DIR, "XDG_DATA_HOME", ".local", "share")


def ensure_config_dir() -> Path:
    return _ensure(config_dir())


def ensure_data_dir() -> Path:
    return _ensure(data_dir())


def _ensure(path: Path) -> Path:
    """Create ``path`` owner-only.

    ``mkdir(mode=...)`` rather than mkdir-then-chmod: the latter leaves the
    directory world-readable for the moment in between, which matters when the
    very next thing written into it is a database password.
    """
    path.mkdir(mode=DIR_MODE, parents=True, exist_ok=True)
    return path


def write_secret_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` readable only by its owner.

    Opens with the mode set at creation time, so — unlike ``write_text`` followed
    by ``chmod`` — the file is never briefly world-readable. The trailing
    ``chmod`` only matters when the file already existed with looser permissions,
    since the mode passed to ``os.open`` is ignored for an existing file.
    """
    _ensure(path.parent)
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, SECRET_FILE_MODE)
    try:
        handle = os.fdopen(fd, "w", encoding="utf-8")
    except BaseException:
        os.close(fd)  # fdopen never took ownership of the descriptor
        raise
    with handle:  # from here the file object owns fd and closes it
        handle.write(content)
    os.chmod(path, SECRET_FILE_MODE)


def user_env_file() -> Path:
    """The per-user ``LOCALGATE_*`` env file, read by ``config.load_settings()``."""
    return config_dir() / "localgate.env"


def describe_mode(path: Path) -> str:
    """``'0600'``-style permissions for ``localgate doctor``, or ``'missing'``."""
    try:
        return format(path.stat().st_mode & 0o777, "04o")
    except OSError:
        return "missing"
