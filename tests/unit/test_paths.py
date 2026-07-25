"""`localgate.paths` — the single door to per-user config/data locations.

Includes two tests that are really about the *test suite* rather than the code:
`test_no_module_outside_paths_resolves_home` (a lint rule) and
`test_tripwire_fires_on_real_home_access` (proof the conftest tripwire works).
Both exist because a test silently writing to the developer's real config
directory has actually happened in this project.
"""

from __future__ import annotations

import re
import stat
from pathlib import Path

import pytest

from localgate import paths

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "localgate"


# ------------------------------------------------------------------- resolution


def test_explicit_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_CONFIG_DIR, str(tmp_path / "explicit"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert paths.config_dir() == tmp_path / "explicit"


def test_xdg_is_used_when_no_override(monkeypatch, tmp_path):
    monkeypatch.delenv(paths.ENV_CONFIG_DIR, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert paths.config_dir() == tmp_path / "xdg" / "localgate"


def test_data_dir_override_and_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path / "explicit"))
    assert paths.data_dir() == tmp_path / "explicit"

    monkeypatch.delenv(paths.ENV_DATA_DIR, raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdgdata"))
    assert paths.data_dir() == tmp_path / "xdgdata" / "localgate"


def test_empty_env_var_is_treated_as_unset(monkeypatch, tmp_path):
    """Per the XDG spec an empty value means "not set", and `if var:` gives us that."""
    monkeypatch.delenv(paths.ENV_CONFIG_DIR, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", "")
    monkeypatch.setattr(paths, "_home", lambda: tmp_path)
    assert paths.config_dir() == tmp_path / ".config" / "localgate"


def test_home_fallback_layout(monkeypatch, tmp_path):
    monkeypatch.delenv(paths.ENV_CONFIG_DIR, raising=False)
    monkeypatch.delenv(paths.ENV_DATA_DIR, raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(paths, "_home", lambda: tmp_path)
    assert paths.config_dir() == tmp_path / ".config" / "localgate"
    assert paths.data_dir() == tmp_path / ".local" / "share" / "localgate"


def test_user_env_file_sits_in_the_config_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_CONFIG_DIR, str(tmp_path))
    assert paths.user_env_file() == tmp_path / "localgate.env"


# ------------------------------------------------------------------------ purity


def test_reading_a_path_creates_nothing(monkeypatch, tmp_path):
    """Merely importing config must not litter a user's filesystem."""
    target = tmp_path / "not-created"
    monkeypatch.setenv(paths.ENV_CONFIG_DIR, str(target))
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(target))
    paths.config_dir()
    paths.data_dir()
    paths.user_env_file()
    assert not target.exists()


def test_ensure_creates_owner_only_dirs(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_CONFIG_DIR, str(tmp_path / "cfg" / "nested"))
    created = paths.ensure_config_dir()
    assert created.is_dir()
    assert stat.S_IMODE(created.stat().st_mode) == paths.DIR_MODE


def test_ensure_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path / "data"))
    first = paths.ensure_data_dir()
    second = paths.ensure_data_dir()
    assert first == second and second.is_dir()


# ------------------------------------------------------------------ secret files


def test_write_secret_text_is_owner_only(tmp_path):
    target = tmp_path / "sub" / "secret.env"
    paths.write_secret_text(target, "LOCALGATE_ADMIN_KEY=abc\n")
    assert target.read_text() == "LOCALGATE_ADMIN_KEY=abc\n"
    assert stat.S_IMODE(target.stat().st_mode) == paths.SECRET_FILE_MODE


def test_write_secret_text_tightens_a_loose_existing_file(tmp_path):
    """The mode passed to os.open is ignored for an existing file, so the
    trailing chmod is what protects a file that was already world-readable."""
    target = tmp_path / "secret.env"
    target.write_text("old\n")
    target.chmod(0o644)
    paths.write_secret_text(target, "new\n")
    assert target.read_text() == "new\n"
    assert stat.S_IMODE(target.stat().st_mode) == paths.SECRET_FILE_MODE


def test_write_secret_text_truncates(tmp_path):
    target = tmp_path / "secret.env"
    paths.write_secret_text(target, "a-long-original-value\n")
    paths.write_secret_text(target, "short\n")
    assert target.read_text() == "short\n"


def test_describe_mode(tmp_path):
    target = tmp_path / "secret.env"
    assert paths.describe_mode(target) == "missing"
    paths.write_secret_text(target, "x")
    assert paths.describe_mode(target) == "0600"


# ---------------------------------------------------- the wall around the suite


def test_tripwire_fires_on_real_home_access():
    """conftest's autouse fixture replaces `_home` with something that raises.

    If this ever stops failing, the suite has lost its protection against a test
    reading or writing the developer's actual config.
    """
    with pytest.raises(AssertionError, match="real home directory"):
        paths._home()


def test_no_module_outside_paths_resolves_home():
    """A lint rule: `paths.py` is the only place allowed to compute a home path.

    Enforced as a test rather than a review habit, because a single
    `Path.home()` added elsewhere silently reopens the hole this module closed.
    """
    pattern = re.compile(r"Path\.home\(\)|os\.path\.expanduser|\.expanduser\(")
    offenders = []
    for source in SRC_ROOT.rglob("*.py"):
        if source.name == "paths.py":
            continue
        for lineno, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if pattern.search(line):
                offenders.append(f"{source.relative_to(SRC_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, "resolve these through localgate.paths instead:\n" + "\n".join(offenders)
