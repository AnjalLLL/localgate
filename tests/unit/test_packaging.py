"""The built wheel must contain the non-Python files the package needs at runtime.

Both omissions fail silently until a real user hits them, which is exactly the
kind of bug a test suite that only ever runs from a source checkout will never
catch:

* no ``alembic.ini`` / ``versions/`` → ``Config(...)`` raises and ``localgate
  serve`` dies during startup migrations;
* no ``dashboard/static/index.html`` → ``mount_dashboard()`` returns ``False``
  and ``/dashboard`` 404s with nothing logged anywhere.

This builds a real wheel, so it is slower than the rest of the unit suite. It is
marked ``slow`` and skipped when the build backend isn't available.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Paths inside the wheel, i.e. relative to the installed `localgate/` package.
REQUIRED_IN_WHEEL = (
    "localgate/py.typed",
    "localgate/db/migrations/alembic.ini",
    "localgate/db/migrations/script.py.mako",
    "localgate/dashboard/static/index.html",
)

pytestmark = pytest.mark.slow


#: `uv build` first — it's what CI and the release workflow use — then the stdlib-ish
#: `python -m build` for anyone without uv on PATH.
_BUILD_COMMANDS = (
    ["uv", "build", "--wheel", "--out-dir", "{out}", str(REPO_ROOT)],
    [sys.executable, "-m", "build", "--wheel", "--outdir", "{out}", str(REPO_ROOT)],
)


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("wheel")
    failures = []
    for template in _BUILD_COMMANDS:
        command = [part.format(out=out) for part in template]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            failures.append(f"{command[0]}: {exc}")
            continue
        if result.returncode == 0:
            wheels = list(out.glob("*.whl"))
            assert wheels, "build reported success but produced no wheel"
            return wheels[0]
        failures.append(f"{command[0]} exited {result.returncode}:\n{result.stderr}")
    pytest.skip("no working wheel builder:\n" + "\n".join(failures))


def test_wheel_contains_runtime_data_files(built_wheel):
    names = set(zipfile.ZipFile(built_wheel).namelist())
    missing = [path for path in REQUIRED_IN_WHEEL if path not in names]
    assert not missing, (
        f"missing from {built_wheel.name}: {missing}. "
        "Check [tool.hatch.build.targets.wheel].artifacts in pyproject.toml."
    )


def test_wheel_contains_at_least_one_migration(built_wheel):
    names = zipfile.ZipFile(built_wheel).namelist()
    versions = [
        n for n in names if n.startswith("localgate/db/migrations/versions/") and n.endswith(".py")
    ]
    assert versions, "no Alembic migration scripts in the wheel — `serve` would fail at startup"
