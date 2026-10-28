"""Sandboxed command execution for the agent's `run_command` tool.

The rule from CODING_AGENT_PLAN.md stands: **never a bare subprocess.** The tool
is only offered when one of these is available, picked in this order:

* **macOS — ``sandbox-exec``** with a Seatbelt profile: writes allowed only in the
  project and temp directories, all network access denied.
* **Linux — bubblewrap (``bwrap``)**: the filesystem mounted read-only except the
  project, a private ``/tmp``, and every namespace unshared (so no network).
* **Any OS, including Windows — Docker or Podman**: the project bind-mounted at
  ``/workspace`` in a throwaway container with ``--network none`` and memory, CPU
  and process limits.

Reads outside the project are not blocked by the first two (toolchains need
them); the environment is scrubbed so the gateway's own secrets are not passed
in, and output only ever goes back to the local model.

Every backend runs an argv list — the command string is interpreted by a shell
*inside* the sandbox, never by a shell on the host.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_OUTPUT_CHARS = 30_000

#: Environment variables passed through to sandboxed commands. Everything else —
#: API keys, LOCALGATE_ADMIN_KEY, cloud credentials — stays out.
_ENV_ALLOWLIST = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "TERM",
    "TMPDIR",
    "USER",
    # Windows: the Docker/Podman CLI needs these to start at all.
    "SYSTEMROOT",
    "USERPROFILE",
    "TEMP",
    "TMP",
    "COMSPEC",
    "PATHEXT",
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMDATA",
    "DOCKER_HOST",
    "DOCKER_CONFIG",
    "CONTAINER_HOST",
)

#: Commands auto mode may run without asking. Matched on the command's leading
#: words; anything else still asks, and plan mode never runs commands.
DEFAULT_AUTO_ALLOW = (
    "pytest",
    "python -m pytest",
    "uv run pytest",
    "ruff",
    "uv run ruff",
    "mypy",
    "uv run mypy",
    "npm test",
    "npm run test",
    "npm run lint",
    "npx tsc",
    "cargo test",
    "cargo check",
    "go test",
    "go vet",
    "make test",
    "git status",
    "git diff",
    "git log",
    "ls",
    "cat",
)

RUN_COMMAND_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "run_command",
        "description": (
            "Run a shell command in the project directory inside a sandbox with no network "
            "access, e.g. to run tests, a linter or a type checker after editing. Returns "
            "the exit code and combined output."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to run."},
                "timeout": {
                    "type": "integer",
                    "description": "Seconds before the command is stopped. Defaults to 120.",
                },
            },
            "required": ["command"],
        },
    },
}

RUN_COMMAND_GUIDANCE = (
    " After changing code, use run_command to run the project's tests or linter and fix "
    "what fails. The sandbox has no network access, so do not try to install packages."
)


def _which(name: str) -> str | None:
    return shutil.which(name)


def _probe(argv: list[str], timeout: float = 10.0) -> bool:
    try:
        completed = subprocess.run(  # noqa: S603 — fixed argv, probe only
            argv, capture_output=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _seatbelt_profile(root: Path) -> str:
    temp_dirs = {Path("/private/tmp"), Path("/private/var/folders")}
    with contextlib.suppress(OSError):
        temp_dirs.add(Path(os.environ.get("TMPDIR", "/private/tmp")).resolve())
    writable = [root.resolve(), *sorted(temp_dirs)]
    subpaths = " ".join(f'(subpath "{_seatbelt_escape(str(p))}")' for p in writable)
    return (
        "(version 1)\n"
        "(allow default)\n"
        "(deny network*)\n"
        "(deny file-write*)\n"
        f"(allow file-write* {subpaths} "
        '(literal "/dev/null") (literal "/dev/stdout") (literal "/dev/stderr") '
        '(regex #"^/dev/tty"))\n'
    )


def _seatbelt_escape(path: str) -> str:
    return path.replace("\\", "\\\\").replace('"', '\\"')


@dataclass
class Sandbox:
    """A detected sandbox backend that can run commands confined to ``root``."""

    kind: str  # "seatbelt" | "bubblewrap" | "docker" | "podman"
    executable: str
    image: str = "python:3.12-slim"

    def argv(self, root: Path, command: str, name: str) -> list[str]:
        root = root.resolve()
        if self.kind == "seatbelt":
            return [self.executable, "-p", _seatbelt_profile(root), "/bin/sh", "-c", command]
        if self.kind == "bubblewrap":
            return [
                self.executable,
                "--ro-bind", "/", "/",
                "--dev", "/dev",
                "--proc", "/proc",
                "--tmpfs", "/tmp",
                "--bind", str(root), str(root),
                "--unshare-all",
                "--die-with-parent",
                "--new-session",
                "--chdir", str(root),
                "/bin/sh", "-c", command,
            ]  # fmt: skip
        # docker / podman
        argv = [
            self.executable,
            "run",
            "--rm",
            "--name", name,
            "--network", "none",
            "--memory", "2g",
            "--cpus", "2",
            "--pids-limit", "256",
            "--mount", f"type=bind,source={root},target=/workspace",
            "--workdir", "/workspace",
        ]  # fmt: skip
        if sys.platform != "win32" and self.kind == "docker":
            # Files the command creates belong to the user, not root (POSIX hosts).
            argv += ["--user", f"{os.getuid()}:{os.getgid()}"]
        return [*argv, self.image, "sh", "-c", command]

    def describe(self) -> str:
        labels = {
            "seatbelt": "macOS sandbox-exec (no network, writes limited to the project)",
            "bubblewrap": "bubblewrap (no network, read-only filesystem except the project)",
            "docker": f"Docker container {self.image} (no network)",
            "podman": f"Podman container {self.image} (no network)",
        }
        return labels.get(self.kind, self.kind)


def detect_sandbox(preference: str = "auto", image: str = "python:3.12-slim") -> Sandbox | None:
    """The strongest sandbox available on this machine, or ``None``.

    ``preference`` is ``auto``, ``off``, or one backend name to require.
    """
    preference = (preference or "auto").lower()
    if preference == "off":
        return None
    candidates: list[str]
    if preference == "auto":
        candidates = []
        if sys.platform == "darwin":
            candidates.append("seatbelt")
        elif sys.platform.startswith("linux"):
            candidates.append("bubblewrap")
        candidates += ["docker", "podman"]
    else:
        candidates = [preference]

    for kind in candidates:
        sandbox = _try(kind, image)
        if sandbox is not None:
            return sandbox
    return None


def _try(kind: str, image: str) -> Sandbox | None:
    if kind == "seatbelt":
        exe = "/usr/bin/sandbox-exec" if Path("/usr/bin/sandbox-exec").exists() else None
        if exe and _probe([exe, "-p", "(version 1)(allow default)", "/usr/bin/true"]):
            return Sandbox("seatbelt", exe)
    elif kind == "bubblewrap":
        exe = _which("bwrap")
        # User namespaces can be disabled (some containers, hardened kernels).
        if exe and _probe([exe, "--ro-bind", "/", "/", "--unshare-all", "/bin/true"]):
            return Sandbox("bubblewrap", exe)
    elif kind in {"docker", "podman"}:
        exe = _which(kind)
        if exe and _probe([exe, "info"], timeout=8.0):
            return Sandbox(kind, exe, image)
    return None


def _scrubbed_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k.upper() in _ENV_ALLOWLIST}
    env.setdefault("PATH", os.defpath)
    env["TERM"] = "dumb"
    env["NO_COLOR"] = "1"
    return env


@dataclass(frozen=True)
class CommandResult:
    exit_code: int | None
    output: str
    timed_out: bool = False

    def render(self) -> str:
        status = "timed out (stopped)" if self.timed_out else f"exit code {self.exit_code}"
        return f"{status}\n{self.output}".rstrip()


async def run_sandboxed(
    sandbox: Sandbox, root: Path, command: str, *, timeout: float = 120.0
) -> CommandResult:
    """Run ``command`` in ``sandbox`` with ``root`` as the working directory."""
    name = f"localgate-run-{uuid.uuid4().hex[:10]}"
    argv = sandbox.argv(root, command, name)
    kwargs: dict[str, Any] = {}
    if sys.platform != "win32":
        kwargs["start_new_session"] = True  # so a timeout can kill the whole group
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(root),
        env=_scrubbed_env(),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        **kwargs,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        await _stop(proc, sandbox, name)
        return CommandResult(None, "", timed_out=True)
    except asyncio.CancelledError:  # the user interrupted the turn (Esc)
        await _stop(proc, sandbox, name)
        raise
    text = stdout.decode("utf-8", errors="replace")
    if len(text) > MAX_OUTPUT_CHARS:
        # Keep the end: test failures and tracebacks are at the bottom.
        text = (
            f"[... {len(text) - MAX_OUTPUT_CHARS:,} chars omitted ...]\n" + text[-MAX_OUTPUT_CHARS:]
        )
    return CommandResult(proc.returncode, text)


async def _stop(proc: asyncio.subprocess.Process, sandbox: Sandbox, name: str) -> None:
    if sandbox.kind in {"docker", "podman"}:
        # Killing the CLI client would leave the container running.
        with contextlib.suppress(OSError):
            killer = await asyncio.create_subprocess_exec(
                sandbox.executable,
                "kill",
                name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()
    if sys.platform != "win32":
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, 9)
    with contextlib.suppress(ProcessLookupError):
        proc.kill()
    with contextlib.suppress(Exception):
        await proc.wait()


def is_auto_allowed(command: str, allow: tuple[str, ...] = DEFAULT_AUTO_ALLOW) -> bool:
    """Whether auto mode may run ``command`` without asking. Anything chaining or
    redirecting (``;``, ``&&``, ``|``, ``>``, backticks, ``$(``) always asks."""
    if any(token in command for token in (";", "&", "|", ">", "<", "`", "$(", "\n")):
        return False
    words = command.split()
    return any(words[: len(prefix.split())] == prefix.split() for prefix in allow)
