"""Run `localgate serve` in the background: started at login, restarted if it
stops — what makes a GPU machine a gateway other devices can rely on.

One small per-user service per OS, no root needed:

* **macOS:** a launchd agent in ~/Library/LaunchAgents.
* **Linux:** a systemd user unit (with `loginctl enable-linger` it also runs
  without anyone logged in).
* **Windows:** a Task Scheduler task that starts at logon.

Rendering the service definition is separate from installing it, so `--dry-run`
and the tests can show exactly what would be written without touching the system.
For a system-wide server unit, `localgate deploy --target systemd` is the tool.
"""

from __future__ import annotations

import os
import plistlib
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from localgate import paths

LABEL = "dev.localgate.serve"
SYSTEMD_UNIT = "localgate.service"
WINDOWS_TASK = "localgate"


def _uid() -> int:
    # A `sys.platform` check (not hasattr) so type-checking on Windows passes.
    if sys.platform == "win32":
        return 0
    return os.getuid()


def serve_command(serve_args: list[str]) -> list[str]:
    """The full command the service runs: this same localgate, wherever it lives."""
    executable = shutil.which("localgate")
    if executable:
        return [str(Path(executable).resolve()), "serve", *serve_args]
    return [sys.executable, "-m", "localgate", "serve", *serve_args]


@dataclass
class ServicePlan:
    platform: str
    #: The file written, if this platform uses one.
    path: Path | None
    content: str
    install: list[list[str]] = field(default_factory=list)
    uninstall: list[list[str]] = field(default_factory=list)
    status: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)


def render_launchd(command: list[str], workdir: Path, logdir: Path) -> str:
    plist = {
        "Label": LABEL,
        "ProgramArguments": command,
        "WorkingDirectory": str(workdir),
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(logdir / "serve.log"),
        "StandardErrorPath": str(logdir / "serve.log"),
        "ProcessType": "Background",
    }
    return plistlib.dumps(plist).decode()


def render_systemd(command: list[str], workdir: Path) -> str:
    return (
        "[Unit]\n"
        "Description=localgate gateway\n"
        "After=network-online.target\n"
        "Wants=network-online.target\n\n"
        "[Service]\n"
        f"ExecStart={shlex.join(command)}\n"
        f"WorkingDirectory={workdir}\n"
        "Restart=on-failure\n"
        "RestartSec=5\n\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def windows_task_command(command: list[str], logdir: Path) -> str:
    line = subprocess.list2cmdline(command)
    return f'cmd /c "{line} >> "{logdir / "serve.log"}" 2>&1"'


def plan(serve_args: list[str], platform: str | None = None) -> ServicePlan:
    """What installing the service on ``platform`` (default: this one) means."""
    platform = platform or sys.platform
    command = serve_command(serve_args)
    # The data dir, never the current directory: a stray ./.env there would
    # silently reconfigure the gateway.
    workdir = paths.data_dir()
    logdir = paths.log_dir()
    if platform == "darwin":
        target = paths.launch_agents_dir() / f"{LABEL}.plist"
        domain = f"gui/{_uid()}"
        return ServicePlan(
            platform="macOS launchd agent",
            path=target,
            content=render_launchd(command, workdir, logdir),
            install=[["launchctl", "bootstrap", domain, str(target)]],
            uninstall=[["launchctl", "bootout", f"{domain}/{LABEL}"]],
            status=["launchctl", "print", f"{domain}/{LABEL}"],
            logs=["tail", "-n", "100", str(logdir / "serve.log")],
        )
    if platform.startswith("linux"):
        target = paths.systemd_user_dir() / SYSTEMD_UNIT
        return ServicePlan(
            platform="systemd user service",
            path=target,
            content=render_systemd(command, workdir),
            install=[
                ["systemctl", "--user", "daemon-reload"],
                ["systemctl", "--user", "enable", "--now", SYSTEMD_UNIT],
            ],
            uninstall=[
                ["systemctl", "--user", "disable", "--now", SYSTEMD_UNIT],
                ["systemctl", "--user", "daemon-reload"],
            ],
            status=["systemctl", "--user", "status", "--no-pager", SYSTEMD_UNIT],
            logs=["journalctl", "--user", "-u", SYSTEMD_UNIT, "-n", "100", "--no-pager"],
        )
    if platform == "win32":
        task = windows_task_command(command, logdir)
        return ServicePlan(
            platform="Windows scheduled task (at logon)",
            path=None,
            content=task,
            install=[
                ["schtasks", "/Create", "/TN", WINDOWS_TASK, "/SC", "ONLOGON", "/TR", task, "/F"],
                ["schtasks", "/Run", "/TN", WINDOWS_TASK],
            ],
            uninstall=[
                ["schtasks", "/End", "/TN", WINDOWS_TASK],
                ["schtasks", "/Delete", "/TN", WINDOWS_TASK, "/F"],
            ],
            status=["schtasks", "/Query", "/TN", WINDOWS_TASK, "/V", "/FO", "LIST"],
            logs=[
                "powershell",
                "-NoProfile",
                "Get-Content",
                "-Tail",
                "100",
                str(logdir / "serve.log"),
            ],
        )
    raise ValueError(f"localgate service doesn't support this platform ({platform}).")
