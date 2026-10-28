"""`localgate service`: the service definitions it writes, per OS. Nothing here
installs anything — rendering is separate from installing on purpose."""

import plistlib
import shlex

import pytest
from typer.testing import CliRunner

from localgate import paths, service
from localgate.cli import app


@pytest.fixture(autouse=True)
def _service_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "launch_agents_dir", lambda: tmp_path / "LaunchAgents")
    monkeypatch.setattr(paths, "systemd_user_dir", lambda: tmp_path / "systemd" / "user")
    monkeypatch.setattr(
        service, "serve_command", lambda args: ["/opt/bin/localgate", "serve", *args]
    )


def test_macos_agent_starts_at_login_and_restarts():
    plan = service.plan(["--share", "--port", "8780"], "darwin")
    plist = plistlib.loads(plan.content.encode())
    assert plist["Label"] == service.LABEL
    assert plist["ProgramArguments"] == ["/opt/bin/localgate", "serve", "--share", "--port", "8780"]
    assert plist["RunAtLoad"] and plist["KeepAlive"]
    assert plist["WorkingDirectory"] == str(paths.data_dir())  # never a stray ./.env
    assert plan.path.name == "dev.localgate.serve.plist"
    assert plan.install[0][:2] == ["launchctl", "bootstrap"]


def test_linux_user_unit_restarts_on_failure():
    plan = service.plan(["--share"], "linux")
    assert "ExecStart=" + shlex.join(["/opt/bin/localgate", "serve", "--share"]) in plan.content
    assert "Restart=on-failure" in plan.content
    assert "WantedBy=default.target" in plan.content
    assert ["systemctl", "--user", "enable", "--now", "localgate.service"] in plan.install


def test_windows_task_runs_at_logon_with_a_log():
    plan = service.plan([], "win32")
    create = plan.install[0]
    assert create[:2] == ["schtasks", "/Create"] and "ONLOGON" in create
    assert "serve.log" in plan.content


def test_unknown_platforms_are_refused():
    with pytest.raises(ValueError):
        service.plan([], "plan9")


def test_dry_run_writes_nothing(tmp_path):
    result = CliRunner().invoke(app, ["service", "install", "--dry-run", "--share"])
    assert result.exit_code == 0, result.output
    assert "serve" in result.stdout and "--share" in result.stdout
    assert not (tmp_path / "LaunchAgents").exists()
    assert not (tmp_path / "systemd").exists()
