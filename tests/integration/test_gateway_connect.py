"""Connecting a machine to a shared gateway: addresses people type, checking a
connection (with messages that say what to do), and the REPL's /connect."""

import io
import socket

import pytest
from httpx import ASGITransport
from rich.console import Console
from typer.testing import CliRunner

from localgate import gateway_client
from localgate.agent import repl
from localgate.cli import app as cli_app


@pytest.mark.parametrize(
    ("typed", "url"),
    [
        ("192.168.1.127", "http://192.168.1.127:8000"),
        ("192.168.1.127:9000", "http://192.168.1.127:9000"),
        ("http://192.168.1.127:8000/v1", "http://192.168.1.127:8000"),
        ("https://gw.example.com", "https://gw.example.com"),
        (" gpu-box.local ", "http://gpu-box.local:8000"),
    ],
)
def test_addresses_people_type_become_urls(typed, url):
    assert gateway_client.normalize_url(typed) == url


async def test_a_good_key_connects_and_names_itself(app, client, api_key):
    _, raw = api_key
    gateway = await gateway_client.verify("http://gw", raw, transport=ASGITransport(app=app))
    assert gateway.key_name == "test-key"
    assert gateway.version


async def test_a_wrong_key_says_who_to_ask(app, client):
    with pytest.raises(gateway_client.GatewayError, match="Dashboard → Share"):
        await gateway_client.verify(
            "http://gw", "lg_not-a-real-key", transport=ASGITransport(app=app)
        )


async def test_nothing_listening_points_at_serve_share():
    with socket.socket() as probe:  # a port nothing is listening on
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with pytest.raises(gateway_client.GatewayError, match="serve --share"):
        await gateway_client.verify(f"http://127.0.0.1:{port}", "lg_x", timeout=3)


def test_login_and_logout_from_the_cli(monkeypatch):
    async def fake_verify(url, api_key, timeout=10.0, transport=None):
        return gateway_client.Gateway(url=url, api_key=api_key, version="1.0.0", key_name="laptop")

    monkeypatch.setattr(gateway_client, "verify", fake_verify)
    runner = CliRunner()
    result = runner.invoke(cli_app, ["login", "--url", "192.168.1.127", "--api-key", "lg_abc"])
    assert result.exit_code == 0, result.output
    assert "Connected to http://192.168.1.127:8000 as laptop" in result.stdout
    assert gateway_client.read_creds() == ("http://192.168.1.127:8000", "lg_abc")
    assert runner.invoke(cli_app, ["logout"]).exit_code == 0
    assert gateway_client.read_creds() is None


async def test_repl_connect_and_disconnect_switch_the_backend(tmp_path, monkeypatch):
    async def fake_verify(url, api_key, timeout=10.0, transport=None):
        return gateway_client.Gateway(url=url, api_key=api_key, version="1.0.0", key_name="laptop")

    monkeypatch.setattr(gateway_client, "verify", fake_verify)
    out = Console(file=io.StringIO(), force_terminal=False, width=120)
    from localgate.agent.loop import AgentSession
    from localgate.backends.fake import FakeBackend

    session = AgentSession(FakeBackend(""), "fake-model", tmp_path)
    answers = iter(["lg_abc"])
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: next(answers))

    await repl._connect_command(out, session, "192.168.1.127", None)
    assert session.backend.name == "localgate"
    assert session.backend.base_url == "http://192.168.1.127:8000"
    assert gateway_client.read_creds() == ("http://192.168.1.127:8000", "lg_abc")
    assert "Connected to http://192.168.1.127:8000 as laptop" in out.file.getvalue()

    await repl._disconnect_command(out, session, None)
    assert session.backend.name == "ollama"
    assert gateway_client.read_creds() is None


def test_first_run_of_code_offers_a_shared_gateway(tmp_path, monkeypatch):
    """No gateway saved and no model server here: `localgate code` asks where
    models should run, connects to the gateway, and starts the session on it."""
    from localgate import cli
    from localgate.agent import repl as repl_module

    async def fake_verify(url, api_key, timeout=10.0, transport=None):
        return gateway_client.Gateway(url=url, api_key=api_key, version="1.0.0", key_name="laptop")

    started: dict = {}

    async def fake_run_repl(backend, model, root, **kwargs):
        started["backend"], started["model"] = backend, model

    async def remote_model(backend, wanted):
        return wanted

    monkeypatch.setattr(gateway_client, "verify", fake_verify)
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    monkeypatch.setattr(cli, "_local_backend_up", lambda settings: False)
    monkeypatch.setattr(cli, "_nearby_gateways", lambda: [])
    monkeypatch.setattr(cli, "_remote_model", remote_model)

    async def supports_tools(self, model):
        return True

    from localgate.backends.localgate_remote import LocalgateBackend

    monkeypatch.setattr(LocalgateBackend, "check_tool_support", supports_tools)
    monkeypatch.setattr(repl_module, "run_repl", fake_run_repl)
    monkeypatch.setenv("LOCALGATE_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/c.db")

    result = CliRunner().invoke(
        cli_app,
        ["code", "--dir", str(tmp_path), "--no-mcp", "--no-memory"],
        input="2\n192.168.1.127\nlg_abc\n",
    )
    assert result.exit_code == 0, result.output
    assert "Where should the models run?" in result.stdout
    assert started["backend"].name == "localgate"
    assert started["backend"].base_url == "http://192.168.1.127:8000"
    assert gateway_client.read_creds() == ("http://192.168.1.127:8000", "lg_abc")


def test_model_names_match_with_or_without_latest():
    available = ["llama3:latest", "qwen2.5-coder:7b"]
    assert gateway_client.match_model("llama3", available) == "llama3:latest"
    assert gateway_client.match_model("qwen2.5-coder:7b", available) == "qwen2.5-coder:7b"
    assert gateway_client.match_model("llama3:latest", ["llama3"]) == "llama3"
    assert gateway_client.match_model("phi4", available) is None
