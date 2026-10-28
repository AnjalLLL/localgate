"""LAN discovery: the CLI's use of it (always), and real mDNS (opt-in, since
CI runners often block multicast: LOCALGATE_TEST_MDNS=1)."""

import asyncio
import os

import pytest
import typer
from typer.testing import CliRunner

from localgate.cli import app
from localgate.core import discovery
from localgate.core.discovery import Gateway

runner = CliRunner()


@pytest.fixture
def two_gateways(monkeypatch):
    async def fake_discover(timeout=3.0):
        return [
            Gateway("big-gpu", "http://192.168.1.20:8000", "1.0.0"),
            Gateway("office", "http://192.168.1.30:8000", "1.0.0"),
        ]

    monkeypatch.setattr(discovery, "discover", fake_discover)


def test_discover_lists_gateways(two_gateways):
    result = runner.invoke(app, ["discover"])
    assert result.exit_code == 0
    assert "big-gpu" in result.stdout and "http://192.168.1.30:8000" in result.stdout


def test_login_accepts_a_discovered_name(two_gateways, monkeypatch):
    from localgate import cli

    assert cli._resolve_gateway_url("big-gpu") == "http://192.168.1.20:8000"
    assert cli._resolve_gateway_url("http://10.0.0.5:8000/") == "http://10.0.0.5:8000"


def test_an_unknown_name_is_explained(two_gateways):
    from localgate import cli

    with pytest.raises(typer.Exit):
        cli._resolve_gateway_url("nope")


def test_missing_zeroconf_gives_an_install_hint(monkeypatch):
    async def unavailable(timeout=3.0):
        raise discovery.DiscoveryUnavailable("pip install 'localgate[discovery]'")

    monkeypatch.setattr(discovery, "discover", unavailable)
    result = runner.invoke(app, ["discover"])
    assert result.exit_code == 2
    assert "localgate[discovery]" in result.stderr


@pytest.mark.skipif(os.environ.get("LOCALGATE_TEST_MDNS") != "1", reason="real mDNS is opt-in")
async def test_a_real_advertisement_is_discovered():
    from localgate.core.netinfo import lan_address

    address = lan_address()
    if address is None:
        pytest.skip("no LAN address")
    ad = discovery.Advertisement(8799, address, name="lg-pytest")
    await ad.start()
    try:
        found = await discovery.discover(3.0)
    finally:
        await ad.close()
    assert any(g.name.startswith("lg-pytest") and g.url.endswith(":8799") for g in found)
    await asyncio.sleep(0)
