"""The built-in internet address (`serve --public`): getting a verified
cloudflared, and trusting Cloudflare's client address only through the tunnel."""

import hashlib
import io
import tarfile
from types import SimpleNamespace

import httpx
import pytest

from localgate import paths
from localgate.core import tunnel
from localgate.core.netinfo import client_address


def test_each_platform_gets_its_release_asset():
    assert tunnel.asset_name("Darwin", "arm64") == "cloudflared-darwin-arm64.tgz"
    assert tunnel.asset_name("Linux", "x86_64") == "cloudflared-linux-amd64"
    assert tunnel.asset_name("Windows", "AMD64") == "cloudflared-windows-amd64.exe"
    with pytest.raises(tunnel.TunnelError):
        tunnel.asset_name("Linux", "mips")


def _release(payload: bytes, digest: str):
    name = tunnel.asset_name()
    if name.endswith(".tgz"):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            info = tarfile.TarInfo("cloudflared")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        blob = buffer.getvalue()
    else:
        blob = payload
    digest = digest or hashlib.sha256(blob).hexdigest()

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.github.com":
            return httpx.Response(
                200,
                json={
                    "assets": [
                        {
                            "name": name,
                            "browser_download_url": "https://dl/x",
                            "digest": f"sha256:{digest}",
                        }
                    ]
                },
            )
        return httpx.Response(200, content=blob)

    return httpx.Client(transport=httpx.MockTransport(handle))


def test_a_verified_download_is_installed_once(monkeypatch):
    monkeypatch.setattr(tunnel.shutil, "which", lambda name: None)
    binary = tunnel.ensure_cloudflared(_release(b"#!/bin/sh\necho cloudflared\n", ""))
    assert binary.parent == paths.data_dir() / "bin"
    assert binary.read_bytes().startswith(b"#!/bin/sh")
    assert tunnel.ensure_cloudflared() == binary  # reused, not downloaded again


def test_a_download_that_fails_its_checksum_is_refused(monkeypatch):
    monkeypatch.setattr(tunnel.shutil, "which", lambda name: None)
    with pytest.raises(tunnel.TunnelError, match="checksum"):
        tunnel.ensure_cloudflared(_release(b"tampered", "0" * 64))
    assert not (paths.data_dir() / "bin").exists() or not any((paths.data_dir() / "bin").iterdir())


def _request(peer: str, headers: dict, behind_tunnel: bool):
    settings = SimpleNamespace(behind_tunnel=behind_tunnel)
    return SimpleNamespace(
        client=SimpleNamespace(host=peer),
        headers=headers,
        app=SimpleNamespace(state=SimpleNamespace(settings=settings)),
    )


def test_cloudflare_client_address_is_used_only_through_the_tunnel():
    via_tunnel = _request("127.0.0.1", {"cf-connecting-ip": "203.0.113.7"}, behind_tunnel=True)
    assert client_address(via_tunnel) == "203.0.113.7"  # an internet visitor, not "localhost"
    no_tunnel = _request("127.0.0.1", {"cf-connecting-ip": "203.0.113.7"}, behind_tunnel=False)
    assert client_address(no_tunnel) == "127.0.0.1"
    spoofed = _request("192.168.1.9", {"cf-connecting-ip": "127.0.0.1"}, behind_tunnel=True)
    assert client_address(spoofed) == "192.168.1.9"  # only the local tunnel process may say so
