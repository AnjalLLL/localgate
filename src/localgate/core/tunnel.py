"""`localgate serve --public`: reach the gateway from any network, no setup.

Uses a Cloudflare *quick tunnel*: `cloudflared` opens an outbound connection to
Cloudflare and gets a random `https://….trycloudflare.com` address that forwards
to this machine — no account, no open ports, no router changes, HTTPS included.
The address changes each time the tunnel starts; for a fixed one, use a named
Cloudflare tunnel or Tailscale (docs/sharing.md).

`cloudflared` is used from PATH when installed; otherwise the official release
binary for this platform is downloaded once from Cloudflare's GitHub releases
into localgate's data directory, and only used if its SHA-256 matches the one
the release publishes.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from localgate import paths

RELEASES_API = "https://api.github.com/repos/cloudflare/cloudflared/releases/latest"
_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


class TunnelError(RuntimeError):
    """The tunnel couldn't be set up; the message says why."""


def asset_name(system: str | None = None, machine: str | None = None) -> str:
    """The cloudflared release asset for this OS and CPU."""
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()
    arch = {"x86_64": "amd64", "amd64": "amd64", "arm64": "arm64", "aarch64": "arm64"}.get(machine)
    if arch is None:
        raise TunnelError(f"No cloudflared build for this CPU ({machine}). Install it yourself.")
    if system == "darwin":
        return f"cloudflared-darwin-{arch}.tgz"
    if system == "linux":
        return f"cloudflared-linux-{arch}"
    if system == "windows" and arch == "amd64":
        return "cloudflared-windows-amd64.exe"
    raise TunnelError(f"No cloudflared build for {system}/{machine}. Install it yourself.")


def _binary_path() -> Path:
    name = "cloudflared.exe" if sys.platform == "win32" else "cloudflared"
    return paths.data_dir() / "bin" / name


def _expected_sha256(release: dict, asset: dict) -> str:
    """The checksum Cloudflare publishes for ``asset`` — GitHub's per-asset
    digest, or the "SHA256 Checksums" list in the release notes."""
    digest = str(asset.get("digest") or "")
    if digest.startswith("sha256:"):
        return digest.split(":", 1)[1].lower()
    match = re.search(
        rf"{re.escape(asset['name'])}\s*:\s*([0-9a-fA-F]{{64}})", release.get("body") or ""
    )
    if match:
        return match.group(1).lower()
    raise TunnelError("Cloudflare's release doesn't list a checksum for cloudflared; not using it.")


def ensure_cloudflared(client: httpx.Client | None = None) -> Path:
    """A usable cloudflared: the installed one, a previously downloaded one, or
    a freshly downloaded and verified official release."""
    installed = shutil.which("cloudflared")
    if installed:
        return Path(installed)
    target = _binary_path()
    if target.exists():
        return target
    name = asset_name()
    own_client = client is None
    http = client or httpx.Client(timeout=120.0, follow_redirects=True)
    try:
        release = http.get(RELEASES_API, headers={"Accept": "application/vnd.github+json"}).json()
        asset = next((a for a in release.get("assets", []) if a.get("name") == name), None)
        if asset is None:
            raise TunnelError(f"Cloudflare's latest release has no {name}.")
        expected = _expected_sha256(release, asset)
        data = http.get(asset["browser_download_url"]).content
    except httpx.HTTPError as exc:
        raise TunnelError(f"Couldn't download cloudflared: {exc}") from exc
    finally:
        if own_client:
            http.close()
    if hashlib.sha256(data).hexdigest() != expected:
        raise TunnelError("The downloaded cloudflared didn't match its published checksum.")
    if name.endswith(".tgz"):
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            member = next((m for m in archive.getmembers() if m.name.endswith("cloudflared")), None)
            extracted = archive.extractfile(member) if member is not None else None
            if extracted is None:
                raise TunnelError("The cloudflared archive didn't contain the program.")
            data = extracted.read()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.chmod(tmp.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    os.replace(tmp, target)
    return target


@dataclass
class Tunnel:
    url: str
    process: subprocess.Popen[str]

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=5)
            if self.process.poll() is None:
                self.process.kill()


def start_quick_tunnel(port: int, binary: Path, timeout: float = 45.0) -> Tunnel:
    """Start a quick tunnel to ``port`` and wait for its public address."""
    process = subprocess.Popen(  # noqa: S603 — our own verified binary, fixed arguments
        [str(binary), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    found: list[str] = []
    ready = threading.Event()

    def read() -> None:
        # Keep draining for the tunnel's whole life, or a full pipe would stall it.
        assert process.stdout is not None
        for line in process.stdout:
            if not found:
                match = _URL_RE.search(line)
                if match:
                    found.append(match.group(0))
                    ready.set()
        ready.set()

    threading.Thread(target=read, name="cloudflared-output", daemon=True).start()
    deadline = time.monotonic() + timeout
    while not ready.wait(0.2) and time.monotonic() < deadline:
        pass
    if not found:
        Tunnel("", process).stop()
        raise TunnelError(
            "Cloudflare didn't hand out a tunnel address in time. Is this machine online? "
            "(`localgate serve` still shares on your own network.)"
        )
    return Tunnel(found[0], process)


class TunnelManager:
    """The gateway's internet address, switchable while it runs (dashboard → Share).

    Opening a tunnel can take a while the first time (downloading cloudflared), so
    `start` returns at once and the work runs in the background; `status` reports
    progress for the Share page to poll.
    """

    def __init__(self, port: int) -> None:
        self.port = port
        self.tunnel: Tunnel | None = None
        self.starting = False
        self.error: str | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def active(self) -> bool:
        return self.tunnel is not None and self.tunnel.process.poll() is None

    @property
    def url(self) -> str | None:
        return self.tunnel.url if self.active and self.tunnel is not None else None

    def adopt(self, tunnel: Tunnel) -> None:
        """Take over a tunnel `serve --public` opened before the server started."""
        self.tunnel = tunnel

    def start(self) -> None:
        if self.active or self.starting:
            return
        self.starting, self.error = True, None
        self._task = asyncio.get_running_loop().create_task(self._open())

    async def _open(self) -> None:
        try:
            self.tunnel = await asyncio.to_thread(self._open_blocking)
        except TunnelError as exc:
            self.error = str(exc)
        except Exception as exc:  # noqa: BLE001 — surfaced on the Share page
            self.error = f"Couldn't open the internet address: {exc}"
        finally:
            self.starting = False

    def _open_blocking(self) -> Tunnel:
        return start_quick_tunnel(self.port, ensure_cloudflared())

    def stop(self) -> None:
        if self.tunnel is not None:
            self.tunnel.stop()
        self.tunnel = None
        self.error = None

    def status(self) -> dict[str, object]:
        return {
            "enabled": self.active,
            "starting": self.starting,
            "url": self.url,
            "error": self.error,
        }
