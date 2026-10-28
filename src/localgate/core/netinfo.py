"""This machine's network addresses, for `serve --share` and the Share page.

Finds the addresses other devices can use to reach the gateway: the LAN
address (the interface the default route uses — found with a UDP "connect",
which sends no packet) and a Tailscale address when Tailscale is running. No
third-party dependency and no network traffic.
"""

from __future__ import annotations

import contextlib
import ipaddress
import shutil
import socket
import subprocess

_TAILSCALE_NET = ipaddress.ip_network("100.64.0.0/10")


def is_loopback_host(host: str) -> bool:
    """Whether binding ``host`` keeps the gateway on this machine only."""
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False  # a hostname that resolves elsewhere: treat as exposed


def lan_address() -> str | None:
    """The address this machine uses on its local network, if any."""
    with contextlib.suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        # Connecting a UDP socket only picks a route; nothing is sent.
        probe.connect(("192.0.2.1", 9))  # TEST-NET-1, never routed anywhere real
        address = probe.getsockname()[0]
        if address and not ipaddress.ip_address(address).is_loopback:
            return str(address)
    return None


def tailscale_address() -> str | None:
    """This machine's Tailscale IPv4 address, when Tailscale is installed and up."""
    binary = shutil.which("tailscale")
    if binary is None:
        return None
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        out = subprocess.run(  # noqa: S603 — fixed argv, resolved binary
            [binary, "ip", "-4"], capture_output=True, text=True, timeout=2, check=False
        ).stdout
        for line in out.split():
            with contextlib.suppress(ValueError):
                if ipaddress.ip_address(line) in _TAILSCALE_NET:
                    return line
    return None


def share_urls(port: int, public_url: str | None = None) -> list[dict[str, str]]:
    """Base URLs other devices can use, best first, each with a label."""
    urls: list[dict[str, str]] = []
    if public_url:
        urls.append({"label": "Public", "url": public_url.rstrip("/")})
    tailscale = tailscale_address()
    if tailscale:
        urls.append({"label": "Tailscale", "url": f"http://{tailscale}:{port}"})
    lan = lan_address()
    if lan:
        urls.append({"label": "Local network", "url": f"http://{lan}:{port}"})
    return urls
