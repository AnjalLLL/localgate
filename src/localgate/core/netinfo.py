"""This machine's network addresses, for `localgate serve` and the Share page.

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
from typing import Any

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


def client_address(request: Any) -> str | None:
    """Who is really calling. Behind `serve --public` every request reaches the
    gateway from the local tunnel process, so the address that matters is the
    one Cloudflare's edge sets in `CF-Connecting-IP` (a client can't forge it;
    Cloudflare overwrites it). Everywhere else: the direct peer."""
    peer = request.client.host if request.client else None
    if via_tunnel(request):
        return str(request.headers["cf-connecting-ip"]).strip()
    return peer


_PRIVATE_NETS = tuple(
    ipaddress.ip_network(n)
    for n in (
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "100.64.0.0/10",
        "fc00::/7",
        "fe80::/10",
    )
)


def via_tunnel(request: Any) -> bool:
    """Whether this request came in through the `serve --public` tunnel."""
    peer = request.client.host if request.client else None
    if not peer or not is_loopback_host(peer) or not request.headers.get("cf-connecting-ip"):
        return False
    state = request.app.state
    manager = getattr(state, "tunnel", None)
    active = bool(manager is not None and manager.active)
    return active or bool(getattr(getattr(state, "settings", None), "behind_tunnel", False))


def is_internet_request(request: Any) -> bool:
    """From the internet — through the tunnel, or from a public address — rather
    than the owner's own network (LAN, Tailscale, this machine)."""
    if via_tunnel(request):
        return True
    address = client_address(request)
    if not address:
        return False
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False  # test clients, unix sockets: local by definition
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not (ip.is_loopback or any(ip in net for net in _PRIVATE_NETS))
