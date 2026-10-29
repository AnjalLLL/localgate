"""Find localgate gateways on the local network (mDNS / Bonjour), optional.

`localgate serve --share` advertises the gateway as `_localgate._tcp`; on
another machine `localgate discover` lists what's out there and
`localgate login --url <name>` connects by name instead of by IP. Needs the
`localgate[discovery]` extra (zeroconf); without it everything else still works.
"""

from __future__ import annotations

import asyncio
import logging
import socket
from dataclasses import dataclass
from typing import Any

from localgate import __version__

SERVICE_TYPE = "_localgate._tcp.local."


class DiscoveryUnavailable(RuntimeError):
    """zeroconf isn't installed."""


def _zeroconf() -> Any:
    # zeroconf logs a traceback for every interface it can't probe ("No route to
    # host" on a VPN or inactive adapter). Harmless — and not for the user's terminal.
    logging.getLogger("zeroconf").setLevel(logging.CRITICAL)
    try:
        from zeroconf import asyncio as zc_asyncio  # noqa: PLC0415
    except ImportError as exc:
        raise DiscoveryUnavailable(
            "Network discovery needs an extra package: pip install 'localgate[discovery]'"
        ) from exc
    return zc_asyncio


@dataclass(frozen=True)
class Gateway:
    name: str
    url: str
    version: str


class Advertisement:
    """This gateway, announced on the local network until `close`."""

    def __init__(self, port: int, address: str, name: str | None = None) -> None:
        self.port = port
        self.address = address
        self.name = name or socket.gethostname().split(".")[0]
        self._zc: Any = None
        self._info: Any = None

    async def start(self) -> None:
        zc_asyncio = _zeroconf()
        from zeroconf import ServiceInfo  # noqa: PLC0415

        self._info = ServiceInfo(
            SERVICE_TYPE,
            f"{self.name}.{SERVICE_TYPE}",
            addresses=[socket.inet_aton(self.address)],
            port=self.port,
            properties={"version": __version__, "path": "/v1"},
            server=f"{self.name}.local.",
        )
        self._zc = zc_asyncio.AsyncZeroconf()
        await self._zc.async_register_service(self._info, allow_name_change=True)

    async def close(self) -> None:
        if self._zc is not None:
            await self._zc.async_unregister_service(self._info)
            await self._zc.async_close()
            self._zc = None


async def discover(timeout: float = 3.0) -> list[Gateway]:
    """Gateways answering on the local network within ``timeout`` seconds."""
    zc_asyncio = _zeroconf()
    from zeroconf import ServiceStateChange  # noqa: PLC0415

    found: dict[str, Gateway] = {}
    pending: set[asyncio.Task[None]] = set()
    zc = zc_asyncio.AsyncZeroconf()

    async def resolve(name: str) -> None:
        info = zc_asyncio.AsyncServiceInfo(SERVICE_TYPE, name)
        if not await info.async_request(zc.zeroconf, 1500):
            return
        addresses = info.parsed_addresses()
        if not addresses:
            return
        props = {k.decode(): (v or b"").decode() for k, v in (info.properties or {}).items()}
        label = name.removesuffix("." + SERVICE_TYPE)
        found[label] = Gateway(
            name=label,
            url=f"http://{addresses[0]}:{info.port}",
            version=props.get("version", "?"),
        )

    def on_change(zeroconf: Any, service_type: str, name: str, state_change: Any) -> None:
        if state_change is ServiceStateChange.Added:
            task = asyncio.ensure_future(resolve(name))
            pending.add(task)
            task.add_done_callback(pending.discard)

    browser = zc_asyncio.AsyncServiceBrowser(zc.zeroconf, [SERVICE_TYPE], handlers=[on_change])
    try:
        await asyncio.sleep(timeout)
        if pending:
            await asyncio.wait(pending, timeout=2)
    finally:
        await browser.async_cancel()
        await zc.async_close()
    return sorted(found.values(), key=lambda g: g.name)
