"""Connecting this machine to a localgate gateway somewhere else.

Shared by `localgate login`, the first-run setup in `localgate code`, and the
REPL's `/connect` and `/disconnect`: where the gateway address and API key are
kept, how an address someone typed becomes a URL, and how a connection is
checked — with an error that says what to do, not just what failed.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from localgate import paths

URL_FILE = "gateway_url"
KEY_FILE = "gateway_key"
DEFAULT_PORT = 8000


class GatewayError(Exception):
    """Connecting failed; the message says why and what to do about it."""


@dataclass(frozen=True)
class Gateway:
    url: str
    api_key: str
    version: str = "unknown"
    key_name: str | None = None


def read_creds() -> tuple[str, str] | None:
    """The saved gateway URL and key, or None when this machine isn't connected."""
    url_file = paths.config_dir() / URL_FILE
    key_file = paths.config_dir() / KEY_FILE
    if not url_file.exists() or not key_file.exists():
        return None
    url = url_file.read_text(encoding="utf-8").strip()
    key = key_file.read_text(encoding="utf-8").strip()
    return (url, key) if url and key else None


def save_creds(url: str, api_key: str) -> None:
    paths.ensure_config_dir()
    paths.write_secret_text(paths.config_dir() / URL_FILE, url + "\n")
    paths.write_secret_text(paths.config_dir() / KEY_FILE, api_key + "\n")


def forget_creds() -> bool:
    """Remove the saved gateway. Returns whether one was saved."""
    removed = False
    for name in (URL_FILE, KEY_FILE):
        path = paths.config_dir() / name
        if path.exists():
            path.unlink()
            removed = True
    return removed


def normalize_url(text: str) -> str:
    """What people type — `192.168.1.20`, `gpu-box:8000`, a full URL — as a base URL.
    No scheme means http; no port means localgate's default 8000."""
    value = text.strip().rstrip("/").removesuffix("/v1")
    if not value:
        raise GatewayError("Enter the gateway's address, e.g. 192.168.1.20 or http://gpu:8000.")
    if "://" not in value:
        value = "http://" + value
    parts = urlsplit(value)
    if not parts.hostname:
        raise GatewayError(f"{text!r} doesn't look like an address.")
    if parts.port is None and parts.scheme == "http" and not parts.path.strip("/"):
        value = f"{parts.scheme}://{parts.hostname}:{DEFAULT_PORT}"
    return value


async def verify(
    url: str,
    api_key: str,
    timeout: float = 10.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Gateway:
    """Check the gateway answers and accepts ``api_key``."""
    api_key = api_key.strip()
    if not api_key:
        raise GatewayError("Enter the API key you were given (it starts with lg_).")
    if api_key.startswith("X-Admin-Key"):
        raise GatewayError("That looks like an admin key. Use the API key you were given.")
    host = urlsplit(url).hostname or url
    async with httpx.AsyncClient(base_url=url, timeout=timeout, transport=transport) as client:
        try:
            live = await client.get("/health/live")
        except httpx.ConnectError as exc:
            raise GatewayError(
                f"Nothing answered at {url}. On the GPU machine, start the gateway with "
                "`localgate serve --share` (plain `localgate serve` only accepts connections "
                f"from that machine itself), and check this device can reach {host}: same "
                "network, or Tailscale / a tunnel for a different one."
            ) from exc
        except httpx.TimeoutException as exc:
            raise GatewayError(
                f"{url} didn't answer in time. Is it on the same network (or Tailscale), and "
                "is a firewall on the GPU machine allowing the port?"
            ) from exc
        except httpx.HTTPError as exc:
            raise GatewayError(f"Couldn't reach {url}: {exc}") from exc
        if live.status_code != 200:
            raise GatewayError(f"{url} answered, but it isn't a localgate gateway.")
        version = str(live.json().get("version", "unknown"))

        resp = await client.get("/v1/usage", headers={"Authorization": f"Bearer {api_key}"})
        if resp.status_code == 401:
            reason = ""
            with contextlib.suppress(ValueError, KeyError, TypeError):
                reason = resp.json()["error"]["message"]
            raise GatewayError(
                (reason or "The gateway didn't accept that API key.")
                + " Ask the gateway's owner for a key (Dashboard → Share)."
            )
        if resp.status_code != 200:
            raise GatewayError(f"The gateway answered {resp.status_code} when checking the key.")
        name = resp.json().get("name")
    return Gateway(url=url, api_key=api_key, version=version, key_name=name)


def match_model(wanted: str, available: list[str]) -> str | None:
    """``wanted`` as the gateway names it: Ollama treats "llama3" and
    "llama3:latest" as the same model, so either spelling matches the other."""
    if wanted in available:
        return wanted
    for name in available:
        if name == f"{wanted}:latest" or f"{name}:latest" == wanted:
            return name
    return None
