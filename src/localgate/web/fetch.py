"""Fetch one web page and reduce it to readable text.

Used by the agent's `fetch_url` tool and by the chat search pipeline. Guards:

* **http(s) only**, and every hop of a redirect is re-checked.
* **No private destinations.** The host is resolved and refused if any address
  is loopback, private, link-local, multicast or reserved — otherwise a model
  (or a page it read) could point the fetcher at the user's router, a cloud
  metadata endpoint or a local admin service. ``allow_private`` exists for
  people who deliberately fetch from their own network.
* **Bounded:** response size, time and output length are all capped.
* **robots.txt** is respected for the page's user agent.

HTML is converted with the standard library's parser: scripts, styles and page
chrome (nav, header, footer, aside, forms) are dropped; headings, paragraphs,
list items and code blocks are kept with light Markdown. No lxml or browser
dependency, so it installs and behaves the same on every OS.
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx

from localgate.web.search import USER_AGENT

MAX_DOWNLOAD_BYTES = 3 * 1024 * 1024
DEFAULT_MAX_CHARS = 12_000
MAX_REDIRECTS = 5
ROBOTS_AGENT = "localgate"

_SKIP_TAGS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "svg",
        "nav",
        "header",
        "footer",
        "aside",
        "form",
        "iframe",
        "template",
        "button",
        "select",
    }
)
_BLOCK_TAGS = frozenset(
    {"p", "div", "section", "article", "main", "br", "tr", "table", "blockquote", "dd", "dt"}
)
_HEADINGS = {"h1": "# ", "h2": "## ", "h3": "### ", "h4": "#### ", "h5": "##### ", "h6": "###### "}


class FetchError(RuntimeError):
    """The page could not (or may not) be fetched."""


@dataclass(frozen=True)
class FetchedPage:
    url: str
    title: str
    text: str
    truncated: bool = False

    def render(self) -> str:
        header = f"# {self.title}\n{self.url}\n\n" if self.title else f"{self.url}\n\n"
        note = "\n\n[truncated]" if self.truncated else ""
        return header + self.text + note


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False
        self._in_pre = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
        elif tag in _HEADINGS:
            self.parts.append("\n\n" + _HEADINGS[tag])
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag == "pre":
            self._in_pre += 1
            self.parts.append("\n```\n")
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
        elif tag == "pre":
            self._in_pre = max(0, self._in_pre - 1)
            self.parts.append("\n```\n")
        elif tag in _HEADINGS or tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            return
        if self._skip_depth:
            return
        self.parts.append(data if self._in_pre else re.sub(r"\s+", " ", data))


def html_to_text(page: str) -> tuple[str, str]:
    """(title, readable text) for an HTML document."""
    parser = _TextExtractor()
    try:
        parser.feed(page)
        parser.close()
    except Exception:  # noqa: BLE001 — malformed HTML still yields what parsed
        pass
    text = "".join(parser.parts)
    lines = [line.rstrip() for line in text.splitlines()]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return " ".join(parser.title.split()), text


def _is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


async def check_destination(url: str, *, allow_private: bool = False) -> None:
    """Raise :class:`FetchError` unless ``url`` is http(s) to a public host."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise FetchError(f"only http and https URLs can be fetched, not {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise FetchError(f"no host in URL {url!r}")
    if allow_private:
        return
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise FetchError(f"could not resolve {host}: {exc}") from exc
    for info in infos:
        address = str(info[4][0]).split("%", 1)[0]  # strip an IPv6 zone id
        if not _is_public(address):
            raise FetchError(f"refusing to fetch {host}: it resolves to a non-public address")


async def _robots_allows(client: httpx.AsyncClient, url: str) -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        resp = await client.get(robots_url, timeout=5.0)
    except httpx.HTTPError:
        return True
    if resp.status_code >= 400:
        return True
    parser = RobotFileParser()
    parser.parse(resp.text.splitlines())
    return parser.can_fetch(ROBOTS_AGENT, url)


async def fetch_page(
    url: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    timeout: float = 15.0,
    allow_private: bool = False,
    respect_robots: bool = True,
) -> FetchedPage:
    """Fetch ``url`` and return it as readable text. Raises :class:`FetchError`."""
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,text/plain,application/json;q=0.9"}
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout, connect=5.0), headers=headers, follow_redirects=False
    ) as client:
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            await check_destination(current, allow_private=allow_private)
            if respect_robots and not await _robots_allows(client, current):
                raise FetchError(f"robots.txt disallows fetching {current}")
            try:
                body, response = await _download(client, current)
            except httpx.HTTPError as exc:
                raise FetchError(f"fetch failed for {current}: {exc}") from exc
            if response.is_redirect and response.headers.get("location"):
                current = urljoin(current, response.headers["location"])
                continue
            if response.status_code >= 400:
                raise FetchError(f"{current} answered HTTP {response.status_code}")
            return _to_page(current, response, body, max_chars)
    raise FetchError(f"too many redirects starting from {url}")


async def _download(client: httpx.AsyncClient, url: str) -> tuple[bytes, httpx.Response]:
    async with client.stream("GET", url) as response:
        chunks: list[bytes] = []
        size = 0
        if not response.is_redirect:
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    break
                chunks.append(chunk)
        return b"".join(chunks), response


def _to_page(url: str, response: httpx.Response, body: bytes, max_chars: int) -> FetchedPage:
    content_type = response.headers.get("content-type", "").lower()
    charset = response.charset_encoding or "utf-8"
    try:
        decoded = body.decode(charset, errors="replace")
    except LookupError:
        decoded = body.decode("utf-8", errors="replace")
    if "html" in content_type or decoded.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
        title, text = html_to_text(decoded)
    elif content_type.startswith(("text/", "application/json", "application/xml")):
        title, text = "", decoded
    else:
        raise FetchError(f"{url} is {content_type or 'binary'}, not a text page")
    truncated = len(text) > max_chars
    return FetchedPage(url=url, title=title, text=text[:max_chars], truncated=truncated)


FETCH_URL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": (
            "Fetch a public web page (for example a result from web_search) and return its "
            "readable text. Use it to read documentation rather than guessing from a snippet."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "An http(s) URL."},
                "max_chars": {
                    "type": "integer",
                    "description": f"Most characters to return. Defaults to {DEFAULT_MAX_CHARS}.",
                },
            },
            "required": ["url"],
        },
    },
}
