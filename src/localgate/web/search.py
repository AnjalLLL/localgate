"""Web search providers, as structured results, plus a fallback chain and cache.

Providers, most local-first first:

- **searxng** — self-hosted metasearch (``docker run searxng/searxng``) with a JSON
  API. Queries go to the user's own server, which fans out to Google/Bing/etc.
  The recommended setup.
- **openserp** — self-hosted, no API key (github.com/karust/openserp).
- **brave** — Brave Search API: official, JSON, has a free tier, needs a key.
- **tavily** — hosted, API key.
- **duckduckgo** — zero-config scrape of the lite HTML page. Always available,
  least robust; the last resort in the default chain.

:class:`SearchChain` tries providers in order and returns the first non-empty
answer, so a SearXNG server that's down degrades to the next provider instead of
failing the tool call. Results are cached in-process for a few minutes: agents
often repeat the same query within a task.
"""

from __future__ import annotations

import html
import re
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx

SNIPPET_MAX_CHARS = 240
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

DEFAULT_SEARXNG_URL = "http://localhost:8888"
DEFAULT_OPENSERP_URL = "http://localhost:7000"

#: Providers that keep the query on the user's own machine or network.
SELF_HOSTED_PROVIDERS = frozenset({"searxng", "openserp"})
SUPPORTED_PROVIDERS = ("searxng", "openserp", "brave", "tavily", "duckduckgo")


class WebSearchError(RuntimeError):
    """The search request itself failed — network error or a non-2xx response."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    provider: str = ""


def _clean(text: Any) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", str(text or ""))).strip()


def truncate_snippet(text: str) -> str:
    text = text.strip()
    if len(text) > SNIPPET_MAX_CHARS:
        return text[: SNIPPET_MAX_CHARS - 3] + "..."
    return text


def format_results(results: list[SearchResult], max_results: int = 5) -> str:
    """Title + short snippet + URL per result, never page content."""
    if not results:
        return "No results."
    return "\n".join(
        f"- {r.title}\n  {truncate_snippet(r.snippet)}\n  {r.url}" for r in results[:max_results]
    )


def _pick(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    return next((str(item[k]) for k in keys if item.get(k)), "")


def _from_dicts(
    items: Any,
    provider: str,
    *,
    title: tuple[str, ...] = ("title",),
    snippet: tuple[str, ...] = ("snippet", "content", "description"),
    url: tuple[str, ...] = ("url", "link"),
    max_results: int,
) -> list[SearchResult]:
    if not isinstance(items, list):
        return []
    out: list[SearchResult] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        result = SearchResult(
            title=_clean(_pick(item, title)),
            url=_pick(item, url).strip(),
            snippet=_clean(_pick(item, snippet)),
            provider=provider,
        )
        if result.title and result.url:
            out.append(result)
        if len(out) >= max_results:
            break
    return out


def _client(timeout: float, **kwargs: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=5.0), **kwargs)


async def searxng_results(
    query: str, base_url: str = DEFAULT_SEARXNG_URL, *, max_results: int = 5, timeout: float = 10.0
) -> list[SearchResult]:
    """SearXNG's JSON API. The instance must allow ``json`` in ``search.formats``
    (settings.yml) — the default Docker image does not, and answers 403."""
    async with _client(timeout) as client:
        try:
            resp = await client.get(
                f"{base_url.rstrip('/')}/search", params={"q": query, "format": "json"}
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            hint = " (enable the json format in SearXNG's settings.yml)" if _is_403(exc) else ""
            raise WebSearchError(f"web search failed (SearXNG at {base_url}): {exc}{hint}") from exc
    body = resp.json()
    return _from_dicts(body.get("results"), "searxng", max_results=max_results)


def _is_403(exc: httpx.HTTPError) -> bool:
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 403


async def openserp_results(
    query: str,
    base_url: str = DEFAULT_OPENSERP_URL,
    *,
    engine: str = "google",
    max_results: int = 5,
    timeout: float = 10.0,
) -> list[SearchResult]:
    async with _client(timeout) as client:
        try:
            resp = await client.get(
                f"{base_url.rstrip('/')}/{engine}/search",
                params={"text": query, "limit": max_results, "format": "json"},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise WebSearchError(
                f"web search failed ({base_url}): {exc} — is OpenSERP running there?"
            ) from exc
    body = resp.json()
    items = body.get("results") if isinstance(body, dict) else None
    return _from_dicts(
        items,
        "openserp",
        snippet=("snippet", "description"),
        max_results=max_results,
    )


async def brave_results(
    query: str, api_key: str, *, max_results: int = 5, timeout: float = 10.0
) -> list[SearchResult]:
    async with _client(timeout) as client:
        try:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": max_results},
                headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise WebSearchError(f"web search failed (Brave): {exc}") from exc
    web = resp.json().get("web") or {}
    return _from_dicts(web.get("results"), "brave", max_results=max_results)


async def tavily_results(
    query: str, api_key: str, *, max_results: int = 5, timeout: float = 10.0
) -> list[SearchResult]:
    async with _client(timeout) as client:
        try:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": api_key, "query": query, "max_results": max_results},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise WebSearchError(f"web search failed: {exc}") from exc
    return _from_dicts(
        resp.json().get("results"), "tavily", snippet=("content",), max_results=max_results
    )


_DDG_LINK_RE = re.compile(
    r'<a[^>]+class="result-link"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL
)
_DDG_LINK_FALLBACK_RE = re.compile(
    r'<a[^>]+rel="nofollow"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL
)
_DDG_SNIPPET_RE = re.compile(r'<td class="result-snippet">(.*?)</td>', re.DOTALL)


def _ddg_target(href: str) -> str:
    """DuckDuckGo sometimes wraps results in a ``/l/?uddg=<target>`` redirect."""
    href = html.unescape(href)
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if parsed.path.startswith("/l/") and "uddg" in parse_qs(parsed.query):
        return unquote(parse_qs(parsed.query)["uddg"][0])
    return href


async def duckduckgo_results(
    query: str, *, max_results: int = 5, timeout: float = 10.0
) -> list[SearchResult]:
    async with _client(timeout, follow_redirects=True) as client:
        try:
            resp = await client.post(
                "https://lite.duckduckgo.com/lite/",
                data={"q": query},
                headers={"User-Agent": USER_AGENT},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise WebSearchError(f"web search failed (DuckDuckGo): {exc}") from exc
    page = resp.text
    links = _DDG_LINK_RE.findall(page) or _DDG_LINK_FALLBACK_RE.findall(page)
    snippets = _DDG_SNIPPET_RE.findall(page)
    out: list[SearchResult] = []
    for i, (href, title) in enumerate(links[:max_results]):
        result = SearchResult(
            title=_clean(title),
            url=_ddg_target(href),
            snippet=_clean(snippets[i]) if i < len(snippets) else "",
            provider="duckduckgo",
        )
        if result.title and result.url:
            out.append(result)
    return out


ProviderFn = Callable[[str, int], Awaitable[list[SearchResult]]]


@dataclass(frozen=True)
class ProviderConfig:
    searxng_url: str | None = None
    openserp_url: str | None = None
    brave_api_key: str | None = None
    tavily_api_key: str | None = None


def build_provider(name: str, config: ProviderConfig) -> ProviderFn:
    """One provider as ``(query, max_results) -> results``. Raises ``ValueError``
    for an unknown provider or one missing a required key."""
    if name == "searxng":
        url = config.searxng_url or DEFAULT_SEARXNG_URL
        return lambda q, n: searxng_results(q, url, max_results=n)
    if name == "openserp":
        url = config.openserp_url or DEFAULT_OPENSERP_URL
        return lambda q, n: openserp_results(q, url, max_results=n)
    if name == "brave":
        if not config.brave_api_key:
            raise ValueError("LOCALGATE_BRAVE_API_KEY is required for the brave provider")
        key = config.brave_api_key
        return lambda q, n: brave_results(q, key, max_results=n)
    if name == "tavily":
        if not config.tavily_api_key:
            raise ValueError("LOCALGATE_SEARCH_API_KEY is required when using tavily")
        key = config.tavily_api_key
        return lambda q, n: tavily_results(q, key, max_results=n)
    if name == "duckduckgo":
        return lambda q, n: duckduckgo_results(q, max_results=n)
    raise ValueError(
        f"unsupported search provider {name!r} — choose from {', '.join(SUPPORTED_PROVIDERS)}"
    )


def parse_provider_list(value: str | None) -> list[str]:
    """``"searxng, duckduckgo"`` -> ``["searxng", "duckduckgo"]``; empty -> DuckDuckGo."""
    names = [part.strip().lower() for part in (value or "").split(",") if part.strip()]
    return names or ["duckduckgo"]


class SearchChain:
    """Providers tried in order; the first one that answers wins.

    A provider that errors or returns nothing falls through to the next. Only if
    every provider fails does the caller see an error, listing each failure.
    """

    def __init__(
        self,
        providers: list[tuple[str, ProviderFn]],
        *,
        cache_ttl: float = 600.0,
        cache_size: int = 256,
    ) -> None:
        if not providers:
            raise ValueError("SearchChain needs at least one provider")
        self.providers = providers
        self.cache_ttl = cache_ttl
        self.cache_size = cache_size
        self._cache: OrderedDict[tuple[str, int], tuple[float, list[SearchResult]]] = OrderedDict()

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.providers]

    @property
    def leaves_machine(self) -> bool:
        """Whether any provider in the chain sends the query to a third party."""
        return any(name not in SELF_HOSTED_PROVIDERS for name in self.names)

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        key = (" ".join(query.lower().split()), max_results)
        cached = self._cache.get(key)
        if cached is not None and time.monotonic() - cached[0] < self.cache_ttl:
            self._cache.move_to_end(key)
            return cached[1]

        errors: list[str] = []
        results: list[SearchResult] = []
        for name, provider in self.providers:
            try:
                results = await provider(query, max_results)
            except WebSearchError as exc:
                errors.append(str(exc))
                continue
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                errors.append(f"{name}: {exc}")
                continue
            if results:
                break
        if not results and errors and len(errors) == len(self.providers):
            raise WebSearchError("; ".join(errors))

        self._cache[key] = (time.monotonic(), results)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return results


def build_chain(provider_names: list[str], config: ProviderConfig) -> SearchChain:
    return SearchChain([(name, build_provider(name, config)) for name in provider_names])


def chain_from_settings(settings: Any) -> SearchChain:
    """The provider chain configured by ``LOCALGATE_SEARCH_*`` settings.

    ``search_base_url``/``search_api_key`` are the original single-provider
    settings and still apply; the provider-specific ones win when set.
    """
    names = parse_provider_list(getattr(settings, "search_provider", None))
    base_url = getattr(settings, "search_base_url", None)
    api_key = getattr(settings, "search_api_key", None)
    config = ProviderConfig(
        searxng_url=getattr(settings, "search_searxng_url", None) or base_url,
        openserp_url=base_url,
        brave_api_key=getattr(settings, "search_brave_api_key", None) or api_key,
        tavily_api_key=api_key,
    )
    return build_chain(names, config)
