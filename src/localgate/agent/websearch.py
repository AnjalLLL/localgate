"""Web search providers for the coding agent — DuckDuckGo by default.

Enabling this is the CLI's first outbound dependency on a third-party service
beyond the user's own inference backend: it sends query text off the user's
machine. The CLI uses DuckDuckGo when `LOCALGATE_SEARCH_PROVIDER` is unset;
that setting can select a different provider.

Providers live in `localgate.web.search` (SearXNG, OpenSERP, Brave, Tavily,
DuckDuckGo); `LOCALGATE_SEARCH_PROVIDER` may name one or a comma-separated
fallback chain such as ``searxng,duckduckgo``. This module is the agent-facing
layer: the tool schema and the ``query -> formatted text`` function.

Results are returned as title + short snippet + URL, never raw page content:
the same paraphrase-not-reproduce discipline applies here as anywhere else web
content enters the model's context.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from localgate.web import search as _search
from localgate.web.search import SearchChain, WebSearchError

__all__ = [
    "SUPPORTED_PROVIDERS",
    "WEB_SEARCH_SCHEMA",
    "SearchChain",
    "SearchFn",
    "WebSearchError",
    "duckduckgo_search",
    "make_search_fn",
    "openserp_search",
    "tavily_search",
]

#: Kept out of `TOOL_SCHEMAS`/`TOOL_NAMES` deliberately, same reasoning as
#: `DELEGATE_TASK_SCHEMA` in `tools.py` — only appended to a session's own
#: schema copy when a `search_fn` is actually configured (see `loop.py`).
WEB_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information not available in the project or the "
            "model's training data (e.g. a library's latest API, a recent CVE, current "
            "docs). Returns a short list of title/snippet/URL results, never full page "
            "content — follow up with the URL yourself if you need more than the snippet."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"],
        },
    },
}

SUPPORTED_PROVIDERS = _search.SUPPORTED_PROVIDERS

#: The self-hosted OpenSERP server's default port (its own documented default).
DEFAULT_OPENSERP_BASE_URL = _search.DEFAULT_OPENSERP_URL

SearchFn = Callable[[str], Awaitable[str]]


async def duckduckgo_search(query: str, *, max_results: int = 5, timeout: float = 10.0) -> str:
    """Free, zero-config web search via DuckDuckGo's HTML lite endpoint."""
    results = await _search.duckduckgo_results(query, max_results=max_results, timeout=timeout)
    return _search.format_results(results, max_results)


async def openserp_search(
    query: str,
    base_url: str,
    *,
    engine: str = "google",
    max_results: int = 5,
    timeout: float = 10.0,
) -> str:
    """One OpenSERP search against a self-hosted instance."""
    results = await _search.openserp_results(
        query, base_url, engine=engine, max_results=max_results, timeout=timeout
    )
    return _search.format_results(results, max_results)


async def tavily_search(
    query: str, api_key: str, *, max_results: int = 5, timeout: float = 10.0
) -> str:
    """One Tavily search, formatted as title + short snippet + URL per result."""
    results = await _search.tavily_results(query, api_key, max_results=max_results, timeout=timeout)
    return _search.format_results(results, max_results)


def make_search_fn(
    provider: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    brave_api_key: str | None = None,
    searxng_url: str | None = None,
    chain: SearchChain | None = None,
) -> SearchFn:
    """Build the tool's execution function. ``provider`` may be one name or a
    comma-separated fallback chain (``"searxng,duckduckgo"``). Raises
    `ValueError` for an unsupported provider, or one missing what it needs
    (Tavily's or Brave's key) — an explicitly-misconfigured feature should fail
    loudly at startup, not silently disable itself.
    """
    if chain is None:
        names = _search.parse_provider_list(provider)
        # LOCALGATE_SEARCH_BASE_URL / _API_KEY are the generic settings; the
        # provider-specific ones win when a chain needs both kinds.
        config = _search.ProviderConfig(
            searxng_url=searxng_url or base_url,
            openserp_url=base_url,
            brave_api_key=brave_api_key or api_key,
            tavily_api_key=api_key,
        )
        chain = _search.build_chain(names, config)
    resolved = chain

    async def _search_fn(query: str) -> str:
        try:
            return _search.format_results(await resolved.search(query))
        except WebSearchError as exc:
            return str(exc)

    _search_fn.chain = resolved  # type: ignore[attr-defined]
    return _search_fn
