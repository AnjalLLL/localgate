"""Optional web search tool for the coding agent — off by default.

Enabling this is the CLI's first outbound dependency on a third-party service
beyond the user's own inference backend: it sends query text off the user's
machine. `LOCALGATE_SEARCH_PROVIDER` must be set for the tool to exist at all
(see `Settings` in `config.py`) — unset means the model never even sees
`web_search` as an option, not that it exists and silently no-ops.

Two providers:

- **`openserp`** (recommended default) — a self-hosted, MIT-licensed, no-API-key
  search API (github.com/karust/openserp). Runs on the user's own machine/network,
  so enabling it doesn't actually send anything to a *third party* — the closest
  fit to this project's local-first stance. `LOCALGATE_SEARCH_BASE_URL` points at
  wherever it's running; defaults to `http://localhost:7000`.
- **`tavily`** — a hosted, paid, API-key-gated service. Kept for anyone who
  already has a key; genuinely leaves the user's machine, unlike `openserp`.

Results are returned as title + short snippet + URL, never raw page content:
the same paraphrase-not-reproduce discipline applies here as anywhere else web
content enters the model's context.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import httpx

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

SUPPORTED_PROVIDERS = ("openserp", "tavily")

#: The self-hosted OpenSERP server's default port (its own documented default).
DEFAULT_OPENSERP_BASE_URL = "http://localhost:7000"

SearchFn = Callable[[str], Awaitable[str]]

_SNIPPET_MAX_CHARS = 240


class WebSearchError(RuntimeError):
    """The search request itself failed — network error or a non-2xx response."""


def _truncate(text: str) -> str:
    text = text.strip()
    if len(text) > _SNIPPET_MAX_CHARS:
        return text[: _SNIPPET_MAX_CHARS - 3] + "..."
    return text


def _format_results(
    results: list[dict[str, Any]],
    *,
    title_keys: tuple[str, ...],
    snippet_keys: tuple[str, ...],
    url_keys: tuple[str, ...],
    max_results: int,
) -> str:
    """Shared title + snippet + URL formatting, tolerant of the small key-naming
    differences between providers (`content` vs `snippet`/`description`, `url`
    vs `link`) rather than assuming one exact response shape.
    """
    if not results:
        return "No results."
    lines: list[str] = []
    for result in results[:max_results]:
        title = next((result[k] for k in title_keys if result.get(k)), "")
        snippet = next((result[k] for k in snippet_keys if result.get(k)), "")
        url = next((result[k] for k in url_keys if result.get(k)), "")
        lines.append(f"- {title.strip()}\n  {_truncate(snippet)}\n  {url}")
    return "\n".join(lines)


async def openserp_search(
    query: str,
    base_url: str,
    *,
    engine: str = "google",
    max_results: int = 5,
    timeout: float = 15.0,
) -> str:
    """One OpenSERP search against a self-hosted instance — no API key, no
    third party involved once the server is running on the user's own machine.
    """
    url = f"{base_url.rstrip('/')}/{engine}/search"
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(
                url, params={"text": query, "limit": max_results, "format": "json"}
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise WebSearchError(
                f"web search failed ({base_url}): {exc} — is OpenSERP running there?"
            ) from exc

    body = resp.json()
    results = body.get("results") if isinstance(body, dict) else None
    if not isinstance(results, list):
        return "No results."
    return _format_results(
        results,
        title_keys=("title",),
        snippet_keys=("snippet", "description"),
        url_keys=("url", "link"),
        max_results=max_results,
    )


async def tavily_search(
    query: str, api_key: str, *, max_results: int = 5, timeout: float = 15.0
) -> str:
    """One Tavily search, formatted as title + short snippet + URL per result."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": api_key, "query": query, "max_results": max_results},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise WebSearchError(f"web search failed: {exc}") from exc

    results = resp.json().get("results") or []
    return _format_results(
        results,
        title_keys=("title",),
        snippet_keys=("content",),
        url_keys=("url",),
        max_results=max_results,
    )


def make_search_fn(
    provider: str, *, api_key: str | None = None, base_url: str | None = None
) -> SearchFn:
    """Build the tool's execution function for `provider`. Raises `ValueError`
    for an unsupported provider, or a provider that's missing what it needs
    (Tavily's key) — an explicitly-misconfigured feature should fail loudly at
    startup, not silently disable itself.
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"unsupported LOCALGATE_SEARCH_PROVIDER {provider!r} — "
            f"choose from {', '.join(SUPPORTED_PROVIDERS)}"
        )

    if provider == "tavily":
        if not api_key:
            raise ValueError("LOCALGATE_SEARCH_API_KEY is required when using tavily")

        async def _search_tavily(query: str) -> str:
            try:
                return await tavily_search(query, api_key)
            except WebSearchError as exc:
                return str(exc)

        return _search_tavily

    resolved_base_url = base_url or DEFAULT_OPENSERP_BASE_URL

    async def _search_openserp(query: str) -> str:
        try:
            return await openserp_search(query, resolved_base_url)
        except WebSearchError as exc:
            return str(exc)

    return _search_openserp
