"""The optional `web_search` tool: schema gating, Tavily formatting, and the
`make_search_fn` factory `cli.py` uses to build (or refuse to build) it.
"""

import httpx
import pytest

from localgate.agent.websearch import (
    WebSearchError,
    make_search_fn,
    openserp_search,
    tavily_search,
)

_RealAsyncClient = httpx.AsyncClient


def _client_factory(handler):
    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return _RealAsyncClient(*args, **kwargs)

    return factory


async def test_tavily_search_formats_title_snippet_url(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "Example", "content": "a short snippet", "url": "https://example.com"}
                ]
            },
        )

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(handler))
    result = await tavily_search("query", "fake-key")
    assert "Example" in result
    assert "a short snippet" in result
    assert "https://example.com" in result


async def test_tavily_search_truncates_long_snippets(monkeypatch):
    long_content = "x" * 500

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"results": [{"title": "T", "content": long_content, "url": "https://x.test"}]},
        )

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(handler))
    result = await tavily_search("query", "fake-key")
    snippet_line = result.splitlines()[1].strip()
    assert len(snippet_line) < 250
    assert snippet_line.endswith("...")


async def test_tavily_search_with_no_results(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(handler))
    assert await tavily_search("query", "fake-key") == "No results."


async def test_tavily_search_raises_web_search_error_on_http_failure(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid key"})

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(handler))
    with pytest.raises(WebSearchError):
        await tavily_search("query", "bad-key")


async def test_tavily_search_never_sends_raw_page_content(monkeypatch):
    """Only title/content/url are read from a result — nothing that would let
    full scraped HTML slip into the model's context.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "T",
                        "content": "snippet",
                        "url": "https://x.test",
                        "raw_content": "<html>a whole scraped page...</html>",
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(handler))
    result = await tavily_search("query", "fake-key")
    assert "raw_content" not in result
    assert "<html>" not in result


def test_make_search_fn_rejects_an_unsupported_provider():
    with pytest.raises(ValueError, match="unsupported"):
        make_search_fn("bing", api_key="some-key")


def test_make_search_fn_requires_a_key_for_tavily():
    with pytest.raises(ValueError, match="API_KEY"):
        make_search_fn("tavily")


def test_make_search_fn_does_not_require_a_key_for_openserp():
    make_search_fn("openserp")  # should not raise


async def test_make_search_fn_returns_a_working_tavily_callable(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"title": "T", "content": "c", "url": "u"}]})

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(handler))
    search = make_search_fn("tavily", api_key="fake-key")
    result = await search("some query")
    assert "T" in result


async def test_make_search_fn_callable_returns_error_text_not_an_exception(monkeypatch):
    """The tool result on failure is text the model can react to, not a crash."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(handler))
    search = make_search_fn("tavily", api_key="fake-key")
    result = await search("some query")
    assert "web search failed" in result


async def test_make_search_fn_returns_a_working_openserp_callable(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"results": [{"title": "T", "snippet": "s", "url": "u"}]}
        )

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(handler))
    search = make_search_fn("openserp", base_url="http://localhost:9999")
    result = await search("some query")
    assert "T" in result


# --------------------------------------------------------------------- openserp


async def test_openserp_search_formats_title_snippet_url(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/google/search"
        assert request.url.params["text"] == "query"
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "Example", "snippet": "a short snippet", "url": "https://example.com"}
                ]
            },
        )

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(handler))
    result = await openserp_search("query", "http://localhost:7000")
    assert "Example" in result
    assert "a short snippet" in result
    assert "https://example.com" in result


async def test_openserp_search_accepts_description_key_as_a_snippet_fallback(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"results": [{"title": "T", "description": "fallback snippet", "url": "u"}]},
        )

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(handler))
    result = await openserp_search("query", "http://localhost:7000")
    assert "fallback snippet" in result


async def test_openserp_search_accepts_link_key_as_a_url_fallback(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"results": [{"title": "T", "snippet": "s", "link": "https://x.test"}]}
        )

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(handler))
    result = await openserp_search("query", "http://localhost:7000")
    assert "https://x.test" in result


async def test_openserp_search_with_no_results(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(handler))
    assert await openserp_search("query", "http://localhost:7000") == "No results."


async def test_openserp_search_raises_web_search_error_on_http_failure(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(handler))
    with pytest.raises(WebSearchError, match="OpenSERP"):
        await openserp_search("query", "http://localhost:7000")


async def test_openserp_search_uses_the_requested_engine(monkeypatch):
    seen_paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        return httpx.Response(200, json={"results": []})

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(handler))
    await openserp_search("query", "http://localhost:7000", engine="duckduckgo")
    assert seen_paths == ["/duckduckgo/search"]
