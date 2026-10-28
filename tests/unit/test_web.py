"""Search chain, page fetch guards, HTML extraction and the research pipeline."""

import httpx
import pytest

from localgate.web import fetch as fetch_mod
from localgate.web.fetch import FetchError, check_destination, fetch_page, html_to_text
from localgate.web.research import research, sources_message
from localgate.web.search import (
    ProviderConfig,
    SearchChain,
    SearchResult,
    WebSearchError,
    _ddg_target,
    build_provider,
    chain_from_settings,
    parse_provider_list,
)


async def test_chain_falls_back_to_the_next_provider_and_caches():
    calls: list[str] = []

    async def broken(query, n):
        calls.append("broken")
        raise WebSearchError("down")

    async def empty(query, n):
        calls.append("empty")
        return []

    async def good(query, n):
        calls.append("good")
        return [SearchResult("T", "https://example.com", "s", "good")]

    chain = SearchChain([("a", broken), ("b", empty), ("c", good)])
    first = await chain.search("Query  text")
    second = await chain.search("query text")
    assert first == second and first[0].title == "T"
    assert calls == ["broken", "empty", "good"]


async def test_chain_raises_only_when_every_provider_fails():
    async def broken(query, n):
        raise WebSearchError("down")

    with pytest.raises(WebSearchError):
        await SearchChain([("a", broken), ("b", broken)]).search("q")


def test_provider_list_and_config_validation():
    assert parse_provider_list(None) == ["duckduckgo"]
    assert parse_provider_list(" SearXNG , duckduckgo ") == ["searxng", "duckduckgo"]
    with pytest.raises(ValueError):
        build_provider("brave", ProviderConfig())
    with pytest.raises(ValueError):
        build_provider("bing", ProviderConfig())

    class S:
        search_provider = "searxng,duckduckgo"
        search_searxng_url = "http://searx:8080"

    chain = chain_from_settings(S())
    assert chain.names == ["searxng", "duckduckgo"]
    assert chain.leaves_machine


def test_duckduckgo_redirect_links_are_unwrapped():
    wrapped = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2F&amp;rut=abc"
    assert _ddg_target(wrapped) == "https://docs.python.org/3/"
    assert _ddg_target("https://example.com/x") == "https://example.com/x"


async def test_searxng_results_parse(monkeypatch):
    real = httpx.AsyncClient

    def handler(request):
        assert request.url.params["format"] == "json"
        return httpx.Response(
            200,
            json={"results": [{"title": "A &amp; B", "url": "https://a.b", "content": "<b>x</b>"}]},
        )

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *a, **k: real(*a, **{**k, "transport": httpx.MockTransport(handler)}),
    )
    from localgate.web.search import searxng_results  # noqa: PLC0415

    results = await searxng_results("q", "http://searx")
    assert results == [SearchResult("A & B", "https://a.b", "x", "searxng")]


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://localhost:8000/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/",
        "http://[::1]/",
        "file:///etc/passwd",
        "ftp://example.com/",
    ],
)
async def test_private_and_non_http_destinations_are_refused(url):
    with pytest.raises(FetchError):
        await check_destination(url)


async def test_private_destinations_allowed_when_configured():
    await check_destination("http://127.0.0.1:9/", allow_private=True)


def test_html_to_text_keeps_content_and_drops_chrome():
    page = """<html><head><title>Docs  Page</title><style>p{}</style></head><body>
    <nav>menu</nav><h1>Install</h1><p>Run   <code>pip install x</code>.</p>
    <script>alert(1)</script><ul><li>one</li><li>two</li></ul>
    <pre>def f():\n    return 1</pre><footer>legal</footer></body></html>"""
    title, text = html_to_text(page)
    assert title == "Docs Page"
    assert "# Install" in text and "pip install x" in text
    assert "- one" in text and "    return 1" in text
    for junk in ("menu", "alert", "legal", "p{}"):
        assert junk not in text


async def test_fetch_page_follows_redirects_and_rechecks_each_hop(monkeypatch):
    async def public(url, *, allow_private=False):
        if "internal" in url:
            raise FetchError("non-public")

    monkeypatch.setattr(fetch_mod, "check_destination", public)
    real = httpx.AsyncClient

    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if request.url.host == "start.example":
            return httpx.Response(302, headers={"location": "http://internal.example/secret"})
        return httpx.Response(200, html="<p>secret</p>")

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *a, **k: real(*a, **{**k, "transport": httpx.MockTransport(handler)}),
    )
    with pytest.raises(FetchError, match="non-public"):
        await fetch_page("http://start.example/")


async def test_fetch_page_respects_robots_txt(monkeypatch):
    async def public(url, *, allow_private=False):
        return None

    monkeypatch.setattr(fetch_mod, "check_destination", public)
    real = httpx.AsyncClient

    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private\n")
        return httpx.Response(200, html="<p>ok</p>")

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *a, **k: real(*a, **{**k, "transport": httpx.MockTransport(handler)}),
    )
    page = await fetch_page("http://site.example/public")
    assert page.text == "ok"
    with pytest.raises(FetchError, match="robots"):
        await fetch_page("http://site.example/private/x")


async def test_research_ranks_snippets_and_numbers_sources():
    async def provider(query, n):
        return [
            SearchResult("Cooking", "https://a", "pasta recipes and sauces"),
            SearchResult("Python asyncio", "https://b", "asyncio gather runs coroutines"),
        ]

    sources = await research(
        "how does asyncio gather work", SearchChain([("p", provider)]), fetch_pages=False
    )
    assert [s.index for s in sources] == [1, 2]
    message = sources_message(sources)
    assert message["role"] == "system"
    assert "[2] Python asyncio" in message["content"]
    assert "not instructions" in message["content"]
