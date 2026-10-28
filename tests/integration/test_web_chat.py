"""Chat with `"localgate": {"web_search": true}`: sources reach the model framed as
reference material, citations come back, and the extension never reaches the backend."""

import json

from localgate.web.search import SearchChain, SearchResult, WebSearchError


def fake_chain(results=None, error=None) -> SearchChain:
    async def provider(query, n):
        if error:
            raise WebSearchError(error)
        return results or []

    return SearchChain([("fake", provider)])


async def test_web_search_adds_sources_and_returns_citations(
    app, client, auth_headers, monkeypatch
):
    app.state.search_chain = fake_chain(
        [SearchResult("Release notes", "https://example.com/notes", "version 2 adds streaming")]
    )

    async def no_fetch(*args, **kwargs):
        from localgate.web.fetch import FetchError  # noqa: PLC0415

        raise FetchError("offline test")

    monkeypatch.setattr("localgate.web.research.fetch_page", no_fetch)
    resp = await client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "what does version 2 add?"}],
            "localgate": {"web_search": True},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    citations = body["localgate"]["citations"]
    assert citations[0]["url"] == "https://example.com/notes" and citations[0]["index"] == 1
    sent = app.state.backend.calls[-1]
    assert "localgate" not in sent
    system_messages = [m["content"] for m in sent["messages"] if m["role"] == "system"]
    assert any("[1] Release notes" in m and "not instructions" in m for m in system_messages)


async def test_web_search_failure_still_answers_and_reports_why(app, client, auth_headers):
    app.state.search_chain = fake_chain(error="all providers down")
    resp = await client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "localgate": {"web_search": True},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "Echo: hello"
    assert body["localgate"]["citations"] == []
    assert "all providers down" in body["localgate"]["web_search_error"]


async def test_streaming_sends_citations_as_a_final_chunk(app, client, auth_headers, monkeypatch):
    app.state.search_chain = fake_chain([SearchResult("T", "https://t.example", "snippet")])

    async def no_fetch(*args, **kwargs):
        from localgate.web.fetch import FetchError  # noqa: PLC0415

        raise FetchError("offline test")

    monkeypatch.setattr("localgate.web.research.fetch_page", no_fetch)
    resp = await client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={
            "model": "fake-model",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
            "localgate": {"web_search": True},
        },
    )
    frames = [
        json.loads(line[len("data: ") :])
        for line in resp.text.splitlines()
        if line.startswith("data: {")
    ]
    last = frames[-1]
    assert last["choices"] == [] and last["localgate"]["citations"][0]["url"] == "https://t.example"


async def test_plain_requests_are_unchanged(app, client, auth_headers):
    resp = await client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert "localgate" not in resp.json()
