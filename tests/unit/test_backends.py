"""The backend registry and the OpenAI-compatible adapters.

The three non-Ollama backends used to declare `InferenceBackend` subclasses that
implemented none of its abstract methods, so `get_backend("vllm")` raised TypeError
at startup. Three of the four advertised backends did not work at all — hence the
instantiation tests here.
"""

import json

import httpx
import pytest

from localgate.backends import available_backends, get_backend, register_backend
from localgate.backends.base import InferenceBackend
from localgate.backends.llamacpp import LlamaCppBackend
from localgate.backends.ollama import OllamaBackend
from localgate.backends.openai_compat import OpenAICompatBackend
from localgate.backends.vllm import VLLMBackend


@pytest.mark.parametrize("backend_type", ["ollama", "vllm", "llamacpp", "openai_compat", "fake"])
def test_every_advertised_backend_can_actually_be_instantiated(backend_type):
    backend = get_backend(backend_type, "http://localhost:1234")
    assert isinstance(backend, InferenceBackend)


@pytest.mark.parametrize(
    ("backend_cls", "expected_port"),
    [(OllamaBackend, 11434), (VLLMBackend, 8000), (LlamaCppBackend, 8080)],
)
def test_backends_default_to_their_own_conventional_port(backend_cls, expected_port):
    assert str(expected_port) in backend_cls().base_url


def test_unknown_backend_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="Unknown backend_type") as exc:
        get_backend("not-a-real-backend")
    assert "ollama" in str(exc.value)  # tell the user what they could have typed


def test_available_backends_lists_the_builtins():
    assert {"ollama", "vllm", "llamacpp", "openai_compat"} <= set(available_backends())


def test_a_plugin_can_register_a_backend():
    """Third parties add backends via the localgate.backends entry point; this is the
    same mechanism, exercised without installing a package."""

    class MyBackend(OpenAICompatBackend):
        name = "my-backend"

    register_backend("my-backend", MyBackend)

    assert "my-backend" in available_backends()
    assert isinstance(get_backend("my-backend", "http://x"), MyBackend)


def test_registering_a_non_backend_is_rejected():
    with pytest.raises(TypeError):
        register_backend("bogus", dict)  # type: ignore[arg-type]


def test_a_plugin_that_only_accepts_base_url_still_works():
    """The plugin contract must stay minimal: a backend whose __init__ takes only
    base_url should not break when localgate grows new per-backend options."""

    class MinimalBackend(InferenceBackend):
        def __init__(self, base_url=None):
            self.base_url = base_url

        async def chat(self, request): ...
        def chat_stream(self, request): ...
        async def embed(self, text, model): ...
        async def list_models(self): ...
        async def health(self): ...

    register_backend("minimal", MinimalBackend)
    backend = get_backend("minimal", "http://x", timeout=5.0, api_key="k")
    assert backend.base_url == "http://x"


async def test_chat_posts_to_the_openai_route_and_returns_the_body():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert json.loads(request.content)["stream"] is False
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    backend = OpenAICompatBackend("http://test")
    backend._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test"
    )

    response = await backend.chat({"model": "m", "messages": []})
    assert response["choices"][0]["message"]["content"] == "hi"
    await backend.aclose()


async def test_chat_stream_parses_sse_and_stops_at_done():
    body = (
        'data: {"choices":[{"delta":{"content":"he"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"llo"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    backend = OpenAICompatBackend("http://test")
    backend._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text=body)),
        base_url="http://test",
    )

    chunks = [chunk async for chunk in backend.chat_stream({"model": "m", "messages": []})]
    assert "".join(c["choices"][0]["delta"]["content"] for c in chunks) == "hello"
    await backend.aclose()


async def test_a_malformed_frame_does_not_kill_the_whole_stream():
    """The caller is better served by the tokens that do parse than by an abort."""
    body = (
        'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
        "data: {not json at all\n\n"
        'data: {"choices":[{"delta":{"content":"!"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    backend = OpenAICompatBackend("http://test")
    backend._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text=body)),
        base_url="http://test",
    )

    chunks = [chunk async for chunk in backend.chat_stream({"model": "m", "messages": []})]
    assert "".join(c["choices"][0]["delta"]["content"] for c in chunks) == "ok!"
    await backend.aclose()


async def test_a_failed_stream_raises_with_the_backend_s_explanation_attached():
    """httpx does not read the body of a streamed response, so a naive
    raise_for_status would produce an exception with an empty .text — throwing away
    exactly the message that says why it failed."""
    backend = OpenAICompatBackend("http://test")
    backend._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(404, text="model 'x' not found")),
        base_url="http://test",
    )

    with pytest.raises(httpx.HTTPStatusError) as exc:
        [chunk async for chunk in backend.chat_stream({"model": "x", "messages": []})]

    assert "model 'x' not found" in exc.value.response.text
    await backend.aclose()


async def test_ollama_embeds_via_its_native_route():
    """Ollama's /v1/embeddings shim has been unreliable across releases; the native
    route takes `prompt` and returns a bare `embedding`, not OpenAI's data[] shape."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embeddings"
        assert json.loads(request.content)["prompt"] == "hello"
        return httpx.Response(200, json={"embedding": [0.1, 0.2]})

    backend = OllamaBackend("http://test")
    backend._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test"
    )

    assert await backend.embed("hello", "nomic-embed-text") == [0.1, 0.2]
    await backend.aclose()


async def test_openai_compat_embeds_via_the_openai_route():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/embeddings"
        assert json.loads(request.content)["input"] == "hello"
        return httpx.Response(200, json={"data": [{"embedding": [0.3]}]})

    backend = VLLMBackend("http://test")
    backend._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test"
    )

    assert await backend.embed("hello", "m") == [0.3]
    await backend.aclose()


async def test_health_is_false_when_the_backend_is_unreachable():
    backend = OpenAICompatBackend("http://test")
    backend._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: (_ for _ in ()).throw(httpx.ConnectError("no"))),
        base_url="http://test",
    )
    assert await backend.health() is False
    await backend.aclose()


async def test_an_upstream_api_key_is_sent_as_a_bearer_token():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer secret"
        return httpx.Response(200, json={"data": []})

    backend = OpenAICompatBackend("http://test", api_key="secret")
    backend._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://test",
        headers={"Authorization": "Bearer secret"},
    )
    assert await backend.list_models() == []
    await backend.aclose()
