"""Ollama's native /api/chat path: used when a request carries `options`/`format`
(so `num_ctx` actually reaches the model), translated to and from the OpenAI shape."""

import json

import httpx

from localgate.backends.ollama import OllamaBackend, from_native_response, to_native_request
from localgate.backends.openai_compat import OpenAICompatBackend


def test_to_native_request_converts_tool_history_and_options():
    request = {
        "model": "qwen",
        "messages": [
            {"role": "system", "content": "sys"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "x = 1"},
        ],
        "tools": [{"type": "function", "function": {"name": "read_file"}}],
        "max_tokens": 512,
        "temperature": 0.2,
        "options": {"num_ctx": 16384},
    }
    body = to_native_request(request, stream=False)
    assert body["options"] == {"num_ctx": 16384, "num_predict": 512, "temperature": 0.2}
    assert body["messages"][1]["content"] == ""
    assert body["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"path": "a.py"}
    assert body["messages"][2]["tool_name"] == "read_file"
    assert body["tools"] == request["tools"]
    assert body["stream"] is False


def test_from_native_response_produces_openai_tool_calls_and_usage():
    body = {
        "model": "qwen",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "a.py"}}}],
        },
        "done": True,
        "prompt_eval_count": 10,
        "eval_count": 3,
    }
    out = from_native_response(body)
    choice = out["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["id"].startswith("call_")
    assert json.loads(call["function"]["arguments"]) == {"path": "a.py"}
    assert out["usage"] == {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}


async def test_chat_with_options_uses_native_route_and_without_uses_v1():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/api/chat":
            sent = json.loads(request.content)
            assert sent["options"]["num_ctx"] == 8192
            return httpx.Response(200, json={"message": {"content": "native"}, "done": True})
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "compat"}}]}
        )

    backend = OllamaBackend("http://ollama")
    backend._client = httpx.AsyncClient(
        base_url="http://ollama", transport=httpx.MockTransport(handler)
    )
    native = await backend.chat({"model": "m", "messages": [], "options": {"num_ctx": 8192}})
    compat = await backend.chat({"model": "m", "messages": []})
    assert native["choices"][0]["message"]["content"] == "native"
    assert compat["choices"][0]["message"]["content"] == "compat"
    assert seen == ["/api/chat", "/v1/chat/completions"]


async def test_native_stream_yields_openai_deltas_with_indexed_tool_calls():
    frames = [
        {"message": {"content": "Hel"}, "done": False},
        {"message": {"content": "lo"}, "done": False},
        {
            "message": {
                "content": "",
                "tool_calls": [{"function": {"name": "git_status", "arguments": {}}}],
            },
            "done": False,
        },
        {"message": {"content": ""}, "done": True, "prompt_eval_count": 5, "eval_count": 2},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="\n".join(json.dumps(f) for f in frames))

    backend = OllamaBackend("http://ollama")
    backend._client = httpx.AsyncClient(
        base_url="http://ollama", transport=httpx.MockTransport(handler)
    )
    request = {"model": "m", "messages": [], "options": {"num_ctx": 1}}
    chunks = [c async for c in backend.chat_stream(request)]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == "Hello"
    calls = [tc for c in chunks for tc in c["choices"][0]["delta"].get("tool_calls", [])]
    assert calls[0]["index"] == 0 and calls[0]["function"]["name"] == "git_status"
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[-1]["usage"]["total_tokens"] == 7


async def test_generic_backend_strips_native_only_keys():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    backend = OpenAICompatBackend("http://vllm")
    backend._client = httpx.AsyncClient(
        base_url="http://vllm", transport=httpx.MockTransport(handler)
    )
    await backend.chat(
        {"model": "m", "messages": [], "options": {"num_ctx": 1}, "format": {}, "keep_alive": "5m"}
    )
    assert "options" not in captured and "format" not in captured and "keep_alive" not in captured
