"""`localgate code --remote`'s backend: it keeps Ollama-native fields and reads
model details from the gateway, so remote sessions size their context like
local ones."""

import json

import httpx

from localgate.agent.loop import AgentSession
from localgate.backends.localgate_remote import LocalgateBackend
from localgate.backends.openai_compat import OpenAICompatBackend

CARDS = {
    "object": "list",
    "data": [
        {
            "id": "qwen2.5-coder:7b",
            "object": "model",
            "context_window": 32768,
            "capabilities": ["completion", "tools"],
            "parameter_size": "7.6B",
            "quantization": "Q4_K_M",
        },
        {"id": "llama3", "object": "model", "capabilities": ["completion"]},
    ],
}


def _with_transport(backend, sent: list[dict]):
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=CARDS)
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    backend._client = httpx.AsyncClient(base_url="http://gw", transport=httpx.MockTransport(handle))
    return backend


async def test_native_fields_reach_the_gateway_but_not_a_generic_server():
    sent: list[dict] = []
    remote = _with_transport(LocalgateBackend("http://gw"), sent)
    generic = _with_transport(OpenAICompatBackend("http://gw"), sent)
    request = {
        "model": "m",
        "messages": [],
        "options": {"num_ctx": 16384},
        "format": {"type": "object"},
    }
    await remote.chat(request)
    await generic.chat(request)
    assert sent[0]["options"] == {"num_ctx": 16384} and "format" in sent[0]
    assert "options" not in sent[1] and "format" not in sent[1]


async def test_model_details_come_from_the_gateway():
    backend = _with_transport(LocalgateBackend("http://gw"), [])
    assert await backend.context_window("qwen2.5-coder:7b") == 32768
    assert await backend.check_tool_support("qwen2.5-coder:7b") is True
    assert await backend.check_tool_support("llama3") is False
    assert await backend.context_window("llama3") is None
    detailed = await backend.list_models_detailed()
    assert detailed[0]["parameter_size"] == "7.6B"


async def test_a_remote_agent_session_sends_num_ctx(tmp_path):
    sent: list[dict] = []
    backend = _with_transport(LocalgateBackend("http://gw"), sent)
    session = AgentSession(backend, "qwen2.5-coder:7b", tmp_path)
    await session.send("hello")
    assert sent[-1]["options"]["num_ctx"] == min(32768, 16384)  # model max vs agent cap
