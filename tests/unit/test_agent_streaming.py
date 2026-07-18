"""Streaming reassembly: OpenAI-shaped delta chunks (content token-by-token,
tool-call fragments keyed by index) must reassemble into the same message shape
the non-streaming path produces.
"""

from collections.abc import AsyncIterator
from typing import Any

import pytest

from localgate.agent.loop import AgentSession
from localgate.backends.base import InferenceBackend


class StreamingScriptedBackend(InferenceBackend):
    """Yields one scripted sequence of stream chunks per call to `chat_stream`."""

    name = "streaming-scripted"

    def __init__(self, turns: list[list[dict[str, Any]]]) -> None:
        self._turns = list(turns)
        self.requests: list[dict[str, Any]] = []

    async def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("this backend only streams")

    async def chat_stream(self, request: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        self.requests.append(request)
        for chunk in self._turns.pop(0):
            yield chunk

    async def embed(self, text: str, model: str) -> list[float]:
        raise NotImplementedError

    async def list_models(self) -> list[str]:
        return ["scripted-model"]

    async def health(self) -> bool:
        return True


def content_chunk(text: str) -> dict[str, Any]:
    return {"choices": [{"delta": {"content": text}}]}


@pytest.fixture
def project(tmp_path):
    (tmp_path / "app.py").write_text("hi\n")
    return tmp_path


async def test_streamed_text_is_reassembled_and_forwarded_token_by_token(project):
    backend = StreamingScriptedBackend([[content_chunk("Hello, "), content_chunk("world!")]])
    tokens: list[str] = []
    session = AgentSession(backend, "scripted-model", project, on_token=tokens.append)

    result = await session.send("say hi")

    assert result == "Hello, world!"
    assert tokens == ["Hello, ", "world!"]


async def test_streamed_tool_call_arguments_are_concatenated_across_fragments(project):
    tool_call_turn = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "read_file", "arguments": ""},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"path"'}}]}}
            ]
        },
        {
            "choices": [
                {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ': "app.py"}'}}]}}
            ]
        },
    ]
    final_turn = [content_chunk("read it, done")]
    backend = StreamingScriptedBackend([tool_call_turn, final_turn])
    session = AgentSession(backend, "scripted-model", project, on_token=lambda _: None)

    result = await session.send("read app.py")

    assert result == "read it, done"
    tool_message = next(m for m in session.messages if m.get("role") == "tool")
    assert tool_message["content"] == "hi\n"


async def test_streaming_and_non_streaming_paths_agree_on_a_plain_text_reply(project):
    streaming_backend = StreamingScriptedBackend([[content_chunk("done")]])
    streaming_session = AgentSession(
        streaming_backend, "scripted-model", project, on_token=lambda _: None
    )
    assert await streaming_session.send("go") == "done"
