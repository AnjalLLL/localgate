"""A deterministic fake backend, used only in tests.

Real assertions about the chat/auth/rate-limit pipeline shouldn't depend on
a real Ollama instance being up — that belongs in a separate, explicitly
opt-in integration test. This backend echoes input deterministically and
produces embeddings via a simple deterministic hash, so retrieval tests can
assert on exact similarity behavior instead of guessing at a real model's
output.
"""
import hashlib
from collections.abc import AsyncIterator
from typing import Any

from localgate.backends.base import InferenceBackend


def _deterministic_embedding(text: str, dims: int = 16) -> list[float]:
    """Same text -> same vector, different text -> different vector. Not semantically
    meaningful like a real embedding model, but good enough to test that retrieval
    picks the right stored chunk for a given query in a controlled test.
    """
    digest = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in digest[:dims]]


class FakeBackend(InferenceBackend):
    def __init__(self, base_url: str = ""):
        self.base_url = base_url
        self.calls: list[dict[str, Any]] = []  # inspectable in tests

    async def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(request)
        last_user_msg = next(
            (m["content"] for m in reversed(request["messages"]) if m["role"] == "user"), ""
        )
        reply = f"Echo: {last_user_msg}"
        prompt_tokens = sum(len(m.get("content", "").split()) for m in request["messages"])
        completion_tokens = len(reply.split())
        return {
            "id": "fake-chatcmpl-1",
            "object": "chat.completion",
            "model": request.get("model", "fake-model"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    async def chat_stream(self, request: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        response = await self.chat(request)
        content = response["choices"][0]["message"]["content"]
        for word in content.split():
            yield {"choices": [{"delta": {"content": word + " "}}]}

    async def embed(self, text: str, model: str) -> list[float]:
        return _deterministic_embedding(text)

    async def list_models(self) -> list[str]:
        return ["fake-model"]

    async def health(self) -> bool:
        return True
