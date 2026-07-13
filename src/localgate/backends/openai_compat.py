"""The generic OpenAI-compatible HTTP backend — and the base for the rest.

vLLM, llama.cpp's server, LM Studio, text-generation-webui and Ollama all expose
the same ``/v1/chat/completions`` contract, so they share one transport
implementation. Where they differ they differ declaratively — a default port, an
embeddings path — which subclasses express by overriding a class attribute rather
than by reimplementing HTTP.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from localgate.backends.base import InferenceBackend


class OpenAICompatBackend(InferenceBackend):
    """Talks to any server that speaks the OpenAI HTTP API."""

    name = "openai_compat"
    default_base_url = "http://localhost:8000"

    #: Path used by :meth:`embed`. Ollama overrides this with its native route.
    embeddings_path = "/v1/embeddings"

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 120.0,
        api_key: str | None = None,
    ) -> None:
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout, headers=headers)

    async def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        resp = await self._client.post("/v1/chat/completions", json={**request, "stream": False})
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    async def chat_stream(self, request: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        async with self._client.stream(
            "POST", "/v1/chat/completions", json={**request, "stream": True}
        ) as resp:
            await _raise_for_stream_status(resp)
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                if not data:
                    continue
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    # One malformed frame shouldn't abort the whole stream: the
                    # caller is better served by the tokens that do parse.
                    continue

    async def embed(self, text: str, model: str) -> list[float]:
        resp = await self._client.post(self.embeddings_path, json={"model": model, "input": text})
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]  # type: ignore[no-any-return]

    async def list_models(self) -> list[str]:
        resp = await self._client.get("/v1/models")
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]

    async def health(self) -> bool:
        try:
            resp = await self._client.get("/v1/models")
        except httpx.HTTPError:
            return False
        return resp.status_code == 200

    async def aclose(self) -> None:
        await self._client.aclose()


async def _raise_for_stream_status(resp: httpx.Response) -> None:
    """``raise_for_status`` for a streamed response, with the body attached.

    httpx doesn't read the body of a streaming response, so the exception it
    raises carries an empty ``.text`` — losing exactly the message the backend
    sent to explain the failure. Reading it first is what lets the error the
    caller finally sees say *why*.
    """
    if resp.is_success:
        return
    await resp.aread()
    resp.raise_for_status()
