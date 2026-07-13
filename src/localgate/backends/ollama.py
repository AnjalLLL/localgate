"""Ollama HTTP adapter.

Talks to Ollama's OpenAI-compatible endpoint (http://localhost:11434/v1/...)
rather than its native /api/*, so request/response shapes need no translation
for chat. Embeddings use the native /api/embeddings endpoint since it's the
more stable of the two for that purpose.
"""
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from localgate.backends.base import InferenceBackend


class OllamaBackend(InferenceBackend):
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=120.0)

    async def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        payload = {**request, "stream": False}
        resp = await self._client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def chat_stream(self, request: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        payload = {**request, "stream": True}
        async with self._client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                yield json.loads(data)

    async def embed(self, text: str, model: str) -> list[float]:
        resp = await self._client.post("/api/embeddings", json={"model": model, "prompt": text})
        resp.raise_for_status()
        return resp.json()["embedding"]

    async def list_models(self) -> list[str]:
        resp = await self._client.get("/v1/models")
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]

    async def health(self) -> bool:
        try:
            resp = await self._client.get("/v1/models")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()
