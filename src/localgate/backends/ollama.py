"""Ollama adapter.

Chat goes through Ollama's OpenAI-compatible ``/v1/*`` surface, so no
translation is needed and the generic transport applies unchanged. Embeddings
are the one exception: Ollama's ``/v1/embeddings`` shim has been less reliable
across releases than its native ``/api/embeddings``, so this backend uses the
native route, which takes ``prompt`` and returns a bare ``embedding`` rather
than OpenAI's ``input``/``data[]`` shape.
"""

from __future__ import annotations

from localgate.backends.openai_compat import OpenAICompatBackend


class OllamaBackend(OpenAICompatBackend):
    name = "ollama"
    default_base_url = "http://localhost:11434"

    async def embed(self, text: str, model: str) -> list[float]:
        resp = await self._client.post("/api/embeddings", json={"model": model, "prompt": text})
        resp.raise_for_status()
        return resp.json()["embedding"]  # type: ignore[no-any-return]
