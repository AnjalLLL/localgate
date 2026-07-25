"""Ollama adapter.

Chat goes through Ollama's OpenAI-compatible ``/v1/*`` surface, so no
translation is needed and the generic transport applies unchanged. Embeddings
are the one exception: Ollama's ``/v1/embeddings`` shim has been less reliable
across releases than its native ``/api/embeddings``, so this backend uses the
native route, which takes ``prompt`` and returns a bare ``embedding`` rather
than OpenAI's ``input``/``data[]`` shape.
"""

from __future__ import annotations

from typing import Any

import httpx

from localgate.backends.openai_compat import OpenAICompatBackend


def _human_size(num_bytes: object) -> str | None:
    """``4.7GB`` rather than a raw byte count — the picker needs this at a
    glance, since picking a 7B vs 34B model is a real tradeoff to see up front.
    """
    if not isinstance(num_bytes, int | float) or num_bytes <= 0:
        return None
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


class OllamaBackend(OpenAICompatBackend):
    name = "ollama"
    default_base_url = "http://localhost:11434"

    async def embed(self, text: str, model: str) -> list[float]:
        resp = await self._client.post("/api/embeddings", json={"model": model, "prompt": text})
        resp.raise_for_status()
        return resp.json()["embedding"]  # type: ignore[no-any-return]

    async def list_models_detailed(self) -> list[dict[str, Any]]:
        """Ollama's native ``/api/tags`` — unlike the OpenAI-compat ``/v1/models``
        shim, it carries size and quantization, which is what a real picker needs.
        """
        resp = await self._client.get("/api/tags")
        resp.raise_for_status()
        out: list[dict[str, Any]] = []
        for entry in resp.json().get("models", []):
            details = entry.get("details") or {}
            out.append(
                {
                    "name": entry.get("name") or entry.get("model", ""),
                    "size_human": _human_size(entry.get("size")),
                    "parameter_size": details.get("parameter_size"),
                    "quantization": details.get("quantization_level"),
                }
            )
        return out

    async def check_tool_support(self, model: str) -> bool | None:
        """Ollama's ``/api/show`` reports a ``capabilities`` list including
        ``"tools"`` on servers new enough to advertise it — cheap and free of
        the 400-on-first-real-turn surprise a live chat call would risk instead.
        """
        try:
            resp = await self._client.post("/api/show", json={"model": model})
            resp.raise_for_status()
        except httpx.HTTPError:
            return None
        capabilities = resp.json().get("capabilities")
        if not isinstance(capabilities, list):
            return None
        return "tools" in capabilities

    async def context_window(self, model: str) -> int | None:
        """``/api/show``'s ``model_info`` carries an architecture-prefixed
        ``<arch>.context_length`` key (e.g. ``qwen2.context_length``) — the key
        name varies by model family, so this scans for the suffix rather than
        hardcoding one architecture.
        """
        try:
            resp = await self._client.post("/api/show", json={"model": model})
            resp.raise_for_status()
        except httpx.HTTPError:
            return None
        info = resp.json().get("model_info")
        if not isinstance(info, dict):
            return None
        for key, value in info.items():
            if key.endswith(".context_length") and isinstance(value, int):
                return value
        return None
