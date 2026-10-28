"""Another localgate gateway as the backend — what `localgate code --remote` uses.

A gateway speaks the OpenAI API, so this is the OpenAI-compatible backend with
two differences that matter to the coding agent:

* Ollama-native fields (`options` such as `num_ctx`, and `format` for JSON tool
  mode) are **kept**: the gateway forwards them to its own Ollama. A generic
  OpenAI client strips them, which silently left remote sessions at Ollama's
  small default context window.
* Model details (context window, capabilities, size) come from the gateway's
  extended `/v1/models`, so the agent can size its context and choose a tool
  mode exactly as it does against a local Ollama.
"""

from __future__ import annotations

import time
from typing import Any

from localgate.backends.openai_compat import OpenAICompatBackend

#: How long the gateway's model listing is trusted before asking again.
_CARDS_SECONDS = 60.0


class LocalgateBackend(OpenAICompatBackend):
    name = "localgate"
    default_base_url = "http://localhost:8000"
    accepts_options = True

    def __init__(
        self, base_url: str | None = None, timeout: float = 120.0, api_key: str | None = None
    ) -> None:
        super().__init__(base_url, timeout, api_key)
        self._cards: dict[str, dict[str, Any]] = {}
        self._cards_at = float("-inf")

    def _body(self, request: dict[str, Any]) -> dict[str, Any]:
        return dict(request)

    async def _model_cards(self) -> dict[str, dict[str, Any]]:
        if time.monotonic() - self._cards_at > _CARDS_SECONDS:
            resp = await self._client.get("/v1/models")
            resp.raise_for_status()
            self._cards = {card["id"]: card for card in resp.json().get("data", [])}
            self._cards_at = time.monotonic()
        return self._cards

    async def list_models(self) -> list[str]:
        return list(await self._model_cards())

    async def list_models_detailed(self) -> list[dict[str, Any]]:
        return [
            {
                "name": model_id,
                "parameter_size": card.get("parameter_size"),
                "quantization": card.get("quantization"),
                "size_human": card.get("size_human"),
            }
            for model_id, card in (await self._model_cards()).items()
        ]

    async def context_window(self, model: str) -> int | None:
        window = (await self._model_cards()).get(model, {}).get("context_window")
        return int(window) if isinstance(window, int) else None

    async def capabilities(self, model: str) -> list[str] | None:
        caps = (await self._model_cards()).get(model, {}).get("capabilities")
        return list(caps) if isinstance(caps, list) else None

    async def check_tool_support(self, model: str) -> bool | None:
        caps = await self.capabilities(model)
        return None if caps is None else "tools" in caps
