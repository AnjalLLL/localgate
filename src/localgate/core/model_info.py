"""What the gateway knows about each model: context window, capabilities, size.

Two uses: ``/v1/models`` reports it so clients (``localgate code --remote``
above all) can size prompts and pick tool modes, and chat requests get a
``num_ctx`` so Ollama doesn't run at its small server default and silently drop
the start of a long prompt. Each lookup is an Ollama ``/api/show`` or
``/api/tags`` call, so results are cached for a few minutes per backend+model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from localgate.backends.base import InferenceBackend

#: How long a model's details are trusted before asking the backend again.
CACHE_SECONDS = 300.0


@dataclass
class ModelDetails:
    context_window: int | None = None
    capabilities: list[str] | None = None
    parameter_size: str | None = None
    quantization: str | None = None
    size_human: str | None = None

    def as_card_fields(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class ModelInfoCache:
    _entries: dict[tuple[int, str], tuple[float, ModelDetails]] = field(default_factory=dict)
    _listing: dict[int, tuple[float, dict[str, dict[str, Any]]]] = field(default_factory=dict)

    async def _detailed_listing(self, backend: InferenceBackend) -> dict[str, dict[str, Any]]:
        now = time.monotonic()
        cached = self._listing.get(id(backend))
        if cached and now - cached[0] < CACHE_SECONDS:
            return cached[1]
        try:
            rows = await backend.list_models_detailed()
        except (httpx.HTTPError, NotImplementedError):
            rows = []
        listing = {str(row.get("name")): row for row in rows if row.get("name")}
        self._listing[id(backend)] = (now, listing)
        return listing

    async def get(self, backend: InferenceBackend, model: str) -> ModelDetails:
        now = time.monotonic()
        key = (id(backend), model)
        cached = self._entries.get(key)
        if cached and now - cached[0] < CACHE_SECONDS:
            return cached[1]
        row = (await self._detailed_listing(backend)).get(model, {})
        details = ModelDetails(
            parameter_size=row.get("parameter_size"),
            quantization=row.get("quantization"),
            size_human=row.get("size_human"),
        )
        try:
            details.context_window = await backend.context_window(model)
            details.capabilities = await backend.capabilities(model)
        except (httpx.HTTPError, NotImplementedError):
            pass
        self._entries[key] = (now, details)
        return details

    def clear(self) -> None:
        self._entries.clear()
        self._listing.clear()


async def apply_default_num_ctx(
    payload: dict[str, Any],
    backend: InferenceBackend,
    model: str,
    default_num_ctx: int,
    cache: ModelInfoCache,
) -> None:
    """Give an Ollama-bound request a context window unless the caller set one:
    ``default_num_ctx``, capped at the model's own maximum when it's known."""
    if not backend.accepts_options:
        return
    options = payload.setdefault("options", {})
    if "num_ctx" in options:
        return
    window = (await cache.get(backend, model)).context_window
    options["num_ctx"] = min(default_num_ctx, window) if window else default_num_ctx
