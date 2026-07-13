"""The interface every inference backend implements.

Supporting a new inference server means writing one subclass of
:class:`InferenceBackend` and registering it under the ``localgate.backends``
entry-point group. Nothing above this layer changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class InferenceBackend(ABC):
    """All backends (Ollama, llama.cpp, vLLM, ...) implement this interface."""

    #: Short identifier used in logs, metric labels, and error messages.
    name: str = "backend"

    @abstractmethod
    async def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        """Non-streaming chat completion. Returns an OpenAI-shaped response body."""

    @abstractmethod
    def chat_stream(self, request: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Streaming chat completion, yielding OpenAI-shaped delta chunks."""

    @abstractmethod
    async def embed(self, text: str, model: str) -> list[float]:
        """Return an embedding vector for ``text``."""

    @abstractmethod
    async def list_models(self) -> list[str]:
        """Return the model names this backend can currently serve."""

    @abstractmethod
    async def health(self) -> bool:
        """Whether the backend is reachable right now."""

    async def aclose(self) -> None:
        """Release held resources (HTTP connections). Safe to call more than once."""
        return None
