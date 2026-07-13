"""Abstract interface every inference backend must implement."""
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class InferenceBackend(ABC):
    """All backends (Ollama, llama.cpp, vLLM, ...) implement this interface."""

    @abstractmethod
    async def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        """Non-streaming chat completion."""

    @abstractmethod
    async def chat_stream(self, request: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Streaming chat completion (yields SSE chunks)."""

    @abstractmethod
    async def embed(self, text: str, model: str) -> list[float]:
        """Returns an embedding vector for the given text."""

    @abstractmethod
    async def list_models(self) -> list[str]:
        """Return available model names."""

    @abstractmethod
    async def health(self) -> bool:
        """Check if the backend is reachable."""
