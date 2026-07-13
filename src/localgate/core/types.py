"""Typed request/response schemas for the OpenAI-compatible surface.

Two rules govern this module:

1. **Requests are validated, never trusted.** Every client-supplied body is parsed
   into one of these models before any handler touches it, so a malformed body is
   a 422 from FastAPI rather than a ``KeyError`` deep in a route.

2. **Unknown fields are passed through, not rejected.** Inference backends keep
   adding sampling knobs (``top_k``, ``min_p``, ``repeat_penalty``, ...) and a
   gateway that dropped them would quietly change the caller's results. The
   request models therefore allow extras and :meth:`ChatCompletionRequest.to_backend_payload`
   forwards them verbatim.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool", "function"]


class ChatMessage(BaseModel):
    """One message in a chat conversation."""

    model_config = ConfigDict(extra="allow")

    role: Role
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None

    def text(self) -> str:
        """Flattens content to plain text.

        Content is a list rather than a string for multimodal messages
        (``[{"type": "text", ...}, {"type": "image_url", ...}]``). Memory and
        token accounting only deal in text, so non-text parts are dropped here.
        """
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        return " ".join(
            part.get("text", "") for part in self.content if part.get("type") == "text"
        ).strip()


class ChatCompletionRequest(BaseModel):
    """Body of ``POST /v1/chat/completions``."""

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[ChatMessage] = Field(..., min_length=1)
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, gt=0)
    stop: str | list[str] | None = None
    user: str | None = None

    def latest_user_text(self) -> str:
        """The text of the most recent user message — the RAG retrieval query."""
        return next((m.text() for m in reversed(self.messages) if m.role == "user"), "")

    def to_backend_payload(self, messages: list[ChatMessage], model: str) -> dict[str, Any]:
        """Rebuilds the outgoing body, substituting the resolved model and
        (memory-augmented) messages while preserving every other field the
        caller sent, including ones this schema doesn't name.
        """
        payload = self.model_dump(exclude_none=True)
        payload["model"] = model
        payload["messages"] = [m.model_dump(exclude_none=True) for m in messages]
        return payload


class CompletionRequest(BaseModel):
    """Body of the legacy ``POST /v1/completions``."""

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    prompt: str | list[str]
    stream: bool = False
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)

    def prompt_text(self) -> str:
        """Collapses the ``list[str]`` batch form into a single prompt.

        The batch form of ``/v1/completions`` is a legacy OpenAI affordance that
        no local backend implements; joining is closer to the caller's intent
        than silently using only the first element.
        """
        if isinstance(self.prompt, list):
            return "\n".join(self.prompt)
        return self.prompt


class EmbeddingsRequest(BaseModel):
    """Body of ``POST /v1/embeddings``."""

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    input: str | list[str]

    def inputs(self) -> list[str]:
        return [self.input] if isinstance(self.input, str) else list(self.input)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "localgate"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class EmbeddingItem(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float]


class EmbeddingsResponse(BaseModel):
    object: Literal["list"] = "list"
    model: str
    data: list[EmbeddingItem]
    usage: Usage


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str | None = "stop"


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: Usage
