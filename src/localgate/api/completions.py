"""``POST /v1/completions`` — the legacy text-completion endpoint.

Local backends have largely dropped this route in favour of chat completions, so
localgate implements it by translating the prompt into a single user message and
forwarding it to the chat path, then translating the answer back. That keeps older
clients and LangChain's ``OpenAI`` (non-chat) LLM working against any backend,
including ones that never implemented ``/v1/completions`` at all.

The translation is lossless in the direction that matters — a raw prompt is exactly
what a chat model receives as a lone user turn — but note it means completions get
the model's chat template applied, which is what a local instruct model expects
anyway.
"""

from __future__ import annotations

import time

import httpx
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from localgate.api.deps import enforce_rate_limit, get_session
from localgate.core.errors import BackendError, describe_backend_failure
from localgate.core.token_counter import count_tokens
from localgate.core.types import CompletionChoice, CompletionRequest, CompletionResponse, Usage
from localgate.db.models import APIKey
from localgate.db.repositories.usage import UsageRepository

router = APIRouter(tags=["completions"])


@router.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(
    body: CompletionRequest,
    request: Request,
    api_key: APIKey = Depends(enforce_rate_limit),
    session: AsyncSession = Depends(get_session),
) -> CompletionResponse:
    settings = request.app.state.settings
    backend = request.app.state.backend
    model = settings.resolve_model(body.model)
    prompt = body.prompt_text()

    payload = body.model_dump(exclude_none=True, exclude={"prompt", "stream"})
    payload |= {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    started = time.perf_counter()
    try:
        response = await backend.chat(payload)
    except httpx.HTTPError as exc:
        raise BackendError(
            describe_backend_failure(exc, settings.backend_url, settings.backend_type)
        ) from exc
    latency_ms = int((time.perf_counter() - started) * 1000)

    text = (response.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    usage = response.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or count_tokens(prompt))
    completion_tokens = int(usage.get("completion_tokens") or count_tokens(text))

    await UsageRepository(session).record(
        api_key_id=api_key.id,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
    )

    return CompletionResponse(
        model=model,
        choices=[CompletionChoice(text=text)],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
