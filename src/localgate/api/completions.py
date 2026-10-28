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

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from localgate.api.deps import enforce_rate_limit, get_session
from localgate.core.backend_pool import call_with_failover
from localgate.core.key_policy import clamp_max_tokens, enforce_key_access
from localgate.core.model_info import apply_default_num_ctx
from localgate.core.streaming import ClientGone, until_disconnected
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
    model = settings.resolve_model(body.model)
    await enforce_key_access(api_key, body.model, model, session)
    prompt = body.prompt_text()

    payload = body.model_dump(exclude_none=True, exclude={"prompt", "stream"})
    payload |= {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    clamp_max_tokens(api_key, payload)
    targets = await request.app.state.pool.route(model)
    await apply_default_num_ctx(
        payload, targets[0].backend, model, settings.default_num_ctx, request.app.state.model_info
    )

    started = time.perf_counter()
    try:
        response = await call_with_failover(
            targets,
            api_key.id,
            api_key.max_concurrent,
            lambda target: until_disconnected(request, target.backend.chat(payload)),
        )
    except ClientGone:
        return Response(status_code=499)  # type: ignore[return-value]
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
