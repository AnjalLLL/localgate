"""``POST /v1/embeddings`` — exposes the backend's embedding model to clients.

The memory layer calls the backend directly rather than looping back through this
route. It exists so a client that already points an OpenAI SDK at localgate can
embed text through the same base URL and the same key as everything else.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from localgate.api.deps import enforce_rate_limit, get_session
from localgate.core.errors import BackendError, describe_backend_failure
from localgate.core.token_counter import count_tokens
from localgate.core.types import EmbeddingItem, EmbeddingsRequest, EmbeddingsResponse, Usage
from localgate.db.models import APIKey
from localgate.db.repositories.usage import UsageRepository

router = APIRouter(tags=["embeddings"])


@router.post("/v1/embeddings", response_model=EmbeddingsResponse)
async def create_embeddings(
    body: EmbeddingsRequest,
    request: Request,
    api_key: APIKey = Depends(enforce_rate_limit),
    session: AsyncSession = Depends(get_session),
) -> EmbeddingsResponse:
    settings = request.app.state.settings
    backend = request.app.state.backend
    model = settings.resolve_model(body.model) if body.model else settings.embedding_model

    inputs = body.inputs()
    try:
        vectors = [await backend.embed(text, model) for text in inputs]
    except httpx.HTTPError as exc:
        raise BackendError(
            describe_backend_failure(exc, settings.backend_url, settings.backend_type)
        ) from exc

    # Embeddings consume backend capacity like any other call, so they are billed
    # against the key. Leaving them out would let a client embed without limit and
    # would make the usage dashboard under-report the load that key actually caused.
    prompt_tokens = sum(count_tokens(text) for text in inputs)
    await UsageRepository(session).record(
        api_key_id=api_key.id, model=model, prompt_tokens=prompt_tokens, completion_tokens=0
    )

    return EmbeddingsResponse(
        model=model,
        data=[EmbeddingItem(index=i, embedding=vector) for i, vector in enumerate(vectors)],
        usage=Usage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens),
    )
