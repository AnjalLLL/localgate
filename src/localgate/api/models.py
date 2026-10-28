"""``GET /v1/models`` — the models this gateway can serve."""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends, Request

from localgate.api.deps import require_api_key
from localgate.core.errors import BackendError, describe_backend_failure
from localgate.core.key_policy import model_allowed
from localgate.core.types import ModelCard, ModelList
from localgate.db.models import APIKey

router = APIRouter(tags=["models"], dependencies=[Depends(require_api_key)])


@router.get("/v1/models", response_model=ModelList, response_model_exclude_none=True)
async def list_models(request: Request, api_key: APIKey = Depends(require_api_key)) -> ModelList:
    """List the backend's models, plus any aliases configured on the gateway.

    Aliases appear alongside the real model ids because a caller passing
    ``model: "fast"`` needs to see that "fast" is a name they can actually use. A
    listing that showed only the backend's own names would make every alias look
    unsupported.
    """
    settings = request.app.state.settings
    backend = request.app.state.backend

    try:
        model_ids = await backend.list_models()
    except httpx.HTTPError as exc:
        raise BackendError(
            describe_backend_failure(exc, settings.backend_url, settings.backend_type)
        ) from exc

    info = request.app.state.model_info
    cards = [
        ModelCard(
            id=model_id,
            owned_by=settings.backend_type,
            **(await info.get(backend, model_id)).as_card_fields(),
        )
        for model_id in model_ids
    ]
    cards.extend(
        ModelCard(id=alias, owned_by=f"localgate-alias:{target}")
        for alias, target in settings.model_aliases.items()
    )
    # A key limited to some models sees only those — the listing is what it can use.
    cards = [c for c in cards if model_allowed(api_key, c.id, settings.resolve_model(c.id))]
    return ModelList(data=cards)
