"""POST /v1/chat/completions — the main OpenAI-compatible endpoint.

Full pipeline per request:
  1. Authenticate the API key, enforce its rate limit
  2. Retrieve relevant memory chunks for this session and inject them as context
  3. Forward the augmented request to the configured backend
  4. Store this turn (user + assistant) as new conversation history and memory chunks
  5. Count tokens and record usage against the calling key
"""
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from localgate.api.deps import get_session, require_api_key
from localgate.core.token_counter import count_message_tokens, count_tokens
from localgate.db.models import APIKey
from localgate.db.repositories.conversations import ConversationRepository
from localgate.db.repositories.embeddings import EmbeddingRepository
from localgate.db.repositories.usage import UsageRepository
from localgate.memory.chunker import chunk_text
from localgate.memory.context_builder import build_augmented_messages
from localgate.memory.embedder import embed_text
from localgate.memory.retriever import retrieve_relevant_context

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    api_key: APIKey = Depends(require_api_key),
    session: AsyncSession = Depends(get_session),
) -> Any:
    settings = request.app.state.settings
    backend = request.app.state.backend
    limiter = request.app.state.rate_limiter

    if not limiter.allow(api_key.id, api_key.rate_limit_per_min):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")

    body = await request.json()
    messages: list[dict] = body["messages"]
    model = body.get("model", settings.default_model)
    session_id = request.headers.get("x-session-id") or str(uuid.uuid4())

    latest_user_message = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

    # --- 2. Retrieve + inject memory ---
    augmented_messages = messages
    if settings.memory_enabled and latest_user_message:
        retrieved = await retrieve_relevant_context(
            session=session,
            backend=backend,
            session_id=session_id,
            query=latest_user_message,
            embedding_model=settings.embedding_model,
            top_k=settings.max_retrieved_chunks,
        )
        augmented_messages = build_augmented_messages(messages, retrieved)

    outgoing_request = {**body, "messages": augmented_messages}

    # --- 3. Forward to backend (streaming path) ---
    if body.get("stream"):
        async def event_stream():
            assistant_text = ""
            async for chunk in backend.chat_stream(outgoing_request):
                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                assistant_text += delta
                import json
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
            await _record_turn(
                session, api_key, session_id, model, latest_user_message, assistant_text, settings, backend
            )

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # --- 3. Forward to backend (non-streaming path) ---
    response = await backend.chat(outgoing_request)
    assistant_text = response.get("choices", [{}])[0].get("message", {}).get("content", "")

    await _record_turn(
        session,
        api_key,
        session_id,
        model,
        latest_user_message,
        assistant_text,
        settings,
        backend,
        reported_usage=response.get("usage"),
    )

    return response


async def _record_turn(
    session: AsyncSession,
    api_key: APIKey,
    session_id: str,
    model: str,
    user_text: str,
    assistant_text: str,
    settings: Any,
    backend: Any,
    reported_usage: dict | None = None,
) -> None:
    """Stores the turn as conversation history + memory chunks, and records token usage."""
    convo_repo = ConversationRepository(session)
    await convo_repo.add_message(session_id, api_key.id, "user", user_text)
    await convo_repo.add_message(session_id, api_key.id, "assistant", assistant_text)

    if settings.memory_enabled:
        embed_repo = EmbeddingRepository(session)
        exchange = f"User: {user_text}\nAssistant: {assistant_text}"
        for chunk in chunk_text(exchange, settings.chunk_size, settings.chunk_overlap):
            vector = await embed_text(backend, chunk, settings.embedding_model)
            await embed_repo.add_chunk(session_id, api_key.id, chunk, vector)

    usage_repo = UsageRepository(session)
    if reported_usage:
        # Prefer the backend's own count — it reflects the full augmented prompt
        # (including injected memory context) and the model's real tokenizer.
        prompt_tokens = reported_usage.get("prompt_tokens", 0)
        completion_tokens = reported_usage.get("completion_tokens", 0)
    else:
        # Fallback approximation (e.g. for streaming, or backends that omit usage).
        prompt_tokens = count_tokens(user_text)
        completion_tokens = count_tokens(assistant_text)
    await usage_repo.record(api_key.id, model, prompt_tokens, completion_tokens)
