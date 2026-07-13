"""``POST /v1/chat/completions`` — the main OpenAI-compatible endpoint.

The pipeline, in order:

1. Authenticate the key and charge the request against its rate limit (dependencies).
2. Resolve the requested model name through the alias table.
3. Retrieve this session's relevant memory and inject it into the prompt.
4. Serve from the prompt cache, or forward to the configured backend.
5. Persist the turn as history and as embedded memory; summarize if the session is long.
6. Record token usage against the calling key.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from localgate.api.deps import enforce_rate_limit, get_session
from localgate.backends.base import InferenceBackend
from localgate.config import Settings
from localgate.core import metrics
from localgate.core.cache import cache_key
from localgate.core.errors import BackendError, describe_backend_failure
from localgate.core.logging import get_logger
from localgate.core.streaming import DONE, extract_delta, sse_error, sse_event
from localgate.core.token_counter import count_message_tokens, count_tokens
from localgate.core.types import ChatCompletionRequest, ChatMessage
from localgate.db.models import APIKey
from localgate.db.repositories.conversations import ConversationRepository, SummaryRepository
from localgate.db.repositories.embeddings import EmbeddingRepository
from localgate.db.repositories.usage import UsageRepository
from localgate.memory.chunker import chunk_text
from localgate.memory.context_builder import build_augmented_messages
from localgate.memory.embedder import embed_text
from localgate.memory.retriever import retrieve_relevant_context
from localgate.memory.summarizer import maybe_summarize

router = APIRouter(tags=["chat"])
logger = get_logger(__name__)

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    # Nginx buffers proxied responses by default, which would hold the entire stream
    # until it completed and defeat the point of streaming.
    "X-Accel-Buffering": "no",
}


@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    api_key: APIKey = Depends(enforce_rate_limit),
    session: AsyncSession = Depends(get_session),
    x_session_id: str | None = Header(
        default=None,
        description="Groups requests into one conversation for memory. Generated if omitted.",
    ),
) -> Any:
    settings: Settings = request.app.state.settings
    backend: InferenceBackend = request.app.state.backend
    cache = request.app.state.cache
    session_factory: async_sessionmaker[AsyncSession] = request.app.state.session_factory

    session_id = x_session_id or str(uuid.uuid4())
    model = settings.resolve_model(body.model)
    user_text = body.latest_user_text()

    messages = await _augment_with_memory(
        session, backend, settings, session_id, user_text, body.messages
    )
    payload = body.to_backend_payload(messages, model)

    key = cache_key(payload)
    if settings.cache_enabled:
        hit = cache.get(key)
        if hit is not None:
            metrics.cache_events_total.labels(outcome="hit").inc()
            logger.info("cache_hit", model=model, session_id=session_id)

            # A cache hit still happened *to this session*, so it is still a turn: it
            # must be billed to the key and written into the conversation's history and
            # memory. Returning early without recording would under-report usage (a
            # caller could replay a cached prompt for free) and would silently break the
            # session's memory — the next turn would find no trace that this one
            # occurred. The cache saves the inference, not the bookkeeping.
            cached_text = _assistant_text(hit)
            await _record_turn(
                session_factory=session_factory,
                backend=backend,
                settings=settings,
                api_key_id=api_key.id,
                session_id=session_id,
                model=model,
                user_text=user_text,
                assistant_text=cached_text,
                prompt_tokens=_reported(
                    hit, "prompt_tokens", count_message_tokens(payload["messages"])
                ),
                completion_tokens=_reported(hit, "completion_tokens", count_tokens(cached_text)),
                latency_ms=0,  # no inference happened; that is the whole point
                cached=True,
            )
            return _replay_cached(hit, model) if body.stream else hit

        metrics.cache_events_total.labels(outcome="miss").inc()

    started = time.perf_counter()

    if body.stream:
        return StreamingResponse(
            _stream_and_record(
                backend=backend,
                settings=settings,
                cache=cache,
                payload=payload,
                key=key,
                api_key_id=api_key.id,
                session_id=session_id,
                model=model,
                user_text=user_text,
                started=started,
                session_factory=session_factory,
            ),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    try:
        response = await backend.chat(payload)
    except httpx.HTTPError as exc:
        metrics.backend_errors_total.labels(backend=settings.backend_type).inc()
        raise BackendError(
            describe_backend_failure(exc, settings.backend_url, settings.backend_type)
        ) from exc

    latency_ms = int((time.perf_counter() - started) * 1000)
    assistant_text = _assistant_text(response)

    if settings.cache_enabled:
        cache.set(key, response)

    await _record_turn(
        session_factory=session_factory,
        backend=backend,
        settings=settings,
        api_key_id=api_key.id,
        session_id=session_id,
        model=model,
        user_text=user_text,
        assistant_text=assistant_text,
        prompt_tokens=_reported(
            response, "prompt_tokens", count_message_tokens(payload["messages"])
        ),
        completion_tokens=_reported(response, "completion_tokens", count_tokens(assistant_text)),
        latency_ms=latency_ms,
    )
    return response


async def _augment_with_memory(
    session: AsyncSession,
    backend: InferenceBackend,
    settings: Settings,
    session_id: str,
    user_text: str,
    messages: list[ChatMessage],
) -> list[ChatMessage]:
    """Prepend recalled context, or return the messages untouched.

    A failure to *recall* must not fail the request. The user asked a question;
    answering it without memory is a degraded result, but refusing to answer at all
    because the embedding model isn't pulled is a worse one. The failure is logged
    and surfaces in ``/health`` rather than being raised at the caller.
    """
    if not settings.memory_enabled or not user_text:
        return messages

    try:
        retrieved = await retrieve_relevant_context(
            session=session,
            backend=backend,
            session_id=session_id,
            query=user_text,
            embedding_model=settings.embedding_model,
            top_k=settings.max_retrieved_chunks,
            min_score=settings.memory_min_score,
        )
        summary = await SummaryRepository(session).latest(session_id)
    except httpx.HTTPError as exc:
        logger.warning(
            "memory_retrieval_failed",
            session_id=session_id,
            embedding_model=settings.embedding_model,
            error=describe_backend_failure(exc, settings.backend_url, settings.backend_type),
        )
        return messages

    if retrieved:
        # Scores are logged because retrieval quality is otherwise impossible to
        # tune: "the model forgot" and "the model recalled the wrong thing" look
        # identical from the outside, and only the scores tell them apart.
        logger.info(
            "memory_retrieved",
            session_id=session_id,
            chunks=len(retrieved),
            top_score=round(retrieved[0].score, 4),
            lowest_score=round(retrieved[-1].score, 4),
        )

    return build_augmented_messages(messages, retrieved, summary.content if summary else None)


async def _stream_and_record(
    backend: InferenceBackend,
    settings: Settings,
    cache: Any,
    payload: dict[str, Any],
    key: str,
    api_key_id: str,
    session_id: str,
    model: str,
    user_text: str,
    started: float,
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[str]:
    """Relay the backend's stream to the client, then persist the completed turn."""
    assistant_text = ""

    try:
        async for chunk in backend.chat_stream(payload):
            assistant_text += extract_delta(chunk)
            yield sse_event(chunk)
    except httpx.HTTPError as exc:
        metrics.backend_errors_total.labels(backend=settings.backend_type).inc()
        # The 200 and its headers are already on the wire, so this can't become a
        # 502. It has to be reported inside the stream — see core/streaming.py.
        yield sse_error(describe_backend_failure(exc, settings.backend_url, settings.backend_type))
        yield DONE
        return

    yield DONE

    latency_ms = int((time.perf_counter() - started) * 1000)
    if settings.cache_enabled and assistant_text:
        cache.set(key, _synthesize_response(assistant_text, model))

    # A streamed response carries no usage block, so these counts are tiktoken's
    # approximation rather than the model's own tokenizer. Documented as such in
    # docs/api-reference.md.
    await _record_turn(
        session_factory=session_factory,
        backend=backend,
        settings=settings,
        api_key_id=api_key_id,
        session_id=session_id,
        model=model,
        user_text=user_text,
        assistant_text=assistant_text,
        prompt_tokens=count_message_tokens(payload["messages"]),
        completion_tokens=count_tokens(assistant_text),
        latency_ms=latency_ms,
    )


async def _record_turn(
    session_factory: async_sessionmaker[AsyncSession],
    backend: InferenceBackend,
    settings: Settings,
    api_key_id: str,
    session_id: str,
    model: str,
    user_text: str,
    assistant_text: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: int,
    cached: bool = False,
) -> None:
    """Persist history, memory and usage for one completed exchange.

    This opens its own database session instead of reusing the request-scoped one
    because the streaming path calls it *after* the response body has been sent, by
    which point FastAPI has already torn down the request's dependencies — the
    injected session would be closed.
    """
    async with session_factory() as session:
        convo_repo = ConversationRepository(session)
        await convo_repo.add_message(session_id, api_key_id, "user", user_text)
        await convo_repo.add_message(session_id, api_key_id, "assistant", assistant_text)

        if settings.memory_enabled and assistant_text:
            await _store_memory(
                session, backend, settings, session_id, api_key_id, user_text, assistant_text
            )
            await maybe_summarize(session, backend, settings, session_id, api_key_id)

        await UsageRepository(session).record(
            api_key_id=api_key_id,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cached=cached,
        )

    metrics.tokens_total.labels(model=model, direction="prompt").inc(prompt_tokens)
    metrics.tokens_total.labels(model=model, direction="completion").inc(completion_tokens)


async def _store_memory(
    session: AsyncSession,
    backend: InferenceBackend,
    settings: Settings,
    session_id: str,
    api_key_id: str,
    user_text: str,
    assistant_text: str,
) -> None:
    """Chunk and embed the exchange so a later turn can recall it.

    This runs after the client already has its answer, so a failure costs nothing
    the caller can see. Losing one turn from memory is worth a log line, not a
    failed request.
    """
    exchange = f"User: {user_text}\nAssistant: {assistant_text}"
    repo = EmbeddingRepository(session)
    try:
        for chunk in chunk_text(exchange, settings.chunk_size, settings.chunk_overlap):
            vector = await embed_text(backend, chunk, settings.embedding_model)
            await repo.add_chunk(session_id, api_key_id, chunk, vector)
    except httpx.HTTPError as exc:
        logger.warning(
            "memory_write_failed",
            session_id=session_id,
            embedding_model=settings.embedding_model,
            error=describe_backend_failure(exc, settings.backend_url, settings.backend_type),
        )


def _assistant_text(response: dict[str, Any]) -> str:
    return (response.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""


def _reported(response: dict[str, Any], field: str, fallback: int) -> int:
    """Prefer the backend's own token count over our approximation.

    The backend counted with the model's real tokenizer, over the full augmented
    prompt including injected memory. tiktoken's cl100k_base is only a stand-in for
    backends that report nothing.
    """
    value = (response.get("usage") or {}).get(field)
    return int(value) if isinstance(value, int) else fallback


def _synthesize_response(content: str, model: str) -> dict[str, Any]:
    """Rebuild a non-streamed body from a stream we just relayed, so it can be cached."""
    tokens = count_tokens(content)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": tokens, "total_tokens": tokens},
    }


def _replay_cached(response: dict[str, Any], model: str) -> StreamingResponse:
    """Replay a cached completion as SSE frames.

    A cache hit on a request that asked for ``stream: true`` still has to *look*
    like a stream: the client is parsing SSE and blocking on ``[DONE]``. The whole
    body arrives in a single frame, which is honest — there is nothing left to wait
    for.
    """
    content = _assistant_text(response)

    async def replay() -> AsyncIterator[str]:
        yield sse_event(
            {
                "id": response.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        yield DONE

    return StreamingResponse(replay(), media_type="text/event-stream", headers=SSE_HEADERS)
