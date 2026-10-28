"""``POST /v1/chat/completions`` — the main OpenAI-compatible endpoint.

The pipeline, in order:

1. Authenticate the key and charge the request against its rate limit (dependencies).
2. Resolve the requested model name through the alias table.
3. Retrieve this session's relevant memory and inject it into the prompt.
4. Serve from the prompt cache, or forward to the configured backend.
5. Persist the turn as history and as embedded memory; summarize if the session is long.
6. Record token usage against the calling key.

Optional, per request: ``"localgate": {"web_search": true}`` searches the web for the
latest user message, adds the numbered sources to the prompt (framed as reference
material, like recalled memory) and returns them as ``localgate.citations`` — on the
response body, or as a final ``choices: []`` chunk when streaming.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.background import BackgroundTask

from localgate.api.deps import enforce_rate_limit, get_session
from localgate.backends.base import InferenceBackend
from localgate.config import Settings
from localgate.core import metrics
from localgate.core.cache import cache_key
from localgate.core.errors import BackendError, describe_backend_failure
from localgate.core.key_policy import clamp_max_tokens, enforce_key_access
from localgate.core.logging import get_logger
from localgate.core.model_info import apply_default_num_ctx
from localgate.core.scheduler import InferenceScheduler, Lease
from localgate.core.streaming import (
    DONE,
    ClientGone,
    extract_delta,
    sse_error,
    sse_event,
    until_disconnected,
)
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
from localgate.web.research import Source, research, sources_message
from localgate.web.search import WebSearchError, chain_from_settings

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
    await enforce_key_access(api_key, body.model, model, session)
    user_text = body.latest_user_text()

    extension = _pop_extension(body)
    store = api_key.store_conversations is not False
    messages = (
        await _augment_with_memory(
            session, backend, settings, session_id, api_key.id, user_text, body.messages
        )
        if store
        else body.messages
    )
    web: dict[str, Any] | None = None
    if extension.get("web_search"):
        sources, error = await _web_sources(request, settings, backend, user_text)
        web = {"citations": [s.to_dict() for s in sources]}
        if error:
            web["web_search_error"] = error
        if sources:
            messages = _insert_after_system(
                messages, ChatMessage(role="system", content=sources_message(sources)["content"])
            )
    payload = body.to_backend_payload(messages, model)
    clamp_max_tokens(api_key, payload)
    await apply_default_num_ctx(
        payload, backend, model, settings.default_num_ctx, request.app.state.model_info
    )

    # Cache entries are tenant-scoped even when two keys submit identical prompts.
    key = cache_key({"_localgate_api_key_id": api_key.id, **payload})
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
                store_conversations=store,
            )
            return _replay_cached(hit, model) if body.stream else hit

        metrics.cache_events_total.labels(outcome="miss").inc()

    started = time.perf_counter()

    scheduler: InferenceScheduler = request.app.state.scheduler
    lease = await scheduler.acquire(api_key.id, api_key.max_concurrent)

    if body.stream:
        return StreamingResponse(
            _stream_and_record(
                lease=lease,
                store_conversations=store,
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
                web=web,
            ),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
            # If the client is gone before the stream even starts, the generator's
            # own cleanup never runs; this frees the slot anyway (release is idempotent).
            background=BackgroundTask(lease.release),
        )

    try:
        response = await until_disconnected(request, backend.chat(payload))
    except httpx.HTTPError as exc:
        metrics.backend_errors_total.labels(backend=settings.backend_type).inc()
        raise BackendError(
            describe_backend_failure(exc, settings.backend_url, settings.backend_type)
        ) from exc
    except ClientGone:
        return Response(status_code=499)  # nobody is listening; the GPU is freed
    finally:
        lease.release()

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
        store_conversations=store,
    )
    if web is not None:
        response = {**response, "localgate": web}
    return response


def _pop_extension(body: ChatCompletionRequest) -> dict[str, Any]:
    """Remove Localgate's own request options so they never reach the backend."""
    extra = body.model_extra if body.model_extra is not None else {}
    value = extra.pop("localgate", None)
    return value if isinstance(value, dict) else {}


def _insert_after_system(messages: list[ChatMessage], message: ChatMessage) -> list[ChatMessage]:
    leading = 0
    for item in messages:
        if item.role != "system":
            break
        leading += 1
    return [*messages[:leading], message, *messages[leading:]]


async def _web_sources(
    request: Request, settings: Settings, backend: InferenceBackend, question: str
) -> tuple[list[Source], str | None]:
    """Search the web for ``question``. Never raises: chat still answers, just
    without sources, and the reason is reported alongside the response."""
    if not settings.web_search_enabled:
        return [], "web search is disabled on this server (LOCALGATE_WEB_SEARCH_ENABLED)"
    chain = getattr(request.app.state, "search_chain", None)
    try:
        if chain is None:
            chain = chain_from_settings(settings)
            request.app.state.search_chain = chain
        sources = await research(
            question,
            chain,
            backend=backend,
            embedding_model=settings.embedding_model,
            max_sources=settings.web_search_max_sources,
            allow_private=settings.fetch_allow_private,
        )
    except (WebSearchError, ValueError) as exc:
        logger.warning("web_search_failed", error=str(exc))
        return [], str(exc)
    logger.info("web_search_sources", count=len(sources))
    return sources, None


async def _augment_with_memory(
    session: AsyncSession,
    backend: InferenceBackend,
    settings: Settings,
    session_id: str,
    api_key_id: str,
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
            api_key_id=api_key_id,
            query=user_text,
            embedding_model=settings.embedding_model,
            top_k=settings.max_retrieved_chunks,
            min_score=settings.memory_min_score,
        )
        summary = await SummaryRepository(session).latest(session_id, api_key_id)
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
    lease: Lease,
    store_conversations: bool,
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
    web: dict[str, Any] | None = None,
) -> AsyncIterator[str]:
    """Relay the backend's stream to the client, then persist the completed turn.

    If the client disconnects mid-stream, the `finally` closes the upstream
    request — so the GPU stops generating for nobody — frees the inference slot,
    and still bills the tokens produced so far.
    """
    assistant_text = ""
    finished = False
    stream = backend.chat_stream(payload)

    def record(text: str) -> Any:
        return _record_turn(
            session_factory=session_factory,
            backend=backend,
            settings=settings,
            api_key_id=api_key_id,
            session_id=session_id,
            model=model,
            user_text=user_text,
            assistant_text=text,
            # A streamed response carries no usage block, so these counts are
            # tiktoken's approximation rather than the model's own tokenizer.
            # Documented as such in docs/api-reference.md.
            prompt_tokens=count_message_tokens(payload["messages"]),
            completion_tokens=count_tokens(text),
            latency_ms=int((time.perf_counter() - started) * 1000),
            store_conversations=store_conversations,
        )

    try:
        try:
            async for chunk in stream:
                assistant_text += extract_delta(chunk)
                yield sse_event(chunk)
        except httpx.HTTPError as exc:
            metrics.backend_errors_total.labels(backend=settings.backend_type).inc()
            # The 200 and its headers are already on the wire, so this can't become a
            # 502. It has to be reported inside the stream — see core/streaming.py.
            yield sse_error(
                describe_backend_failure(exc, settings.backend_url, settings.backend_type)
            )
            yield DONE
            return

        if web is not None:
            yield sse_event({"object": "chat.completion.chunk", "choices": [], "localgate": web})
        yield DONE
        finished = True
    finally:
        close = getattr(stream, "aclose", None)
        if close is not None:
            with contextlib.suppress(Exception):
                await close()
        lease.release()
        if not finished and assistant_text:
            # Cut short (client gone, or backend failed midway). Bill what was
            # generated. The surrounding task is being cancelled, so the write runs
            # as its own task — awaiting it here would be cancelled with us.
            _keep(asyncio.ensure_future(record(assistant_text)))

    if settings.cache_enabled and assistant_text:
        cache.set(key, _synthesize_response(assistant_text, model))
    await record(assistant_text)


#: Background writes started from cancelled streams, kept referenced until done.
_pending_writes: set[asyncio.Future[Any]] = set()


def _keep(future: asyncio.Future[Any]) -> None:
    _pending_writes.add(future)
    future.add_done_callback(_pending_writes.discard)


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
    store_conversations: bool = True,
) -> None:
    """Persist history, memory and usage for one completed exchange. Keys set not
    to store conversations get usage only — no history, no memory.

    This opens its own database session instead of reusing the request-scoped one
    because the streaming path calls it *after* the response body has been sent, by
    which point FastAPI has already torn down the request's dependencies — the
    injected session would be closed.
    """
    async with session_factory() as session:
        if store_conversations:
            convo_repo = ConversationRepository(session)
            await convo_repo.add_message(session_id, api_key_id, "user", user_text)
            await convo_repo.add_message(session_id, api_key_id, "assistant", assistant_text)

        if store_conversations and settings.memory_enabled and assistant_text:
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
