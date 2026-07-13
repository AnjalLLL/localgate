"""Rolling summarization for long-running sessions.

Retrieval alone degrades as a session grows. Chunk-level similarity search can
find *an* exchange, but it loses the through-line — "we already decided on
Postgres", "the user's name is Ana" — because no single chunk states it. A
summary keeps that narrative in bounded space.

Summarization is **incremental**: each run summarizes only the messages newer than
the previous summary and folds the previous summary in as context. Re-summarizing
the entire history every time would make cost grow with the square of session
length, which is precisely the problem this is here to solve.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from localgate.backends.base import InferenceBackend
from localgate.core.logging import get_logger
from localgate.db.models import ConversationSummary
from localgate.db.repositories.conversations import ConversationRepository, SummaryRepository
from localgate.db.repositories.embeddings import EmbeddingRepository
from localgate.memory.embedder import embed_text

logger = get_logger(__name__)

#: Messages left un-summarized at the tail. They are still in the live context
#: window, so summarizing them would only duplicate what the model can already see.
KEEP_RECENT_MESSAGES = 4

SUMMARY_INSTRUCTION = (
    "You are maintaining a running summary of a conversation so that a model with a "
    "small context window can recall it later.\n\n"
    "Write a dense, factual summary in under 200 words. Preserve: decisions reached, "
    "stated preferences and constraints, names, numbers, and unresolved questions. "
    "Drop: pleasantries, restatements, and anything the assistant merely speculated "
    "about. Write plain prose, no preamble, no headings."
)


async def maybe_summarize(
    session: AsyncSession,
    backend: InferenceBackend,
    settings: Any,
    session_id: str,
    api_key_id: str,
) -> ConversationSummary | None:
    """Summarize a session's older turns if it has grown past the threshold.

    Returns the new summary, or ``None`` if the session was short enough to leave
    alone. Never raises: a failure to summarize degrades recall on a future
    request, which is not worth failing the current one over.
    """
    threshold = getattr(settings, "summarize_after_messages", 0)
    if not threshold or not settings.memory_enabled:
        return None

    convo_repo = ConversationRepository(session)
    summary_repo = SummaryRepository(session)

    previous = await summary_repo.latest(session_id)
    pending = await convo_repo.messages_after(
        session_id, previous.covers_until if previous else None
    )

    # Only the messages past the live tail are candidates: the tail is still inside
    # the model's own context window.
    to_summarize = pending[:-KEEP_RECENT_MESSAGES] if len(pending) > KEEP_RECENT_MESSAGES else []
    if len(pending) < threshold or not to_summarize:
        return None

    transcript = "\n".join(f"{m.role}: {m.content}" for m in to_summarize)
    prompt = (
        f"Previous summary:\n{previous.content}\n\nNew messages:\n{transcript}"
        if previous
        else f"Conversation:\n{transcript}"
    )

    try:
        response = await backend.chat(
            {
                "model": settings.resolve_model(None),
                "messages": [
                    {"role": "system", "content": SUMMARY_INSTRUCTION},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,  # a summary should not be creative
            }
        )
        content = (response.get("choices") or [{}])[0].get("message", {}).get("content", "")
    except Exception as exc:  # noqa: BLE001 — see docstring
        logger.warning("summarization_failed", session_id=session_id, error=str(exc))
        return None

    if not content.strip():
        return None

    summary = await summary_repo.add(
        session_id=session_id,
        api_key_id=api_key_id,
        content=content.strip(),
        covers_until=to_summarize[-1].created_at,
        message_count=len(to_summarize),
    )

    # Store the summary as a retrievable chunk too, so a query that matches the
    # gist of an old exchange can surface the summary even when no verbatim chunk
    # scores well.
    try:
        vector = await embed_text(backend, summary.content, settings.embedding_model)
        await EmbeddingRepository(session).add_chunk(
            session_id, api_key_id, summary.content, vector, kind="summary"
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("summary_embedding_failed", session_id=session_id, error=str(exc))

    logger.info(
        "session_summarized",
        session_id=session_id,
        messages_summarized=len(to_summarize),
        summary_chars=len(summary.content),
    )
    return summary
