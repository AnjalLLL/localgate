"""Wires the coding agent into the same conversation-history and RAG-memory
tables the HTTP `/v1/chat/completions` endpoint uses, so a coding session
survives across `localgate code` invocations the way an API client's
`X-Session-Id` does — retrievable later via the same `ConversationRepository`
`GET /v1/conversations/{session_id}` reads from.

The CLI talks to the backend directly, bypassing HTTP and API-key auth entirely
(see loop.py's module docstring) — but the conversation/memory tables have a
``NOT NULL`` ``api_key_id`` foreign key. Rather than relax that constraint for
every caller, this module provisions one dedicated, reusable ``APIKey`` row to
own local CLI sessions; its raw key is discarded immediately since it's never
used to authenticate anything, only to satisfy the foreign key.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from localgate.backends.base import InferenceBackend
from localgate.config import Settings
from localgate.core.types import ChatMessage
from localgate.db.repositories.conversations import ConversationRepository, SummaryRepository
from localgate.db.repositories.embeddings import EmbeddingRepository
from localgate.db.repositories.keys import APIKeyRepository
from localgate.memory.chunker import chunk_text
from localgate.memory.context_builder import build_augmented_messages
from localgate.memory.embedder import embed_text
from localgate.memory.retriever import retrieve_relevant_context
from localgate.memory.summarizer import maybe_summarize

#: The one APIKey row every local `localgate code` session's history is attributed
#: to. Never sent anywhere as a credential.
LOCAL_AGENT_KEY_NAME = "localgate-code (local)"

_SESSION_MARKER_DIR = ".localgate"
_SESSION_MARKER_FILE = "session_id"


def project_session_id(root: Path) -> str:
    """A session id for this project directory, minted once and reused.

    Persisted at ``.localgate/session_id`` so re-running ``localgate code`` in the
    same project resumes the same memory context automatically, per
    CODING_AGENT_PLAN.md Phase 5.
    """
    marker = root / _SESSION_MARKER_DIR / _SESSION_MARKER_FILE
    if marker.is_file():
        existing = marker.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    session_id = str(uuid.uuid4())
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(session_id + "\n", encoding="utf-8")
    return session_id


async def get_or_create_local_agent_key_id(session: AsyncSession, settings: Settings) -> str:
    """The ``api_key_id`` local CLI sessions are attributed to."""
    repo = APIKeyRepository(session)
    for key in await repo.list_all():
        if key.name == LOCAL_AGENT_KEY_NAME:
            return key.id
    key, _raw_key = await repo.create(LOCAL_AGENT_KEY_NAME, settings.default_rate_limit_per_min)
    return key.id


class AgentMemory:
    """Per-session memory: injects recalled context into outgoing turns, and
    records each turn into conversation history, chunked and embedded, with
    rolling summarization — the same three steps `chat.py` performs per request.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        backend: InferenceBackend,
        settings: Settings,
        session_id: str,
        api_key_id: str,
    ) -> None:
        self._session_factory = session_factory
        self._backend = backend
        self._settings = settings
        self.session_id = session_id
        self._api_key_id = api_key_id

    async def augment(self, messages: list[dict]) -> list[dict]:
        """Retrieve context relevant to the latest user turn and inject it as a
        framed system message, without mutating the caller's own history — the
        same request-scoped augmentation `chat.py` does, applied in-process.
        """
        if not self._settings.memory_enabled:
            return messages
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        query = last_user.get("content") if last_user else None
        if not isinstance(query, str) or not query:
            return messages

        async with self._session_factory() as db_session:
            retrieved = await retrieve_relevant_context(
                db_session,
                self._backend,
                self.session_id,
                query,
                self._settings.embedding_model,
                top_k=self._settings.max_retrieved_chunks,
                min_score=self._settings.memory_min_score,
            )
            summary = await SummaryRepository(db_session).latest(self.session_id)

        if not retrieved and summary is None:
            return messages

        chat_messages = [ChatMessage(**m) for m in messages]
        augmented = build_augmented_messages(
            chat_messages, retrieved, summary.content if summary else None
        )
        return [m.model_dump(exclude_none=True) for m in augmented]

    async def record_turn(self, user_text: str, assistant_text: str) -> None:
        """Persist the turn, chunk and embed it, and fold it into the rolling
        summary once the session has grown past `summarize_after_messages`.
        """
        async with self._session_factory() as db_session:
            convo = ConversationRepository(db_session)
            await convo.add_message(self.session_id, self._api_key_id, "user", user_text)
            if assistant_text:
                await convo.add_message(
                    self.session_id, self._api_key_id, "assistant", assistant_text
                )

            if not self._settings.memory_enabled:
                return

            exchange = f"User: {user_text}\nAssistant: {assistant_text}"
            chunks = chunk_text(exchange, self._settings.chunk_size, self._settings.chunk_overlap)
            embeddings_repo = EmbeddingRepository(db_session)
            for chunk in chunks:
                vector = await embed_text(self._backend, chunk, self._settings.embedding_model)
                await embeddings_repo.add_chunk(
                    self.session_id, self._api_key_id, chunk, vector, kind="turn"
                )

            await maybe_summarize(
                db_session, self._backend, self._settings, self.session_id, self._api_key_id
            )
