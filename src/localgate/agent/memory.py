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

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from localgate.backends.base import InferenceBackend
from localgate.config import Settings
from localgate.core.logging import get_logger
from localgate.core.types import ChatMessage
from localgate.db.repositories.conversations import ConversationRepository, SummaryRepository
from localgate.db.repositories.embeddings import EmbeddingRepository
from localgate.db.repositories.keys import APIKeyRepository
from localgate.db.repositories.usage import UsageRepository
from localgate.memory.chunker import chunk_text
from localgate.memory.context_builder import build_augmented_messages
from localgate.memory.embedder import embed_text
from localgate.memory.retriever import retrieve_relevant_context
from localgate.memory.summarizer import maybe_summarize

#: The one APIKey row every local `localgate code` session's history is attributed
#: to. Never sent anywhere as a credential.
logger = get_logger(__name__)

LOCAL_AGENT_KEY_NAME = "localgate-code (local)"

_SESSION_MARKER_DIR = ".localgate"
_SESSION_MARKER_FILE = "session_id"
_SESSIONS_INDEX_FILE = "sessions.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _legacy_marker(root: Path) -> Path:
    return root / _SESSION_MARKER_DIR / _SESSION_MARKER_FILE


def _sessions_index_path(root: Path) -> Path:
    return root / _SESSION_MARKER_DIR / _SESSIONS_INDEX_FILE


def _load_session_store(root: Path) -> dict[str, Any]:
    """The full `{"current": id, "sessions": [{"id", "created_at"}, ...]}` index
    for this project, migrating the old single-id `.localgate/session_id` marker
    (still written alongside for anything else that reads it directly) the first
    time a project with only that legacy file is touched.
    """
    index_path = _sessions_index_path(root)
    if index_path.is_file():
        try:
            store = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            store = None
        if isinstance(store, dict) and isinstance(store.get("sessions"), list):
            return store

    legacy = _legacy_marker(root)
    if legacy.is_file():
        existing = legacy.read_text(encoding="utf-8").strip()
        if existing:
            return {"current": existing, "sessions": [{"id": existing, "created_at": _now_iso()}]}
    return {"current": None, "sessions": []}


def _save_session_store(root: Path, store: dict[str, Any]) -> None:
    index_path = _sessions_index_path(root)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(store, indent=2) + "\n", encoding="utf-8")
    current = store.get("current")
    if current:
        _legacy_marker(root).write_text(f"{current}\n", encoding="utf-8")


def project_session_id(root: Path) -> str:
    """The *current* session id for this project directory, minting one (and
    starting the project's session index) the first time it's called.

    Persisted at ``.localgate/sessions.json`` (with ``.localgate/session_id`` kept
    in sync for backward compatibility) so re-running ``localgate code`` in the
    same project resumes the same memory context automatically, per
    CODING_AGENT_PLAN.md Phase 5. Use `/resume` in the REPL to switch to a
    different past session for this project.
    """
    store = _load_session_store(root)
    if store.get("current"):
        return str(store["current"])
    session_id = str(uuid.uuid4())
    store["sessions"].append({"id": session_id, "created_at": _now_iso()})
    store["current"] = session_id
    _save_session_store(root, store)
    return session_id


def list_project_sessions(root: Path) -> list[dict[str, str]]:
    """Every session id ever used in this project, most-recently-created first."""
    sessions = _load_session_store(root).get("sessions", [])
    return sorted(sessions, key=lambda s: s["created_at"], reverse=True)


def set_current_project_session(root: Path, session_id: str) -> None:
    """Switch this project's "current" session — what the next `project_session_id`
    call (and the next `localgate code` invocation) will resume.
    """
    store = _load_session_store(root)
    if session_id not in {s["id"] for s in store["sessions"]}:
        store["sessions"].append({"id": session_id, "created_at": _now_iso()})
    store["current"] = session_id
    _save_session_store(root, store)


def start_new_project_session(root: Path) -> str:
    """Mint a brand-new session id for this project and make it current."""
    session_id = str(uuid.uuid4())
    set_current_project_session(root, session_id)
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

        try:
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
        except httpx.HTTPError as exc:
            # Memory is an enhancement, never a precondition — `chat.py` treats a
            # failed embedding the same way. Most often the embedding model just
            # isn't pulled; letting that propagate would kill the whole turn, and
            # `cli.py`'s handler would misreport it as "this model doesn't support
            # tool calling".
            logger.warning(
                "agent_memory_retrieval_failed",
                session_id=self.session_id,
                embedding_model=self._settings.embedding_model,
                error=str(exc),
            )
            return messages

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
            try:
                for chunk in chunks:
                    vector = await embed_text(self._backend, chunk, self._settings.embedding_model)
                    await embeddings_repo.add_chunk(
                        self.session_id, self._api_key_id, chunk, vector, kind="turn"
                    )

                await maybe_summarize(
                    db_session, self._backend, self._settings, self.session_id, self._api_key_id
                )
            except httpx.HTTPError as exc:
                # The conversation rows above are already committed, which is the
                # part that matters; only recall for *future* turns is degraded.
                logger.warning(
                    "agent_memory_store_failed",
                    session_id=self.session_id,
                    embedding_model=self._settings.embedding_model,
                    error=str(exc),
                )

    async def record_usage(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Durable token accounting for `localgate code`, so `keys usage` (and any
        future `/usage`-style route) reflects local CLI activity too, not just
        requests that went through the HTTP API. Tokens are the same
        `count_message_tokens`/`count_tokens` approximation `chat.py` falls back to
        when a backend doesn't report its own usage — direct-to-backend CLI calls
        never get a reported count to prefer.
        """
        async with self._session_factory() as db_session:
            await UsageRepository(db_session).record(
                self._api_key_id, model, prompt_tokens, completion_tokens
            )

    async def load_history(self, limit: int = 80) -> list[dict[str, str]]:
        """The stored turns for `self.session_id`, oldest-first, in chat-message
        shape — used by `/resume` to rehydrate `AgentSession.messages`.
        """
        async with self._session_factory() as db_session:
            messages = await ConversationRepository(db_session).recent(self.session_id, limit=limit)
        return [{"role": m.role, "content": m.content} for m in messages]
