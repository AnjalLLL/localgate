"""Operator endpoints behind the dashboard: sign-in, overview, models, sessions,
agent tool-call activity, a memory-retrieval inspector and a search-provider test.

Everything here is under ``/admin`` and needs the admin key — as the
``X-Admin-Key`` header, or the httpOnly session cookie that ``POST
/admin/session`` issues to the dashboard. Nothing is dashboard-private: the same
routes are scriptable with the header.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Query, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from localgate.api.deps import get_session, require_admin
from localgate.backends.base import InferenceBackend
from localgate.core.auth import (
    ADMIN_COOKIE,
    ADMIN_SESSION_SECONDS,
    admin_session_subject,
    constant_time_equals,
    issue_admin_session,
)
from localgate.core.errors import AuthenticationError, LocalgateError, RateLimitError
from localgate.db.repositories.conversations import ConversationRepository, SummaryRepository
from localgate.db.repositories.embeddings import EmbeddingRepository
from localgate.db.repositories.keys import APIKeyRepository
from localgate.db.repositories.ledgers import LedgerRepository
from localgate.db.repositories.tool_calls import ToolCallRepository
from localgate.db.repositories.usage import UsageRepository
from localgate.db.repositories.users import UserRepository
from localgate.memory.retriever import retrieve_relevant_context
from localgate.web.search import WebSearchError, chain_from_settings

session_router = APIRouter(tags=["admin"])
router = APIRouter(tags=["admin"], dependencies=[Depends(require_admin)])


class NotFound(LocalgateError):
    status_code = 404
    error_type = "not_found_error"


class SignIn(BaseModel):
    """Either an account's email and password, or the admin key."""

    email: str | None = Field(default=None, max_length=320)
    password: str | None = Field(default=None, max_length=1024)
    admin_key: str | None = Field(default=None, max_length=1024)


#: Failed sign-ins allowed per client address inside the window before refusing.
_MAX_FAILED_SIGN_INS = 10
_FAILED_SIGN_IN_WINDOW = 10 * 60
_failed_sign_ins: dict[str, list[float]] = defaultdict(list)


def _too_many_failures(client: str, now: float) -> bool:
    recent = [t for t in _failed_sign_ins[client] if now - t < _FAILED_SIGN_IN_WINDOW]
    _failed_sign_ins[client] = recent
    return len(recent) >= _MAX_FAILED_SIGN_INS


@session_router.get("/session/methods")
async def sign_in_methods(session: AsyncSession = Depends(get_session)) -> dict[str, bool]:
    """Public: whether any dashboard account exists, so the sign-in page can say
    how to create the first one. Reveals nothing else."""
    return {"accounts": await UserRepository(session).count() > 0}


@session_router.post("/session", status_code=204)
async def sign_in(
    body: SignIn,
    request: Request,
    response: Response,
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Exchange an email and password (or the admin key) for an httpOnly session
    cookie — the dashboard's login."""
    settings = request.app.state.settings
    now = time.time()
    client = request.client.host if request.client else "unknown"
    if _too_many_failures(client, now):
        raise RateLimitError(
            "Too many failed sign-ins. Wait a few minutes and try again.",
            code="too_many_attempts",
        )
    subject: str | None = None
    if body.email and body.password:
        user = await UserRepository(session).authenticate(body.email, body.password)
        subject = user.email if user is not None else None
    elif body.admin_key and constant_time_equals(body.admin_key, settings.admin_key):
        subject = ""
    if subject is None:
        _failed_sign_ins[client].append(now)
        if body.email:
            raise AuthenticationError("Wrong email or password.", code="invalid_credentials")
        raise AuthenticationError("Invalid admin key.", code="invalid_admin_key")
    _failed_sign_ins.pop(client, None)
    response.set_cookie(
        ADMIN_COOKIE,
        issue_admin_session(settings.admin_key, now, subject=subject),
        max_age=ADMIN_SESSION_SECONDS,
        httponly=True,
        samesite="strict",
        secure=request.url.scheme == "https",
        path="/",
    )
    response.status_code = 204
    return response


@session_router.delete("/session", status_code=204)
async def sign_out(response: Response) -> Response:
    response.delete_cookie(ADMIN_COOKIE, path="/")
    response.status_code = 204
    return response


@router.get("/session")
async def whoami(request: Request) -> dict[str, Any]:
    """200 when the caller is signed in — the dashboard's session check — with the
    account's email when it signed in with one."""
    cookie = request.cookies.get(ADMIN_COOKIE)
    subject = (
        admin_session_subject(cookie, request.app.state.settings.admin_key, time.time())
        if cookie
        else None
    )
    return {"admin": True, "email": subject or None}


@router.get("/overview")
async def overview(request: Request, session: AsyncSession = Depends(get_session)) -> dict:
    """Headline numbers for the overview page, in one round trip."""
    backend: InferenceBackend = request.app.state.backend
    usage = UsageRepository(session)
    keys = await APIKeyRepository(session).list_all()
    tool_rows = await ToolCallRepository(session).recent(limit=500)
    try:
        healthy = await asyncio.wait_for(backend.health(), timeout=3.0)
    except (asyncio.TimeoutError, httpx.HTTPError):
        healthy = False
    tool_ok = sum(1 for r in tool_rows if r.success)
    return {
        "backend": {
            "type": request.app.state.settings.backend_type,
            "url": request.app.state.settings.backend_url,
            "healthy": healthy,
        },
        "usage": await usage.totals(),
        "daily": await usage.daily_totals(days=14),
        "keys": {"total": len(keys), "active": sum(1 for k in keys if not k.revoked)},
        "agent": {
            "recent_tool_calls": len(tool_rows),
            "success_rate": round(tool_ok / len(tool_rows), 3) if tool_rows else None,
        },
        "queue": request.app.state.scheduler.stats(),
    }


@router.get("/models")
async def models(request: Request) -> list[dict[str, Any]]:
    """Models the backend serves, with tool support and context window when the
    backend can report them cheaply."""
    backend: InferenceBackend = request.app.state.backend
    try:
        listed = await backend.list_models_detailed()
    except httpx.HTTPError as exc:
        raise LocalgateError(f"Couldn't list models from the backend: {exc}") from exc
    limit = asyncio.Semaphore(4)

    async def describe(entry: dict[str, Any]) -> dict[str, Any]:
        async with limit:
            try:
                tools, window, caps = await asyncio.gather(
                    backend.check_tool_support(entry["name"]),
                    backend.context_window(entry["name"]),
                    backend.capabilities(entry["name"]),
                )
            except httpx.HTTPError:
                tools, window, caps = None, None, None
        # Embedding-only models can't chat; the dashboard labels them and keeps
        # them out of the chat model picker.
        can_chat = caps is None or "completion" in caps
        return {
            **entry,
            "tool_support": tools,
            "context_window": window,
            "capabilities": caps,
            "can_chat": can_chat,
        }

    return list(await asyncio.gather(*(describe(m) for m in listed)))


@router.get("/sessions")
async def sessions(
    limit: int = Query(default=100, ge=1, le=1000), session: AsyncSession = Depends(get_session)
) -> list[dict]:
    return await ConversationRepository(session).list_all_sessions(limit=limit)


@router.get("/sessions/{api_key_id}/{session_id}")
async def session_detail(
    api_key_id: str,
    session_id: str,
    limit: int = Query(default=300, ge=1, le=2000),
    session: AsyncSession = Depends(get_session),
) -> dict:
    messages = await ConversationRepository(session).recent(session_id, api_key_id, limit=limit)
    ledger = await LedgerRepository(session).get(session_id, api_key_id)
    if not messages and ledger is None:
        raise NotFound(f"No session {session_id!r} for key {api_key_id!r}.")
    summary = await SummaryRepository(session).latest(session_id, api_key_id)
    return {
        "api_key_id": api_key_id,
        "session_id": session_id,
        "messages": [
            {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
            for m in messages
        ],
        "summary": summary.content if summary else None,
        "ledger": ledger,
        "memory_chunks": await EmbeddingRepository(session).count(session_id, api_key_id),
    }


@router.get("/tool-calls")
async def tool_calls(
    limit: int = Query(default=200, ge=1, le=2000), session: AsyncSession = Depends(get_session)
) -> dict:
    """Recent agent tool calls plus success rates per tool and per model — the
    numbers that say which local models are reliable at tool calling."""
    rows = await ToolCallRepository(session).recent(limit=limit)
    by_tool: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_model: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        for bucket, key in ((by_tool, row.tool_name), (by_model, row.model or "unknown")):
            bucket[key][0] += 1
            bucket[key][1] += int(row.success)

    def rates(bucket: dict[str, list[int]]) -> list[dict[str, Any]]:
        return sorted(
            (
                {"name": name, "calls": total, "ok": ok, "success_rate": round(ok / total, 3)}
                for name, (total, ok) in bucket.items()
            ),
            key=lambda item: -item["calls"],
        )

    return {
        "recent": [
            {
                "created_at": r.created_at.isoformat(),
                "tool_name": r.tool_name,
                "model": r.model,
                "session_id": r.session_id,
                "run_id": r.run_id,
                "success": r.success,
                "timed_out": r.timed_out,
                "duration_ms": r.duration_ms,
                "arguments": r.arguments,
                "error": r.error,
            }
            for r in rows
        ],
        "by_tool": rates(by_tool),
        "by_model": rates(by_model),
    }


class MemoryQuery(BaseModel):
    api_key_id: str
    session_id: str
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=8, ge=1, le=50)


@router.post("/memory/search")
async def memory_search(
    body: MemoryQuery, request: Request, session: AsyncSession = Depends(get_session)
) -> dict:
    """What retrieval would recall for ``query`` in one session, with scores —
    for tuning ``memory_min_score`` against real data."""
    settings = request.app.state.settings
    try:
        chunks = await retrieve_relevant_context(
            session,
            request.app.state.backend,
            body.session_id,
            body.api_key_id,
            body.query,
            settings.embedding_model,
            top_k=body.top_k,
            min_score=0.0,
        )
    except httpx.HTTPError as exc:
        raise LocalgateError(f"Embedding failed ({settings.embedding_model}): {exc}") from exc
    return {
        "min_score": settings.memory_min_score,
        "results": [{"score": round(c.score, 4), "content": c.content} for c in chunks],
    }


class SearchProbe(BaseModel):
    query: str = Field(..., min_length=1, max_length=300)


@router.post("/search/test")
async def search_test(body: SearchProbe, request: Request) -> dict:
    """Run the configured search provider chain once — to check the setup works."""
    settings = request.app.state.settings
    try:
        chain = chain_from_settings(settings)
    except ValueError as exc:
        return {"providers": [], "error": str(exc), "results": []}
    started = time.perf_counter()
    try:
        results = await chain.search(body.query)
        error = None
    except WebSearchError as exc:
        results, error = [], str(exc)
    return {
        "providers": chain.names,
        "leaves_machine": chain.leaves_machine,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "error": error,
        "results": [{"title": r.title, "url": r.url, "snippet": r.snippet} for r in results],
    }
