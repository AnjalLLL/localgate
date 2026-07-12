"""POST /v1/chat/completions — the main OpenAI-compatible endpoint.

Phase-1 version: no auth, no memory injection, no token accounting yet.
Just proves the request/response round-trip through a real backend.
"""
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    body = await request.json()
    backend = request.app.state.backend

    if body.get("stream"):
        async def event_stream():
            async for chunk in backend.chat_stream(body):
                import json
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return await backend.chat(body)
