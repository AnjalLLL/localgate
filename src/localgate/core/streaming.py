"""Server-Sent Events framing for the OpenAI streaming protocol.

The wire format is fixed by what OpenAI clients expect: each chunk is a line
``data: {json}`` followed by a blank line, and the stream terminates with the
literal sentinel ``data: [DONE]``. Clients block waiting for that sentinel, so
every exit path — success, backend failure, cancellation — has to emit it.
"""

from __future__ import annotations

import json
from typing import Any

DONE = "data: [DONE]\n\n"


def sse_event(payload: dict[str, Any]) -> str:
    """Frame one JSON payload as an SSE ``data:`` event."""
    return f"data: {json.dumps(payload, default=str)}\n\n"


def sse_error(message: str, error_type: str = "backend_error") -> str:
    """Frame an error in the same envelope every other error path uses.

    A stream that has already sent its HTTP 200 headers cannot retroactively
    become a 502, so a mid-stream failure has to be reported *inside* the stream.
    Reusing the ``{"error": {...}}`` shape means a client parses failures the same
    way whether or not it asked for streaming.
    """
    return sse_event({"error": {"message": message, "type": error_type}})


def extract_delta(chunk: dict[str, Any]) -> str:
    """Pull the incremental text out of a streaming chunk.

    Tolerant by design: a chunk carrying only a role or a finish_reason — both of
    which real backends emit — has no content, and that is not an error.
    """
    choices = chunk.get("choices") or [{}]
    delta = choices[0].get("delta") or {}
    return delta.get("content") or ""
