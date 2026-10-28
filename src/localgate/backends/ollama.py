"""Ollama adapter.

Chat goes through Ollama's OpenAI-compatible ``/v1/*`` surface by default, so no
translation is needed and the generic transport applies unchanged. The one
thing that surface cannot do is set per-request model options — above all
``num_ctx``. Without it Ollama runs every model at its small server default and
silently drops the *oldest* tokens of a long prompt: the system prompt, the tool
rules and the user's original request. So a request carrying ``options`` or
``format`` (Ollama-native extensions) goes to native ``/api/chat`` instead and is
translated to and from the OpenAI shape here, invisibly to callers. Embeddings
are the one exception: Ollama's ``/v1/embeddings`` shim has been less reliable
across releases than its native ``/api/embeddings``, so this backend uses the
native route, which takes ``prompt`` and returns a bare ``embedding`` rather
than OpenAI's ``input``/``data[]`` shape.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from localgate.backends.openai_compat import OpenAICompatBackend, _raise_for_stream_status

#: OpenAI sampling fields and their Ollama ``options`` equivalents.
_OPTION_FIELDS = {
    "max_tokens": "num_predict",
    "temperature": "temperature",
    "top_p": "top_p",
    "stop": "stop",
    "seed": "seed",
}


def _human_size(num_bytes: object) -> str | None:
    """``4.7GB`` rather than a raw byte count — the picker needs this at a
    glance, since picking a 7B vs 34B model is a real tradeoff to see up front.
    """
    if not isinstance(num_bytes, int | float) or num_bytes <= 0:
        return None
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _text_content(content: Any) -> str:
    if isinstance(content, list):
        return " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return content or ""


def _native_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw) if raw else {}
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def to_native_request(request: dict[str, Any], *, stream: bool) -> dict[str, Any]:
    """An OpenAI-shaped chat request as an ``/api/chat`` body."""
    messages: list[dict[str, Any]] = []
    for message in request.get("messages", []):
        item: dict[str, Any] = {
            "role": message.get("role", "user"),
            "content": _text_content(message.get("content")),
        }
        if message.get("tool_calls"):
            item["tool_calls"] = [
                {
                    "function": {
                        "name": (call.get("function") or {}).get("name", ""),
                        "arguments": _native_arguments(
                            (call.get("function") or {}).get("arguments")
                        ),
                    }
                }
                for call in message["tool_calls"]
            ]
        if message.get("role") == "tool" and message.get("name"):
            item["tool_name"] = message["name"]
        messages.append(item)

    options = dict(request.get("options") or {})
    for openai_key, native_key in _OPTION_FIELDS.items():
        if request.get(openai_key) is not None:
            options.setdefault(native_key, request[openai_key])

    body: dict[str, Any] = {
        "model": request.get("model"),
        "messages": messages,
        "stream": stream,
        "options": options,
    }
    if request.get("tools"):
        body["tools"] = request["tools"]
    for key in ("format", "keep_alive"):
        if request.get(key) is not None:
            body[key] = request[key]
    return body


def _openai_tool_calls(native: list[dict[str, Any]], start: int = 0) -> list[dict[str, Any]]:
    return [
        {
            "index": start + offset,
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {
                "name": (call.get("function") or {}).get("name", ""),
                "arguments": json.dumps((call.get("function") or {}).get("arguments") or {}),
            },
        }
        for offset, call in enumerate(native)
    ]


def _finish_reason(body: dict[str, Any], has_tool_calls: bool) -> str:
    if has_tool_calls:
        return "tool_calls"
    return "length" if body.get("done_reason") == "length" else "stop"


def _usage(body: dict[str, Any]) -> dict[str, int]:
    prompt = int(body.get("prompt_eval_count") or 0)
    completion = int(body.get("eval_count") or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def from_native_response(body: dict[str, Any]) -> dict[str, Any]:
    """An ``/api/chat`` response as an OpenAI chat completion."""
    native = body.get("message") or {}
    message: dict[str, Any] = {"role": "assistant", "content": native.get("content") or ""}
    tool_calls = _openai_tool_calls(native.get("tool_calls") or [])
    if tool_calls:
        for call in tool_calls:
            call.pop("index")
        message["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "model": body.get("model"),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": _finish_reason(body, bool(tool_calls)),
            }
        ],
        "usage": _usage(body),
    }


class OllamaBackend(OpenAICompatBackend):
    name = "ollama"
    default_base_url = "http://localhost:11434"
    accepts_options = True

    @staticmethod
    def _wants_native(request: dict[str, Any]) -> bool:
        return bool(request.get("options")) or request.get("format") is not None

    async def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self._wants_native(request):
            return await super().chat(request)
        resp = await self._client.post("/api/chat", json=to_native_request(request, stream=False))
        resp.raise_for_status()
        return from_native_response(resp.json())

    async def chat_stream(self, request: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        if not self._wants_native(request):
            async for chunk in super().chat_stream(request):
                yield chunk
            return
        body = to_native_request(request, stream=True)
        tool_index = 0
        async with self._client.stream("POST", "/api/chat", json=body) as resp:
            await _raise_for_stream_status(resp)
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    frame = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if frame.get("error"):
                    raise httpx.HTTPStatusError(
                        str(frame["error"]), request=resp.request, response=resp
                    )
                native = frame.get("message") or {}
                delta: dict[str, Any] = {}
                if native.get("content"):
                    delta["content"] = native["content"]
                if native.get("tool_calls"):
                    calls = _openai_tool_calls(native["tool_calls"], tool_index)
                    tool_index += len(calls)
                    delta["tool_calls"] = calls
                out: dict[str, Any] = {"choices": [{"index": 0, "delta": delta}]}
                if frame.get("done"):
                    out["choices"][0]["finish_reason"] = _finish_reason(frame, tool_index > 0)
                    out["usage"] = _usage(frame)
                yield out
                if frame.get("done"):
                    break

    async def embed(self, text: str, model: str) -> list[float]:
        resp = await self._client.post("/api/embeddings", json={"model": model, "prompt": text})
        resp.raise_for_status()
        return resp.json()["embedding"]  # type: ignore[no-any-return]

    async def list_models_detailed(self) -> list[dict[str, Any]]:
        """Ollama's native ``/api/tags`` — unlike the OpenAI-compat ``/v1/models``
        shim, it carries size and quantization, which is what a real picker needs.
        """
        resp = await self._client.get("/api/tags")
        resp.raise_for_status()
        out: list[dict[str, Any]] = []
        for entry in resp.json().get("models", []):
            details = entry.get("details") or {}
            out.append(
                {
                    "name": entry.get("name") or entry.get("model", ""),
                    "size_human": _human_size(entry.get("size")),
                    "parameter_size": details.get("parameter_size"),
                    "quantization": details.get("quantization_level"),
                }
            )
        return out

    async def check_tool_support(self, model: str) -> bool | None:
        """Ollama's ``/api/show`` reports a ``capabilities`` list including
        ``"tools"`` on servers new enough to advertise it — cheap and free of
        the 400-on-first-real-turn surprise a live chat call would risk instead.
        """
        try:
            resp = await self._client.post("/api/show", json={"model": model})
            resp.raise_for_status()
        except httpx.HTTPError:
            return None
        capabilities = resp.json().get("capabilities")
        if not isinstance(capabilities, list):
            return None
        return "tools" in capabilities

    async def capabilities(self, model: str) -> list[str] | None:
        """``/api/show``'s ``capabilities`` list, as reported by the server."""
        try:
            resp = await self._client.post("/api/show", json={"model": model})
            resp.raise_for_status()
        except httpx.HTTPError:
            return None
        caps = resp.json().get("capabilities")
        return [str(c) for c in caps] if isinstance(caps, list) else None

    async def context_window(self, model: str) -> int | None:
        """``/api/show``'s ``model_info`` carries an architecture-prefixed
        ``<arch>.context_length`` key (e.g. ``qwen2.context_length``) — the key
        name varies by model family, so this scans for the suffix rather than
        hardcoding one architecture.
        """
        try:
            resp = await self._client.post("/api/show", json={"model": model})
            resp.raise_for_status()
        except httpx.HTTPError:
            return None
        info = resp.json().get("model_info")
        if not isinstance(info, dict):
            return None
        for key, value in info.items():
            if key.endswith(".context_length") and isinstance(value, int):
                return value
        return None
