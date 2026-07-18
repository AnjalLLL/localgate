"""The agent's turn loop: ask the model, run any tool calls, feed results back, repeat.

This talks to an :class:`~localgate.backends.base.InferenceBackend` directly rather
than over HTTP — the same pattern ``cli.py`` already uses for ``localgate health``.
That sidesteps API-key management entirely for what is, for now, a local dev tool
running on the same machine as the backend.

:class:`AgentSession` holds the conversation and runs it turn by turn, so the REPL
can keep one session alive across many user inputs. :func:`run_agent` is a thin
convenience wrapper around a single-turn session, used by the single-shot
``localgate code "task"`` invocation.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from localgate.agent.tools import TOOL_SCHEMAS, ToolCallResult, execute_tool_call
from localgate.backends.base import InferenceBackend

SYSTEM_PROMPT = (
    "You are a coding assistant working directly in the user's project directory. "
    "Use the available tools to inspect and modify the project as needed to complete "
    "the user's task. Prefer reading a file before overwriting it. When you are done, "
    "reply with plain text summarizing what you did — do not call a tool in the same "
    "turn as your final summary. If you cannot use structured tool calls, respond with "
    'a single JSON object of the form {"name": "<tool>", "arguments": {...}} and '
    "nothing else — no prose, no markdown fence."
)

#: Some models (observed live with qwen2.5-coder via Ollama's OpenAI-compat shim)
#: never populate `tool_calls` at all — they print the tool call as plain-text JSON
#: in `content` instead, despite advertising tool-calling support. `_FENCE_RE` and
#: `_TAG_RE` strip the wrappers some other models use for the same failure mode: a
#: markdown code fence, or some fine-tune-specific XML tag. `_TAG_RE` matches any
#: single wrapping tag by name (`<tool_call>`, `<tool_request>`, ...) rather than
#: one hardcoded name — the same model was observed emitting both across
#: otherwise-identical live requests, so a fixed tag name isn't reliable.
_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)
_TAG_RE = re.compile(r"^<(\w+)>\s*(.*?)\s*</\1>$", re.DOTALL)


def _strip_wrapper(text: str) -> str:
    """Peel off one layer of fence and/or tag wrapping, in either order/nesting."""
    text = text.strip()
    for _ in range(2):  # a fence-around-a-tag or tag-around-a-fence is one pass each
        fence_match = _FENCE_RE.match(text)
        if fence_match:
            text = fence_match.group(1).strip()
            continue
        tag_match = _TAG_RE.match(text)
        if tag_match:
            text = tag_match.group(2).strip()
            continue
        break
    return text


def _as_synthetic_tool_call(content: str, known_names: frozenset[str]) -> dict[str, Any] | None:
    """If ``content`` is *only* a disguised tool call, return it in the same shape
    a real ``tool_calls`` entry has. Otherwise return ``None`` — a final answer
    that happens to contain JSON (e.g. showing the user a config file) must not
    be misread as an action the model didn't actually request.

    The whole-string parse is itself the main guard: JSON embedded partway
    through prose fails ``json.loads`` on the full string and falls through here
    to a real answer, rather than requiring separate prose-detection logic.
    """
    cleaned = _strip_wrapper(content)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    name = parsed.get("name")
    arguments: Any = parsed.get("arguments")
    if not isinstance(name, str):
        function = parsed.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            arguments = function.get("arguments")
    if not isinstance(name, str) or name not in known_names:
        return None

    if isinstance(arguments, dict):
        arguments_json = json.dumps(arguments)
    elif isinstance(arguments, str):
        try:
            if not isinstance(json.loads(arguments), dict):
                return None
        except json.JSONDecodeError:
            return None
        arguments_json = arguments
    else:
        return None

    return {
        "id": f"synthetic-{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {"name": name, "arguments": arguments_json},
    }


class AgentTurnLimitExceeded(RuntimeError):
    """Raised when the model keeps calling tools without ever finishing a turn."""


#: Called before a write_file call actually runs: (path, new_content) -> proceed?
ConfirmWrite = Callable[[str, str], bool]
#: Called with a short human-readable line as each tool call happens.
OnEvent = Callable[[str], None]
#: Called with each streamed text fragment as the model produces it.
OnToken = Callable[[str], None]
#: Executes one tool call. Same shape as `tools.execute_tool_call`; Phase 3's
#: expanded tool set plugs in here without AgentSession needing to change.
ToolExecutor = Callable[[Path, str, str, dict[str, Any]], ToolCallResult]
#: Rewrites the outgoing message list for one payload — e.g. injecting recalled
#: memory — without mutating the session's own history. Called fresh before every
#: backend call, same request-scoped augmentation `chat.py` does per HTTP call.
Augment = Callable[[list[dict[str, Any]]], Awaitable[list[dict[str, Any]]]]


def _parse_arguments(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def _stream_completion(
    backend: InferenceBackend, payload: dict[str, Any], on_token: OnToken
) -> dict[str, Any]:
    """Consume a streamed chat completion and reassemble it into one message.

    OpenAI-shaped tool-call deltas arrive fragment by fragment, keyed by index —
    the id and function name are usually whole in the first fragment, but
    `arguments` is typically streamed character by character and must be
    concatenated, not replaced.
    """
    content_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}

    async for chunk in backend.chat_stream(payload):
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}

        text = delta.get("content")
        if text:
            content_parts.append(text)
            on_token(text)

        for tc_delta in delta.get("tool_calls") or []:
            index = tc_delta.get("index", 0)
            entry = tool_calls.setdefault(
                index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
            )
            if tc_delta.get("id"):
                entry["id"] = tc_delta["id"]
            fn_delta = tc_delta.get("function") or {}
            if fn_delta.get("name"):
                entry["function"]["name"] += fn_delta["name"]
            if fn_delta.get("arguments"):
                entry["function"]["arguments"] += fn_delta["arguments"]

    content = "".join(content_parts)
    message: dict[str, Any] = {"role": "assistant", "content": content or None}
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    return message


class AgentSession:
    """One conversation with the model, across as many turns as the caller sends.

    A fresh session starts with just the system prompt; each call to :meth:`send`
    appends a user turn, runs the tool-call loop to completion, and returns the
    model's final text. History accumulates in ``self.messages`` until :meth:`reset`.
    """

    def __init__(
        self,
        backend: InferenceBackend,
        model: str,
        root: Path,
        *,
        confirm_write: ConfirmWrite | None = None,
        on_event: OnEvent | None = None,
        on_token: OnToken | None = None,
        max_turns: int = 20,
        tool_schemas: list[dict[str, Any]] | None = None,
        tool_executor: ToolExecutor | None = None,
        augment: Augment | None = None,
    ) -> None:
        self.backend = backend
        self.model = model
        self.root = root
        self.confirm_write = confirm_write
        self.on_event = on_event
        self.on_token = on_token
        self.max_turns = max_turns
        self.tool_schemas = tool_schemas if tool_schemas is not None else TOOL_SCHEMAS
        self.tool_executor = tool_executor if tool_executor is not None else execute_tool_call
        self.augment = augment
        self._known_tool_names = frozenset(s["function"]["name"] for s in self.tool_schemas)
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def reset(self) -> None:
        """Drop history, starting a new conversation in the same session."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def send(self, user_input: str) -> str:
        """Run one user turn to completion and return the model's final reply."""
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(self.max_turns):
            outgoing = await self.augment(self.messages) if self.augment else self.messages
            payload = {"model": self.model, "messages": outgoing, "tools": self.tool_schemas}
            if self.on_token is not None:
                message = await _stream_completion(self.backend, payload, self.on_token)
            else:
                response = await self.backend.chat(payload)
                message = response["choices"][0]["message"]

            tool_calls = message.get("tool_calls") or []
            if not tool_calls and message.get("content"):
                synthetic = _as_synthetic_tool_call(message["content"], self._known_tool_names)
                if synthetic is not None:
                    if self.on_event is not None:
                        self.on_event(
                            f"(model isn't using structured tool calls — "
                            f"parsed {synthetic['function']['name']} from plain text)"
                        )
                    message = {"role": "assistant", "content": None, "tool_calls": [synthetic]}
                    tool_calls = [synthetic]

            self.messages.append(message)

            if not tool_calls:
                return message.get("content") or ""

            for call in tool_calls:
                self.messages.append(await self._run_tool_call(call))

        raise AgentTurnLimitExceeded(
            f"Stopped after {self.max_turns} turns without a final answer — "
            "the model may be looping."
        )

    async def _run_tool_call(self, call: dict[str, Any]) -> dict[str, Any]:
        fn = call["function"]
        name = fn["name"]
        arguments = _parse_arguments(fn.get("arguments", ""))

        if name == "write_file" and self.confirm_write is not None:
            path = arguments.get("path", "?")
            if not self.confirm_write(path, arguments.get("content", "")):
                return {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": f"User declined to write {path}. Ask before trying again.",
                }

        if self.on_event is not None:
            args_repr = ", ".join(f"{k}={v!r}" for k, v in _summarize(arguments).items())
            self.on_event(f"{name}({args_repr})")

        result = self.tool_executor(self.root, call["id"], name, arguments)
        return {
            "role": "tool",
            "tool_call_id": result.tool_call_id,
            "name": result.name,
            "content": result.content,
        }


async def run_agent(
    backend: InferenceBackend,
    model: str,
    root: Path,
    task: str,
    *,
    confirm_write: ConfirmWrite | None = None,
    on_event: OnEvent | None = None,
    on_token: OnToken | None = None,
    max_turns: int = 20,
    augment: Augment | None = None,
) -> str:
    """Run one task to completion and return the model's final plain-text reply.

    A convenience wrapper around a single-turn :class:`AgentSession`, for the
    single-shot ``localgate code "task"`` invocation.
    """
    session = AgentSession(
        backend,
        model,
        root,
        confirm_write=confirm_write,
        on_event=on_event,
        on_token=on_token,
        max_turns=max_turns,
        augment=augment,
    )
    return await session.send(task)


def _summarize(arguments: dict[str, Any]) -> dict[str, Any]:
    """Truncate long values (e.g. file content) so status lines stay one line."""
    return {
        k: (v if not isinstance(v, str) or len(v) <= 40 else v[:37] + "...")
        for k, v in arguments.items()
    }
