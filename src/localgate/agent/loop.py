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

from localgate.agent.mcp import McpRegistry, is_mcp_tool_name
from localgate.agent.tools import (
    DELEGATE_TASK_SCHEMA,
    READ_ONLY_TOOL_NAMES,
    TOOL_SCHEMAS,
    ToolCallResult,
    execute_tool_call,
)
from localgate.agent.websearch import WEB_SEARCH_SCHEMA, SearchFn
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

#: Appended to `SYSTEM_PROMPT` only when the corresponding tool is actually on
#: this session's schema — no point spending tokens (or risking confusion)
#: steering the model around a tool it doesn't have. This is the concrete
#: answer to "does the model decide when to search/delegate on its own?" — yes,
#: and this is the only place that decision gets any guidance beyond each
#: tool's own one-line `description`.
_DELEGATE_GUIDANCE = (
    " Use delegate_task only for a large, self-contained sub-exploration (e.g. "
    '"find every call site of foo() across the project and summarize them") — a '
    "single-file edit or a quick lookup should stay in this turn instead of being delegated."
)
_SEARCH_GUIDANCE = (
    " Use web_search only for information that isn't in this project or your training "
    "data (a library's current version, a recent CVE, docs published after your "
    "training cutoff) — don't search for things you can find by reading the project."
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
        confirm_search: Callable[[str], bool] | None = None,
        confirm_delegate: Callable[[str], bool] | None = None,
        on_event: OnEvent | None = None,
        on_token: OnToken | None = None,
        max_turns: int = 20,
        tool_schemas: list[dict[str, Any]] | None = None,
        tool_executor: ToolExecutor | None = None,
        augment: Augment | None = None,
        allow_delegation: bool = False,
        search_fn: SearchFn | None = None,
        mcp_registry: McpRegistry | None = None,
    ) -> None:
        self.backend = backend
        self.model = model
        self.root = root
        self.confirm_write = confirm_write
        #: Asked before a web_search/delegate_task call actually runs, same
        #: shape as `confirm_write` — `None` means always allowed (e.g. a
        #: delegated sub-agent's own session never re-confirms its parent's
        #: already-approved delegation).
        self.confirm_search = confirm_search
        self.confirm_delegate = confirm_delegate
        self.on_event = on_event
        self.on_token = on_token
        self.max_turns = max_turns
        self.tool_schemas = list(tool_schemas) if tool_schemas is not None else list(TOOL_SCHEMAS)
        #: Whether *this* session may call `delegate_task` — never propagated to
        #: a delegated sub-agent's own session (see `_run_delegated_task`), which
        #: is the depth-1 cap: a sub-agent cannot itself spawn a sub-agent.
        self.allow_delegation = allow_delegation
        if allow_delegation:
            self.tool_schemas.append(DELEGATE_TASK_SCHEMA)
        #: Executes `web_search`, if the CLI configured a provider — `None` means
        #: the tool isn't in `self.tool_schemas` at all, not that it silently no-ops.
        self.search_fn = search_fn
        if search_fn is not None:
            self.tool_schemas.append(WEB_SEARCH_SCHEMA)
        #: Every connected MCP server's tools, if any were configured — see
        #: `agent/mcp.py`. `None`/empty means no `mcp__*` tools are offered.
        self.mcp_registry = mcp_registry
        if mcp_registry is not None:
            self.tool_schemas.extend(mcp_registry.tool_schemas)
        self.tool_executor = tool_executor if tool_executor is not None else execute_tool_call
        self.augment = augment
        self._known_tool_names = frozenset(s["function"]["name"] for s in self.tool_schemas)
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt()}]

    def system_prompt(self) -> str:
        """`SYSTEM_PROMPT` plus tool-specific steering, but only for tools this
        session actually has — see `_DELEGATE_GUIDANCE`/`_SEARCH_GUIDANCE`.
        """
        prompt = SYSTEM_PROMPT
        if self.allow_delegation:
            prompt += _DELEGATE_GUIDANCE
        if self.search_fn is not None:
            prompt += _SEARCH_GUIDANCE
        return prompt

    def reset(self) -> None:
        """Drop history, starting a new conversation in the same session."""
        self.messages = [{"role": "system", "content": self.system_prompt()}]

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

        if name not in self._known_tool_names:
            # Not just a typo guard: this is the actual boundary that keeps a
            # delegated sub-agent confined to its `allowed_tools` — its schema
            # simply excludes anything it wasn't granted, and a model that
            # hallucinates a call for a tool it was never offered (structured
            # `tool_calls` aren't otherwise checked against the schema) hits
            # this rather than `self.tool_executor`.
            return {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": name,
                "content": f"Unknown tool: {name}",
            }

        if name == "write_file" and self.confirm_write is not None:
            path = arguments.get("path", "?")
            if not self.confirm_write(path, arguments.get("content", "")):
                return {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": f"User declined to write {path}. Ask before trying again.",
                }

        if name == "delegate_task" and self.allow_delegation:
            task = arguments.get("task", "")
            if self.confirm_delegate is not None and not self.confirm_delegate(task):
                return {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": "User declined this delegation. Do the sub-task directly instead.",
                }
            if self.on_event is not None:
                self.on_event(f"delegate_task({_summarize(arguments)})")
            content = await self._run_delegated_task(arguments)
            return {"role": "tool", "tool_call_id": call["id"], "name": name, "content": content}

        if name == "web_search" and self.search_fn is not None:
            query = arguments.get("query", "")
            if self.confirm_search is not None and not self.confirm_search(query):
                return {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": "User declined this search. Ask before trying again.",
                }
            if self.on_event is not None:
                self.on_event(f"web_search(query={query!r})")
            content = await self.search_fn(query) if query else "web_search requires a query."
            return {"role": "tool", "tool_call_id": call["id"], "name": name, "content": content}

        if is_mcp_tool_name(name) and self.mcp_registry is not None:
            if self.on_event is not None:
                self.on_event(f"{name}({_summarize(arguments)})")
            content = await self.mcp_registry.call_tool(name, arguments)
            return {"role": "tool", "tool_call_id": call["id"], "name": name, "content": content}

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

    async def _run_delegated_task(self, arguments: dict[str, Any]) -> str:
        """Run one sub-task to completion in a fresh, isolated `AgentSession` and
        return only its final text — the parent's `messages` never see the
        sub-agent's own tool calls, just this summary.

        Shares `root`, `confirm_write`, and `tool_executor` with the parent, so a
        delegated write hits the same path-safety checks and the same
        confirmation UI as any other write — delegation changes who's asking,
        not whether it's asked.
        """
        task = arguments.get("task")
        if not isinstance(task, str) or not task.strip():
            return "delegate_task requires a non-empty 'task' string."

        requested = arguments.get("allowed_tools")
        if isinstance(requested, list) and requested and all(isinstance(t, str) for t in requested):
            allowed = frozenset(requested) & self._known_tool_names - {"delegate_task"}
            if not allowed:
                allowed = READ_ONLY_TOOL_NAMES
        else:
            allowed = READ_ONLY_TOOL_NAMES

        requested_turns = arguments.get("max_turns")
        sub_max_turns = 10
        if isinstance(requested_turns, int) and requested_turns > 0:
            sub_max_turns = requested_turns

        sub_schemas = [s for s in TOOL_SCHEMAS if s["function"]["name"] in allowed]
        sub_confirm_write = self.confirm_write if "write_file" in allowed else None

        sub_session = AgentSession(
            self.backend,
            self.model,
            self.root,
            confirm_write=sub_confirm_write,
            tool_executor=self.tool_executor,
            max_turns=sub_max_turns,
            tool_schemas=sub_schemas,
            allow_delegation=False,  # depth 1 — a sub-agent cannot itself delegate
        )
        try:
            return await sub_session.send(task)
        except AgentTurnLimitExceeded as exc:
            return f"Sub-agent stopped without finishing: {exc}"


async def run_agent(
    backend: InferenceBackend,
    model: str,
    root: Path,
    task: str,
    *,
    confirm_write: ConfirmWrite | None = None,
    confirm_search: Callable[[str], bool] | None = None,
    confirm_delegate: Callable[[str], bool] | None = None,
    on_event: OnEvent | None = None,
    on_token: OnToken | None = None,
    max_turns: int = 20,
    augment: Augment | None = None,
    allow_delegation: bool = False,
    search_fn: SearchFn | None = None,
    mcp_registry: McpRegistry | None = None,
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
        confirm_search=confirm_search,
        confirm_delegate=confirm_delegate,
        on_event=on_event,
        on_token=on_token,
        max_turns=max_turns,
        augment=augment,
        allow_delegation=allow_delegation,
        search_fn=search_fn,
        mcp_registry=mcp_registry,
    )
    return await session.send(task)


def _summarize(arguments: dict[str, Any]) -> dict[str, Any]:
    """Truncate long values (e.g. file content) so status lines stay one line."""
    return {
        k: (v if not isinstance(v, str) or len(v) <= 40 else v[:37] + "...")
        for k, v in arguments.items()
    }
