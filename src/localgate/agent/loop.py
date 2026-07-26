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
    list_directory,
    read_file,
)
from localgate.agent.websearch import WEB_SEARCH_SCHEMA, SearchFn
from localgate.backends.base import InferenceBackend

SYSTEM_PROMPT = (
    "You are a coding agent. Execute tasks by calling tools.\n"
    "RULES:\n"
    "- Call ONE tool per response. No explanation, no markdown, no code blocks.\n"
    "- read_file BEFORE write_file. Always read first, then write the full file.\n"
    "- Write COMPLETE code. No placeholders like '/* add here */' or '...'.\n"
    "- When finished, say only: Done: <1 sentence summary>\n"
    'FORMAT: {"name": "tool_name", "arguments": {...}}\n'
    "Respond with ONLY the JSON above. Nothing else."
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
#: Matches a JSON object embedded anywhere in prose — used as a fallback when the
#: whole-string parse fails but the model clearly emitted a tool call inside text.
_EMBEDDED_JSON_RE = re.compile(
    r'\{[^{}]*"name"\s*:\s*"[^"]+?"[^{}]*"arguments"\s*:\s*\{', re.DOTALL
)


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


def _extract_json_object(text: str, start: int) -> str | None:
    """Extract a balanced JSON object starting at `start` (which must be '{')."""
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            if in_string:
                escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _try_parse_tool_call(text: str, known_names: frozenset[str]) -> dict[str, Any] | None:
    """Try to parse a single JSON object as a tool call."""
    try:
        parsed = json.loads(text)
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


def _as_synthetic_tool_call(content: str, known_names: frozenset[str]) -> dict[str, Any] | None:
    """Extract a tool call from model output — first tries the whole string
    (wrapped or not), then scans for an embedded JSON tool call in prose.

    Small models often mix prose with a JSON tool call. We extract the FIRST
    valid tool call found anywhere in the output, so the agent can still act
    even when the model won't shut up.
    """
    # First: try the whole string (original strict path)
    cleaned = _strip_wrapper(content)
    result = _try_parse_tool_call(cleaned, known_names)
    if result is not None:
        return result

    # Second: try to find a code fence containing a tool call
    for fence_match in re.finditer(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL):
        inner = fence_match.group(1).strip()
        result = _try_parse_tool_call(inner, known_names)
        if result is not None:
            return result

    # Third: scan for embedded JSON objects that look like tool calls
    for match in _EMBEDDED_JSON_RE.finditer(content):
        obj_str = _extract_json_object(content, match.start())
        if obj_str is not None:
            result = _try_parse_tool_call(obj_str, known_names)
            if result is not None:
                return result

    # Fourth: extract a named code block as a write_file call.
    # Models often say "### `contact.html`\n```html\n...code...\n```"
    # which IS a write_file action expressed as prose.
    if "write_file" in known_names:
        write = _extract_code_block_as_write(content)
        if write is not None:
            return write

    return None


#: Matches a filename hint before a code block. Handles real model output like:
#: "### File: `src/contact.html`\n\n```html\n...\n```"
#: "### index.html\n```html\n...\n```"
#: "#### `styles.css`:\n\n```css\n...\n```"
_FILENAME_HINT_RE = re.compile(
    r"(?:^|\n)"
    r"[#*/ <!-]*"
    r"(?:File:\s*|file:\s*)?"
    r"[`\"']?"
    r"((?:src/|\./)?"
    r"[\w][\w./-]*\."
    r"(?:html|css|js|jsx|tsx|ts|py|json|md|txt|yml|yaml|toml|cfg|ini|sh))"
    r"[`\"']?"
    r"[: )*#>-]*"
    r"\s*\n"
    r"\n?"
    r"```\w*\s*\n"
    r"(.*?)"
    r"\n```",
    re.DOTALL,
)


def _extract_code_block_as_write(content: str) -> dict[str, Any] | None:
    """If model output contains a named code block (filename hint + fence),
    synthesize a write_file tool call from the first one found.
    """
    for match in _FILENAME_HINT_RE.finditer(content):
        path = match.group(1).strip()
        code = match.group(2)
        if len(code.strip()) < 50:
            continue
        return {
            "id": f"synthetic-{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": "write_file",
                "arguments": json.dumps({"path": path, "content": code}),
            },
        }
    return None


def _extract_all_code_blocks_as_writes(content: str) -> list[dict[str, Any]]:
    """Extract ALL named code blocks from model output as write_file calls."""
    writes: list[dict[str, Any]] = []
    for match in _FILENAME_HINT_RE.finditer(content):
        path = match.group(1).strip()
        code = match.group(2)
        if len(code.strip()) < 50:
            continue
        writes.append(
            {
                "id": f"synthetic-{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps({"path": path, "content": code}),
                },
            }
        )
    return writes


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
    backend: InferenceBackend,
    payload: dict[str, Any],
    on_token: OnToken,
    known_tool_names: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Consume a streamed chat completion and reassemble it into one message.

    OpenAI-shaped tool-call deltas arrive fragment by fragment, keyed by index —
    the id and function name are usually whole in the first fragment, but
    `arguments` is typically streamed character by character and must be
    concatenated, not replaced.

    Early-stops if the accumulated text content contains a complete tool call
    JSON — prevents the model from generating thousands of tokens of prose
    when it already emitted the one action we need.
    """
    content_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    content_len = 0
    early_stopped = False

    async for chunk in backend.chat_stream(payload):
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}

        text = delta.get("content")
        if text:
            content_parts.append(text)
            content_len += len(text)
            on_token(text)

            # Early stop: if content is getting long and contains a JSON tool call,
            # stop consuming the stream. But NOT if we're inside a code fence
            # (model may be writing file contents as named blocks).
            if known_tool_names and not tool_calls and content_len > 200:
                joined = "".join(content_parts)
                if _EMBEDDED_JSON_RE.search(joined) and "```" not in joined:
                    early_stopped = True
                    break

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
    if early_stopped:
        message["_early_stopped"] = True
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
        auto_read: bool = False,
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
        self.auto_read = auto_read
        self._known_tool_names = frozenset(s["function"]["name"] for s in self.tool_schemas)
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt()}]

    def system_prompt(self) -> str:
        """`SYSTEM_PROMPT` plus tool-specific steering and the project root path."""
        prompt = SYSTEM_PROMPT
        prompt += f"\n\nProject directory: {self.root}"
        if self.allow_delegation:
            prompt += _DELEGATE_GUIDANCE
        if self.search_fn is not None:
            prompt += _SEARCH_GUIDANCE
        return prompt

    def reset(self) -> None:
        """Drop history, starting a new conversation in the same session."""
        self.messages = [{"role": "system", "content": self.system_prompt()}]

    def _build_project_context(self) -> str:
        """Build a compact project context to inject on the first turn."""
        parts: list[str] = [f"## Project: {self.root}\n\nFiles in this directory:"]
        try:
            entries = list_directory(self.root, ".")
        except (OSError, ValueError):
            return ""
        for entry in entries[:30]:
            parts.append(entry)

        # Auto-read small key files at root level
        _AUTO_READ_EXTS = {".html", ".py", ".js", ".ts", ".css", ".jsx", ".tsx"}
        _MAX_FILE_LINES = 150
        _MAX_TOTAL_CHARS = 4000
        total_chars = 0
        for entry in entries:
            if "/" in entry:
                continue
            ext = Path(entry).suffix.lower()
            if ext not in _AUTO_READ_EXTS:
                continue
            try:
                content = read_file(self.root, entry)
            except (OSError, ValueError):
                continue
            lines = content.splitlines()
            if len(lines) > _MAX_FILE_LINES:
                continue
            if total_chars + len(content) > _MAX_TOTAL_CHARS:
                break
            total_chars += len(content)
            parts.append(f"\nContents of {entry}:\n```\n{content}\n```")

        return "\n".join(parts)

    async def send(self, user_input: str) -> str:
        """Run one user turn to completion and return the model's final reply.

        Output is buffered silently during the tool loop — only tool events
        (via on_event) are shown. The final answer is streamed at the end.
        """
        if len(self.messages) == 1:
            context = self._build_project_context()
            if context:
                user_input = f"{context}\n\nUser request: {user_input}"
            if self.auto_read:
                self._auto_read_first_file()
        self.messages.append({"role": "user", "content": user_input})

        def _noop(_text: str) -> None:
            pass

        for _ in range(self.max_turns):
            outgoing = await self.augment(self.messages) if self.augment else self.messages
            payload = {
                "model": self.model,
                "messages": outgoing,
                "tools": self.tool_schemas,
                "max_tokens": 4096,
            }
            # Buffer silently — don't stream model prose to user
            if self.on_token is not None:
                message = await _stream_completion(
                    self.backend, payload, _noop, self._known_tool_names
                )
            else:
                response = await self.backend.chat(payload)
                message = response["choices"][0]["message"]

            content = message.get("content") or ""
            # Truncate only prose without code blocks — code blocks need full content
            if len(content) > 2000 and "```" not in content:
                message["content"] = content[:2000]
                content = message["content"]

            tool_calls = message.get("tool_calls") or []
            if not tool_calls and content:
                synthetic = _as_synthetic_tool_call(content, self._known_tool_names)
                if synthetic is not None:
                    # If it's a code-block write, grab ALL code blocks as writes
                    if (
                        synthetic["function"]["name"] == "write_file"
                        and "write_file" in self._known_tool_names
                    ):
                        all_writes = _extract_all_code_blocks_as_writes(content)
                        tool_calls = all_writes if len(all_writes) > 1 else [synthetic]
                    else:
                        tool_calls = [synthetic]
                    if self.on_event is not None:
                        for tc in tool_calls:
                            fn = tc["function"]
                            self.on_event(f"{fn['name']}(...)")
                    message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls,
                    }

            self.messages.append(message)

            if not tool_calls:
                if self.auto_read and not self._has_done_work():
                    first = self._pick_first_file()
                    if first:
                        await self._force_read(first)
                        continue
                # Final answer — show it to user
                if self.on_token and content:
                    self.on_token(content)
                return content

            for call in tool_calls:
                self.messages.append(await self._run_tool_call(call))
            self.messages.append({"role": "user", "content": "Next step. One tool call only."})

        raise AgentTurnLimitExceeded(
            f"Stopped after {self.max_turns} turns without a final answer — "
            "the model may be looping."
        )

    def _has_done_work(self) -> bool:
        """Check if any tool calls have been made in this conversation."""
        return any(msg.get("role") == "tool" for msg in self.messages)

    def _pick_first_file(self) -> str | None:
        """Pick the most relevant file to auto-read from the project root."""
        _PRIORITY_EXTS = (".html", ".py", ".js", ".ts", ".jsx", ".tsx", ".css")
        try:
            entries = list_directory(self.root, ".")
        except (OSError, ValueError):
            return None
        for entry in entries:
            if "/" in entry:
                continue
            if Path(entry).suffix.lower() in _PRIORITY_EXTS:
                return entry
        return None

    def _auto_read_first_file(self) -> None:
        """Pre-read the first relevant file so the model has code context."""
        first = self._pick_first_file()
        if not first:
            return
        try:
            content = read_file(self.root, first)
        except (OSError, ValueError):
            return
        call_id = f"auto-{uuid.uuid4().hex[:8]}"
        self.messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": first}),
                        },
                    }
                ],
            }
        )
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": "read_file",
                "content": content,
            }
        )
        if self.on_event is not None:
            self.on_event(f"read_file(path={first!r})")

    async def _force_read(self, path: str) -> None:
        """Force-inject a read_file when model refuses to use tools."""
        try:
            content = read_file(self.root, path)
        except (OSError, ValueError):
            return
        call_id = f"force-{uuid.uuid4().hex[:8]}"
        self.messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": path}),
                        },
                    }
                ],
            }
        )
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": "read_file",
                "content": content,
            }
        )
        if self.on_event is not None:
            self.on_event(f"read_file(path={path!r})")
        self.messages.append(
            {
                "role": "user",
                "content": (
                    f"I read {path} for you. Now write_file to make changes. "
                    "JSON only, no explanation."
                ),
            }
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
