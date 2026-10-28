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

import asyncio
import contextlib
import json
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from localgate.agent.context import (
    CompactionReport,
    ContextManager,
    SessionLedger,
    ledger_message,
    make_llm_ledger_updater,
)
from localgate.agent.input import project_files
from localgate.agent.instructions import load_instructions
from localgate.agent.mcp import McpRegistry, is_mcp_tool_name
from localgate.agent.tool_executor import (
    ToolCallTracker,
    execute_tool_call_with_timeout,
)
from localgate.agent.tools import (
    DEFAULT_SUB_AGENT_EXTRAS,
    MUTATING_TOOL_NAMES,
    READ_ONLY_TOOL_NAMES,
    SYNTAX_WARNING,
    TOOL_SCHEMAS,
    ToolCallResult,
    apply_edit,
    delegate_task_schema,
    ensure_visible,
    execute_tool_call,
    list_directory,
    read_file,
    read_file_page,
    resolve_within,
    search_files,
    unescape_text_arguments,
    validate_arguments,
)
from localgate.agent.websearch import WEB_SEARCH_SCHEMA, SearchFn
from localgate.backends.base import InferenceBackend


def _normalize_backend_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize empty assistant tool-call content for Ollama compatibility."""
    normalized: list[dict[str, Any]] = []
    for message in messages:
        item = dict(message)
        if "content" in item and item["content"] is None:
            item["content"] = ""
        normalized.append(item)
    return normalized


SYSTEM_PROMPT = (
    "You are a coding agent working inside one project directory. You act only "
    "through tools; a change you did not make with a tool did not happen.\n"
    "How to work:\n"
    "- Find the relevant code first with search_files, list_directory and read_file. "
    "You may call several read-only tools at once.\n"
    "- To change an existing file: read it, then call edit_file with an exact snippet "
    "copied from the file. Use create_file for new files. Use write_file only when "
    "most of a file changes.\n"
    "- Write complete, working code. Never leave placeholders such as '...' or "
    "'TODO: implement'.\n"
    "- If a tool returns an error, read it and fix the call instead of repeating it.\n"
    "- When the task is done, reply in plain text with a short summary of what you "
    "changed and anything left to do.\n"
    "If native tool calling is unavailable, reply with exactly one JSON object: "
    '{"name": "tool_name", "arguments": {...}}'
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
_MUTATION_REQUEST_RE = re.compile(
    r"\b(add|build|change|create|delete|edit|enhance|expand|extend|fix|implement|"
    r"improve|make|modify|move|polish|redesign|refactor|remove|rename|replace|"
    r"restyle|rewrite|update|upgrade|write)\b",
    re.IGNORECASE,
)
#: Matches a JSON object embedded anywhere in prose — used as a fallback when the
#: whole-string parse fails but the model clearly emitted a tool call inside text.
#: A reply announcing work still to do ("Next, I need to update main.py") rather
#: than finishing — small models often stop there without making the call.
_NEXT_STEP_RE = re.compile(
    r"\b(next,? (i|we)('ll| will| need to| should)|now (i|we)('ll| will| need to)|"
    r"i('ll| will) now|let me now|then (i|we)('ll| will| need to))\b",
    re.IGNORECASE,
)
#: Calls whose result can only change if a file changes (or, for search, not at all
#: within one task) — repeating one verbatim is a loop, not progress.
_REPEATABLE_TOOLS = READ_ONLY_TOOL_NAMES | {"run_command", "web_search", "fetch_url"}
#: A model echoing a tool result back as its "answer" (seen with small models).
_ECHOED_TOOL_RESPONSE_RE = re.compile(r"<tool_response>.*?(</tool_response>|$)", re.DOTALL)
#: Path-like tokens in prose ("main.py", "src/app.ts") for `_claimed_but_unchanged`.
_PATH_TOKEN_RE = re.compile(r"[\w./-]*\w\.[A-Za-z]{1,8}\b")
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

    # Fourth: Aider-style SEARCH/REPLACE blocks are an edit_file call in prose.
    if "edit_file" in known_names:
        edits = _extract_search_replace_edits(content)
        if edits:
            return edits[0]

    # Fifth: extract a named code block as a write_file call.
    # Models often say "### `contact.html`\n```html\n...code...\n```"
    # which IS a write_file action expressed as prose.
    if "write_file" in known_names:
        write = _extract_code_block_as_write(content)
        if write is not None:
            return write

    return None


_TOOL_NAME_FIELD_RE = re.compile(r'"name"\s*:\s*"(\w+)"')


def _malformed_tool_call_error(content: str, known_names: frozenset[str]) -> str | None:
    """A nudge for a reply that tried to call a tool in text but wrote invalid
    JSON — typically an unescaped ``"`` inside file content. Without it the
    broken call reads as a final answer and nothing is written."""
    for match in _TOOL_NAME_FIELD_RE.finditer(content):
        name = match.group(1)
        if name not in known_names:
            continue
        start = content.rfind("{", 0, match.start())
        if start == -1:
            continue
        candidate = _extract_json_object(content, start) or content[start:]
        try:
            json.loads(candidate)
        except json.JSONDecodeError as exc:
            return (
                f"Your {name} call was not valid JSON ({exc.msg} at char {exc.pos}), so it "
                f"did not run and no file was changed. Call {name} again with valid JSON — "
                'escape every double quote inside a string value as \\".'
            )
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


#: A fenced block labelled by a comment on its first line — the other common way
#: models name a file: ```html / <!-- src/index.html -->, /* styles.css */,
#: // app.js or # main.py. The label line is dropped from the written content.
_PATH_COMMENT_BLOCK_RE = re.compile(
    r"```\w*[ \t]*\n"
    r"[ \t]*(?:<!--|/\*|//|#)[ \t]*(?:[Ff]ile:\s*)?"
    r"((?:\./)?[\w][\w./-]*\."
    r"(?:html|css|js|jsx|tsx|ts|py|json|md|txt|yml|yaml|toml|cfg|ini|sh))"
    r"[ \t]*(?:-->|\*/)?[ \t]*\n"
    r"(.*?)"
    r"\n```",
    re.DOTALL,
)


def _named_code_blocks(content: str) -> list[tuple[str, str]]:
    """(path, code) for every code block the reply names a file for, in order.

    A block named both ways (a heading and a comment label) is counted once —
    both matches end at the same closing fence.
    """
    found: dict[int, tuple[int, str, str]] = {}
    for pattern in (_PATH_COMMENT_BLOCK_RE, _FILENAME_HINT_RE):
        for match in pattern.finditer(content):
            code = match.group(2)
            if len(code.strip()) < 50:
                continue
            found.setdefault(match.end(), (match.start(), match.group(1).strip(), code))
    return [(path, code) for _start, path, code in sorted(found.values())]


def _write_call(path: str, code: str) -> dict[str, Any]:
    return {
        "id": f"synthetic-{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": "write_file",
            "arguments": json.dumps({"path": path, "content": code}),
        },
    }


def _extract_code_block_as_write(content: str) -> dict[str, Any] | None:
    """If model output contains a named code block (filename hint + fence, or a
    path comment as the block's first line), synthesize a write_file tool call
    from the first one found.
    """
    blocks = _named_code_blocks(content)
    return _write_call(*blocks[0]) if blocks else None


#: A SEARCH/REPLACE block — the edit format Aider popularized, which many coding
#: models emit unprompted:
#:
#:     path/to/file.py
#:     <<<<<<< SEARCH
#:     old lines
#:     =======
#:     new lines
#:     >>>>>>> REPLACE
_SEARCH_REPLACE_RE = re.compile(
    r"(?:^|\n)[`*# ]*([\w./\\-]+\.\w+)[`* ]*\s*\n(?:```\w*\s*\n)?"
    r"<{5,9} ?SEARCH\s*\n(.*?)\n?={5,9}\s*\n(.*?)\n?>{5,9} ?REPLACE",
    re.DOTALL,
)


def _extract_search_replace_edits(content: str) -> list[dict[str, Any]]:
    """Every SEARCH/REPLACE block in ``content`` as an edit_file call."""
    edits: list[dict[str, Any]] = []
    for match in _SEARCH_REPLACE_RE.finditer(content):
        path, old, new = match.group(1).strip(), match.group(2), match.group(3)
        if not old.strip():
            continue
        edits.append(
            {
                "id": f"synthetic-{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "arguments": json.dumps(
                        {"path": path.replace("\\", "/"), "old_string": old, "new_string": new}
                    ),
                },
            }
        )
    return edits


def _extract_all_code_blocks_as_writes(content: str) -> list[dict[str, Any]]:
    """Extract ALL named code blocks from model output as write_file calls."""
    return [_write_call(path, code) for path, code in _named_code_blocks(content)]


#: Fence language tag -> the file extension that kind of code lives in.
_LANGUAGE_EXTENSIONS = {
    "html": ".html", "htm": ".html", "css": ".css", "scss": ".scss",
    "js": ".js", "javascript": ".js", "jsx": ".jsx", "ts": ".ts", "typescript": ".ts",
    "tsx": ".tsx", "py": ".py", "python": ".py", "json": ".json", "go": ".go",
    "rs": ".rs", "rust": ".rs", "java": ".java", "rb": ".rb", "ruby": ".rb",
    "php": ".php", "vue": ".vue", "svelte": ".svelte",
}  # fmt: skip
_FENCED_BLOCK_RE = re.compile(r"```(\w*)[^\n]*\n(.*?)\n```", re.DOTALL)
#: A block at least this long reads as file content rather than an illustration.
_FILE_LIKE_BLOCK_LINES = 8


class AgentTurnLimitExceeded(RuntimeError):
    """Raised when the model keeps calling tools without ever finishing a turn."""


class AgentToolUseRequired(RuntimeError):
    """Raised when a coding model repeatedly answers without using project tools."""


#: Called before a write_file call actually runs: (path, new_content) -> proceed?
ConfirmWrite = Callable[[str, str | None], bool]
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

            # Early stop: if content is getting long and contains a COMPLETE JSON tool call,
            # stop consuming the stream. But NOT if we're inside a code fence
            # (model may be writing file contents as named blocks).
            # Check every 100 chars to avoid parsing on every token.
            if known_tool_names and not tool_calls and content_len > 200 and content_len % 100 < 10:
                joined = "".join(content_parts)
                if "```" not in joined:
                    # Only stop if we have a complete, parseable tool call
                    synthetic = _as_synthetic_tool_call(joined, known_tool_names)
                    if synthetic is not None:
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


@dataclass
class AsyncTool:
    """A tool implemented outside `tools.py` — web fetch, sandboxed commands,
    the to-do list — offered to the model only when it is configured."""

    schema: dict[str, Any]
    run: Callable[[dict[str, Any]], Awaitable[str]]
    #: Asked before each call: (tool name, arguments) -> proceed? ``None`` = never ask.
    confirm: Callable[[str, dict[str, Any]], bool] | None = None
    read_only: bool = True
    timeout: float | None = None
    #: Appended to the system prompt while the tool is offered.
    guidance: str = ""

    @property
    def name(self) -> str:
        return str(self.schema["function"]["name"])


@dataclass
class ToolLogEntry:
    """One executed tool call, for durable logging. Arguments are already
    redacted to a one-line summary: file contents never reach the log."""

    run_id: str
    session_id: str
    model: str
    tool_name: str
    arguments: dict[str, Any]
    success: bool
    duration_ms: int
    error: str | None = None
    timed_out: bool = False


#: Receives each :class:`ToolLogEntry` — e.g. `AgentMemory.record_tool_call`.
OnToolLog = Callable[[ToolLogEntry], Awaitable[None]]

#: Tool results longer than this are cut, with a note, before entering history.
MAX_TOOL_RESULT_CHARS = 16_000
#: Agent sampling temperature when no settings are given (see `agent_temperature`).
DEFAULT_TEMPERATURE = 0.2
#: Context budget used when the model's window is unknown.
DEFAULT_NUM_CTX = 16_384


def _cap_tool_result(content: str) -> str:
    if len(content) <= MAX_TOOL_RESULT_CHARS:
        return content
    omitted = len(content) - MAX_TOOL_RESULT_CHARS
    return (
        content[:MAX_TOOL_RESULT_CHARS]
        + f"\n[... {omitted:,} more characters omitted; narrow the request to see them]"
    )


class AgentSession:
    """One conversation with the model, across as many turns as the caller sends.

    A fresh session starts with just the system prompt; each call to :meth:`send`
    appends a user turn, runs the tool-call loop to completion, and returns the
    model's final text. History accumulates in ``self.messages`` until :meth:`reset`,
    compacted against the context budget as it grows (see `agent/context.py`).
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
        confirm_mcp: Callable[[str, dict[str, Any]], bool] | None = None,
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
        settings: Any | None = None,
        session_id: str | None = None,
        extra_tools: list[AsyncTool] | None = None,
        num_ctx: int | None = None,
        native_tools: bool = True,
        on_tool_log: OnToolLog | None = None,
        ledger: SessionLedger | None = None,
        agent_roles: dict[str, Any] | None = None,
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
        self.confirm_mcp = confirm_mcp
        self.on_event = on_event
        self.on_token = on_token
        #: Called with the reply text so far while a reply streams in — before it is
        #: known whether it ends as a tool call or the final answer.
        self.on_draft: OnToken | None = None
        self.max_turns = max_turns
        self.tool_schemas = list(tool_schemas) if tool_schemas is not None else list(TOOL_SCHEMAS)
        #: Whether *this* session may call `delegate_task` — never propagated to
        #: a delegated sub-agent's own session (see `_run_delegated_task`), which
        #: is the depth-1 cap: a sub-agent cannot itself spawn a sub-agent.
        self.allow_delegation = allow_delegation
        #: Named sub-agent roles (`agent/roles.py`), offered through delegate_task.
        self.agent_roles: dict[str, Any] = dict(agent_roles or {})
        if allow_delegation:
            self.tool_schemas.append(delegate_task_schema(self.agent_roles))
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
        self.extra_tools: dict[str, AsyncTool] = {t.name: t for t in extra_tools or []}
        self.tool_schemas.extend(t.schema for t in self.extra_tools.values())
        self.tool_executor = tool_executor if tool_executor is not None else execute_tool_call
        self.augment = augment
        self.auto_read = auto_read
        self.settings = settings
        self.session_id = session_id or str(uuid.uuid4())
        #: Explicit context window to request; resolved from the model on first send.
        self.num_ctx = num_ctx
        self._num_ctx_resolved = num_ctx is not None
        #: False for models without native tool calling: tools are described in the
        #: prompt and replies are grammar-constrained to a JSON tool call instead.
        self.native_tools = native_tools
        self.on_tool_log = on_tool_log
        self.ledger = ledger or SessionLedger()
        self.context: ContextManager | None = None
        self.last_compaction: CompactionReport | None = None
        self.run_id = ""
        self.tool_tracker = ToolCallTracker()
        self._task_tool_calls = 0
        self._task_mutations = 0
        self._read_paths: set[str] = set()
        self._last_mutation_error: str | None = None
        self._broken_files: set[str] = set()
        self._task_changed_paths: set[str] = set()
        self._task_mutation_attempts = 0
        self._seen_results: dict[tuple[str, str], tuple[int, str]] = {}
        self._task_is_mutation = False
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
        for tool in self.extra_tools.values():
            prompt += tool.guidance
        if not self.native_tools:
            prompt += _json_tool_protocol(self.tool_schemas)
        prompt += load_instructions(self.root)
        return prompt

    def reset(self) -> None:
        """Drop history, starting a new conversation in the same session."""
        self.messages = [{"role": "system", "content": self.system_prompt()}]
        self.ledger = SessionLedger()

    def switch_model(self, model: str, *, native_tools: bool = True) -> None:
        """Change model; the context window and tool protocol are re-derived."""
        self.model = model
        self.native_tools = native_tools
        self.num_ctx = None
        self._num_ctx_resolved = False
        self.context = None
        self.refresh_tools()

    def refresh_tools(self) -> None:
        """Recompute derived state after `tool_schemas` changes."""
        self._known_tool_names = frozenset(s["function"]["name"] for s in self.tool_schemas)
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0] = {"role": "system", "content": self.system_prompt()}

    # -- context window -------------------------------------------------------------

    async def _resolve_context(self) -> None:
        """Pick the context window once: the model's own maximum, capped by the
        configured agent budget so a 128K-capable model doesn't reserve 128K of
        VRAM. Only Ollama needs it sent per request; other servers fix it at load."""
        if self._num_ctx_resolved:
            if self.context is None and self.num_ctx:
                self.context = self._make_context_manager(self.num_ctx)
            return
        self._num_ctx_resolved = True
        cap = getattr(self.settings, "agent_num_ctx", None) or DEFAULT_NUM_CTX
        try:
            window = await self.backend.context_window(self.model)
        except Exception:  # noqa: BLE001 — unknown window is the safe answer
            window = None
        if window:
            self.num_ctx = min(window, cap)
        elif self.backend.accepts_options:
            self.num_ctx = cap
        budget = self.num_ctx or window
        if budget:
            self.context = self._make_context_manager(budget)

    def _make_context_manager(self, budget: int) -> ContextManager:
        options = {"num_ctx": self.num_ctx} if self.num_ctx else None
        updater = make_llm_ledger_updater(self.backend, self.model, options)
        # Leave room for the reply itself.
        return ContextManager(max(1024, budget - 4096), updater=updater)

    def _request_options(self) -> dict[str, Any]:
        return {"num_ctx": self.num_ctx} if self.num_ctx else {}

    async def compact(self, focus: str | None = None, *, force: bool = True) -> CompactionReport:
        """Compact history now (the `/compact` command), or when over budget."""
        await self._resolve_context()
        manager = self.context or self._make_context_manager(DEFAULT_NUM_CTX)
        self.messages, report = await manager.compact(
            self.messages, self.tool_schemas, self.ledger, focus=focus, force=force
        )
        self.last_compaction = report
        return report

    def context_usage(self) -> int:
        return ContextManager.usage(self.messages, self.tool_schemas, self.ledger)

    def _outgoing(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """History plus the pinned ledger, right after the system prompt."""
        pinned = ledger_message(self.ledger)
        if pinned is None or not messages or messages[0].get("role") != "system":
            return messages
        return [messages[0], pinned, *messages[1:]]

    # -- the turn loop ---------------------------------------------------------------

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
            # The model has seen it in full, so it counts as read for edits.
            self._read_paths.add(entry)

        return "\n".join(parts)

    def _payload(self, outgoing: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _normalize_backend_messages(outgoing),
            "max_tokens": 4096,
            "temperature": getattr(self.settings, "agent_temperature", DEFAULT_TEMPERATURE),
        }
        if self.native_tools:
            payload["tools"] = self.tool_schemas
        else:
            payload["format"] = _json_tool_format(self._known_tool_names)
        options = self._request_options()
        if options:
            payload["options"] = options
        return payload

    async def send(self, user_input: str, attachments: Sequence[str] = ()) -> str:
        """Run one user turn to completion and return the model's final reply.

        Output is buffered silently during the tool loop — only tool events
        (via on_event) and, when set, the in-progress text (via on_draft) are
        shown. The final answer goes to on_token at the end. ``attachments``
        are project files (`@path` in the REPL) whose contents ride along with
        the message and count as read.
        """
        await self._resolve_context()
        require_tool_evidence = self.auto_read and bool(_MUTATION_REQUEST_RE.search(user_input))
        self.run_id = uuid.uuid4().hex
        self.tool_tracker = ToolCallTracker()
        self._task_tool_calls = 0
        self._task_mutations = 0
        self._read_paths = set()
        self._last_mutation_error = None
        self._broken_files = set()
        self._task_changed_paths = set()
        self._task_mutation_attempts = 0
        self._seen_results = {}
        self._task_is_mutation = bool(_MUTATION_REQUEST_RE.search(user_input))
        self.ledger.observe_user(user_input)
        prose_without_tools = 0
        unverified_claim_nudged = False
        empty_reply_nudged = False
        malformed_call_nudged = False
        attached = self._attachment_text(attachments)
        if attached:
            user_input = f"{user_input}\n\n{attached}"
        if len(self.messages) == 1:
            context = self._build_project_context()
            if context:
                user_input = f"{context}\n\nUser request: {user_input}"
            if self.auto_read:
                self._auto_read_first_file()
        self.messages.append({"role": "user", "content": user_input})

        def _noop(_text: str) -> None:
            pass

        def _draft_callback() -> OnToken:
            on_draft = self.on_draft
            if on_draft is None:
                return _noop
            parts: list[str] = []

            def collect(text: str) -> None:
                parts.append(text)
                on_draft("".join(parts))

            return collect

        for _ in range(self.max_turns):
            if self.context is not None and self.context.needs_compaction(
                self.messages, self.tool_schemas, self.ledger
            ):
                report = await self.compact(force=False)
                if report.changed and self.on_event is not None:
                    self.on_event(report.describe())

            history = self._outgoing(self.messages)
            outgoing = await self.augment(history) if self.augment else history
            payload = self._payload(outgoing)
            if self.on_token is not None and self.native_tools:
                message = await _stream_completion(
                    self.backend, payload, _draft_callback(), self._known_tool_names
                )
            else:
                response = await self.backend.chat(payload)
                message = response["choices"][0]["message"]

            content = message.get("content") or ""
            tool_calls = message.get("tool_calls") or []
            if not self.native_tools and not tool_calls:
                final, constrained_call = _parse_constrained_reply(content, self._known_tool_names)
                if constrained_call is not None:
                    tool_calls = [constrained_call]
                    message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
                elif final is not None:
                    content = final
                    message = {"role": "assistant", "content": final}
            if not tool_calls and content:
                synthetic = _as_synthetic_tool_call(content, self._known_tool_names)
                if synthetic is not None:
                    synthetic_name = synthetic["function"]["name"]
                    if synthetic_name == "edit_file":
                        tool_calls = _extract_search_replace_edits(content) or [synthetic]
                    elif synthetic_name == "write_file":
                        all_writes = _extract_all_code_blocks_as_writes(content)
                        tool_calls = all_writes if len(all_writes) > 1 else [synthetic]
                        tool_calls = [self._retarget_write(call) for call in tool_calls]
                    else:
                        tool_calls = [synthetic]
                    message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls,
                    }
            message.pop("_early_stopped", None)

            self.messages.append(message)

            if not tool_calls:
                malformed = _malformed_tool_call_error(content, self._known_tool_names)
                if malformed and not malformed_call_nudged:
                    malformed_call_nudged = True
                    self.messages.append({"role": "user", "content": malformed})
                    continue
                if require_tool_evidence and self._task_tool_calls == 0:
                    prose_without_tools += 1
                    if prose_without_tools == 1:
                        self.messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "You have not used a project tool for this task. Inspect the "
                                    "workspace and perform the requested action with tools. Do not "
                                    "claim completion without tool evidence."
                                ),
                            }
                        )
                        continue
                    raise AgentToolUseRequired(
                        "The selected model answered without using project tools twice; "
                        "choose a model with reliable tool-calling support."
                    )
                # A model that ends right after failed edits (or leaving a file that
                # no longer parses) usually still claims success. Tell it once what
                # actually happened, with the error, before accepting its answer.
                problem = self._unfinished_work(content)
                answer_text = _ECHOED_TOOL_RESPONSE_RE.sub("", content).strip()
                if not answer_text and self._task_tool_calls and not empty_reply_nudged:
                    # An empty reply mid-task is a stall, not an answer.
                    empty_reply_nudged = True
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "Your reply was empty. Continue the task with a tool "
                            "call, or give your final answer.",
                        }
                    )
                    continue
                if problem and not unverified_claim_nudged:
                    unverified_claim_nudged = True
                    self.messages.append({"role": "user", "content": problem})
                    continue
                # Final answer — show it to user
                if self.on_token and content:
                    self.on_token(content)
                return content

            for result in await self._run_tool_calls(tool_calls):
                self.messages.append(result)

            # The only nudge left: a model repeating the same call (or failing
            # over and over) is told so once, instead of being pushed every turn.
            if self.tool_tracker.is_stuck_loop():
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "PROGRESS CHECK: you are repeating the same tool call or every "
                            "recent call failed. Use a different approach, ask the user for "
                            "clarification, or give your final answer now."
                        ),
                    }
                )
                self.tool_tracker = ToolCallTracker()

        raise AgentTurnLimitExceeded(
            f"Stopped after {self.max_turns} turns without a final answer — "
            "the model may be looping."
        )

    async def _run_tool_calls(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run one turn's calls. Built-in read-only calls with no confirmation
        step run concurrently; anything that writes, asks, or has side effects
        runs in order so confirmations and checkpoints stay sequential."""
        if len(calls) > 1 and all(
            (call.get("function") or {}).get("name") in READ_ONLY_TOOL_NAMES for call in calls
        ):
            return list(await asyncio.gather(*(self._run_tool_call(call) for call in calls)))
        return [await self._run_tool_call(call) for call in calls]

    def _claimed_but_unchanged(self, answer: str) -> list[str]:
        """Project files an edit task's final answer names but didn't change.

        Small models often finish one file of a multi-file change and report
        all of them done. Only checked once something was changed, so a plain
        question about a file is never second-guessed.
        """
        if self._task_mutations == 0 or not self._task_is_mutation:
            return []
        found: list[str] = []
        for token in sorted(set(_PATH_TOKEN_RE.findall(answer))):
            path = token.strip("`'\".,:;()")
            if path in self._task_changed_paths or path in found:
                continue
            try:
                exists = resolve_within(self.root, path).is_file()
            except ValueError:
                continue
            if exists:
                found.append(path)
        return found

    def _unfinished_work(self, final_answer: str = "") -> str | None:
        if self._broken_files:
            names = ", ".join(sorted(self._broken_files))
            return (
                f"CHECK: {names} still has a syntax error, so the task is not done. Read "
                "it and fix it, or tell the user plainly that it is broken."
            )
        unchanged = self._claimed_but_unchanged(final_answer)
        if unchanged:
            names = ", ".join(unchanged)
            return (
                f"CHECK: your answer mentions {names}, but {names} was not changed in this "
                "task. Do not rely on memory: call read_file on it now, and if it still "
                "needs the change, make it with edit_file. Then give your final answer."
            )
        if self._task_is_mutation and _NEXT_STEP_RE.search(final_answer):
            return (
                "You described a next step instead of doing it. Do it now with a tool "
                "call; give your final answer only when every step is done."
            )
        if self._task_is_mutation and self._task_mutation_attempts == 0:
            target = next(
                (p for p in reversed(self.ledger.files_read) if p in self._read_paths), None
            )
            first_step = f" Start with {target}: call edit_file on it now." if target else ""
            return (
                "CHECK: this task asked for a change, but no file has been changed yet. Do not "
                "describe a plan — make the change with a tool call." + first_step + " If no "
                "change is needed, say so plainly and explain why."
            )
        if self._task_mutations == 0 and self._has_project_code(final_answer):
            return (
                "CHECK: your reply contains code for this project, but no file was changed. "
                "If that code belongs in the project, apply it now: edit_file for a small "
                "change, write_file to replace a whole file, using the real paths from the file "
                "list. If the user only asked to see code, say so in one line without repeating it."
            )
        if self._task_mutations == 0 and self._last_mutation_error:
            return (
                "CHECK: no file was changed in this task; your last change failed with: "
                f"{self._last_mutation_error[:300]}\nRetry it correctly, or tell the user "
                "plainly that the change was not made."
            )
        return None

    def _has_done_work(self) -> bool:
        """Check if any tool calls have been made in this conversation."""
        return self._task_tool_calls > 0

    def _existing_target(self, path: str) -> str:
        """Models often guess a directory for a file that lives elsewhere
        ("src/index.html" when it's at the root). A labelled code block for a path
        that doesn't exist goes to the one project file with that name, if exactly
        one exists; otherwise the path is kept (a genuinely new file)."""
        try:
            if resolve_within(self.root, path).exists():
                return path
        except ValueError:
            return path
        name = PurePosixPath(path).name
        matches = [f for f in project_files(self.root) if PurePosixPath(f).name == name]
        return matches[0] if len(matches) == 1 else path

    def _retarget_write(self, call: dict[str, Any]) -> dict[str, Any]:
        fn = call["function"]
        arguments = _parse_arguments(fn["arguments"])
        path = arguments.get("path")
        if not isinstance(path, str):
            return call
        target = self._existing_target(path)
        if target == path:
            return call
        arguments["path"] = target
        return {**call, "function": {**fn, "arguments": json.dumps(arguments)}}

    def _has_project_code(self, answer: str) -> bool:
        """Whether ``answer`` holds file-sized code in a language this project
        uses — the sign of a change written into chat instead of into a file."""
        blocks = [
            (lang.lower(), code)
            for lang, code in _FENCED_BLOCK_RE.findall(answer)
            if code.count("\n") + 1 >= _FILE_LIKE_BLOCK_LINES
        ]
        if not blocks:
            return False
        extensions = {PurePosixPath(f).suffix for f in project_files(self.root)}
        return any(_LANGUAGE_EXTENSIONS.get(lang) in extensions for lang, _code in blocks)

    def _attachment_text(self, attachments: Sequence[str]) -> str:
        """File contents for the user's `@path` attachments, marked as read."""
        blocks: list[str] = []
        for path in attachments:
            try:
                content = read_file(self.root, path)
            except (OSError, ValueError):
                continue
            self._read_paths.add(path)
            self.ledger.observe_tool("read_file", {"path": path}, True, content)
            blocks.append(f"Contents of {path} (attached by the user):\n```\n{content}\n```")
        return "\n\n".join(blocks)

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

    def _inject_read(self, path: str, content: str, prefix: str) -> None:
        call_id = f"{prefix}-{uuid.uuid4().hex[:8]}"
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
                "content": _cap_tool_result(content),
            }
        )
        self._read_paths.add(path)
        self.ledger.observe_tool("read_file", {"path": path}, True, "")
        if self.on_event is not None:
            self.on_event(f"read_file(path={path!r})")

    def _auto_read_first_file(self) -> None:
        """Pre-read the first relevant file so the model has code context."""
        first = self._pick_first_file()
        if not first:
            return
        try:
            content = read_file(self.root, first)
        except (OSError, ValueError):
            return
        self._inject_read(first, content, "auto")

    async def _force_read(self, path: str) -> None:
        """Force-inject a read_file when model refuses to use tools."""
        try:
            content = read_file(self.root, path)
        except (OSError, ValueError):
            return
        self._inject_read(path, content, "force")
        self.messages.append(
            {
                "role": "user",
                "content": f"I read {path} for you. Now make the change with a tool call.",
            }
        )

    async def _log_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        success: bool,
        started: float,
        *,
        error: str | None = None,
        timed_out: bool = False,
    ) -> None:
        self.tool_tracker.record(name, success, arguments)
        if self.on_tool_log is None:
            return
        entry = ToolLogEntry(
            run_id=self.run_id,
            session_id=self.session_id,
            model=self.model,
            tool_name=name,
            arguments=_summarize(arguments),
            success=success,
            duration_ms=int((time.monotonic() - started) * 1000),
            error=error[:500] if error else None,
            timed_out=timed_out,
        )
        with contextlib.suppress(Exception):  # logging must never break a turn
            await self.on_tool_log(entry)

    async def _run_tool_call(self, call: dict[str, Any]) -> dict[str, Any]:
        result = await self._execute_tool_call(call)
        result["content"] = _cap_tool_result(str(result.get("content") or ""))
        name = (call.get("function") or {}).get("name", "")
        if name in _REPEATABLE_TOOLS and not result["content"].startswith("You already made"):
            arguments = _parse_arguments((call.get("function") or {}).get("arguments", ""))
            key = (name, json.dumps(arguments, sort_keys=True))
            self._seen_results[key] = (self._task_mutations, result["content"])
        return result

    async def _execute_tool_call(self, call: dict[str, Any]) -> dict[str, Any]:
        fn = call["function"]
        name = fn["name"]
        arguments = _parse_arguments(fn.get("arguments", ""))
        started = time.monotonic()
        self._task_tool_calls += 1
        if name in MUTATING_TOOL_NAMES or name == "delegate_task":
            # Any attempt — even refused, declined or delegated — means the model did
            # try to make the change, so the "nothing changed yet" check stays quiet.
            self._task_mutation_attempts += 1

        def reply(content: str) -> dict[str, Any]:
            return {"role": "tool", "tool_call_id": call["id"], "name": name, "content": content}

        # The same read-only call (or command) again with no file changed since can only
        # return the same thing. Small models loop like this; hand back the earlier
        # result and say so, instead of re-running it.
        repeat_key = (name, json.dumps(arguments, sort_keys=True))
        if name in _REPEATABLE_TOOLS and repeat_key in self._seen_results:
            mutations_then, previous = self._seen_results[repeat_key]
            if mutations_then == self._task_mutations:
                await self._log_tool(name, arguments, False, started, error="repeated call")
                return reply(
                    "You already made this exact call and no file has changed since, so the "
                    "result is the same (shown below). Act on it now — e.g. fix the code with "
                    f"edit_file — instead of repeating it.\n\n{previous[:3000]}"
                )

        if name not in self._known_tool_names:
            # Not just a typo guard: this is the actual boundary that keeps a
            # delegated sub-agent confined to its `allowed_tools` — its schema
            # simply excludes anything it wasn't granted, and a model that
            # hallucinates a call for a tool it was never offered (structured
            # `tool_calls` aren't otherwise checked against the schema) hits
            # this rather than `self.tool_executor`.
            await self._log_tool(name, arguments, False, started, error="unknown tool")
            return reply(f"Unknown tool: {name}")

        # Full replacements and deletes need a prior read: they could clobber content
        # the model never saw. edit_file doesn't — its exact-match old_string is
        # itself proof the model knows what it is changing.
        if name in MUTATING_TOOL_NAMES:
            arguments = unescape_text_arguments(arguments)
            call = {**call, "function": {**fn, "arguments": json.dumps(arguments)}}
        if name == "create_file" and isinstance(arguments.get("path"), str):
            try:
                exists = resolve_within(self.root, arguments["path"]).exists()
            except ValueError:
                exists = False  # the executor returns the precise boundary error
            if exists:
                message = (
                    f"{arguments['path']} already exists, so create_file was refused and nothing "
                    "changed. To change it, read it and use edit_file."
                )
                await self._log_tool(name, arguments, False, started, error="already exists")
                self._last_mutation_error = message
                return reply(message)
        needs_prior_read = name in {"update_file", "delete_file"}
        if name == "write_file" and isinstance(arguments.get("path"), str):
            try:
                candidate = resolve_within(self.root, arguments["path"])
                ensure_visible(self.root, candidate)
                needs_prior_read = candidate.is_file()
            except ValueError:
                # The normal confirmation/executor path returns the useful boundary error.
                pass
        if needs_prior_read:
            path = arguments.get("path")
            if not isinstance(path, str) or path not in self._read_paths:
                await self._log_tool(name, arguments, False, started, error="not read first")
                if not isinstance(path, str):
                    return reply(f"Error: {name} needs a 'path' string.")
                # Rather than make a small model work out that it must read and
                # retry, read for it: the change can be retried on the next turn.
                try:
                    current = read_file_page(self.root, path)
                except (OSError, ValueError) as exc:
                    return reply(f"Error: {name} failed: {type(exc).__name__}: {exc}")
                self._read_paths.add(path)
                self.ledger.observe_tool("read_file", {"path": path}, True, "")
                self._last_mutation_error = f"{name} on {path} was refused: not read first"
                return reply(
                    f"Error: {path} was not changed, because a file must be read before "
                    f"{name}. Its current content is below. Check that your change still "
                    f"applies, then call {name} again.\n\n{current}"
                )

        if name in MUTATING_TOOL_NAMES and self.confirm_write is not None:
            path = arguments.get("path", "?")
            if name == "delete_file":
                preview: str | None = None
            elif name == "edit_file":
                try:
                    preview = apply_edit(
                        self.root,
                        str(path),
                        arguments.get("old_string", ""),
                        arguments.get("new_string", ""),
                        bool(arguments.get("replace_all", False)),
                    )
                except (OSError, ValueError, TypeError) as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    await self._log_tool(name, arguments, False, started, error=message)
                    self.ledger.observe_tool(name, arguments, False, message)
                    self._last_mutation_error = message
                    return reply(message)
            else:
                preview = arguments.get("content", "")
            if not self.confirm_write(path, preview):
                await self._log_tool(name, arguments, False, started, error="declined")
                return reply(f"User declined to write {path}. Ask before trying again.")

        if name == "delegate_task" and self.allow_delegation:
            task = arguments.get("task", "")
            if self.confirm_delegate is not None and not self.confirm_delegate(task):
                await self._log_tool(name, arguments, False, started, error="declined")
                return reply("User declined this delegation. Do the sub-task directly instead.")
            if self.on_event is not None:
                self.on_event(f"delegate_task({_summarize(arguments)})")
            content = await self._run_delegated_task(arguments)
            await self._log_tool(name, arguments, True, started)
            return reply(content)

        if name == "web_search" and self.search_fn is not None:
            query = arguments.get("query", "")
            if self.confirm_search is not None and not self.confirm_search(query):
                await self._log_tool(name, arguments, False, started, error="declined")
                return reply("User declined this search. Ask before trying again.")
            if self.on_event is not None:
                self.on_event(f"web_search(query={query!r})")
            if not query:
                await self._log_tool(name, arguments, False, started, error="no query")
                return reply("web_search requires a query.")
            timeout = self.settings.tool_timeout_search if self.settings is not None else None
            content, timed_out = await _with_timeout(self.search_fn(query), timeout)
            if timed_out:
                content = (
                    f"Web search timed out after {timeout}s. "
                    "Try a more specific query or skip this search."
                )
            await self._log_tool(name, arguments, not timed_out, started, timed_out=timed_out)
            return reply(content)

        if name in self.extra_tools:
            tool = self.extra_tools[name]
            if tool.confirm is not None and not tool.confirm(name, arguments):
                await self._log_tool(name, arguments, False, started, error="declined")
                return reply(f"User declined this {name} call.")
            if self.on_event is not None:
                self.on_event(f"{name}({_summarize(arguments)})")
            problem = validate_arguments(tool.schema, arguments)
            if problem:
                await self._log_tool(name, arguments, False, started, error=problem)
                return reply(f"Invalid arguments: {problem}")
            try:
                content, timed_out = await _with_timeout(tool.run(arguments), tool.timeout)
            except Exception as exc:  # noqa: BLE001 — failures go back to the model
                message = f"{type(exc).__name__}: {exc}"
                await self._log_tool(name, arguments, False, started, error=message)
                self.ledger.observe_tool(name, arguments, False, message)
                return reply(message)
            if timed_out:
                content = f"Tool '{name}' timed out after {tool.timeout}s."
            await self._log_tool(name, arguments, not timed_out, started, timed_out=timed_out)
            return reply(content)

        if is_mcp_tool_name(name) and self.mcp_registry is not None:
            if self.confirm_mcp is not None and not self.confirm_mcp(name, arguments):
                await self._log_tool(name, arguments, False, started, error="declined")
                return reply("User declined this MCP tool call.")
            if self.on_event is not None:
                self.on_event(f"{name}({_summarize(arguments)})")
            timeout = self.settings.tool_timeout_read if self.settings is not None else None
            content, timed_out = await _with_timeout(
                self.mcp_registry.call_tool(name, arguments), timeout
            )
            if timed_out:
                content = f"Tool '{name}' timed out after {timeout}s. Try a simpler operation."
            await self._log_tool(name, arguments, not timed_out, started, timed_out=timed_out)
            return reply(content)

        if self.on_event is not None:
            args_repr = ", ".join(f"{k}={v!r}" for k, v in _summarize(arguments).items())
            self.on_event(f"{name}({args_repr})")

        if self.settings is not None:
            result = await execute_tool_call_with_timeout(
                self.root,
                call["id"],
                name,
                arguments,
                self.settings,
                executor=self.tool_executor,
            )
        else:
            result = self.tool_executor(self.root, call["id"], name, arguments)

        await self._log_tool(
            name,
            arguments,
            not result.is_error,
            started,
            error=result.content if result.is_error else None,
            timed_out=result.timed_out,
        )
        self.ledger.observe_tool(name, arguments, not result.is_error, result.content)
        if not result.is_error and name == "read_file":
            path = arguments.get("path")
            if isinstance(path, str):
                self._read_paths.add(path)
        if name in MUTATING_TOOL_NAMES:
            path_arg = arguments.get("path")
            if result.is_error:
                self._last_mutation_error = result.content
            elif isinstance(path_arg, str):
                self._task_changed_paths.add(path_arg)
                if SYNTAX_WARNING in result.content:
                    self._broken_files.add(path_arg)
                else:
                    self._broken_files.discard(path_arg)
        if not result.is_error and name in MUTATING_TOOL_NAMES:
            self._task_mutations += 1
            path = arguments.get("path")
            if isinstance(path, str) and name != "delete_file":
                # The model just wrote it, so its content is known: a follow-up
                # edit needn't re-read the file first.
                self._read_paths.add(path)

        content = result.content
        if not result.is_error and name == "edit_file":
            content += _remaining_references(self.root, arguments)
        return {
            "role": "tool",
            "tool_call_id": result.tool_call_id,
            "name": result.name,
            "content": content,
        }

    async def _run_delegated_task(self, arguments: dict[str, Any]) -> str:
        """Run one sub-task to completion in a fresh, isolated `AgentSession` and
        return only its final text — the parent's `messages` never see the
        sub-agent's own tool calls, just this summary.

        Shares `root`, `confirm_write`, and `tool_executor` with the parent, so a
        delegated write hits the same path-safety checks and the same
        confirmation UI as any other write — delegation changes who's asking,
        not whether it's asked. It also inherits search, MCP and extra tools
        (filtered to what it was granted), timeouts and tool logging.
        """
        task = arguments.get("task")
        if not isinstance(task, str) or not task.strip():
            return "delegate_task requires a non-empty 'task' string."

        role = self.agent_roles.get(str(arguments.get("agent") or ""))
        delegable = self._known_tool_names - {"delegate_task"}
        requested = arguments.get("allowed_tools")
        if isinstance(requested, list) and requested and all(isinstance(t, str) for t in requested):
            allowed = frozenset(requested) & delegable
        elif role is not None and role.tools:
            allowed = frozenset(role.tools) & delegable
        else:
            allowed = frozenset()
        if not allowed:
            allowed = (READ_ONLY_TOOL_NAMES | DEFAULT_SUB_AGENT_EXTRAS) & delegable

        requested_turns = arguments.get("max_turns")
        sub_max_turns = role.max_turns if role is not None else 10
        if isinstance(requested_turns, int) and requested_turns > 0:
            sub_max_turns = requested_turns

        may_write = bool(allowed & MUTATING_TOOL_NAMES)
        sub_session = AgentSession(
            self.backend,
            role.model if role is not None and role.model else self.model,
            self.root,
            confirm_write=self.confirm_write if may_write else None,
            confirm_search=None,  # the delegation itself was the approval
            confirm_mcp=self.confirm_mcp,
            tool_executor=self.tool_executor,
            max_turns=sub_max_turns,
            tool_schemas=[],
            allow_delegation=False,  # depth 1 — a sub-agent cannot itself delegate
            search_fn=self.search_fn if "web_search" in allowed else None,
            mcp_registry=self.mcp_registry,
            extra_tools=[t for n, t in self.extra_tools.items() if n in allowed],
            settings=self.settings,
            session_id=f"{self.session_id}:sub:{uuid.uuid4().hex[:6]}",
            num_ctx=self.num_ctx if (role is None or not role.model) else None,
            on_tool_log=self.on_tool_log,
        )
        # TOOL_SCHEMAS and the parent's list overlap; keep the first of each name.
        by_name: dict[str, dict[str, Any]] = {}
        for schema in [*TOOL_SCHEMAS, *self.tool_schemas]:
            by_name.setdefault(schema["function"]["name"], schema)
        sub_session.tool_schemas = [s for n, s in by_name.items() if n in allowed]
        sub_session.refresh_tools()
        if role is not None and role.prompt:
            sub_session.messages[0]["content"] = (
                sub_session.system_prompt() + f"\n\nYour role: {role.prompt}"
            )
        if self.on_event is not None:
            parent_event = self.on_event
            sub_session.on_event = lambda text: parent_event(f"  ↳ {text}")

        timeout = getattr(self.settings, "tool_timeout_delegate", None)
        try:
            content, timed_out = await _with_timeout(sub_session.send(task), timeout)
        except (AgentTurnLimitExceeded, AgentToolUseRequired) as exc:
            return f"Sub-agent stopped without finishing: {exc}"
        if timed_out:
            return f"Sub-agent timed out after {timeout}s without finishing."
        return content


_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")


def _remaining_references(root: Path, arguments: dict[str, Any]) -> str:
    """After an edit that removes an identifier (a rename, a deleted function),
    where it is still used — so a multi-file change isn't left half done."""
    old = arguments.get("old_string")
    new = arguments.get("new_string")
    if not isinstance(old, str) or not isinstance(new, str):
        return ""
    removed = set(_IDENTIFIER_RE.findall(old)) - set(_IDENTIFIER_RE.findall(new))
    removed -= {"def", "class", "return", "self", "import", "from", "None", "True", "False"}
    if not removed or len(removed) > 3:
        return ""
    notes: list[str] = []
    for identifier in sorted(removed):
        try:
            hits = search_files(root, rf"\b{re.escape(identifier)}\b", max_results=10)
        except (OSError, ValueError):
            continue
        if hits:
            places = ", ".join(hit.split(": ", 1)[0] for hit in hits)
            notes.append(f"'{identifier}' still appears at {places}")
    if not notes:
        return ""
    return "\nNote: " + "; ".join(notes) + ". Update those too if they refer to what you changed."


async def _with_timeout(coro: Awaitable[Any], timeout: float | None) -> tuple[str, bool]:
    """Await ``coro`` with an optional timeout: (text result, timed out?)."""
    if timeout is None:
        return str(await coro or ""), False
    try:
        return str(await asyncio.wait_for(coro, timeout=timeout) or ""), False
    except asyncio.TimeoutError:
        return "", True


def _json_tool_protocol(tool_schemas: list[dict[str, Any]]) -> str:
    """Tool descriptions for a model without native tool calling."""
    lines = [
        "\n\nYou cannot call tools natively. Every reply must be ONE JSON object:",
        '{"name": "<tool name>", "arguments": {...}} to call a tool, or',
        '{"name": "final_answer", "arguments": {"text": "<your answer>"}} when done.',
        "Available tools:",
    ]
    for schema in tool_schemas:
        function = schema["function"]
        params = ", ".join((function.get("parameters") or {}).get("properties") or {})
        lines.append(f"- {function['name']}({params}): {function.get('description', '')}")
    return "\n".join(lines)


def _json_tool_format(names: frozenset[str]) -> dict[str, Any]:
    """A JSON schema Ollama enforces with a grammar, so even a model with no tool
    support always produces a parseable tool call or final answer."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": sorted(names | {"final_answer"})},
            "arguments": {"type": "object"},
        },
        "required": ["name", "arguments"],
    }


def _parse_constrained_reply(
    content: str, known_names: frozenset[str]
) -> tuple[str | None, dict[str, Any] | None]:
    """(final answer, tool call) from a grammar-constrained reply."""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None, None
    if not isinstance(parsed, dict):
        return None, None
    if parsed.get("name") == "final_answer":
        arguments = parsed.get("arguments") or {}
        text = arguments.get("text") if isinstance(arguments, dict) else None
        return (str(text) if text is not None else ""), None
    return None, _try_parse_tool_call(content, known_names)


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
