"""Tools available to the coding agent: filesystem access, a project-wide search,
and read-only git awareness.

Every tool is confined to a single project root. The model only ever sees paths
relative to that root, and every path is resolved and checked against it before
any filesystem operation runs — a `../../etc/passwd` or an absolute path from the
model is rejected rather than followed. Paths matching `.gitignore` or
`.localgateignore` are invisible to every tool too, layered on top of that same
check, so secrets and generated directories stay out of the model's reach even if
it asks by name. This is the one piece of this module that must not have a bug:
it's the entire difference between "the agent edits your project" and "the agent
edits your filesystem."

There is deliberately no `run_command`/shell-execution tool. That's the single
riskiest capability a coding agent can have, and CODING_AGENT_PLAN.md calls for
adding it — if ever — behind mandatory per-call confirmation and a real sandbox,
not as a peer to these four read/write tools.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from localgate.agent import gitutil
from localgate.agent.ignore import is_ignored, load_patterns

#: Default page size for `read_file`. A whole large file in one tool result
#: crowds the conversation out of a small local context window.
READ_PAGE_LINES = 400

#: Tools that change the workspace. Every one is previewed, confirmed by policy,
#: checkpointed and tracked.
MUTATING_TOOL_NAMES = frozenset(
    {"write_file", "create_file", "update_file", "edit_file", "delete_file"}
)

#: OpenAI-style tool definitions, passed as `tools` on the chat completions request.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a text file in the project. Long files are returned in pages of "
                f"{READ_PAGE_LINES} lines; the footer says which offset to pass to read on."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the project root, e.g. 'src/app.py'.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "1-based line to start from. Defaults to 1.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": f"Lines to return. Defaults to {READ_PAGE_LINES}.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Change part of an existing file by replacing an exact snippet. Read the "
                "file first. old_string must match the file exactly, including indentation, "
                "and be unique unless replace_all is true. Prefer this over rewriting a "
                "whole file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Project-relative path."},
                    "old_string": {"type": "string", "description": "Exact text to replace."},
                    "new_string": {"type": "string", "description": "Replacement text."},
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every occurrence. Defaults to false.",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a new text file. Fails if the path already exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Project-relative path."},
                    "content": {"type": "string", "description": "Complete file contents."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_file",
            "description": "Replace an existing text file after reading it first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Project-relative path."},
                    "content": {"type": "string", "description": "Complete new contents."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete an existing file after reading it first.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Project-relative path."}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create or overwrite a text file in the project with the given content. "
                "Always read the file first if it already exists, so the write is an "
                "intentional full replacement rather than a guess."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the project root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The complete new contents of the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and subdirectories at a path in the project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the project root. Defaults to '.'.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": (
                "Search file contents in the project for a regular expression, like grep. "
                "Returns matching lines as 'path:line: text'. Use this to find relevant code "
                "without already knowing the exact filename."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "A regular expression."},
                    "path": {
                        "type": "string",
                        "description": (
                            "Directory to search under, relative to the project root. "
                            "Defaults to '.'."
                        ),
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Show which files are modified, staged, or untracked — read-only.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Show unstaged changes, optionally limited to one file — read-only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Limit the diff to this file, relative to the project root.",
                    }
                },
                "required": [],
            },
        },
    },
]

TOOL_NAMES = frozenset(schema["function"]["name"] for schema in TOOL_SCHEMAS)
_SCHEMAS_BY_NAME = {schema["function"]["name"]: schema for schema in TOOL_SCHEMAS}

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> str | None:
    """A short, precise complaint about ``arguments``, or ``None`` if they fit.

    Deliberately small: required keys and top-level types are what small models
    actually get wrong, and a one-line answer is what they can act on. Booleans
    are excluded from "integer" because ``True`` is an ``int`` in Python.
    """
    params = (schema.get("function") or {}).get("parameters") or {}
    properties: dict[str, Any] = params.get("properties") or {}
    missing = [key for key in params.get("required", []) if key not in arguments]
    if missing:
        return f"missing required {', '.join(repr(m) for m in missing)}"
    for key, value in arguments.items():
        expected = (properties.get(key) or {}).get("type")
        allowed = _JSON_TYPES.get(expected) if isinstance(expected, str) else None
        if allowed is None:
            continue
        if isinstance(value, bool) and expected in {"integer", "number"}:
            return f"{key!r} must be {expected}, got boolean"
        if not isinstance(value, allowed):
            return f"{key!r} must be {expected}, got {type(value).__name__}"
    return None


#: What a delegated sub-agent gets when the parent doesn't specify `allowed_tools`
#: — read-only, so delegation never grants a bigger trust boundary than the
#: top-level agent already has without the parent explicitly opting in per call.
READ_ONLY_TOOL_NAMES = frozenset(
    {"read_file", "list_directory", "search_files", "git_status", "git_diff"}
)
assert not READ_ONLY_TOOL_NAMES & MUTATING_TOOL_NAMES


#: Kept out of `TOOL_SCHEMAS`/`TOOL_NAMES` deliberately — this tool only exists
#: on a session `AgentSession` constructs with `allow_delegation=True`, and a
#: delegated sub-agent's own session never gets it (depth 1, no recursion). See
#: `AgentSession._run_delegated_task` in loop.py for the execution side.
def delegate_task_schema(roles: dict[str, Any] | None = None) -> dict[str, Any]:
    """The delegate_task tool, listing any named sub-agent roles it can use."""
    schema: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "delegate_task",
            "description": (
                "Hand off a self-contained sub-task (e.g. 'find every place foo() is called and "
                "summarize the call sites') to a fresh sub-agent, and get back only its final "
                "summary — the sub-agent's own exploration doesn't clutter this conversation. "
                "Only use this for sub-tasks that genuinely benefit from an isolated context; a "
                "single-file edit or a quick lookup should stay in the current turn instead. The "
                "sub-agent cannot itself delegate further."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The sub-task to hand off, as a complete, self-contained "
                        "instruction — the sub-agent has no memory of this conversation.",
                    },
                    "allowed_tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Tool names the sub-agent may use. Defaults to read-only "
                            f"({', '.join(sorted(READ_ONLY_TOOL_NAMES))}). Include 'edit_file' "
                            "only if this sub-task is actually expected to edit files — its "
                            "writes still go through the same confirmation as any other write."
                        ),
                    },
                    "max_turns": {
                        "type": "integer",
                        "description": (
                            "Tool-call turns before the sub-agent gives up. Defaults to 10."
                        ),
                    },
                },
                "required": ["task"],
            },
        },
    }
    if roles:
        listing = "; ".join(f"{name}: {role.description}" for name, role in sorted(roles.items()))
        schema["function"]["parameters"]["properties"]["agent"] = {
            "type": "string",
            "enum": sorted(roles),
            "description": f"Optional specialist to run the sub-task. {listing}",
        }
    return schema


DELEGATE_TASK_SCHEMA: dict[str, Any] = delegate_task_schema()

#: Beyond read-only file tools, what a sub-agent gets by default when the parent
#: has them: research tools, which only read.
DEFAULT_SUB_AGENT_EXTRAS = frozenset({"web_search", "fetch_url"})


class PathEscapeError(ValueError):
    """Raised when a tool argument would resolve outside the project root."""


class IgnoredPathError(ValueError):
    """Raised when a tool argument matches .gitignore/.localgateignore."""


def resolve_within(root: Path, relative: str) -> Path:
    """Resolve ``relative`` against ``root`` and refuse anything that escapes it.

    Handles both `../` traversal and absolute paths (which `Path.__truediv__`
    would otherwise happily accept, silently discarding ``root``).
    """
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise PathEscapeError("path must be a non-empty string without NUL bytes")
    raw = Path(relative)
    if raw.is_absolute():
        raise PathEscapeError(f"absolute paths are not allowed: {relative!r}")

    root = root.resolve()
    lexical = root / raw
    # Reject every existing symlink component, even one that resolves back inside
    # the root. This keeps the policy stable if a link target changes later.
    current = root
    for part in raw.parts:
        if part in ("", "."):
            continue
        current = current / part
        if current.is_symlink():
            raise PathEscapeError(f"symlinks are not allowed in tool paths: {relative!r}")

    candidate = lexical.resolve()
    with contextlib.suppress(ValueError):
        candidate.relative_to(root)
        return candidate
    raise PathEscapeError(f"{relative!r} resolves outside the project root ({root})")


def ensure_visible(root: Path, candidate: Path) -> None:
    """Refuse a path excluded by `.gitignore`/`.localgateignore` (or inside `.git`)."""
    root = root.resolve()
    candidate = candidate.resolve()
    if is_ignored(root, candidate, load_patterns(root)):
        relative = candidate.relative_to(root)
        raise IgnoredPathError(f"{relative} is excluded by .gitignore/.localgateignore")


def read_file(root: Path, path: str) -> str:
    target = resolve_within(root, path)
    ensure_visible(root, target)
    if target.is_dir():
        raise IsADirectoryError(f"{path} is a directory; use list_directory to see what is in it")
    if not target.is_file():
        raise FileNotFoundError(f"No such file: {path}")
    return target.read_text(encoding="utf-8", errors="replace")


#: Text arguments a model writes into files, checked by `unescape_text_arguments`.
_TEXT_ARGUMENTS = ("content", "new_string", "old_string")


def unescape_text_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Undo a common small-model mistake: file text sent JSON-escaped twice, so it
    arrives as one line full of literal ``\\n`` sequences instead of newlines.

    Only applied when a value has no real newline at all but several literal ones —
    a single ``\\n`` inside a string literal in code is left alone.
    """
    fixed = dict(arguments)
    for key in _TEXT_ARGUMENTS:
        value = fixed.get(key)
        if isinstance(value, str) and "\n" not in value and value.count("\\n") >= 2:
            fixed[key] = (
                value.replace("\\r\\n", "\n")
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace('\\"', '"')
            )
    return fixed


def read_file_page(root: Path, path: str, offset: int = 1, limit: int = READ_PAGE_LINES) -> str:
    """`read_file` for the model: one page of lines, plus a footer when there is
    more, so a 3,000-line file costs one page of context rather than all of it.
    A file that fits in one page comes back exactly as stored.
    """
    text = read_file(root, path)
    lines = text.splitlines(keepends=True)
    offset = max(1, offset)
    limit = max(1, limit)
    if offset == 1 and len(lines) <= limit:
        return text
    if offset > len(lines) and lines:
        raise ValueError(f"offset {offset} is past the end of {path} ({len(lines)} lines)")
    end = min(len(lines), offset - 1 + limit)
    page = "".join(lines[offset - 1 : end])
    footer = f"\n[lines {offset}-{end} of {len(lines)}"
    if end < len(lines):
        footer += (
            f"; call read_file with offset={end + 1} for more, or use search_files to "
            "find specific text]"
        )
    else:
        footer += "]"
    return page + footer


def _read_exact(target: Path) -> str:
    """File text with its newlines untouched (no universal-newline translation),
    so an edit to a CRLF file on any OS stays CRLF."""
    with open(target, encoding="utf-8", errors="replace", newline="") as handle:
        return handle.read()


def _match_newlines(existing: str, snippet: str) -> str:
    if "\r\n" in existing and "\r\n" not in snippet:
        return snippet.replace("\n", "\r\n")
    return snippet


def apply_edit(
    root: Path, path: str, old_string: str, new_string: str, replace_all: bool = False
) -> str:
    """The file's contents after an `edit_file` call, without writing anything —
    used both to preview the change for confirmation and to perform it."""
    target = resolve_within(root, path)
    ensure_visible(root, target)
    if not target.is_file():
        raise FileNotFoundError(f"No such file: {path}")
    if not old_string:
        raise ValueError("old_string must not be empty; use create_file for new files")
    current = _read_exact(target)
    old = _match_newlines(current, old_string)
    new = _match_newlines(current, new_string)
    count = current.count(old)
    if count == 0:
        hint = ""
        first_line = old_string.strip().splitlines()[0].strip() if old_string.strip() else ""
        if first_line:
            for lineno, line in enumerate(current.splitlines(), start=1):
                if first_line in line:
                    hint = f" A similar line is at line {lineno}; re-read it and copy it exactly."
                    break
        raise ValueError(f"old_string was not found in {path}.{hint}")
    if count > 1 and not replace_all:
        raise ValueError(
            f"old_string occurs {count} times in {path}, so it is ambiguous and nothing was "
            "changed. Call edit_file again with the pair for the occurrence you mean:\n"
            + _describe_matches(current, old, new)
        )
    return current.replace(old, new) if replace_all else current.replace(old, new, 1)


def _describe_matches(text: str, needle: str, replacement: str, limit: int = 5) -> str:
    """Each occurrence of ``needle`` as a complete, ready-to-send edit: a unique
    ``old_string`` (the match extended to the start of the line above it) and the
    matching ``new_string`` with the caller's intended change already applied."""
    out: list[str] = []
    start = 0
    while len(out) < limit:
        index = text.find(needle, start)
        if index < 0:
            break
        lineno = text.count("\n", 0, index) + 1
        line_start = text.rfind("\n", 0, index) + 1
        prev_start = text.rfind("\n", 0, max(0, line_start - 1)) + 1 if line_start else 0
        snippet = text[prev_start : index + len(needle)]
        if text.count(snippet) != 1:
            snippet = text[line_start : index + len(needle)]
        fixed = snippet[: len(snippet) - len(needle)] + replacement
        out.append(
            f"- match at line {lineno}: use old_string {json.dumps(snippet)} with "
            f"new_string {json.dumps(fixed)}"
        )
        start = index + max(1, len(needle))
    return "\n".join(out)


def edit_file(
    root: Path, path: str, old_string: str, new_string: str, replace_all: bool = False
) -> None:
    updated = apply_edit(root, path, old_string, new_string, replace_all)
    write_file(root, path, updated)


def _atomic_write(target: Path, content: str) -> None:
    """Write text without exposing a partially truncated destination.

    ``newline=""`` writes the text exactly as given on every OS: Windows would
    otherwise turn each ``\n`` into ``\r\n`` and POSIX would never produce one,
    so the same edit would give different bytes depending on the platform.
    """
    mode = target.stat().st_mode & 0o777 if target.exists() else None
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temp_name, mode)
        os.replace(temp_name, target)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)


def write_file(root: Path, path: str, content: str) -> None:
    target = resolve_within(root, path)
    ensure_visible(root, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Re-resolve after mkdir to close the obvious symlink-swap window.
    target = resolve_within(root, path)
    ensure_visible(root, target)
    if target.is_file():
        # Models write "\n"; keep an existing CRLF file CRLF on every OS.
        with contextlib.suppress(OSError):
            content = _match_newlines(_read_exact(target), content)
    _atomic_write(target, content)


def create_file(root: Path, path: str, content: str) -> None:
    target = resolve_within(root, path)
    ensure_visible(root, target)
    if target.exists():
        raise FileExistsError(f"Path already exists: {path}")
    write_file(root, path, content)


def update_file(root: Path, path: str, content: str) -> None:
    target = resolve_within(root, path)
    ensure_visible(root, target)
    if not target.is_file():
        raise FileNotFoundError(f"No such file: {path}")
    write_file(root, path, content)


def delete_file(root: Path, path: str) -> None:
    target = resolve_within(root, path)
    ensure_visible(root, target)
    if not target.is_file():
        raise FileNotFoundError(f"No such file: {path}")
    target.unlink()


def list_directory(root: Path, path: str = ".") -> list[str]:
    target = resolve_within(root, path)
    ensure_visible(root, target)
    if not target.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    root = root.resolve()
    patterns = load_patterns(root)
    entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
    return [
        f"{entry.name}/" if entry.is_dir() else entry.name
        for entry in entries
        if not entry.is_symlink() and not is_ignored(root, entry, patterns)
    ]


def search_files(root: Path, pattern: str, path: str = ".", max_results: int = 200) -> list[str]:
    """Grep-like content search under ``path``, skipping ignored and binary files."""
    start = resolve_within(root, path)
    ensure_visible(root, start)
    if not start.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    root = root.resolve()
    patterns = load_patterns(root)
    regex = re.compile(pattern)
    results: list[str] = []

    for file_path in sorted(start.rglob("*")):
        if file_path.is_symlink() or not file_path.is_file():
            continue
        try:
            canonical = file_path.resolve(strict=True)
            canonical.relative_to(root)
        except (OSError, ValueError):
            continue
        if is_ignored(root, canonical, patterns):
            continue
        try:
            raw = file_path.read_bytes()
        except OSError:
            continue
        if b"\0" in raw:
            continue  # a NUL byte is the cheap, standard heuristic for "this is binary"
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        relative = file_path.relative_to(root).as_posix()
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                results.append(f"{relative}:{lineno}: {line.strip()}")
                if len(results) >= max_results:
                    return results
    return results


def git_status(root: Path) -> str:
    if not gitutil.is_repo(root):
        return "Not a git repository."
    root = root.resolve()
    patterns = load_patterns(root)
    visible: list[str] = []
    for status_code, path in gitutil.status_entries(root):
        try:
            target = resolve_within(root, path)
        except ValueError:
            continue
        if not is_ignored(root, target, patterns):
            visible.append(f"{status_code} {path}")
    return "\n".join(visible) or "Working tree clean."


def git_diff(root: Path, path: str | None = None) -> str:
    if not gitutil.is_repo(root):
        return "Not a git repository."
    root = root.resolve()
    if path is not None:
        ensure_visible(root, resolve_within(root, path))
        return gitutil.diff(root, path) or "No changes."
    patterns = load_patterns(root)
    chunks: list[str] = []
    for changed in gitutil.diff_paths(root):
        try:
            target = resolve_within(root, changed)
        except ValueError:
            continue
        if not is_ignored(root, target, patterns):
            diff = gitutil.diff(root, changed)
            if diff:
                chunks.append(diff)
    return "".join(chunks) or "No changes."


def syntax_problem(root: Path, path: str) -> str | None:
    """A one-line syntax error for a just-written Python or JSON file, else None.

    Cheap, needs no sandbox, and catches the most common small-model mistake —
    a snippet edit that leaves the file unparseable — on the very next turn.
    """
    suffix = Path(path).suffix.lower()
    if suffix not in {".py", ".json"}:
        return None
    try:
        text = read_file(root, path)
    except (OSError, ValueError):
        return None
    if suffix == ".py":
        try:
            compile(text, path, "exec", dont_inherit=True)
        except SyntaxError as exc:
            return f"line {exc.lineno}: {exc.msg}"
        except ValueError:
            return None
        return None
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        return f"line {exc.lineno}: {exc.msg}"
    return None


SYNTAX_WARNING = "WARNING: the file now has a syntax error"


@dataclass(frozen=True)
class ToolCallResult:
    """The outcome of executing one tool call, ready to feed back to the model."""

    tool_call_id: str
    name: str
    content: str
    is_error: bool = False
    timed_out: bool = False


def execute_tool_call(
    root: Path, tool_call_id: str, name: str, arguments: dict[str, Any]
) -> ToolCallResult:
    """Run one tool call and turn any failure into a result the model can react to.

    A tool that raises would kill the whole agent loop over something as mundane
    as a typo'd filename; the model should see "No such file: foo.py" as a tool
    result and try again, not crash the CLI.
    """
    schema = _SCHEMAS_BY_NAME.get(name)
    if schema is not None:
        problem = validate_arguments(schema, arguments)
        if problem:
            return ToolCallResult(tool_call_id, name, f"Invalid arguments: {problem}", True)
    # Only warn about a syntax error the change introduced, not one already there.
    parsed_before = True
    if name in MUTATING_TOOL_NAMES and name != "delete_file":
        with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
            if resolve_within(root, arguments["path"]).is_file():
                parsed_before = syntax_problem(root, arguments["path"]) is None
    try:
        if name == "read_file":
            content = read_file_page(
                root,
                arguments["path"],
                int(arguments.get("offset") or 1),
                int(arguments.get("limit") or READ_PAGE_LINES),
            )
        elif name == "edit_file":
            edit_file(
                root,
                arguments["path"],
                arguments["old_string"],
                arguments["new_string"],
                bool(arguments.get("replace_all", False)),
            )
            content = f"Edited {arguments['path']}"
        elif name == "write_file":
            write_file(root, arguments["path"], arguments["content"])
            content = f"Wrote {len(arguments['content'])} bytes to {arguments['path']}"
        elif name == "create_file":
            create_file(root, arguments["path"], arguments["content"])
            content = f"Created {arguments['path']}"
        elif name == "update_file":
            update_file(root, arguments["path"], arguments["content"])
            content = f"Updated {arguments['path']}"
        elif name == "delete_file":
            delete_file(root, arguments["path"])
            content = f"Deleted {arguments['path']}"
        elif name == "list_directory":
            content = "\n".join(list_directory(root, arguments.get("path", ".")))
        elif name == "search_files":
            matches = search_files(root, arguments["pattern"], arguments.get("path", "."))
            content = "\n".join(matches) if matches else "No matches."
        elif name == "git_status":
            content = git_status(root)
        elif name == "git_diff":
            content = git_diff(root, arguments.get("path"))
        else:
            return ToolCallResult(tool_call_id, name, f"Unknown tool: {name}", is_error=True)
    except Exception as exc:  # noqa: BLE001 — any failure becomes a tool-result the model reads
        return ToolCallResult(tool_call_id, name, f"{type(exc).__name__}: {exc}", is_error=True)
    if parsed_before and name in MUTATING_TOOL_NAMES and name != "delete_file":
        problem = syntax_problem(root, arguments["path"])
        if problem:
            content += f"\n{SYNTAX_WARNING} ({problem}). Read the file and fix it with edit_file."
    return ToolCallResult(tool_call_id, name, content)
