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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from localgate.agent import gitutil
from localgate.agent.ignore import is_ignored, load_patterns

#: OpenAI-style tool definitions, passed as `tools` on the chat completions request.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a text file in the project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the project root, e.g. 'src/app.py'.",
                    }
                },
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


class PathEscapeError(ValueError):
    """Raised when a tool argument would resolve outside the project root."""


class IgnoredPathError(ValueError):
    """Raised when a tool argument matches .gitignore/.localgateignore."""


def resolve_within(root: Path, relative: str) -> Path:
    """Resolve ``relative`` against ``root`` and refuse anything that escapes it.

    Handles both `../` traversal and absolute paths (which `Path.__truediv__`
    would otherwise happily accept, silently discarding ``root``).
    """
    root = root.resolve()
    candidate = (root / relative).resolve()
    with contextlib.suppress(ValueError):
        candidate.relative_to(root)
        return candidate
    raise PathEscapeError(f"{relative!r} resolves outside the project root ({root})")


def ensure_visible(root: Path, candidate: Path) -> None:
    """Refuse a path excluded by `.gitignore`/`.localgateignore` (or inside `.git`)."""
    if is_ignored(root, candidate, load_patterns(root)):
        relative = candidate.relative_to(root.resolve())
        raise IgnoredPathError(f"{relative} is excluded by .gitignore/.localgateignore")


def read_file(root: Path, path: str) -> str:
    target = resolve_within(root, path)
    ensure_visible(root, target)
    if not target.is_file():
        raise FileNotFoundError(f"No such file: {path}")
    return target.read_text(encoding="utf-8", errors="replace")


def write_file(root: Path, path: str, content: str) -> None:
    target = resolve_within(root, path)
    ensure_visible(root, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


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
        if not is_ignored(root, entry, patterns)
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
        if not file_path.is_file() or is_ignored(root, file_path, patterns):
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
    return gitutil.status(root) or "Working tree clean."


def git_diff(root: Path, path: str | None = None) -> str:
    if not gitutil.is_repo(root):
        return "Not a git repository."
    if path is not None:
        ensure_visible(root, resolve_within(root, path))
    return gitutil.diff(root, path) or "No changes."


@dataclass(frozen=True)
class ToolCallResult:
    """The outcome of executing one tool call, ready to feed back to the model."""

    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


def execute_tool_call(
    root: Path, tool_call_id: str, name: str, arguments: dict[str, Any]
) -> ToolCallResult:
    """Run one tool call and turn any failure into a result the model can react to.

    A tool that raises would kill the whole agent loop over something as mundane
    as a typo'd filename; the model should see "No such file: foo.py" as a tool
    result and try again, not crash the CLI.
    """
    try:
        if name == "read_file":
            content = read_file(root, arguments["path"])
        elif name == "write_file":
            write_file(root, arguments["path"], arguments["content"])
            content = f"Wrote {len(arguments['content'])} bytes to {arguments['path']}"
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
    return ToolCallResult(tool_call_id, name, content)
