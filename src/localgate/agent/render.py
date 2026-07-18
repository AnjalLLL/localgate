"""Terminal rendering for the coding agent: colored diffs and status lines.

Kept separate from `cli.py` so the line-classification logic (`diff_lines`) is
testable without a terminal — golden-file tests assert on the returned
``(text, style)`` pairs directly rather than parsing captured ANSI output.
"""

from __future__ import annotations

import difflib

from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text

_STYLE_FOR_PREFIX = (
    ("+++", "dim"),
    ("---", "dim"),
    ("@@", "cyan"),
    ("+", "green"),
    ("-", "red"),
)


def diff_lines(path: str, old_content: str, new_content: str) -> list[tuple[str, str]]:
    """A unified diff between ``old_content`` and ``new_content``, one (line, style) pair
    per line. ``old_content`` of ``""`` renders as an all-additions diff, which is
    exactly right for a brand-new file.
    """
    diff = difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=path,
        tofile=path,
    )
    lines: list[tuple[str, str]] = []
    for raw in diff:
        line = raw.rstrip("\n")
        style = "default"
        for prefix, prefix_style in _STYLE_FOR_PREFIX:
            if line.startswith(prefix):
                style = prefix_style
                break
        lines.append((line, style))
    return lines


def print_diff(console: Console, path: str, old_content: str, new_content: str) -> None:
    """Render a colored unified diff for a pending write, or the new file's
    contents with syntax highlighting when there's nothing to diff against.
    """
    if not old_content:
        console.print(f"[dim]--- new file {path} ---[/dim]")
        console.print(Syntax(new_content, _lexer_for(path), theme="ansi_dark", line_numbers=True))
        return
    console.rule(f"[bold]{path}[/bold]")
    for line, style in diff_lines(path, old_content, new_content):
        console.print(Text(line, style=style))
    console.rule()


def _lexer_for(path: str) -> str:
    suffix = path.rsplit(".", 1)[-1] if "." in path else ""
    return suffix or "text"
