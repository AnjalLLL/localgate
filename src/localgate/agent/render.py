"""Terminal rendering for the coding agent: colored diffs and status lines.

Kept separate from `cli.py` so the line-classification logic (`diff_lines`) is
testable without a terminal — golden-file tests assert on the returned
``(text, style)`` pairs directly rather than parsing captured ANSI output.
"""

from __future__ import annotations

import difflib

from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import Markdown
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


def print_diff(
    console: Console,
    path: str,
    old_content: str,
    new_content: str,
    *,
    syntax_theme: str = "ansi_dark",
) -> None:
    """Render a colored unified diff for a pending write, or the new file's
    contents with syntax highlighting when there's nothing to diff against.
    """
    if not old_content:
        console.print(f"[dim]--- new file {path} ---[/dim]")
        console.print(Syntax(new_content, _lexer_for(path), theme=syntax_theme, line_numbers=True))
        return
    console.rule(f"[bold]{path}[/bold]")
    for line, style in diff_lines(path, old_content, new_content):
        console.print(Text(line, style=style))
    console.rule()


_TODO_MARKS = {
    "done": ("☑", "green"),
    "in_progress": ("◐", "bold yellow"),
    "pending": ("☐", "dim"),
}


def todo_checklist(items: list[tuple[str, str]]) -> Text:
    """The agent's to-do list as a checklist: ☑ done, ◐ in progress, ☐ pending."""
    text = Text()
    for content, status in items:
        mark, style = _TODO_MARKS.get(status, _TODO_MARKS["pending"])
        text.append(f"  {mark} ", style=style)
        text.append(content, style="dim strike" if status == "done" else style)
        text.append("\n")
    text.rstrip()
    return text


class DraftView:
    """A reply that is still streaming in, as Markdown — only the last
    screenful, so a long reply scrolls instead of overflowing a transient
    live display (which can't erase lines that scrolled off the screen)."""

    def __init__(self, code_theme: str) -> None:
        self.text = ""
        self.code_theme = code_theme

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        lines = self.text.splitlines()
        room = max(3, console.size.height - 4)
        head, tail = lines[:-room], lines[-room:]
        # Cutting inside a code block would turn its tail into prose; reopen it.
        if sum(line.lstrip().startswith("```") for line in head) % 2:
            tail = ["```", *tail]
        yield Markdown("\n".join(tail), code_theme=self.code_theme)


def _lexer_for(path: str) -> str:
    suffix = path.rsplit(".", 1)[-1] if "." in path else ""
    return suffix or "text"
