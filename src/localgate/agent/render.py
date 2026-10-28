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
    Either is cut at ``MAX_DIFF_LINES`` with a count of what's left, so a
    whole-file rewrite doesn't scroll the confirmation prompt's context away.
    """
    if not old_content:
        lines = new_content.splitlines()
        console.print(f"[dim]--- new file {path} ({len(lines)} lines) ---[/dim]")
        shown = "\n".join(lines[:MAX_DIFF_LINES])
        console.print(Syntax(shown, _lexer_for(path), theme=syntax_theme, line_numbers=True))
        _print_more(console, len(lines) - MAX_DIFF_LINES)
        return
    diff = diff_lines(path, old_content, new_content)
    added = sum(1 for line, _ in diff if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line, _ in diff if line.startswith("-") and not line.startswith("---"))
    console.rule(f"[bold]{path}[/bold] [green]+{added}[/green] [red]-{removed}[/red]")
    for line, style in diff[:MAX_DIFF_LINES]:
        console.print(Text(line, style=style))
    _print_more(console, len(diff) - MAX_DIFF_LINES)
    console.rule()


#: Diff and new-file previews longer than this are cut, with a count of the rest.
MAX_DIFF_LINES = 40


def _print_more(console: Console, remaining: int) -> None:
    if remaining > 0:
        console.print(f"[dim]  … {remaining} more line{'s' if remaining != 1 else ''}[/dim]")


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


#: Code blocks longer than this are summarized in the chat instead of printed:
#: the agent's job is to put code in files, and a whole file in the terminal
#: buries everything else.
MAX_SHOWN_CODE_LINES = 12


def collapse_code(
    text: str, max_lines: int = MAX_SHOWN_CODE_LINES, *, streaming: bool = False
) -> tuple[str, int]:
    """``text`` with each fenced code block over ``max_lines`` replaced by a
    one-line note; returns it with the number of blocks hidden. With
    ``streaming``, a block that hasn't closed yet is summarized too."""
    lines = text.split("\n")
    out: list[str] = []
    hidden = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.lstrip().startswith("```"):
            out.append(line)
            i += 1
            continue
        language = line.strip()[3:].strip() or "code"
        end = i + 1
        while end < len(lines) and not lines[end].lstrip().startswith("```"):
            end += 1
        body = lines[i + 1 : end]
        closed = end < len(lines)
        if closed and len(body) <= max_lines:
            out.extend(lines[i : end + 1])
        elif closed:
            hidden += 1
            out.append(f"*⋯ {len(body)} lines of {language} hidden · /last shows them*")
        elif streaming:
            hidden += 1
            out.append(f"*⋯ writing {language}… {len(body)} lines*")
        else:
            out.extend(lines[i:])
        i = end + 1
    return "\n".join(out), hidden


class DraftView:
    """A reply that is still streaming in, as Markdown — long code collapsed to
    a line, and only the last screenful, so a long reply scrolls instead of
    overflowing a transient live display (which can't erase lines that
    scrolled off the screen)."""

    def __init__(self, code_theme: str, reserve: int = 4) -> None:
        self.text = ""
        self.code_theme = code_theme
        #: Screen rows kept free below the draft (for a footer drawn under it).
        self.reserve = reserve

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        if not self.text:
            return
        lines = collapse_code(self.text, streaming=True)[0].splitlines()
        room = max(3, console.size.height - self.reserve)
        head, tail = lines[:-room], lines[-room:]
        # Cutting inside a code block would turn its tail into prose; reopen it.
        if sum(line.lstrip().startswith("```") for line in head) % 2:
            tail = ["```", *tail]
        yield Markdown("\n".join(tail), code_theme=self.code_theme)


def _lexer_for(path: str) -> str:
    suffix = path.rsplit(".", 1)[-1] if "." in path else ""
    return suffix or "text"
