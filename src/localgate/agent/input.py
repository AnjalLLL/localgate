"""The REPL's input line: `/` command and `@` file completion, `@file`
attachments, and multi-line editing.

Pure prompt_toolkit pieces with no terminal I/O of their own, so the completer
and the mention parser are testable by feeding them a `Document` directly.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from prompt_toolkit.application.current import get_app
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent

from localgate.agent.ignore import is_ignored, load_patterns
from localgate.agent.tools import IgnoredPathError, PathEscapeError, ensure_visible, resolve_within

#: `@path` at the start of the input or after whitespace (so `me@host` isn't one).
MENTION_RE = re.compile(r"(?<!\S)@([\w./-]+)")
#: Attachments bigger than this are skipped — the agent can still read_file them.
MAX_ATTACHMENT_BYTES = 100_000
#: Stop indexing a huge tree here; completion is a convenience, not a search.
MAX_INDEXED_FILES = 5_000
_INDEX_TTL_SECONDS = 10.0

#: Yields (command, description) pairs — built-ins plus whatever custom
#: commands exist right now, so new command files show up without a restart.
CommandsProvider = Callable[[], Iterable[tuple[str, str]]]


def project_files(root: Path) -> list[str]:
    """Visible project files, relative and POSIX-style, skipping ignored dirs."""
    root = root.resolve()
    patterns = load_patterns(root)
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d != ".git"
            and not (current / d).is_symlink()
            and not is_ignored(root, current / d, patterns)
        )
        for name in sorted(filenames):
            path = current / name
            if path.is_symlink() or is_ignored(root, path, patterns):
                continue
            found.append(path.relative_to(root).as_posix())
            if len(found) >= MAX_INDEXED_FILES:
                return found
    return found


class AgentCompleter(Completer):
    """`/` at the start completes command names; `@` completes file paths."""

    def __init__(self, root: Path, commands: CommandsProvider) -> None:
        self.root = root
        self.commands = commands
        self._files: list[str] = []
        self._indexed_at = float("-inf")

    def _file_index(self) -> list[str]:
        now = time.monotonic()
        if now - self._indexed_at > _INDEX_TTL_SECONDS:
            self._files = project_files(self.root)
            self._indexed_at = now
        return self._files

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text = document.text_before_cursor
        if text.startswith("/") and " " not in text:
            for command, description in self.commands():
                name = command.split()[0]
                if name.startswith(text):
                    yield Completion(name, start_position=-len(text), display_meta=description)
            return

        word = document.get_word_before_cursor(WORD=True)
        if not word.startswith("@"):
            return
        needle = word[1:].lower()
        files = self._file_index()
        # Prefix matches first, then anything containing the text.
        prefix = [f for f in files if f.lower().startswith(needle)]
        contains = [f for f in files if needle in f.lower() and f not in prefix]
        for path in (prefix + contains)[:50]:
            yield Completion(f"@{path}", start_position=-len(word), display=path)


def mentioned_files(root: Path, text: str) -> list[str]:
    """The `@path` mentions in ``text`` that name a visible, attachable project file."""
    found: list[str] = []
    for match in MENTION_RE.finditer(text):
        relative = match.group(1).rstrip(".,;:")
        try:
            target = resolve_within(root, relative)
            ensure_visible(root, target)
        except (PathEscapeError, IgnoredPathError, ValueError):
            continue
        if not target.is_file() or relative in found:
            continue
        try:
            if target.stat().st_size > MAX_ATTACHMENT_BYTES:
                continue
        except OSError:
            continue
        found.append(relative)
    return found


def input_key_bindings() -> KeyBindings:
    """Multi-line editing: a trailing backslash or Esc then Enter adds a newline
    (most terminals send plain Enter for Shift+Enter, so that can't be used)."""
    bindings = KeyBindings()

    @Condition
    def ends_with_backslash() -> bool:
        return get_app().current_buffer.document.text_before_cursor.endswith("\\")

    @bindings.add("enter", filter=ends_with_backslash)
    def _continue_line(event: KeyPressEvent) -> None:
        buffer = event.current_buffer
        buffer.delete_before_cursor(1)
        buffer.insert_text("\n")

    @bindings.add("escape", "enter")
    def _newline(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    return bindings
