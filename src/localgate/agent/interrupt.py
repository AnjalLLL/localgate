"""Keyboard input while the agent works: type ahead, switch mode, or interrupt.

While a turn runs, stdin is put in raw mode and read key by key. Esc (or Ctrl+C)
cancels the turn's asyncio task — Esc first clears a half-typed line, if there
is one. Printable keys build a line the REPL shows in its footer; Enter queues
it to run after the turn, Shift+Tab cycles the write mode. Anything that needs
to read a normal line from the terminal mid-turn (the [y/N] confirmations) must
call `paused()` around it — in raw mode Enter never ends a line, so the prompt
would hang.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from typing import Any

from prompt_toolkit.input import Input, create_input
from prompt_toolkit.keys import Keys

_SUBMIT_KEYS = {Keys.ControlM, Keys.ControlJ}
_BACKSPACE_KEYS = {Keys.ControlH, Keys.Backspace}


class InterruptWatcher:
    """Reads keys during a turn. A no-op when stdin isn't a terminal."""

    def __init__(
        self,
        *,
        on_change: Callable[[], None] | None = None,
        on_submit: Callable[[str], None] | None = None,
        on_cycle_mode: Callable[[], object] | None = None,
    ) -> None:
        self.task: asyncio.Future[Any] | None = None
        self.interrupted = False
        #: The line being typed for the next message.
        self.buffer = ""
        self.on_change = on_change
        self.on_submit = on_submit
        self.on_cycle_mode = on_cycle_mode
        self._input: Input | None = None
        self._active: list[AbstractContextManager[None]] = []

    def start(self, task: asyncio.Future[Any]) -> None:
        self.task = task
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return
        self._input = create_input()
        self._enter()

    def _enter(self) -> None:
        if self._input is None or self._active:
            return
        raw = self._input.raw_mode()
        raw.__enter__()
        attached = self._input.attach(self._on_keys)
        attached.__enter__()
        self._active = [raw, attached]

    def _exit(self) -> None:
        while self._active:
            with contextlib.suppress(Exception):
                self._active.pop().__exit__(None, None, None)

    def stop(self) -> None:
        self._exit()
        if self._input is not None:
            with contextlib.suppress(Exception):
                self._input.close()
        self._input = None

    @contextlib.contextmanager
    def paused(self) -> Iterator[None]:
        """Give the terminal back for a normal line read."""
        was_active = bool(self._active)
        self._exit()
        try:
            yield
        finally:
            if was_active:
                self._enter()

    def interrupt(self) -> None:
        self.interrupted = True
        if self.task is not None and not self.task.done():
            self.task.cancel()

    def _on_keys(self) -> None:
        if self._input is None:
            return
        # A lone Esc sits in the parser waiting for a possible escape sequence;
        # flushing hands it over now instead of on the next keypress.
        for press in [*self._input.read_keys(), *self._input.flush_keys()]:
            self.handle_key(press.key, press.data)

    def handle_key(self, key: Keys | str, data: str = "") -> None:
        """Apply one keypress (separate from `_on_keys` so it's testable)."""
        changed = True
        if key == Keys.ControlC:
            self.interrupt()
        elif key == Keys.Escape:
            if self.buffer:
                self.buffer = ""
            else:
                self.interrupt()
        elif key in _SUBMIT_KEYS:
            text = self.buffer.strip()
            self.buffer = ""
            if text and self.on_submit is not None:
                self.on_submit(text)
        elif key in _BACKSPACE_KEYS:
            self.buffer = self.buffer[:-1]
        elif key == Keys.ControlU:
            self.buffer = ""
        elif key == Keys.BackTab:
            if self.on_cycle_mode is not None:
                self.on_cycle_mode()
        elif key == Keys.BracketedPaste:
            self.buffer += data.replace("\r\n", " ").replace("\n", " ")
        elif not isinstance(key, Keys) and key.isprintable():
            self.buffer += key
        else:
            changed = False
        if changed and self.on_change is not None:
            self.on_change()
