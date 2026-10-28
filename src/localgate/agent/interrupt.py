"""Esc (or Ctrl+C) cancels whatever the agent is doing, without leaving the REPL.

While a turn runs, stdin is put in raw mode and watched for those keys; one of
them cancels the turn's asyncio task. Anything that needs to read a normal
line from the terminal mid-turn (the [y/N] confirmations) must call `paused()`
around it — in raw mode Enter never ends a line, so the prompt would hang.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from collections.abc import Iterator
from contextlib import AbstractContextManager
from typing import Any

from prompt_toolkit.input import Input, create_input
from prompt_toolkit.keys import Keys

_INTERRUPT_KEYS = {Keys.Escape, Keys.ControlC}


class InterruptWatcher:
    """Cancels ``task`` on Esc/Ctrl+C. A no-op when stdin isn't a terminal."""

    def __init__(self) -> None:
        self.task: asyncio.Future[Any] | None = None
        self.interrupted = False
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

    def _on_keys(self) -> None:
        if self._input is None:
            return
        # A lone Esc sits in the parser waiting for a possible escape sequence;
        # flushing hands it over now instead of on the next keypress.
        keys = [*self._input.read_keys(), *self._input.flush_keys()]
        if any(key.key in _INTERRUPT_KEYS for key in keys):
            self.interrupted = True
            if self.task is not None and not self.task.done():
                self.task.cancel()
