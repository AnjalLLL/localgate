"""`!command` in the REPL: run a shell command yourself, not through the agent.

It runs in your own shell and environment (not the agent's sandbox — you typed
it), with output streamed as it arrives. The captured output is handed back so
the REPL can show it to the model on the next turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

#: Output kept for the model; the tail, since errors and summaries come last.
MAX_SHELL_OUTPUT_CHARS = 8_000


async def run_shell(
    root: Path, command: str, on_line: Callable[[str], None]
) -> tuple[int | None, str]:
    """Run ``command`` in ``root``; returns (exit code, captured output).

    Cancelling the awaiting task (Esc in the REPL) kills the whole process group.
    """
    kwargs: dict[str, Any] = {}
    if sys.platform != "win32":
        kwargs["start_new_session"] = True
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=str(root),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        **kwargs,
    )
    assert proc.stdout is not None
    lines: list[str] = []
    try:
        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            lines.append(line)
            on_line(line)
        code = await proc.wait()
    except asyncio.CancelledError:
        _kill(proc)
        with contextlib.suppress(Exception):
            await proc.wait()
        raise
    output = "\n".join(lines)
    if len(output) > MAX_SHELL_OUTPUT_CHARS:
        output = "[... earlier output omitted ...]\n" + output[-MAX_SHELL_OUTPUT_CHARS:]
    return code, output


def _kill(proc: asyncio.subprocess.Process) -> None:
    if sys.platform != "win32":
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError):
        proc.kill()


def shell_note(command: str, code: int | None, output: str) -> str:
    """How a `!command` run is described to the model on the next turn."""
    status = "was interrupted" if code is None else f"exited with code {code}"
    body = output.strip() or "(no output)"
    return f"I ran `{command}` in the project directory; it {status}. Output:\n```\n{body}\n```"
