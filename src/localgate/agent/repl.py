"""Interactive REPL for `localgate code`, and the write-safety plumbing it shares
with the single-shot invocation: a dirty-tree gate, colored diffs before every
write, a spinner while waiting on the model, and `/undo`.

Kept out of `cli.py` because none of this is Typer-specific — it is pure asyncio
and Rich, and is exercised directly in tests without going through the CLI layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from localgate.agent import gitutil
from localgate.agent.loop import AgentSession, AgentTurnLimitExceeded
from localgate.agent.memory import AgentMemory
from localgate.agent.render import print_diff
from localgate.agent.tools import ToolCallResult, execute_tool_call

HELP_TEXT = "[dim]/exit  /clear  /model <name>  /undo[/dim]"


class WriteGate:
    """Confirms writes with a diff, gates once on a dirty tree, and tracks what
    changed so `/undo` and `--auto-commit` have something to act on.
    """

    def __init__(
        self,
        console: Console,
        root: Path,
        *,
        auto_approve: bool = False,
        force: bool = False,
        auto_commit: bool = False,
    ) -> None:
        self.console = console
        self.root = root
        self.auto_approve = auto_approve
        self.force = force
        self.auto_commit = auto_commit
        self.is_repo = gitutil.is_repo(root)
        self._dirty_checked = False
        self.last_written_path: str | None = None
        self.writes_this_turn: list[str] = []

    def _dirty_tree_ok(self) -> bool:
        """Warn about a dirty tree once per session; the user opts back in or bails."""
        if self._dirty_checked or not self.is_repo:
            self._dirty_checked = True
            return True
        self._dirty_checked = True
        if self.force or not gitutil.is_dirty(self.root):
            return True
        self.console.print(
            "[yellow]! uncommitted changes exist in this project — agent writes may be "
            "hard to distinguish from your own edits.[/yellow]"
        )
        return typer.confirm("Continue anyway?", default=False)

    def confirm_write(self, path: str, content: str) -> bool:
        if not self._dirty_tree_ok():
            return False

        target = self.root / path
        old_content = ""
        if target.is_file():
            try:
                old_content = target.read_text(encoding="utf-8", errors="replace")
            except OSError:
                old_content = ""
        print_diff(self.console, path, old_content, content)

        if self.auto_approve:
            return True
        return typer.confirm(f"Write {path}?", default=False)

    def tracking_executor(
        self, root: Path, tool_call_id: str, name: str, arguments: dict[str, Any]
    ) -> ToolCallResult:
        """Wraps the default tool executor to record successful writes."""
        result = execute_tool_call(root, tool_call_id, name, arguments)
        if name == "write_file" and not result.is_error:
            path = arguments.get("path", "?")
            self.last_written_path = path
            self.writes_this_turn.append(path)
        return result

    def after_turn(self, summary: str) -> None:
        """Auto-commit whatever was written this turn, if that's enabled."""
        if self.auto_commit and self.is_repo and self.writes_this_turn:
            message = f"{gitutil.AGENT_COMMIT_PREFIX} {summary[:60]}"
            gitutil.commit_all(self.root, message)
        self.writes_this_turn = []

    def undo(self) -> str:
        if not self.is_repo:
            return "Not a git repository — nothing to undo automatically."
        if self.auto_commit:
            message = gitutil.last_commit_message(self.root)
            if message is None or not message.startswith(gitutil.AGENT_COMMIT_PREFIX):
                return "The last commit wasn't made by the agent — refusing to reset it."
            gitutil.reset_hard_last(self.root)
            return f"Reset the last agent commit: {message}"
        if self.last_written_path is None:
            return "Nothing written yet this session to undo."
        outcome = gitutil.undo_file(self.root, self.last_written_path)
        self.last_written_path = None
        return outcome


async def run_turn(
    console: Console,
    session: AgentSession,
    gate: WriteGate,
    user_input: str,
    memory: AgentMemory | None = None,
) -> str:
    """Run one turn with a spinner that clears the moment the model produces
    anything — streamed text or a tool-call event — auto-commit afterward, and
    record the turn into conversation history/memory if a session is attached.
    """
    status = console.status("[dim]thinking...[/dim]", spinner="dots")
    status.start()
    stopped = False

    def stop() -> None:
        nonlocal stopped
        if not stopped:
            status.stop()
            stopped = True

    def on_token(text: str) -> None:
        stop()
        console.print(text, end="")

    def on_event(line: str) -> None:
        stop()
        console.print(f"[cyan]  {line}[/cyan]")

    session.on_token = on_token
    session.on_event = on_event
    if memory is not None:
        session.augment = memory.augment
    try:
        result = await session.send(user_input)
    finally:
        stop()

    gate.after_turn(user_input)
    console.print()
    if memory is not None:
        await memory.record_turn(user_input, result)
    return result


async def run_repl(
    backend: Any,
    model: str,
    root: Path,
    *,
    auto_approve: bool = False,
    force: bool = False,
    auto_commit: bool = False,
    memory: AgentMemory | None = None,
) -> None:
    """A persistent chat session in `root`, until `/exit` or EOF (Ctrl+D)."""
    console = Console()
    gate = WriteGate(console, root, auto_approve=auto_approve, force=force, auto_commit=auto_commit)
    session = AgentSession(
        backend, model, root, confirm_write=gate.confirm_write, tool_executor=gate.tracking_executor
    )

    console.print(f"[bold]localgate code[/bold] — {root}")
    console.print(HELP_TEXT)
    console.print()

    while True:
        try:
            line = console.input("[bold green]> [/bold green]")
        except EOFError:
            console.print()
            break
        line = line.strip()
        if not line:
            continue

        if line == "/exit":
            break
        if line == "/clear":
            session.reset()
            gate.writes_this_turn = []
            console.print("[dim]conversation cleared[/dim]")
            continue
        if line.startswith("/model"):
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                session.model = parts[1].strip()
                console.print(f"[dim]model set to {session.model}[/dim]")
            else:
                console.print(f"[dim]current model: {session.model}[/dim]")
            continue
        if line == "/undo":
            console.print(f"[yellow]{gate.undo()}[/yellow]")
            continue
        if line.startswith("/"):
            console.print(f"[dim]unknown command {line!r} — {HELP_TEXT}[/dim]")
            continue

        try:
            await run_turn(console, session, gate, line, memory=memory)
        except AgentTurnLimitExceeded as exc:
            console.print(f"[red]{exc}[/red]")
        except KeyboardInterrupt:
            console.print("\n[yellow]cancelled — session still open[/yellow]")


async def run_single_shot(
    backend: Any,
    model: str,
    root: Path,
    task: str,
    *,
    auto_approve: bool = False,
    force: bool = False,
    auto_commit: bool = False,
    memory: AgentMemory | None = None,
) -> str:
    """One task, one turn, with the same diff/spinner/streaming UI as the REPL."""
    console = Console()
    gate = WriteGate(console, root, auto_approve=auto_approve, force=force, auto_commit=auto_commit)
    session = AgentSession(
        backend, model, root, confirm_write=gate.confirm_write, tool_executor=gate.tracking_executor
    )
    return await run_turn(console, session, gate, task, memory=memory)
