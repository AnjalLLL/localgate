"""Interactive REPL for `localgate code`, and the write-safety plumbing it shares
with the single-shot invocation: a dirty-tree gate, colored diffs before every
write, a spinner while waiting on the model, and `/undo`.

Kept out of `cli.py` because none of this is Typer-specific — it is pure asyncio
and Rich, and is exercised directly in tests without going through the CLI layer.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, NamedTuple

import httpx
import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import get_app
from prompt_toolkit.filters import to_filter
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory, History, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.layout import BufferControl, HSplit, VerticalAlign
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from localgate import __version__, paths
from localgate.agent import gitutil
from localgate.agent import theme as theme_mod
from localgate.agent.commands import command_dirs, load_commands
from localgate.agent.input import (
    AgentCompleter,
    completing_command_or_path,
    input_key_bindings,
    mentioned_files,
)
from localgate.agent.instructions import (
    INIT_PROMPT,
    INSTRUCTION_FILES,
    project_instructions_file,
)
from localgate.agent.interrupt import InterruptWatcher
from localgate.agent.loop import AgentSession, AgentTurnLimitExceeded, AsyncTool
from localgate.agent.mcp import McpRegistry
from localgate.agent.memory import AgentMemory, list_project_sessions, set_current_project_session
from localgate.agent.render import DraftView, collapse_code, print_diff, todo_checklist
from localgate.agent.roles import AgentRole
from localgate.agent.sandbox import (
    RUN_COMMAND_GUIDANCE,
    RUN_COMMAND_SCHEMA,
    Sandbox,
    is_auto_allowed,
    run_sandboxed,
)
from localgate.agent.shell import run_shell, shell_note
from localgate.agent.theme import THEMES
from localgate.agent.todo import TODO_GUIDANCE, TODO_WRITE_SCHEMA, TodoList
from localgate.agent.tools import (
    MUTATING_TOOL_NAMES,
    ToolCallResult,
    ensure_visible,
    execute_tool_call,
    resolve_within,
)
from localgate.agent.userconfig import UserConfig
from localgate.agent.websearch import SearchFn
from localgate.core.token_counter import count_message_tokens, count_tokens
from localgate.integrations.caniollama import (
    CaniollamaClient,
    get_compat_for_models,
    overall_pass_rate,
)
from localgate.voice.terminal import VoiceIO
from localgate.web.fetch import FETCH_URL_SCHEMA, FetchError, fetch_page

#: (command, one-line description) — the single source of truth for both the
#: startup banner and `/help`, so a new command only needs to be added here once.
SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/exit", "End the session."),
    ("/clear", "Clear conversation history (keep model/settings)."),
    ("/model [name]", "Show a picker, or switch directly to <name>."),
    ("/theme [name]", "Show or set the color theme: sunrise, dark, light, none."),
    ("/mode [name]", "Show or set the write mode: manual, auto, plan (or shift+tab)."),
    ("/usage", "Request/token totals for this session."),
    ("/context", "How full the conversation is vs. the model's context window."),
    ("/compact [focus]", "Compact history into the session ledger now (keep <focus>)."),
    ("/ledger", "Show the session ledger: goal, decisions, files changed, to-dos."),
    ("/todos", "Show the agent's current to-do list."),
    ("/agents", "List the sub-agent roles delegate_task can use."),
    ("/voice [on|off]", "Read replies aloud with Piper (local text-to-speech)."),
    ("/listen", "Dictate your next message: speak, then press Enter (local Whisper)."),
    ("/config [key value]", "View, or set and persist, a preference."),
    ("/resume", "Pick a past session for this project to resume."),
    ("/last", "Show the last reply in full, including code hidden from the chat."),
    ("/undo", "Revert the most recent write."),
    ("/rewind [n]", "Step back through the last n writes (default 1)."),
    ("/mcp", "List connected MCP servers and their tools."),
    ("/tools", "List every tool available this session, and what's off and why."),
    ("/init", "Have the agent write LOCALGATE.md: standing instructions for this project."),
    ("/commands", "List custom commands from .localgate/commands/*.md."),
    ("/help", "Show this list."),
]

#: (keys, what they do) — shown by `/help` under the commands.
SHORTCUTS: list[tuple[str, str]] = [
    ("@path", "Attach a file to your message (tab completes paths)."),
    ("!command", "Run a shell command yourself; the agent sees its output next turn."),
    ("esc / ctrl+c", "Interrupt the agent mid-turn (the session stays open)."),
    ("\\ then enter", "Continue on a new line (or esc then enter)."),
    ("↑ / ↓, ctrl+r", "Recall or search earlier prompts."),
    ("shift+tab", "Cycle the write mode."),
]

_QUICK_COMMANDS = "/model  /theme  /mode  /tools  /help  /exit"

#: The three write-handling modes, cycled by Shift+Tab (in a real terminal) or
#: set once at startup via `--auto-approve`/`--plan`. Order matters: cycling
#: always advances through this tuple, wrapping back to "manual".
MODE_ORDER: tuple[str, ...] = ("manual", "auto", "plan")

_MODE_LABELS: dict[str, str] = {
    "manual": "manual accept",
    "auto": "auto-accept",
    "plan": "plan mode",
}


def mode_label(mode: str) -> str:
    return _MODE_LABELS.get(mode, mode)


@dataclass
class SessionUsage:
    """In-process token/request totals for the current `localgate code`
    invocation — deliberately not a DB query, since the durable `UsageRecord`
    table is keyed by the one shared local-agent API key across every project,
    not by this particular run. See `AgentMemory.record_usage` for the durable
    side, which this doesn't replace.
    """

    request_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def record(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.request_count += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens


@dataclass
class Checkpoint:
    """The state of one file immediately before an agent write — enough to put
    it back. `previous_content is None` means the file didn't exist yet, so
    rewinding it means deleting it, not restoring empty content.
    """

    path: str
    previous_content: str | None


class SingleShotResult(NamedTuple):
    """`run_single_shot`'s outcome — `cli.py` uses `declined_a_write` to give a
    non-interactive caller a distinct exit code from a plain success.
    """

    text: str
    declined_a_write: bool


def describe_backend_error(exc: httpx.HTTPStatusError) -> str:
    """A readable message for a failed backend request — the inference server's
    own error body (e.g. Ollama's "does not support tools") is far more useful
    than the raw status line, when it's there to read.
    """
    detail: str | None = None
    try:
        body = exc.response.json()
    except ValueError:
        body = None
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            detail = error.get("message")
        elif isinstance(error, str):
            detail = error
    if detail is None:
        detail = exc.response.text.strip() or str(exc)
    return f"{exc.response.status_code}: {detail}"


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
        theme_name: str = theme_mod.DEFAULT_THEME,
        plan_mode: bool = False,
    ) -> None:
        self.console = console
        self.root = root
        self.auto_approve = auto_approve
        self.force = force
        self.auto_commit = auto_commit
        self.theme_name = theme_name
        #: When set, `confirm_write` buffers writes instead of applying or
        #: prompting for them one at a time — see `flush_plan`. Kept as an
        #: independent flag rather than folded into `auto_approve` so existing
        #: callers/tests that only ever set `auto_approve` are unaffected;
        #: `mode()`/`set_mode()` below are what let the REPL treat all three
        #: states (manual/auto/plan) as one cycle.
        self.plan_mode = plan_mode
        self.pending_writes: list[tuple[str, str | None]] = []
        self.is_repo = gitutil.is_repo(root)
        self._dirty_paths_at_start = (
            {path for _code, path in gitutil.status_entries(root)} if self.is_repo else set()
        )
        self._dirty_checked = False
        self.last_written_path: str | None = None
        self.writes_this_turn: list[str] = []
        #: Set once a write is explicitly turned down (not a dirty-tree bailout) —
        #: `run_single_shot` surfaces this to `cli.py` for a distinct exit code,
        #: since a script piping `localgate code` should be able to tell "the
        #: agent gave up" apart from "a write was withheld".
        self.declined_a_write = False
        #: Every write this session, oldest first — independent of `--auto-commit`,
        #: so `/rewind` has something to step through even when auto-commit is
        #: off. In-process only (not git objects): simpler and just as effective
        #: for "undo the last few writes", the actual thing `/rewind` needs to do.
        self.checkpoints: list[Checkpoint] = []

    def mode(self) -> str:
        if self.plan_mode:
            return "plan"
        return "auto" if self.auto_approve else "manual"

    def set_mode(self, mode: str) -> None:
        self.plan_mode = mode == "plan"
        self.auto_approve = mode == "auto"

    def cycle_mode(self) -> str:
        current = MODE_ORDER.index(self.mode())
        new_mode = MODE_ORDER[(current + 1) % len(MODE_ORDER)]
        self.set_mode(new_mode)
        return new_mode

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

    def _target(self, path: str) -> Path:
        target = resolve_within(self.root, path)
        ensure_visible(self.root, target)
        return target

    def confirm_write(self, path: str, content: str | None) -> bool:
        """Validate and preview a create/update/delete before it can run."""
        try:
            target = self._target(path)
        except ValueError as exc:
            self.console.print(f"[red]Refusing unsafe path {path!r}: {exc}[/red]")
            return False
        if not self._dirty_tree_ok():
            return False

        existed = target.is_file()
        old_content = ""
        if existed:
            try:
                old_content = target.read_text(encoding="utf-8", errors="replace")
            except OSError:
                old_content = ""
        print_diff(
            self.console,
            path,
            old_content,
            content or "",
            syntax_theme=theme_mod.syntax_theme_for(self.theme_name),
        )

        if self.plan_mode:
            self.pending_writes.append((path, content))
            self.console.print(f"[dim]queued for plan review: {path}[/dim]")
            return False

        destructive = existed or content is None
        if self.auto_approve and not destructive:
            self.checkpoints.append(Checkpoint(path, old_content if existed else None))
            return True
        action = "Delete" if content is None else "Overwrite" if existed else "Create"
        approved = typer.confirm(f"{action} {path}?", default=False)
        if approved:
            self.checkpoints.append(Checkpoint(path, old_content if existed else None))
        else:
            self.declined_a_write = True
        return approved

    def confirm_search(self, query: str) -> bool:
        """Only prompts in manual mode — auto/plan mode run web searches without
        asking, same as they do for writes. Defaults to yes (unlike writes):
        a search is read-only and reversible, so the friction only needs to be
        "you can see and stop it," not "you must opt in every time."
        """
        if self.mode() != "manual":
            return True
        return typer.confirm(f"Search the web for {query!r}?", default=True)

    def confirm_delegate(self, task: str) -> bool:
        """Same reasoning as `confirm_search` — manual mode only, defaults to yes."""
        if self.mode() != "manual":
            return True
        preview = task if len(task) <= 80 else task[:77] + "..."
        return typer.confirm(f"Delegate sub-task {preview!r}?", default=True)

    def confirm_fetch(self, _name: str, arguments: dict[str, Any]) -> bool:
        """Like `confirm_search`: manual mode asks, defaulting to yes."""
        if self.mode() != "manual":
            return True
        return typer.confirm(f"Fetch {arguments.get('url', '?')}?", default=True)

    def confirm_command(self, _name: str, arguments: dict[str, Any]) -> bool:
        """Commands run in a sandbox, but still change files in the project.
        Plan mode never runs them; auto mode runs only allow-listed commands
        (tests, linters, read-only git) without asking; everything else asks."""
        command = str(arguments.get("command", ""))
        if self.plan_mode:
            self.console.print(f"[dim]plan mode: not running {command!r}[/dim]")
            return False
        if self.auto_approve and is_auto_allowed(command):
            return True
        self.console.print(f"[bold]$ {command}[/bold]")
        return typer.confirm("Run this command in the sandbox?", default=False)

    def confirm_mcp(self, name: str, arguments: dict[str, Any]) -> bool:
        """MCP tools are externally supplied, so their side effects are unknown."""
        preview = str(arguments)
        if len(preview) > 100:
            preview = preview[:97] + "..."
        return typer.confirm(f"Allow MCP tool {name} with {preview}?", default=False)

    def flush_plan(self) -> None:
        """Show every write queued this turn together and apply the ones the
        user picks — the batch-review step that makes plan mode different from
        just declining writes one at a time.
        """
        if not self.pending_writes:
            return

        self.console.print(
            f"[bold]{len(self.pending_writes)} pending write(s) from this plan:[/bold]"
        )
        for path, content in self.pending_writes:
            try:
                target = self._target(path)
            except ValueError as exc:
                self.console.print(f"[red]Skipping unsafe path {path!r}: {exc}[/red]")
                continue
            old_content = ""
            if target.is_file():
                try:
                    old_content = target.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    old_content = ""
            print_diff(
                self.console,
                path,
                old_content,
                content or "",
                syntax_theme=theme_mod.syntax_theme_for(self.theme_name),
            )

        raw_choice = typer.prompt("Apply [a]ll / [n]one / [p]ick individually", default="n")
        choice = raw_choice.strip().lower()
        if choice.startswith("a"):
            selected = list(self.pending_writes)
        elif choice.startswith("p"):
            selected = [
                (path, content)
                for path, content in self.pending_writes
                if typer.confirm(f"Write {path}?", default=False)
            ]
        else:
            selected = []

        for path, content in selected:
            try:
                target = self._target(path)
            except ValueError:
                continue
            existed = target.is_file()
            old_content = ""
            if existed:
                try:
                    old_content = target.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    old_content = ""
            operation = "delete_file" if content is None else "write_file"
            arguments = {"path": path} if content is None else {"path": path, "content": content}
            result = execute_tool_call(self.root, "plan-apply", operation, arguments)
            if not result.is_error:
                self.checkpoints.append(Checkpoint(path, old_content if existed else None))
                self.last_written_path = path
                self.writes_this_turn.append(path)

        skipped = len(self.pending_writes) - len(selected)
        self.console.print(f"[dim]{len(selected)} written, {skipped} discarded[/dim]")
        self.pending_writes = []

    def tracking_executor(
        self, root: Path, tool_call_id: str, name: str, arguments: dict[str, Any]
    ) -> ToolCallResult:
        """Wraps the default tool executor to record successful writes."""
        result = execute_tool_call(root, tool_call_id, name, arguments)
        if name in MUTATING_TOOL_NAMES and not result.is_error:
            path = arguments.get("path", "?")
            self.last_written_path = path
            self.writes_this_turn.append(path)
        return result

    def after_turn(self, summary: str) -> None:
        """Auto-commit whatever was written this turn, if that's enabled."""
        if self.auto_commit and self.is_repo and self.writes_this_turn:
            safe_paths = [
                path for path in self.writes_this_turn if path not in self._dirty_paths_at_start
            ]
            skipped = sorted(set(self.writes_this_turn) - set(safe_paths))
            if skipped:
                self.console.print(
                    "[yellow]Not auto-committing paths that were already dirty: "
                    f"{', '.join(skipped)}[/yellow]"
                )
            message = f"{gitutil.AGENT_COMMIT_PREFIX} {summary[:60]}"
            if safe_paths:
                gitutil.commit_paths(self.root, message, safe_paths)
        self.writes_this_turn = []

    def undo(self) -> str:
        if not self.checkpoints:
            return "Nothing written yet this session to undo."
        checkpoint = self.checkpoints.pop()
        self._restore_checkpoint(checkpoint)
        self.last_written_path = None
        if self.auto_commit and self.is_repo:
            gitutil.commit_paths(
                self.root,
                f"{gitutil.AGENT_COMMIT_PREFIX} undo {checkpoint.path}",
                [checkpoint.path],
            )
        return f"Restored {checkpoint.path} to its pre-agent state."

    def _restore_checkpoint(self, checkpoint: Checkpoint) -> None:
        target = self._target(checkpoint.path)
        if checkpoint.previous_content is None:
            if target.exists():
                target.unlink()
        else:
            execute_tool_call(
                self.root,
                "checkpoint-restore",
                "write_file",
                {"path": checkpoint.path, "content": checkpoint.previous_content},
            )

    def rewind(self, steps: int = 1) -> str:
        """Step back through the last `steps` checkpoints, restoring each file's
        content from immediately before that write (or deleting it, if the write
        created the file). Unlike `undo`, this doesn't touch git history — it
        only restores file *content*; if `--auto-commit` already committed the
        writes being rewound, the working tree will end up dirty relative to
        that commit, which `/undo` (or a manual `git` command) can still clean up.
        """
        if not self.checkpoints:
            return "Nothing to rewind — no checkpoints recorded this session."
        steps = max(1, min(steps, len(self.checkpoints)))
        reverted: list[str] = []
        for _ in range(steps):
            checkpoint = self.checkpoints.pop()
            self._restore_checkpoint(checkpoint)
            reverted.append(checkpoint.path)
        return f"Rewound {len(reverted)} checkpoint(s): {', '.join(reverted)}"


async def _switch_model(
    console: Console,
    session: AgentSession,
    name: str,
    known: list[dict[str, Any]] | None = None,
    caniollama_client: CaniollamaClient | None = None,
) -> None:
    """Validate, warn, and apply a model switch — shared by `/model <name>` and
    picking a numbered entry from `/model`'s list.
    """
    if known is None:
        try:
            known = await session.backend.list_models_detailed()
        except httpx.HTTPError:
            known = None  # backend couldn't be asked — don't block the switch on it
    if known:
        names = {m["name"] for m in known}
        if name not in names:
            console.print(
                f"[red]no such model: {name!r} — run /model to see available models[/red]"
            )
            return

    if len(session.messages) > 1:
        console.print(
            "[yellow]Switching models mid-session — the next turn will resend full "
            "history without cache, and the new model may not support tool calling.[/yellow]"
        )
        answer = console.input("[yellow]Continue? [y/N] [/yellow]").strip().lower()
        if answer not in ("y", "yes"):
            console.print("[dim]model unchanged[/dim]")
            return

    # Prefer registry data when available, fall back to backend's check_tool_support
    warned = False
    if caniollama_client:
        compat = await caniollama_client.get_compat(name)
        if compat is not None:
            rate = overall_pass_rate(compat)
            if rate < 0.5:  # Threshold for unreliable
                console.print(
                    f"[red]{name}: registry reports {rate:.0%} structured-pass rate "
                    f"across {compat.total_reports} runs — switching will likely misbehave.[/red]"
                )
                answer = console.input("[yellow]Switch anyway? [y/N] [/yellow]").strip().lower()
                if answer not in ("y", "yes"):
                    return
                warned = True

    # Fallback to backend's check_tool_support if registry had no data
    supports_tools = await session.backend.check_tool_support(name)
    native = supports_tools is not False
    if not warned and supports_tools is False:
        if session.backend.name == "ollama":
            console.print(
                f"[yellow]{name} doesn't advertise tool-calling support — Localgate will use "
                "JSON tool mode (grammar-constrained replies) for it, which is slower and "
                "less reliable than native tools.[/yellow]"
            )
        else:
            console.print(
                f"[red]{name} doesn't advertise tool-calling support — switching would likely "
                "400 on the next turn.[/red]"
            )
        answer = console.input("[yellow]Switch anyway? [y/N] [/yellow]").strip().lower()
        if answer not in ("y", "yes"):
            return

    session.switch_model(name, native_tools=native or session.backend.name != "ollama")
    console.print(f"[dim]model set to {session.model}[/dim]")


async def _model_picker(
    console: Console, session: AgentSession, caniollama_client: CaniollamaClient | None = None
) -> None:
    """`/model` with no argument: a numbered, selectable list with size/quant info."""
    try:
        models = await session.backend.list_models_detailed()
    except httpx.HTTPError as exc:
        console.print(f"[red]couldn't list models: {exc}[/red]")
        return
    if not models:
        console.print("[dim]no models found[/dim]")
        return

    # Fan out caniollama lookups concurrently if the client is available
    compat_map = {}
    if caniollama_client:
        model_names = [m["name"] for m in models]
        compat_map = await get_compat_for_models(caniollama_client, model_names)

    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("model")
    table.add_column("params", justify="right", style="dim")
    table.add_column("quant", style="dim")
    table.add_column("size", justify="right", style="dim")
    if caniollama_client:
        table.add_column("tool calling")
    missing_compat = False
    for i, m in enumerate(models, start=1):
        current = m["name"] == session.model
        name = f"[bold green]● {escape(m['name'])}[/bold green]" if current else escape(m["name"])
        if "embed" in m["name"]:
            name += " [dim](embeddings only)[/dim]"
        row = [
            str(i),
            name,
            str(m.get("parameter_size") or ""),
            str(m.get("quantization") or ""),
            str(m.get("size_human") or ""),
        ]
        if caniollama_client:
            compat = compat_map.get(m["name"])
            if compat is not None:
                rate = overall_pass_rate(compat)
                color = "green" if rate >= 0.7 else "red"
                label = "reliable" if rate >= 0.7 else "unreliable"
                row.append(f"[{color}]{label} {rate:.0%}[/{color}]")
            else:
                missing_compat = True
                row.append("[dim]unknown[/dim]")
        table.add_row(*row)
    console.print(table)
    if missing_compat:
        console.print("[dim]  unknown? run: caniollama check <model>[/dim]")

    choice = console.input(
        "[bold green]❯ pick a model[/bold green] [dim](number or name, blank to cancel)[/dim] "
    ).strip()
    if not choice:
        return

    target: str | None = None
    if choice.isdigit() and 1 <= int(choice) <= len(models):
        target = models[int(choice) - 1]["name"]
    elif choice in {m["name"] for m in models}:
        target = choice
    if target is None:
        console.print(f"[red]no such model: {choice!r}[/red]")
        return
    await _switch_model(console, session, target, models, caniollama_client)


_THEME_NOTES = {
    "sunrise": "warm amber, for a dark terminal (default)",
    "dark": "classic green, for a dark terminal",
    "light": "deeper colors, for a light-background terminal",
    "none": "no colors",
}


def _apply_theme(
    theme_name: str,
    no_color: bool,
    gate: WriteGate,
    prompt_session: PromptSession[str] | None,
) -> Console:
    """Switch every part of the UI to ``theme_name``; returns the new console."""
    console = theme_mod.make_console(theme_name, no_color=no_color)
    gate.console = console
    gate.theme_name = theme_name
    if prompt_session is not None:
        prompt_session.style = theme_mod.prompt_style(theme_name, no_color=no_color)
    return console


def _theme_picker(console: Console, current: str) -> str | None:
    """`/theme` with no argument: show available themes and let user pick."""
    for i, name in enumerate(THEMES, start=1):
        marker = "[bold green]●[/bold green]" if name == current else " "
        swatch = "" if name == "none" else _theme_swatch(name)
        note = _THEME_NOTES.get(name, "")
        console.print(f" {marker} {i}. [bold]{name:<8}[/bold] {swatch} [dim]{note}[/dim]")
    choice = console.input(
        "[bold green]❯ pick a theme[/bold green] [dim](number or name, blank to cancel)[/dim] "
    ).strip()
    if not choice:
        return None
    if choice.isdigit() and 1 <= int(choice) <= len(THEMES):
        return THEMES[int(choice) - 1]
    if choice.lower() in THEMES:
        return choice.lower()
    console.print(f"[red]invalid choice: {choice!r}[/red]")
    return None


def _theme_swatch(theme_name: str) -> str:
    """Three colored blocks previewing a theme's accent, secondary and success."""
    p = theme_mod.palette_for(theme_name)
    return "".join(f"[{c}]■[/{c}]" for c in (p.accent, p.secondary, p.success))


def _set_config_value(cfg: UserConfig, key: str, raw_value: str) -> UserConfig:
    """Validate and apply one `/config <key> <value>` change. Raises `ValueError`
    with a message fit to show the user directly on anything invalid.
    """
    if key == "theme":
        if raw_value not in THEMES:
            raise ValueError(f"theme must be one of {', '.join(THEMES)}")
        return cfg.with_override(theme=raw_value)
    if key == "default_model":
        return cfg.with_override(default_model=raw_value or None)
    if key == "auto_approve":
        truthy = raw_value.strip().lower() in ("1", "true", "yes", "on")
        return cfg.with_override(auto_approve=truthy)
    if key == "max_turns":
        try:
            turns = int(raw_value)
        except ValueError as exc:
            raise ValueError("max_turns must be an integer") from exc
        if turns < 1:
            raise ValueError("max_turns must be at least 1")
        return cfg.with_override(max_turns=turns)
    known = ", ".join(f.name for f in fields(cfg))
    raise ValueError(f"unknown config key {key!r} — choose from {known}")


def _print_config(console: Console, cfg: UserConfig) -> None:
    console.print("[dim]~/.config/localgate/config.toml:[/dim]")
    for f in fields(cfg):
        console.print(f"  {f.name} = {getattr(cfg, f.name)}")


def _render_startup_banner(
    root: Path,
    model: str,
    mode: str,
    session: AgentSession,
    search_fn: Any,
    mcp_registry: Any,
) -> Panel:
    """Build a Rich Panel with project info, model, files, tools, and commands."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", no_wrap=True)
    grid.add_column()
    grid.add_row("project", f"[bold]{escape(str(root))}[/bold]")
    grid.add_row("model", f"[cyan]{escape(model)}[/cyan]")
    if not session.native_tools:
        grid.add_row(
            "",
            "[yellow]no native tool calling — using slower JSON mode; "
            "try /model qwen2.5-coder[/yellow]",
        )
    grid.add_row("mode", f"[bold green]{mode}[/bold green] [dim](shift+tab to cycle)[/dim]")

    # Files
    try:
        entries = sorted(root.iterdir())
        names = [
            e.name + ("/" if e.is_dir() else "") for e in entries if not e.name.startswith(".")
        ]
        if names:
            shown = " · ".join(escape(n) for n in names[:8])
            more = f" [dim]+{len(names) - 8} more[/dim]" if len(names) > 8 else ""
            grid.add_row("files", shown + more)
    except OSError:
        pass

    # Tools
    caps: list[str] = ["read", "edit", "search", "git", "todos"]
    if search_fn is not None:
        caps.extend(["web_search", "fetch_url"])
    if getattr(session, "sandbox", None) is not None:
        caps.append("run_command")
    if session.allow_delegation:
        caps.append("delegate")
    if mcp_registry is not None and mcp_registry.servers():
        names = [name for name, _ in mcp_registry.servers()]
        caps.extend(f"mcp:{n}" for n in names)
    grid.add_row("tools", " · ".join(caps))
    instructions = project_instructions_file(root)
    grid.add_row(
        "memory",
        f"{instructions.name}"
        if instructions
        else "[dim]none — /init to create LOCALGATE.md[/dim]",
    )

    tips = Text.from_markup(
        "[dim]Ask for a concrete change, e.g. [/dim]"
        '"add a dark theme to styles.css"\n'
        "[dim]@ attach a file · ! run a shell command · esc interrupts · "
        "\\ + enter for a new line[/dim]\n"
        f"[dim]{_QUICK_COMMANDS}[/dim]"
    )
    return Panel(
        Group(grid, Text(""), tips),
        title=f"[bold green]◆ localgate code[/bold green] [dim]v{__version__}[/dim]",
        title_align="left",
        subtitle="[dim]github.com/AnjalLLL/localgate[/dim]",
        subtitle_align="right",
        border_style="green",
        padding=(1, 2),
    )


def _print_tools(console: Console, session: AgentSession) -> None:
    """What the model can actually call this session, and — for the opt-in
    ones — why not, if they're off. The direct answer to "what decides when
    the agent searches/delegates": nothing but the model, from this exact list
    plus each tool's own description (see `system_prompt()` for the little
    extra steering the always-on tools don't get).
    """
    builtin = [
        "read_file",
        "create_file",
        "update_file",
        "edit_file",
        "delete_file",
        "write_file",
        "list_directory",
        "search_files",
        "git_status",
        "git_diff",
    ]
    console.print(f"[bold]built-in:[/bold] {', '.join(builtin)}")

    if session.allow_delegation:
        console.print("[bold]delegate_task:[/bold] enabled")
    else:
        console.print("[dim]delegate_task: disabled — enable with --allow-delegation[/dim]")

    if session.search_fn is not None:
        console.print("[bold]web_search:[/bold] enabled (DuckDuckGo by default)")
    else:
        console.print("[dim]web_search: disabled — enable with LOCALGATE_SEARCH_PROVIDER[/dim]")

    if "fetch_url" in session.extra_tools:
        console.print("[bold]fetch_url:[/bold] enabled (public http(s) pages only)")
    sandbox = getattr(session, "sandbox", None)
    if sandbox is not None:
        console.print(f"[bold]run_command:[/bold] {sandbox.describe()}")
    else:
        console.print(
            "[dim]run_command: disabled — no sandbox found (install bubblewrap on Linux, "
            "or Docker/Podman on any OS; macOS uses sandbox-exec)[/dim]"
        )
    if session.mcp_registry is not None and session.mcp_registry.servers():
        for server_name, tool_names in session.mcp_registry.servers():
            console.print(f"[bold]mcp:{server_name}:[/bold] {', '.join(tool_names)}")
    else:
        console.print(
            "[dim]mcp: no servers connected — see ~/.config/localgate/mcp_servers.json[/dim]"
        )


def _print_help(console: Console, root: Path) -> None:
    for cmd, desc in SLASH_COMMANDS:
        console.print(f"  [bold]{cmd:<22}[/bold] [dim]{desc}[/dim]")
    custom = load_commands(root)
    if custom:
        console.print("\n[bold]custom commands[/bold]")
        for command in custom.values():
            console.print(
                f"  [bold]/{command.name:<21}[/bold] [dim]{escape(command.description)}[/dim]"
            )
    console.print("\n[bold]shortcuts[/bold]")
    for keys, desc in SHORTCUTS:
        console.print(f"  [bold]{escape(keys):<22}[/bold] [dim]{desc}[/dim]")


def _print_custom_commands(console: Console, root: Path) -> None:
    custom = load_commands(root)
    if not custom:
        user_dir, project_dir = command_dirs(root)
        console.print(
            "[dim]no custom commands — add a Markdown file to "
            f"{escape(str(project_dir))} (this project) or {escape(str(user_dir))} "
            "(everywhere); its name becomes the command and its text the prompt, "
            "with $ARGUMENTS replaced by what you type after it.[/dim]"
        )
        return
    for command in custom.values():
        console.print(
            f"  [bold]/{command.name:<21}[/bold] [dim]{escape(command.description)} "
            f"({escape(str(command.source))})[/dim]"
        )


async def _run_user_shell(console: Console, root: Path, command: str) -> str | None:
    """`!command`: run it with output streamed, Esc to stop; returns the note
    for the model, or ``None`` if the command couldn't start."""
    console.print(f"[bold]$ {escape(command)}[/bold]")
    watcher = InterruptWatcher()
    task = asyncio.ensure_future(
        run_shell(root, command, lambda line: console.print(Text(line), highlight=False))
    )
    watcher.start(task)
    try:
        code, output = await task
    except asyncio.CancelledError:
        if not watcher.interrupted:
            raise
        console.print("[yellow]⏹ interrupted[/yellow]")
        return shell_note(command, None, "")
    except OSError as exc:
        console.print(f"[red]couldn't run it: {exc}[/red]")
        return None
    finally:
        watcher.stop()
    color = "green" if code == 0 else "red"
    console.print(f"[{color}]exit code {code}[/{color}] [dim]· the agent sees this next turn[/dim]")
    return shell_note(command, code, output)


async def _resume_command(
    console: Console, session: AgentSession, memory: AgentMemory | None, root: Path
) -> None:
    """List this project's past sessions and, on a pick, switch `memory` to it
    and rehydrate `session.messages` so the model actually resumes the
    conversation rather than just changing where new turns get logged.
    """
    if memory is None:
        console.print("[dim]memory is disabled (--no-memory) — nothing to resume.[/dim]")
        return
    entries = list_project_sessions(root)
    if not entries:
        console.print("[dim]no past sessions for this project.[/dim]")
        return

    console.print("[dim]sessions for this project (most recent first):[/dim]")
    for i, entry in enumerate(entries, start=1):
        marker = "*" if entry["id"] == memory.session_id else " "
        console.print(f"  {marker}{i}. {entry['id'][:8]}  {entry['created_at']}")

    choice = console.input(
        "[bold green]resume which (number, blank to cancel): [/bold green]"
    ).strip()
    if not choice:
        return
    if not choice.isdigit() or not (1 <= int(choice) <= len(entries)):
        console.print(f"[red]invalid selection: {choice!r}[/red]")
        return

    chosen = entries[int(choice) - 1]["id"]
    set_current_project_session(root, chosen)
    memory.session_id = chosen
    history = await memory.load_history()
    session.messages = [{"role": "system", "content": session.system_prompt()}, *history]
    session.ledger = await memory.load_ledger()
    console.print(
        f"[dim]resumed session {chosen[:8]} ({len(history)} prior message(s) loaded)[/dim]"
    )


async def _print_context(console: Console, session: AgentSession) -> None:
    """`/context`: where the context window goes — system prompt, ledger, history,
    tool schemas — against the window this session actually requests."""
    import json  # noqa: PLC0415

    system = count_message_tokens(session.messages[:1])
    history = count_message_tokens(session.messages[1:])
    tools = count_tokens(json.dumps(session.tool_schemas))
    ledger = 0 if session.ledger.is_empty() else count_tokens(session.ledger.render())
    used = system + history + tools + ledger
    window = session.num_ctx or await session.backend.context_window(session.model)
    if window:
        console.print(
            f"[dim]{used:,} / {window:,} tokens ({used / window * 100:.0f}% of context)[/dim]"
        )
    else:
        console.print(f"[dim]{used:,} tokens in the current conversation[/dim]")
    console.print(
        f"[dim]  system {system:,} · ledger {ledger:,} · history {history:,} · "
        f"tool schemas {tools:,}[/dim]"
    )
    if session.last_compaction is not None and session.last_compaction.changed:
        console.print(f"[dim]  last: {session.last_compaction.describe()}[/dim]")


def _make_prompt_session(
    gate: WriteGate, session: AgentSession | None = None
) -> PromptSession[str] | None:
    """A prompt_toolkit session whose Shift+Tab cycles `gate`'s write mode, with
    the current mode always visible in the bottom toolbar.

    Only for real interactive terminals — a key binding is meaningless without
    a terminal to bind it in, so piped/non-tty input (including every test in
    this suite) falls back to plain `Console.input` via `_read_line` below,
    with `/mode` as the always-available equivalent of the key binding.
    """
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None

    bindings = KeyBindings()

    @bindings.add("s-tab")
    def _cycle(event: Any) -> None:
        gate.cycle_mode()
        event.app.invalidate()

    # The toolbar redraws on every keystroke; recount tokens only when history changes.
    usage_cache: dict[int, int] = {}

    def _bottom_toolbar() -> FormattedText:
        # The rule closes the input frame opened by `_prompt_message`, so the
        # prompt reads as one compact box with the status line right under it.
        fragments: list[tuple[str, str]] = [("class:frame", _rule() + "\n")]
        if session is not None:
            fragments.append(("class:bottom-toolbar.model", f" {session.model}"))
        parts = [f"{mode_label(gate.mode())} (shift+tab)"]
        if session is not None and session.num_ctx:
            key = len(session.messages)
            if key not in usage_cache:
                usage_cache.clear()
                usage_cache[key] = session.context_usage()
            parts.append(f"context {usage_cache[key] / session.num_ctx:.0%}")
        parts.append("@ file · ! shell · /help")
        fragments.append(("", " · " + " · ".join(parts)))
        return FormattedText(fragments)

    completer = None
    if session is not None:
        root = session.root

        def _all_commands() -> list[tuple[str, str]]:
            custom = [(f"/{c.name}", c.description) for c in load_commands(root).values()]
            return [*SLASH_COMMANDS, *custom]

        completer = AgentCompleter(root, _all_commands)
    try:
        history: History = FileHistory(str(paths.ensure_data_dir() / "prompt_history"))
    except OSError:
        history = InMemoryHistory()
    prompt_session: PromptSession[str] = PromptSession(
        key_bindings=merge_key_bindings([bindings, input_key_bindings()]),
        bottom_toolbar=_bottom_toolbar,
        completer=completer,
        # Completion pops up only for a `/command` or an `@path`. That also decides
        # when prompt_toolkit holds blank lines open under the input for the menu:
        # always reserving them left a big gap between the prompt and the toolbar.
        complete_while_typing=completing_command_or_path,
        history=history,
        multiline=False,
        style=theme_mod.prompt_style(gate.theme_name),
    )
    _keep_toolbar_under_input(prompt_session)
    return prompt_session


def _keep_toolbar_under_input(prompt_session: PromptSession[str]) -> None:
    """Once prompt_toolkit learns the cursor position it stretches the input
    window over the rest of the screen, pinning the toolbar to the bottom row
    with a block of blank lines above it. Stop the input window from stretching
    and let the layout end right after the toolbar instead."""
    for window in prompt_session.layout.find_all_windows():
        control = window.content
        if isinstance(control, BufferControl) and control.buffer is prompt_session.default_buffer:
            window.dont_extend_height = to_filter(True)
    root = prompt_session.layout.container
    if isinstance(root, HSplit):
        root.align = VerticalAlign.TOP


def _rule() -> str:
    try:
        width = get_app().output.get_size().columns
    except Exception:  # noqa: BLE001 — no running app: fall back to a sane width
        width = 80
    return "─" * max(10, width)


def _prompt_message() -> FormattedText:
    return FormattedText([("class:frame", _rule() + "\n"), ("class:prompt", "❯ ")])


async def _read_line(console: Console, prompt_session: PromptSession[str] | None) -> str:
    if prompt_session is not None:
        return await prompt_session.prompt_async(_prompt_message)
    return console.input("[bold green]❯ [/bold green]")


async def run_turn(
    console: Console,
    session: AgentSession,
    gate: WriteGate,
    user_input: str,
    memory: AgentMemory | None = None,
    usage: SessionUsage | None = None,
    attachments: Sequence[str] = (),
) -> str:
    """Run one turn with a spinner that clears the moment the model streams
    real output, but otherwise stays visible (relabeled "working...") between
    tool calls — a slow one (a web search, a delegated sub-task, an MCP call)
    would otherwise leave a dead gap with no sign anything is still happening.
    While a reply streams in it is shown live, then replaced by the tool call
    or the final answer it turned out to be. Esc/Ctrl+C interrupts the turn.
    Auto-commits afterward, and records the turn into conversation
    history/memory if a session is attached.
    """
    status = console.status("[dim]thinking...[/dim]", spinner="dots")
    status.start()
    stopped = False
    started = time.monotonic()
    tool_calls = 0
    hidden_code = 0
    code_theme = theme_mod.syntax_theme_for(gate.theme_name)
    draft = DraftView(code_theme)
    draft_live: Live | None = None
    watcher = InterruptWatcher()

    def stop() -> None:
        nonlocal stopped, draft_live
        if draft_live is not None:
            draft_live.stop()  # transient: the draft disappears
            draft_live = None
        if not stopped:
            status.stop()
            stopped = True

    def restart() -> None:
        nonlocal stopped
        status.update("[dim]working...[/dim]")
        status.start()
        stopped = False

    def on_draft(text: str) -> None:
        nonlocal stopped, draft_live
        if draft_live is None:
            if not stopped:
                status.stop()
                stopped = True
            draft_live = Live(draft, console=console, transient=True, refresh_per_second=8)
            draft_live.start()
        draft.text = text

    def on_token(text: str) -> None:
        # The loop hands over the final answer in one piece, so it can be laid
        # out as Markdown — headings, lists, highlighted code — not raw ``###``.
        # Long code blocks are summarized; `/last` prints the reply in full.
        nonlocal hidden_code
        stop()
        shown, hidden_code = collapse_code(text)
        console.print()
        console.print(Markdown(shown, code_theme=code_theme))

    def on_event(line: str) -> None:
        nonlocal tool_calls
        stop()
        name, paren, args = line.partition("(")
        if paren and name.isidentifier():
            tool_calls += 1
            color = "bold green" if name in MUTATING_TOOL_NAMES else "cyan"
            console.print(
                f"  [{color}]●[/{color}] [bold]{escape(name)}[/bold] "
                f"[dim]{escape(args.removesuffix(')'))}[/dim]"
            )
        else:
            console.print(f"  [dim]● {escape(line)}[/dim]")
        restart()

    original_confirm_write = session.confirm_write
    original_confirm_search = session.confirm_search
    original_confirm_delegate = session.confirm_delegate
    original_confirm_mcp = session.confirm_mcp

    def with_visible_prompt(callback: Any) -> Any:
        def visible(*args: Any, **kwargs: Any) -> bool:
            stop()
            try:
                with watcher.paused():
                    return bool(callback(*args, **kwargs))
            finally:
                restart()

        return visible

    if original_confirm_write is not None:
        session.confirm_write = with_visible_prompt(original_confirm_write)
    if original_confirm_search is not None:
        session.confirm_search = with_visible_prompt(original_confirm_search)
    if original_confirm_delegate is not None:
        session.confirm_delegate = with_visible_prompt(original_confirm_delegate)
    if original_confirm_mcp is not None:
        session.confirm_mcp = with_visible_prompt(original_confirm_mcp)
    # run_command and fetch_url ask through their own `confirm` — without this the
    # spinner redraws over the [y/N] prompt and the turn looks hung.
    original_tool_confirms = {name: tool.confirm for name, tool in session.extra_tools.items()}
    for tool in session.extra_tools.values():
        if tool.confirm is not None:
            tool.confirm = with_visible_prompt(tool.confirm)

    session.on_token = on_token
    session.on_event = on_event
    session.on_draft = on_draft
    if memory is not None:
        session.augment = memory.augment
    for path in attachments:
        console.print(f"  [cyan]●[/cyan] [bold]attached[/bold] [dim]{escape(path)}[/dim]")
    history_before = len(session.messages)
    interrupted = False
    task = asyncio.ensure_future(session.send(user_input, attachments))
    watcher.start(task)
    try:
        result = await task
    except asyncio.CancelledError:
        if not watcher.interrupted:
            raise
        interrupted = True
        result = ""
        _forget_interrupted_turn(session, history_before, user_input)
    finally:
        watcher.stop()
        stop()
        session.on_draft = None
        session.confirm_write = original_confirm_write
        session.confirm_search = original_confirm_search
        session.confirm_delegate = original_confirm_delegate
        session.confirm_mcp = original_confirm_mcp
        for name, confirm in original_tool_confirms.items():
            session.extra_tools[name].confirm = confirm

    if gate.plan_mode:
        gate.flush_plan()

    # One summary line: what changed, how much work it took, how long.
    summary: list[str] = []
    if interrupted:
        summary.append("[yellow]⏹ interrupted[/yellow]")
    if gate.writes_this_turn:
        names = ", ".join(escape(n) for n in dict.fromkeys(gate.writes_this_turn))
        summary.append(f"[bold green]✓ wrote {names}[/bold green]")
    elif hidden_code:
        # Say it plainly: code in the chat is not code in the project.
        summary.append("[yellow]⚠ no files changed — the code above is only in the reply[/yellow]")
    else:
        summary.append("[dim]no files changed[/dim]")
    if tool_calls:
        summary.append(f"[dim]{tool_calls} tool call{'s' if tool_calls != 1 else ''}[/dim]")
    summary.append(f"[dim]{time.monotonic() - started:.1f}s[/dim]")
    console.print()
    console.print("  " + " [dim]·[/dim] ".join(summary))

    # A freshly written LOCALGATE.md (by /init or by hand-off) applies from the next turn.
    if any(Path(path).name in INSTRUCTION_FILES for path in gate.writes_this_turn):
        session.refresh_tools()

    gate.after_turn(user_input)
    console.print()
    if interrupted:
        return result
    if usage is not None:
        prompt_tokens = count_message_tokens(session.messages[:-1])
        completion_tokens = count_tokens(result)
        usage.record(prompt_tokens, completion_tokens)
        if memory is not None:
            await memory.record_usage(session.model, prompt_tokens, completion_tokens)
    if memory is not None:
        await memory.record_turn(user_input, result)
        with contextlib.suppress(Exception):
            await memory.save_ledger(session.ledger)
    return result


def _forget_interrupted_turn(session: AgentSession, history_before: int, user_input: str) -> None:
    """Drop a cancelled turn's half-finished messages — an assistant tool call
    with no result would make the next request invalid — and leave a short note
    so the model knows the request was cut off. On the very first turn, roll
    back completely so the next one still gets the project overview."""
    del session.messages[history_before:]
    if history_before <= 1:
        return
    session.messages.append({"role": "user", "content": user_input})
    session.messages.append(
        {"role": "assistant", "content": "(Interrupted by the user before I finished.)"}
    )


async def build_agent_session(
    backend: Any,
    model: str,
    root: Path,
    gate: WriteGate,
    console: Console,
    *,
    max_turns: int = 20,
    allow_delegation: bool = False,
    search_fn: SearchFn | None = None,
    mcp_registry: McpRegistry | None = None,
    settings: Any | None = None,
    session_id: str | None = None,
    memory: AgentMemory | None = None,
    sandbox: Sandbox | None = None,
    agent_roles: dict[str, AgentRole] | None = None,
    native_tools: bool = True,
) -> AgentSession:
    """The `AgentSession` both the REPL and single-shot runs use: the write gate's
    confirmations, the optional tools this machine supports (web fetch, sandboxed
    commands, the to-do list), durable tool logging, and the persisted ledger."""
    todos = TodoList()
    extra_tools: list[AsyncTool] = []
    if getattr(settings, "agent_todo_tool", False):
        extra_tools.append(AsyncTool(TODO_WRITE_SCHEMA, todos.write, guidance=TODO_GUIDANCE))
    if search_fn is not None:
        allow_private = bool(getattr(settings, "fetch_allow_private", False))

        async def fetch(arguments: dict[str, Any]) -> str:
            try:
                page = await fetch_page(
                    str(arguments["url"]),
                    max_chars=int(arguments.get("max_chars") or 12_000),
                    allow_private=allow_private,
                )
            except FetchError as exc:
                return f"Could not fetch: {exc}"
            return page.render()

        extra_tools.append(
            AsyncTool(
                FETCH_URL_SCHEMA,
                fetch,
                confirm=gate.confirm_fetch,
                timeout=float(getattr(settings, "tool_timeout_search", 20.0)) + 10,
            )
        )
    if sandbox is not None:
        default_timeout = float(getattr(settings, "tool_timeout_command", 120.0))

        async def run_command(arguments: dict[str, Any]) -> str:
            timeout = arguments.get("timeout")
            limit = float(timeout) if isinstance(timeout, int) and timeout > 0 else default_timeout
            result = await run_sandboxed(sandbox, root, str(arguments["command"]), timeout=limit)
            return result.render()

        extra_tools.append(
            AsyncTool(
                RUN_COMMAND_SCHEMA,
                run_command,
                confirm=gate.confirm_command,
                read_only=False,
                guidance=RUN_COMMAND_GUIDANCE,
            )
        )

    session = AgentSession(
        backend,
        model,
        root,
        confirm_write=gate.confirm_write,
        confirm_search=gate.confirm_search,
        confirm_delegate=gate.confirm_delegate,
        confirm_mcp=gate.confirm_mcp,
        tool_executor=gate.tracking_executor,
        max_turns=max_turns,
        allow_delegation=allow_delegation,
        search_fn=search_fn,
        mcp_registry=mcp_registry,
        auto_read=True,
        settings=settings,
        session_id=session_id,
        extra_tools=extra_tools,
        native_tools=native_tools,
        on_tool_log=memory.record_tool_call if memory is not None else None,
        agent_roles=agent_roles,
    )
    session.sandbox = sandbox  # type: ignore[attr-defined]
    session.todos = todos  # type: ignore[attr-defined]

    def show_todos(current: TodoList) -> None:
        session.ledger.open_todos = current.pending()
        if current.items:
            gate.console.print(todo_checklist(current.items))  # follows /theme

    todos.on_change = show_todos
    if memory is not None:
        # A missing table or a bad row just starts a fresh ledger.
        with contextlib.suppress(Exception):
            session.ledger = await memory.load_ledger()
    return session


async def run_repl(
    backend: Any,
    model: str,
    root: Path,
    *,
    auto_approve: bool = False,
    force: bool = False,
    auto_commit: bool = False,
    memory: AgentMemory | None = None,
    theme: str = theme_mod.DEFAULT_THEME,
    no_color: bool = False,
    max_turns: int = 20,
    user_config: UserConfig | None = None,
    plan_mode: bool = False,
    allow_delegation: bool = False,
    search_fn: SearchFn | None = None,
    mcp_registry: McpRegistry | None = None,
    settings: Any | None = None,
    session_id: str | None = None,
    sandbox: Sandbox | None = None,
    agent_roles: dict[str, AgentRole] | None = None,
    native_tools: bool = True,
    voice: VoiceIO | None = None,
) -> None:
    """A persistent chat session in `root`, until `/exit` or EOF (Ctrl+D)."""
    console = theme_mod.make_console(theme, no_color=no_color)
    theme_name = theme
    cfg = user_config or UserConfig()
    usage = SessionUsage()
    gate = WriteGate(
        console,
        root,
        auto_approve=auto_approve,
        force=force,
        auto_commit=auto_commit,
        theme_name=theme_name,
        plan_mode=plan_mode,
    )
    session = await build_agent_session(
        backend,
        model,
        root,
        gate,
        console,
        max_turns=max_turns,
        allow_delegation=allow_delegation,
        search_fn=search_fn,
        mcp_registry=mcp_registry,
        settings=settings,
        session_id=session_id,
        memory=memory,
        sandbox=sandbox,
        agent_roles=agent_roles,
        native_tools=native_tools,
    )
    prompt_session = _make_prompt_session(gate, session)
    if prompt_session is not None and no_color:
        prompt_session.style = theme_mod.prompt_style(theme_name, no_color=True)

    # Initialize caniollama client if settings are available
    caniollama_client = None
    if settings is not None:
        caniollama_client = CaniollamaClient(
            registry_url=settings.caniollama_registry_url,
            enabled=settings.caniollama_enabled,
            timeout=settings.caniollama_timeout,
        )

    banner = _render_startup_banner(
        root, model, mode_label(gate.mode()), session, search_fn, mcp_registry
    )
    console.print(banner)
    #: `!command` results waiting to be shown to the model with the next message.
    shell_notes: list[str] = []
    #: The last full reply, for `/last` (the chat shows it with long code collapsed).
    last_reply = ""

    while True:
        try:
            line = await _read_line(console, prompt_session)
        except EOFError:
            console.print()
            break
        line = line.strip()
        if not line:
            continue
        if line == "/listen":
            if voice is None:
                console.print("[dim]voice is off on this server (LOCALGATE_VOICE_ENABLED)[/dim]")
                continue
            heard = await voice.listen()
            if not heard:
                continue
            console.print(f"[dim]you said:[/dim] {heard}")
            line = heard

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
                await _switch_model(
                    console, session, parts[1].strip(), caniollama_client=caniollama_client
                )
            else:
                await _model_picker(console, session, caniollama_client)
            continue
        if line.startswith("/theme"):
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                requested = parts[1].strip().lower()
                if requested not in THEMES:
                    console.print(
                        f"[red]invalid theme {requested!r} — choose from {', '.join(THEMES)}[/red]"
                    )
                else:
                    theme_name = requested
            else:
                theme_name = _theme_picker(console, theme_name) or theme_name
            if theme_name != cfg.theme:
                console = _apply_theme(theme_name, no_color, gate, prompt_session)
                cfg = cfg.with_override(theme=theme_name)
                cfg.save()
                console.print(f"[green]theme set to {theme_name}[/green]")
                if theme_name == "light":
                    console.print(
                        "[dim]Colors for a light-background terminal. The background itself "
                        "is set in your terminal app's settings.[/dim]"
                    )
            continue
        if line.startswith("/mode"):
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                requested = parts[1].strip().lower()
                if requested not in MODE_ORDER:
                    console.print(
                        f"[red]invalid mode {requested!r} — choose from "
                        f"{', '.join(MODE_ORDER)}[/red]"
                    )
                else:
                    gate.set_mode(requested)
                    console.print(f"[dim]mode set to {mode_label(gate.mode())}[/dim]")
            else:
                console.print(f"[dim]current mode: {mode_label(gate.mode())}[/dim]")
            continue
        if line == "/undo":
            console.print(f"[yellow]{gate.undo()}[/yellow]")
            continue
        if line.startswith("/rewind"):
            parts = line.split(maxsplit=1)
            steps = 1
            if len(parts) == 2:
                if not parts[1].strip().isdigit():
                    console.print(f"[red]/rewind expects a number, got {parts[1]!r}[/red]")
                    continue
                steps = int(parts[1].strip())
            console.print(f"[yellow]{gate.rewind(steps)}[/yellow]")
            continue
        if line == "/last":
            if last_reply:
                console.print(
                    Markdown(last_reply, code_theme=theme_mod.syntax_theme_for(theme_name))
                )
            else:
                console.print("[dim]no reply yet[/dim]")
            continue
        if line == "/help":
            _print_help(console, root)
            continue
        if line == "/commands":
            _print_custom_commands(console, root)
            continue
        if line == "/usage":
            console.print(
                f"[dim]{usage.request_count} request(s) this session — "
                f"{usage.prompt_tokens:,} prompt + {usage.completion_tokens:,} completion = "
                f"{usage.total_tokens:,} tokens (approximate)[/dim]"
            )
            continue
        if line == "/context":
            await _print_context(console, session)
            continue
        if line.startswith("/compact"):
            focus = line[len("/compact") :].strip() or None
            report = await session.compact(focus)
            console.print(f"[dim]{report.describe()}[/dim]")
            continue
        if line == "/ledger":
            ledger = session.ledger
            console.print(
                ledger.render() if not ledger.is_empty() else "[dim]ledger is empty[/dim]"
            )
            continue
        if line == "/todos":
            todos = getattr(session, "todos", None)
            if todos is None or not todos.items:
                console.print("[dim]no to-dos[/dim]")
            else:
                console.print(todo_checklist(todos.items))
            continue
        if line == "/agents":
            if not session.allow_delegation:
                console.print("[dim]delegation is off — start with --allow-delegation[/dim]")
            for role in session.agent_roles.values():
                model_note = f"  [dim]model: {role.model}[/dim]" if role.model else ""
                console.print(f"  [bold]{role.name}[/bold] — {role.description}{model_note}")
            continue
        if line.startswith("/config"):
            parts = line.split(maxsplit=2)
            if len(parts) == 1:
                _print_config(console, cfg)
            elif len(parts) == 3:
                try:
                    cfg = _set_config_value(cfg, parts[1], parts[2])
                except ValueError as exc:
                    console.print(f"[red]{exc}[/red]")
                else:
                    cfg.save()
                    if parts[1] == "theme":
                        theme_name = cfg.theme
                        console = _apply_theme(theme_name, no_color, gate, prompt_session)
                    console.print(f"[dim]{parts[1]} = {getattr(cfg, parts[1])}[/dim]")
            else:
                console.print("[dim]usage: /config [key value][/dim]")
            continue
        if line == "/resume":
            await _resume_command(console, session, memory, root)
            continue
        if line == "/mcp":
            if session.mcp_registry is None or not session.mcp_registry.servers():
                console.print("[dim]no MCP servers connected[/dim]")
            else:
                for server_name, tool_names in session.mcp_registry.servers():
                    console.print(f"[bold]{server_name}[/bold]")
                    for tool_name in tool_names:
                        console.print(f"  {tool_name}")
            continue
        if line == "/tools":
            _print_tools(console, session)
            continue
        if line.startswith("/voice"):
            if voice is None:
                console.print("[dim]voice is off on this server (LOCALGATE_VOICE_ENABLED)[/dim]")
                continue
            arg = line[len("/voice") :].strip().lower()
            voice.speak_enabled = (
                (not voice.speak_enabled) if not arg else arg in {"on", "1", "yes"}
            )
            state = "on — replies are read aloud" if voice.speak_enabled else "off"
            console.print(f"[dim]voice {state}[/dim]")
            continue
        if line.startswith("!"):
            command = line[1:].strip()
            if command:
                note = await _run_user_shell(console, root, command)
                if note:
                    shell_notes.append(note)
            continue
        if line == "/init":
            line = INIT_PROMPT
        elif line.startswith("/"):
            name, _, arguments = line[1:].partition(" ")
            custom = load_commands(root).get(name.lower())
            if custom is None:
                console.print(f"[dim]unknown command {line!r} — try /help[/dim]")
                continue
            console.print(f"[dim]/{custom.name} — {escape(custom.description)}[/dim]")
            line = custom.expand(arguments.strip())

        attachments = mentioned_files(root, line)
        if shell_notes:
            line = "\n\n".join([*shell_notes, line])
            shell_notes.clear()
        try:
            reply = await run_turn(
                console,
                session,
                gate,
                line,
                memory=memory,
                usage=usage,
                attachments=attachments,
            )
            if reply:
                last_reply = reply
            if voice is not None and voice.speak_enabled and reply:
                await voice.say(reply)
        except AgentTurnLimitExceeded as exc:
            console.print(f"[red]{exc}[/red]")
        except httpx.HTTPStatusError as exc:
            detail = describe_backend_error(exc)
            console.print(f"[red]Backend rejected the request — {detail}[/red]")
            if exc.response.status_code == 400:
                console.print(
                    "[yellow]This usually means the current model doesn't support tool "
                    "calling — try /model <tool-capable-model>, e.g. "
                    "/model qwen2.5-coder:7b.[/yellow]"
                )
        except httpx.HTTPError as exc:
            console.print(f"[red]Couldn't reach the backend: {exc}[/red]")
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
    theme: str = theme_mod.DEFAULT_THEME,
    no_color: bool = False,
    max_turns: int = 20,
    plan_mode: bool = False,
    allow_delegation: bool = False,
    search_fn: SearchFn | None = None,
    mcp_registry: McpRegistry | None = None,
    settings: Any | None = None,
    session_id: str | None = None,
    sandbox: Sandbox | None = None,
    agent_roles: dict[str, AgentRole] | None = None,
    native_tools: bool = True,
    voice: VoiceIO | None = None,
) -> SingleShotResult:
    """One task, one turn, with the same diff/spinner/streaming UI as the REPL."""
    console = theme_mod.make_console(theme, no_color=no_color)
    gate = WriteGate(
        console,
        root,
        auto_approve=auto_approve,
        force=force,
        auto_commit=auto_commit,
        theme_name=theme,
        plan_mode=plan_mode,
    )
    session = await build_agent_session(
        backend,
        model,
        root,
        gate,
        console,
        max_turns=max_turns,
        allow_delegation=allow_delegation,
        search_fn=search_fn,
        mcp_registry=mcp_registry,
        settings=settings,
        session_id=session_id,
        memory=memory,
        sandbox=sandbox,
        agent_roles=agent_roles,
        native_tools=native_tools,
    )
    text = await run_turn(console, session, gate, task, memory=memory, usage=SessionUsage())
    if voice is not None and voice.speak_enabled and text:
        await voice.say(text)
    return SingleShotResult(text, gate.declined_a_write)
