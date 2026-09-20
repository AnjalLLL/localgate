"""Interactive REPL for `localgate code`, and the write-safety plumbing it shares
with the single-shot invocation: a dirty-tree gate, colored diffs before every
write, a spinner while waiting on the model, and `/undo`.

Kept out of `cli.py` because none of this is Typer-specific — it is pure asyncio
and Rich, and is exercised directly in tests without going through the CLI layer.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, NamedTuple

import httpx
import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.panel import Panel

from localgate.agent import gitutil
from localgate.agent import theme as theme_mod
from localgate.agent.loop import AgentSession, AgentTurnLimitExceeded
from localgate.agent.mcp import McpRegistry
from localgate.agent.memory import AgentMemory, list_project_sessions, set_current_project_session
from localgate.agent.render import print_diff
from localgate.agent.theme import THEMES
from localgate.agent.tools import (
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

#: (command, one-line description) — the single source of truth for both the
#: startup banner and `/help`, so a new command only needs to be added here once.
SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/exit", "End the session."),
    ("/clear", "Clear conversation history (keep model/settings)."),
    ("/model [name]", "Show a picker, or switch directly to <name>."),
    ("/theme [name]", "Show or set the color theme: dark, light, none."),
    ("/mode [name]", "Show or set the write mode: manual, auto, plan (or shift+tab)."),
    ("/usage", "Request/token totals for this session."),
    ("/context", "How full the conversation is vs. the model's context window."),
    ("/config [key value]", "View, or set and persist, a preference."),
    ("/resume", "Pick a past session for this project to resume."),
    ("/undo", "Revert the most recent write."),
    ("/rewind [n]", "Step back through the last n writes (default 1)."),
    ("/mcp", "List connected MCP servers and their tools."),
    ("/tools", "List every tool available this session, and what's off and why."),
    ("/help", "Show this list."),
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
        theme_name: str = "dark",
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
        if (
            name in {"write_file", "create_file", "update_file", "delete_file"}
            and not result.is_error
        ):
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
    if not warned:
        supports_tools = await session.backend.check_tool_support(name)
        if supports_tools is False:
            console.print(
                f"[red]{name} doesn't advertise tool-calling support — switching would likely "
                "400 on the next turn.[/red]"
            )
            answer = console.input("[yellow]Switch anyway? [y/N] [/yellow]").strip().lower()
            if answer not in ("y", "yes"):
                return

    session.model = name
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

    console.print(f"[dim]current model: {session.model}[/dim]")
    for i, m in enumerate(models, start=1):
        details = (m.get("parameter_size"), m.get("quantization"), m.get("size_human"))
        extra = "  ".join(str(v) for v in details if v)
        marker = "*" if m["name"] == session.model else " "

        # Add caniollama compat annotation if available
        compat_suffix = ""
        if caniollama_client:
            compat = compat_map.get(m["name"])
            if compat is not None:
                rate = overall_pass_rate(compat)
                label = "reliable" if rate >= 0.7 else "unreliable"
                color = "green" if rate >= 0.7 else "red"
                compat_suffix = (
                    f"  [{color}]tool calling: {label} "
                    f"({rate:.0%}, {compat.total_reports} reports)[/{color}]"
                )
            else:
                compat_suffix = (
                    f"  [dim]no compat data yet — run: caniollama check {m['name']}[/dim]"
                )

        suffix = f"  [dim]{extra}[/dim]" if extra else ""
        console.print(f"  {marker}{i}. {m['name']}{suffix}{compat_suffix}")

    choice = console.input(
        "[bold green]select a model (number or name, blank to cancel): [/bold green]"
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


def _theme_picker(console: Console, current: str) -> str | None:
    """`/theme` with no argument: show available themes and let user pick."""
    console.print(f"[dim]current theme: {current}[/dim]")
    for i, name in enumerate(THEMES, start=1):
        marker = "*" if name == current else " "
        console.print(f"  {marker}{i}. {name}")
    choice = console.input(
        "[bold green]select a theme (number or name, blank to cancel): [/bold green]"
    ).strip()
    if not choice:
        return None
    if choice.isdigit() and 1 <= int(choice) <= len(THEMES):
        return THEMES[int(choice) - 1]
    if choice.lower() in THEMES:
        return choice.lower()
    console.print(f"[red]invalid choice: {choice!r}[/red]")
    return None


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
    lines: list[str] = []
    lines.append(f"[bold]project:[/bold] {root}")
    lines.append(f"[bold]model:[/bold]   {model}")
    lines.append(f"[bold]mode:[/bold]    {mode}")

    # Files
    try:
        entries = sorted(root.iterdir())
        visible = [
            e.name + ("/" if e.is_dir() else "") for e in entries if not e.name.startswith(".")
        ][:12]
        if visible:
            lines.append("")
            lines.append(f"[dim]files:[/dim] {', '.join(visible)}")
    except OSError:
        pass

    # Tools
    caps: list[str] = ["read", "write", "search", "git"]
    if search_fn is not None:
        caps.append("web_search")
    if session.allow_delegation:
        caps.append("delegate")
    if mcp_registry is not None and mcp_registry.servers():
        names = [name for name, _ in mcp_registry.servers()]
        caps.extend(f"mcp:{n}" for n in names)
    lines.append(f"[dim]tools:[/dim] {', '.join(caps)}")

    # Quick command reference
    lines.append("")
    lines.append(f"[dim]{_QUICK_COMMANDS}[/dim]")
    lines.append("[dim]shift+tab cycles mode[/dim]")

    return Panel(
        "\n".join(lines),
        title="[bold green]localgate code[/bold green]",
        subtitle="[dim]github.com/AnjalLLL/localgate[/dim]",
        border_style="green",
        padding=(0, 1),
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

    if session.mcp_registry is not None and session.mcp_registry.servers():
        for server_name, tool_names in session.mcp_registry.servers():
            console.print(f"[bold]mcp:{server_name}:[/bold] {', '.join(tool_names)}")
    else:
        console.print(
            "[dim]mcp: no servers connected — see ~/.config/localgate/mcp_servers.json[/dim]"
        )


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
    console.print(
        f"[dim]resumed session {chosen[:8]} ({len(history)} prior message(s) loaded)[/dim]"
    )


def _make_prompt_session(gate: WriteGate) -> PromptSession[str] | None:
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

    def _bottom_toolbar() -> str:
        return f" mode: {mode_label(gate.mode())}  (shift+tab to cycle)"

    return PromptSession(key_bindings=bindings, bottom_toolbar=_bottom_toolbar)


async def _read_line(console: Console, prompt_session: PromptSession[str] | None) -> str:
    if prompt_session is not None:
        return await prompt_session.prompt_async("> ")
    return console.input("[bold green]> [/bold green]")


async def run_turn(
    console: Console,
    session: AgentSession,
    gate: WriteGate,
    user_input: str,
    memory: AgentMemory | None = None,
    usage: SessionUsage | None = None,
) -> str:
    """Run one turn with a spinner that clears the moment the model streams
    real output, but otherwise stays visible (relabeled "working...") between
    tool calls — a slow one (a web search, a delegated sub-task, an MCP call)
    would otherwise leave a dead gap with no sign anything is still happening.
    Auto-commits afterward, and records the turn into conversation
    history/memory if a session is attached.
    """
    status = console.status("[dim]thinking...[/dim]", spinner="dots")
    status.start()
    stopped = False

    def stop() -> None:
        nonlocal stopped
        if not stopped:
            status.stop()
            stopped = True

    def restart() -> None:
        nonlocal stopped
        status.update("[dim]working...[/dim]")
        status.start()
        stopped = False

    def on_token(text: str) -> None:
        stop()
        console.print(text, end="")

    def on_event(line: str) -> None:
        nonlocal stopped
        stop()
        if "wrote " in line or "write_file" in line:
            console.print(f"[bold green]  ✓ {line}[/bold green]")
        else:
            console.print(f"[cyan]  → {line}[/cyan]")
        restart()

    original_confirm_write = session.confirm_write
    original_confirm_search = session.confirm_search
    original_confirm_delegate = session.confirm_delegate
    original_confirm_mcp = session.confirm_mcp

    def with_visible_prompt(callback: Any) -> Any:
        def visible(*args: Any, **kwargs: Any) -> bool:
            stop()
            try:
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

    session.on_token = on_token
    session.on_event = on_event
    if memory is not None:
        session.augment = memory.augment
    try:
        result = await session.send(user_input)
    finally:
        stop()
        session.confirm_write = original_confirm_write
        session.confirm_search = original_confirm_search
        session.confirm_delegate = original_confirm_delegate
        session.confirm_mcp = original_confirm_mcp

    if gate.plan_mode:
        gate.flush_plan()

    # Show a summary of files written this turn
    if gate.writes_this_turn:
        names = ", ".join(gate.writes_this_turn)
        console.print(f"[bold green]  files written: {names}[/bold green]")

    gate.after_turn(user_input)
    console.print()
    if usage is not None:
        prompt_tokens = count_message_tokens(session.messages[:-1])
        completion_tokens = count_tokens(result)
        usage.record(prompt_tokens, completion_tokens)
        if memory is not None:
            await memory.record_usage(session.model, prompt_tokens, completion_tokens)
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
    theme: str = "dark",
    no_color: bool = False,
    max_turns: int = 20,
    user_config: UserConfig | None = None,
    plan_mode: bool = False,
    allow_delegation: bool = False,
    search_fn: SearchFn | None = None,
    mcp_registry: McpRegistry | None = None,
    settings: Any | None = None,
    session_id: str | None = None,
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
    )
    prompt_session = _make_prompt_session(gate)

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

    while True:
        try:
            line = await _read_line(console, prompt_session)
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
                    console = theme_mod.make_console(theme_name, no_color=no_color)
                    gate.console = console
                    gate.theme_name = theme_name
                    cfg = cfg.with_override(theme=theme_name)
                    cfg.save()
                    console.print(f"[dim]theme set to {theme_name}[/dim]")
            else:
                picked = _theme_picker(console, theme_name)
                if picked is not None:
                    theme_name = picked
                    console = theme_mod.make_console(theme_name, no_color=no_color)
                    gate.console = console
                    gate.theme_name = theme_name
                    cfg = cfg.with_override(theme=theme_name)
                    cfg.save()
                    console.print(f"[dim]theme set to {theme_name}[/dim]")
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
        if line == "/help":
            for cmd, desc in SLASH_COMMANDS:
                console.print(f"  [bold]{cmd:<22}[/bold] [dim]{desc}[/dim]")
            continue
        if line == "/usage":
            console.print(
                f"[dim]{usage.request_count} request(s) this session — "
                f"{usage.prompt_tokens:,} prompt + {usage.completion_tokens:,} completion = "
                f"{usage.total_tokens:,} tokens (approximate)[/dim]"
            )
            continue
        if line == "/context":
            used = count_message_tokens(session.messages)
            window = await session.backend.context_window(session.model)
            if window:
                pct = used / window * 100
                console.print(f"[dim]{used:,} / {window:,} tokens ({pct:.0f}% of context)[/dim]")
            else:
                console.print(f"[dim]{used:,} tokens in the current conversation[/dim]")
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
                        console = theme_mod.make_console(theme_name, no_color=no_color)
                        gate.console = console
                        gate.theme_name = theme_name
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
        if line.startswith("/"):
            console.print(f"[dim]unknown command {line!r} — try /help[/dim]")
            continue

        try:
            await run_turn(console, session, gate, line, memory=memory, usage=usage)
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
    theme: str = "dark",
    no_color: bool = False,
    max_turns: int = 20,
    plan_mode: bool = False,
    allow_delegation: bool = False,
    search_fn: SearchFn | None = None,
    mcp_registry: McpRegistry | None = None,
    settings: Any | None = None,
    session_id: str | None = None,
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
    )
    text = await run_turn(console, session, gate, task, memory=memory, usage=SessionUsage())
    return SingleShotResult(text, gate.declined_a_write)
