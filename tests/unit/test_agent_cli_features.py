"""The Claude Code-style REPL features: custom commands, LOCALGATE.md, `@file`
completion and attachments, `!command`, the to-do checklist, and interrupting
a turn — each exercised without a real terminal."""

import asyncio
import io
import json
import sys
from typing import Any

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from rich.console import Console

from localgate import paths
from localgate.agent import repl
from localgate.agent.commands import load_commands
from localgate.agent.input import AgentCompleter, mentioned_files
from localgate.agent.instructions import load_instructions
from localgate.agent.loop import AgentSession
from localgate.agent.render import todo_checklist
from localgate.agent.shell import run_shell, shell_note
from localgate.backends.base import InferenceBackend


class ScriptedBackend(InferenceBackend):
    name = "scripted"

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    async def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        message = self._responses.pop(0)
        return {"choices": [{"index": 0, "message": message, "finish_reason": "stop"}]}

    async def chat_stream(self, request: dict[str, Any]):
        # `repl.run_turn` streams; replay the scripted message as one chunk.
        response = await self.chat(request)
        message = response["choices"][0]["message"]
        if message.get("tool_calls"):
            delta = {
                "tool_calls": [
                    {"index": i, "id": tc["id"], "function": tc["function"]}
                    for i, tc in enumerate(message["tool_calls"])
                ]
            }
        else:
            delta = {"content": message.get("content") or ""}
        yield {"choices": [{"delta": delta}]}

    async def embed(self, text: str, model: str) -> list[float]:
        raise NotImplementedError

    async def list_models(self) -> list[str]:
        return ["scripted-model"]

    async def health(self) -> bool:
        return True


def final_text(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=100)


@pytest.fixture
def project(tmp_path):
    (tmp_path / "app.py").write_text("print('hi')\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("x = 1\n")
    return tmp_path


# ------------------------------------------------------------- custom commands


def test_project_command_file_becomes_a_command_with_arguments(project):
    commands_dir = project / ".localgate" / "commands"
    commands_dir.mkdir(parents=True)
    (commands_dir / "review.md").write_text(
        "---\ndescription: Review the diff\n---\nReview git diff, focusing on $ARGUMENTS.\n"
    )
    command = load_commands(project)["review"]
    assert command.description == "Review the diff"
    assert command.expand("error handling") == "Review git diff, focusing on error handling."


def test_command_without_placeholder_gets_arguments_appended(project):
    commands_dir = project / ".localgate" / "commands"
    commands_dir.mkdir(parents=True)
    (commands_dir / "explain.md").write_text("Explain this project.\n")
    command = load_commands(project)["explain"]
    assert command.description == "Explain this project."
    assert command.expand("briefly") == "Explain this project.\n\nbriefly"


def test_project_command_overrides_a_user_command_of_the_same_name(project):
    user_dir = paths.config_dir() / "commands"
    user_dir.mkdir(parents=True)
    (user_dir / "test.md").write_text("user version")
    project_dir = project / ".localgate" / "commands"
    project_dir.mkdir(parents=True)
    (project_dir / "test.md").write_text("project version")
    assert load_commands(project)["test"].template == "project version"


# ------------------------------------------------------------- LOCALGATE.md


def test_localgate_md_is_part_of_the_system_prompt(project):
    (project / "LOCALGATE.md").write_text("Always use tabs.")
    session = AgentSession(ScriptedBackend([]), "scripted-model", project)
    assert "Always use tabs." in session.messages[0]["content"]


def test_agents_md_is_used_when_there_is_no_localgate_md(project):
    (project / "AGENTS.md").write_text("Run make test.")
    assert "Run make test." in load_instructions(project)


def test_no_instruction_files_adds_nothing(project):
    assert load_instructions(project) == ""


# ------------------------------------------------------------- input line


def _completions(completer: AgentCompleter, text: str) -> list[str]:
    return [c.text for c in completer.get_completions(Document(text), CompleteEvent())]


def test_slash_completes_command_names(project):
    completer = AgentCompleter(project, lambda: [("/model [name]", "x"), ("/mode", "y")])
    assert _completions(completer, "/mod") == ["/model", "/mode"]
    assert _completions(completer, "/model qwen") == []


def test_at_completes_project_files_but_not_ignored_ones(project):
    (project / ".gitignore").write_text("secret.txt\n")
    (project / "secret.txt").write_text("s")
    completer = AgentCompleter(project, lambda: [])
    assert _completions(completer, "look at @src/") == ["@src/main.py"]
    assert "@secret.txt" not in _completions(completer, "@")
    assert "@src/main.py" in _completions(completer, "@main")


def test_mentions_resolve_to_existing_visible_files_only(project):
    text = "compare @app.py with @src/main.py, @missing.py, @../etc/passwd and me@host.py"
    assert mentioned_files(project, text) == ["app.py", "src/main.py"]


async def test_attached_file_reaches_the_model_and_counts_as_read(project):
    backend = ScriptedBackend(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "w",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": json.dumps({"path": "app.py", "content": "new\n"}),
                        },
                    }
                ],
            },
            final_text("done"),
        ]
    )
    session = AgentSession(backend, "scripted-model", project, confirm_write=lambda *_: True)
    assert await session.send("update @app.py", attachments=["app.py"]) == "done"
    first_user = [m for m in backend.requests[0]["messages"] if m["role"] == "user"][-1]
    assert "print('hi')" in first_user["content"]
    # No read_file call was needed before overwriting it.
    assert (project / "app.py").read_text() == "new\n"


# ------------------------------------------------------------- ! shell


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell syntax")
async def test_run_shell_streams_lines_and_reports_the_exit_code(project):
    lines: list[str] = []
    code, output = await run_shell(project, "echo one; echo two; exit 3", lines.append)
    assert (code, lines) == (3, ["one", "two"])
    assert "exited with code 3" in shell_note("x", code, output)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell syntax")
async def test_cancelling_run_shell_stops_the_command(project):
    task = asyncio.ensure_future(run_shell(project, "sleep 30", lambda _: None))
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)


async def test_shell_output_is_shown_to_the_model_with_the_next_message(project, monkeypatch):
    inputs = iter(["!echo from-shell", "what happened?", "/exit"])
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: next(inputs))
    backend = ScriptedBackend([final_text("it printed from-shell")])
    await repl.run_repl(backend, "scripted-model", project, auto_approve=True)
    user_messages = [m for m in backend.requests[0]["messages"] if m["role"] == "user"]
    assert "I ran `echo from-shell`" in user_messages[-1]["content"]
    assert "what happened?" in user_messages[-1]["content"]


async def test_custom_command_runs_its_prompt_in_the_repl(project, monkeypatch):
    commands_dir = project / ".localgate" / "commands"
    commands_dir.mkdir(parents=True)
    (commands_dir / "greet.md").write_text("Say hello to $ARGUMENTS.")
    inputs = iter(["/greet the team", "/exit"])
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: next(inputs))
    backend = ScriptedBackend([final_text("hello team")])
    await repl.run_repl(backend, "scripted-model", project, auto_approve=True)
    user_messages = [m for m in backend.requests[0]["messages"] if m["role"] == "user"]
    assert "Say hello to the team." in user_messages[-1]["content"]


# ------------------------------------------------------------- todos


def test_todo_checklist_marks_each_status():
    text = todo_checklist([("plan", "done"), ("build", "in_progress"), ("ship", "pending")])
    assert text.plain == "  ☑ plan\n  ◐ build\n  ☐ ship"


# ------------------------------------------------------------- interrupting


async def test_interrupting_a_turn_keeps_the_session_usable(project, monkeypatch):
    """Esc cancels the running turn; history gets no dangling tool call, and the
    next turn works normally."""

    class SlowBackend(ScriptedBackend):
        async def chat(self, request):
            if not self.requests:
                self.requests.append(request)
                await asyncio.sleep(30)  # interrupted here
            return await super().chat(request)

    def interrupt_immediately(self, task):
        self.interrupted = True
        asyncio.get_running_loop().call_later(0.05, task.cancel)

    monkeypatch.setattr(repl.InterruptWatcher, "start", interrupt_immediately)
    backend = SlowBackend([final_text("second answer")])
    out = console()
    gate = repl.WriteGate(out, project, auto_approve=True)
    session = AgentSession(backend, "scripted-model", project)
    session.messages.append({"role": "user", "content": "earlier"})
    session.messages.append({"role": "assistant", "content": "earlier reply"})

    assert await repl.run_turn(out, session, gate, "slow request") == ""
    assert "interrupted" in out.file.getvalue()
    assert session.messages[-1]["content"] == "(Interrupted by the user before I finished.)"

    monkeypatch.setattr(repl.InterruptWatcher, "start", lambda self, task: None)
    assert await repl.run_turn(out, session, gate, "try again") == "second answer"


# ------------------------------------------------- code written into the chat

_PAGE = "\n".join(
    [
        "<!DOCTYPE html>",
        '<html lang="en">',
        '<head><link rel="stylesheet" href="styles.css"></head>',
        "<body>",
        '  <nav class="navbar"><a href="/">Home</a> <a href="#contact">Contact</a></nav>',
        '  <section id="contact"><h2>Contact</h2></section>',
        "</body>",
        "</html>",
    ]
)


@pytest.fixture
def site(tmp_path):
    (tmp_path / "index.html").write_text("<html><body>old</body></html>\n")
    (tmp_path / "styles.css").write_text("body {}\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.js").write_text("// app\n")
    return tmp_path


async def test_a_path_comment_inside_the_block_becomes_a_write_to_the_real_file(site):
    """The shape qwen2.5-coder produced live: the file named in a comment on the
    block's first line, under a directory that doesn't hold that file."""
    reply = f"Update index.html as follows:\n\n```html\n<!-- src/index.html -->\n{_PAGE}\n```\n"
    backend = ScriptedBackend([final_text(reply), final_text("Updated index.html.")])
    session = AgentSession(backend, "scripted-model", site, confirm_write=lambda *_: True)

    assert await session.send("add a contact section") == "Updated index.html."
    assert (site / "index.html").read_text().rstrip() == _PAGE  # label line dropped
    assert not (site / "src" / "index.html").exists()  # no stray copy


async def test_css_comment_label_creates_a_new_file_where_it_says(site):
    css = "\n".join(["/* theme.css */", *[f".c{i} {{ color: red; }}" for i in range(8)]])
    backend = ScriptedBackend([final_text(f"```css\n{css}\n```"), final_text("done")])
    session = AgentSession(backend, "scripted-model", site, confirm_write=lambda *_: True)

    await session.send("add a theme stylesheet")
    assert (site / "theme.css").read_text().startswith(".c0")


async def test_code_in_the_reply_without_any_change_is_sent_back_once(site):
    unlabelled = f"Here is the updated page:\n\n```html\n{_PAGE}\n```"
    backend = ScriptedBackend([final_text(unlabelled), final_text("I only showed the code.")])
    session = AgentSession(backend, "scripted-model", site)

    # No change verb in the request: the old check would have accepted the reply.
    assert await session.send("its just a stale page, where is the css") == (
        "I only showed the code."
    )
    nudge = [m for m in backend.requests[1]["messages"] if m["role"] == "user"][-1]
    assert "no file was changed" in nudge["content"]


async def test_short_snippets_in_a_reply_are_not_second_guessed(site):
    backend = ScriptedBackend(
        [final_text("Use `display: flex`:\n\n```css\nnav { display: flex; }\n```")]
    )
    session = AgentSession(backend, "scripted-model", site)
    await session.send("how do I lay out the nav?")
    assert len(backend.requests) == 1


def test_long_code_is_collapsed_in_the_chat_and_short_code_kept():
    from localgate.agent.render import collapse_code

    long_block = "```html\n" + "\n".join(f"<p>{i}</p>" for i in range(30)) + "\n```"
    shown, hidden = collapse_code(f"Intro\n\n{long_block}\n\n```css\na {{}}\n```\nBye")
    assert hidden == 1
    assert "30 lines of html hidden" in shown
    assert "<p>3</p>" not in shown and "a {}" in shown and "Bye" in shown


def test_an_unfinished_block_is_summarized_while_streaming():
    from localgate.agent.render import collapse_code

    shown, hidden = collapse_code("Writing:\n```html\n<div>\n<p>", streaming=True)
    assert hidden == 1 and "writing html… 2 lines" in shown and "<div>" not in shown


# ------------------------------------------------- typing while a turn runs


def test_footer_keys_type_queue_cycle_and_interrupt():
    from prompt_toolkit.keys import Keys

    from localgate.agent.interrupt import InterruptWatcher

    submitted: list[str] = []
    cycled: list[bool] = []
    watcher = InterruptWatcher(
        on_submit=submitted.append, on_cycle_mode=lambda: cycled.append(True)
    )
    for ch in "add tests":
        watcher.handle_key(ch)
    watcher.handle_key(Keys.Backspace)
    watcher.handle_key(Keys.ControlM)
    assert submitted == ["add test"] and watcher.buffer == ""

    watcher.handle_key(Keys.BackTab)
    assert cycled == [True]

    watcher.handle_key("x")
    watcher.handle_key(Keys.Escape)  # first Esc clears the half-typed line…
    assert watcher.buffer == "" and not watcher.interrupted
    watcher.handle_key(Keys.Escape)  # …the next one interrupts
    assert watcher.interrupted


async def test_messages_typed_during_a_turn_run_next_in_order(project, monkeypatch):
    """Simulate typing into the footer mid-turn: both messages queue, then run
    as their own turns before the prompt asks for anything new."""

    def type_two_messages(self, task):
        self.on_submit("second request")
        self.on_submit("third request")

    started = {"done": False}
    original_start = repl.InterruptWatcher.start

    def start_once(self, task):
        if not started["done"]:
            started["done"] = True
            type_two_messages(self, task)
        return original_start(self, task)

    monkeypatch.setattr(repl.InterruptWatcher, "start", start_once)
    inputs = iter(["first request", "/exit"])
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: next(inputs))
    backend = ScriptedBackend([final_text("one"), final_text("two"), final_text("three")])
    await repl.run_repl(backend, "scripted-model", project, auto_approve=True)

    asked = [
        [m for m in request["messages"] if m["role"] == "user"][-1]["content"]
        for request in backend.requests
    ]
    assert [a.rsplit("User request: ", 1)[-1] for a in asked] == [
        "first request",
        "second request",
        "third request",
    ]


# ------------------------------------- small context windows (seen with llama3)


def _write(path: str, content: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"w{abs(hash(content)) % 10_000}",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps({"path": path, "content": content}),
                },
            }
        ],
    }


async def test_a_compaction_placeholder_is_never_written_and_never_asked_about(project):
    asked: list[str] = []
    # After the refusal nothing has changed, so the loop also nudges once more.
    backend = ScriptedBackend(
        [_write("app.py", "<500 chars elided>"), final_text("done"), final_text("left it alone")]
    )
    session = AgentSession(
        backend,
        "scripted-model",
        project,
        confirm_write=lambda path, _c: asked.append(path) or True,
    )
    await session.send("rewrite app.py")
    assert (project / "app.py").read_text() == "print('hi')\n"
    assert asked == []  # the user never saw a prompt for it
    tool_reply = [m for m in backend.requests[1]["messages"] if m["role"] == "tool"][-1]
    assert "placeholder" in tool_reply["content"]


async def test_rewriting_one_file_over_and_over_is_stopped(project):
    asked: list[str] = []
    backend = ScriptedBackend(
        [
            _write("app.py", "v1 = 1\n"),
            _write("app.py", "v2 = 2\n"),
            _write("app.py", "v3 = 3\n"),
            final_text("done"),
        ]
    )
    session = AgentSession(
        backend,
        "scripted-model",
        project,
        confirm_write=lambda path, _c: asked.append(path) or True,
    )
    await session.send("update app.py")
    assert asked == ["app.py", "app.py"]  # the third rewrite was refused without asking
    assert (project / "app.py").read_text() == "v2 = 2\n"
    tool_reply = [m for m in backend.requests[-1]["messages"] if m["role"] == "tool"][-1]
    assert "already rewritten 2 times" in tool_reply["content"]


def test_the_reply_budget_scales_with_the_window():
    from localgate.agent.loop import reply_budget

    assert reply_budget(8192) == 2048  # llama3: ~6K left for context, not ~4K
    assert reply_budget(32768) == 4096
    assert reply_budget(2048) == 1024


def test_half_escaped_html_is_repaired_but_code_strings_are_not():
    from localgate.agent.tools import unescape_text_arguments

    html = (
        "<html>\n<body>\n<div class='nav'>\\n<a href='#'>Home</a>\\n<a href='#a'>About</a>\\n</div>"
    )
    fixed = unescape_text_arguments({"path": "src/index.html", "content": html})["content"]
    assert "\\n" not in fixed and fixed.count("\n") == 5

    code = 'import sys\nprint("a\\nb")\nprint("c\\nd")\n'
    assert unescape_text_arguments({"path": "app.py", "content": code})["content"] == code
