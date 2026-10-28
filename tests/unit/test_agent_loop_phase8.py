"""Agent-loop behavior added for reliable tool calling and context management:
edit_file previews, parallel reads, result caps, tool logging, num_ctx, JSON tool
mode, SEARCH/REPLACE fallback, sub-agent inheritance, ledger pinning and compaction."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from localgate.agent.loop import (
    MAX_TOOL_RESULT_CHARS,
    AgentSession,
    AsyncTool,
    ToolLogEntry,
)
from localgate.agent.roles import AgentRole
from localgate.backends.base import InferenceBackend


class Scripted(InferenceBackend):
    name = "scripted"

    def __init__(self, responses: list[dict[str, Any]], window: int | None = None) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.window = window

    async def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        message = self.responses.pop(0)
        return {"choices": [{"index": 0, "message": message}]}

    async def chat_stream(self, request: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        raise NotImplementedError
        yield {}

    async def embed(self, text: str, model: str) -> list[float]:
        raise NotImplementedError

    async def list_models(self) -> list[str]:
        return ["m"]

    async def health(self) -> bool:
        return True

    async def context_window(self, model: str) -> int | None:
        return self.window


def call(name: str, **arguments: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"c-{name}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def calls(*pairs: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"c{i}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
            for i, (name, args) in enumerate(pairs)
        ],
    }


def text(content: str) -> dict[str, Any]:
    return {"role": "assistant", "content": content}


@pytest.fixture
def project(tmp_path):
    (tmp_path / "app.py").write_text("def f():\n    return 1\n", newline="\n")
    return tmp_path


async def test_edit_file_previews_the_full_new_content_and_applies_it(project):
    previews: list[tuple[str, str | None]] = []

    def confirm(path: str, content: str | None) -> bool:
        previews.append((path, content))
        return True

    backend = Scripted(
        [
            call("read_file", path="app.py"),
            call("edit_file", path="app.py", old_string="return 1", new_string="return 2"),
            text("Changed f to return 2."),
        ]
    )
    session = AgentSession(backend, "m", project, confirm_write=confirm)
    await session.send("make f return 2")
    assert previews == [("app.py", "def f():\n    return 2\n")]
    assert (project / "app.py").read_text() == "def f():\n    return 2\n"
    assert session.ledger.files_changed == {"app.py": "modified"}


async def test_edit_file_needs_no_prior_read_but_update_file_does(project):
    # Nested, so it is not among the small root files shown in the first prompt.
    (project / "pkg").mkdir()
    (project / "pkg" / "mod.py").write_text("x = 1\n", newline="\n")
    (project / "pkg" / "other.py").write_text("y = 1\n", newline="\n")
    backend = Scripted(
        [
            call("edit_file", path="pkg/mod.py", old_string="x = 1", new_string="x = 2"),
            call("update_file", path="pkg/other.py", content="y = 2\n"),
            call("update_file", path="pkg/other.py", content="y = 2\n"),
            text("ok"),
        ]
    )
    session = AgentSession(backend, "m", project)
    await session.send("edit")
    assert (project / "pkg" / "mod.py").read_text() == "x = 2\n"
    refusal = next(m for m in session.messages if m.get("name") == "update_file")
    # Refused, but the file is read for the model so the retry succeeds.
    assert "was not changed" in refusal["content"] and "y = 1" in refusal["content"]
    assert (project / "pkg" / "other.py").read_text() == "y = 2\n"


async def test_files_shown_in_the_first_prompt_count_as_read(project):
    backend = Scripted(
        [
            call("edit_file", path="app.py", old_string="return 1", new_string="return 2"),
            text("done"),
        ]
    )
    session = AgentSession(backend, "m", project, auto_read=False)
    await session.send("edit")
    assert (project / "app.py").read_text() == "def f():\n    return 2\n"


async def test_an_edit_that_breaks_python_is_reported_and_challenged(project):
    backend = Scripted(
        [
            call("read_file", path="app.py"),
            call("edit_file", path="app.py", old_string="def f():", new_string="def def f():"),
            text("Done!"),
            text("app.py is broken; I could not fix it."),
        ]
    )
    session = AgentSession(backend, "m", project)
    answer = await session.send("rename")
    edit_result = [m for m in session.messages if m.get("role") == "tool"][1]
    assert "syntax error" in edit_result["content"]
    assert any("still has a syntax error" in (m.get("content") or "") for m in session.messages)
    assert answer == "app.py is broken; I could not fix it."


async def test_no_forced_user_nudges_between_tool_rounds(project):
    backend = Scripted(
        [
            call("read_file", path="app.py"),
            call("list_directory"),
            call("search_files", pattern="def"),
            call("git_status"),
            call("read_file", path="app.py"),
            text("done"),
        ]
    )
    session = AgentSession(backend, "m", project)
    await session.send("look around")
    injected = [m for m in session.messages[2:] if m["role"] == "user"]
    assert injected == []


async def test_read_only_calls_in_one_turn_run_concurrently(project, monkeypatch):
    (project / "b.py").write_text("b\n")
    active = 0
    peak = 0

    def slow_executor(root, call_id, name, arguments):
        from localgate.agent.tools import execute_tool_call  # noqa: PLC0415

        return execute_tool_call(root, call_id, name, arguments)

    async def fake_timeout(root, call_id, name, arguments, settings, executor=None):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return slow_executor(root, call_id, name, arguments)

    monkeypatch.setattr("localgate.agent.loop.execute_tool_call_with_timeout", fake_timeout)

    class S:
        tool_timeout_read = 5.0

    backend = Scripted(
        [calls(("read_file", {"path": "app.py"}), ("read_file", {"path": "b.py"})), text("x")]
    )
    session = AgentSession(backend, "m", project, settings=S())
    await session.send("read both")
    assert peak == 2


async def test_long_tool_results_are_capped(project):
    (project / "big.txt").write_text("y" * (MAX_TOOL_RESULT_CHARS * 2))
    backend = Scripted([call("read_file", path="big.txt"), text("ok")])
    session = AgentSession(backend, "m", project)
    await session.send("read it")
    tool_message = next(m for m in session.messages if m.get("role") == "tool")
    assert len(tool_message["content"]) < MAX_TOOL_RESULT_CHARS + 200
    assert "omitted" in tool_message["content"]


async def test_every_tool_call_is_logged_with_redacted_arguments(project):
    log: list[ToolLogEntry] = []

    async def record(entry: ToolLogEntry) -> None:
        log.append(entry)

    backend = Scripted(
        [
            call("create_file", path="new.py", content="x" * 500),
            call("read_file", path="missing.py"),
            text("ok"),
        ]
    )
    session = AgentSession(backend, "m", project, on_tool_log=record)
    await session.send("create")
    assert [(e.tool_name, e.success) for e in log] == [
        ("create_file", True),
        ("read_file", False),
    ]
    assert len(log[0].arguments["content"]) <= 40
    assert log[0].run_id == session.run_id and log[0].model == "m"


async def test_ollama_requests_carry_num_ctx_capped_by_settings(project):
    class S:
        agent_num_ctx = 8192

    backend = Scripted([text("hi")], window=131072)
    backend.name = "ollama"
    session = AgentSession(backend, "m", project, settings=S())
    await session.send("hello")
    assert backend.requests[0]["options"] == {"num_ctx": 8192}


async def test_unknown_window_on_generic_backend_sends_no_options(project):
    backend = Scripted([text("hi")])
    session = AgentSession(backend, "m", project)
    await session.send("hello")
    assert "options" not in backend.requests[0]


async def test_json_tool_mode_constrains_and_parses_replies(project):
    backend = Scripted(
        [
            text(json.dumps({"name": "read_file", "arguments": {"path": "app.py"}})),
            text(json.dumps({"name": "final_answer", "arguments": {"text": "f returns 1"}})),
        ]
    )
    session = AgentSession(backend, "m", project, native_tools=False)
    answer = await session.send("what does f return?")
    assert answer == "f returns 1"
    first = backend.requests[0]
    assert "tools" not in first
    assert "final_answer" in first["format"]["properties"]["name"]["enum"]
    assert "cannot call tools natively" in first["messages"][0]["content"]


async def test_search_replace_blocks_in_prose_become_edits(project):
    prose = (
        "Here is the change:\n\napp.py\n<<<<<<< SEARCH\n    return 1\n=======\n"
        "    return 3\n>>>>>>> REPLACE\n"
    )
    backend = Scripted([call("read_file", path="app.py"), text(prose), text("done")])
    session = AgentSession(backend, "m", project)
    await session.send("change it")
    assert (project / "app.py").read_text() == "def f():\n    return 3\n"


async def test_extra_tools_are_offered_confirmed_and_validated(project):
    seen: list[dict[str, Any]] = []

    async def echo(arguments: dict[str, Any]) -> str:
        seen.append(arguments)
        return "echoed"

    schema = {
        "type": "function",
        "function": {
            "name": "echo",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        },
    }
    backend = Scripted([call("echo"), call("echo", value="a"), text("ok")])
    session = AgentSession(
        backend, "m", project, extra_tools=[AsyncTool(schema, echo, guidance=" Use echo.")]
    )
    await session.send("echo")
    contents = [m["content"] for m in session.messages if m.get("role") == "tool"]
    assert contents[0].startswith("Invalid arguments")
    assert contents[1] == "echoed"
    assert seen == [{"value": "a"}]
    assert "Use echo." in session.messages[0]["content"]


async def test_sub_agent_inherits_search_and_uses_a_role_model(project):
    searched: list[str] = []

    async def search(query: str) -> str:
        searched.append(query)
        return "- result"

    backend = Scripted(
        [
            call("delegate_task", task="look up X", agent="researcher"),
            call("web_search", query="X docs"),
            text("X is documented at example.com"),
            text("Summary: X is documented."),
        ]
    )
    roles = {
        "researcher": AgentRole(
            "researcher",
            "web research",
            tools=("web_search",),
            model="small-model",
            prompt="Research things.",
        )
    }
    session = AgentSession(
        backend, "big-model", project, allow_delegation=True, search_fn=search, agent_roles=roles
    )
    await session.send("research X")
    assert searched == ["X docs"]
    assert backend.requests[1]["model"] == "small-model"
    sub_tools = {t["function"]["name"] for t in backend.requests[1]["tools"]}
    assert sub_tools == {"web_search"}
    assert "Your role:" in backend.requests[1]["messages"][0]["content"]


async def test_sub_agent_timeout_is_enforced(project):
    class Slow(Scripted):
        async def chat(self, request):
            if len(self.requests) >= 1:
                await asyncio.sleep(1)
            return await super().chat(request)

    class S:
        tool_timeout_delegate = 0.05
        tool_timeout_read = 5.0

    backend = Slow([call("delegate_task", task="slow"), text("never"), text("parent done")])
    session = AgentSession(backend, "m", project, allow_delegation=True, settings=S())
    await session.send("delegate")
    tool_message = next(m for m in session.messages if m.get("name") == "delegate_task")
    assert "timed out" in tool_message["content"]


async def test_ledger_is_pinned_after_the_system_prompt(project):
    backend = Scripted([call("read_file", path="app.py"), text("ok"), text("ok again")])
    session = AgentSession(backend, "m", project)
    await session.send("first goal")
    await session.send("second request")
    pinned = backend.requests[-1]["messages"][1]
    assert pinned["role"] == "system"
    assert "Original goal (verbatim): first goal" in pinned["content"]
    assert "Current request (verbatim): second request" in pinned["content"]
    assert "app.py" in pinned["content"]


async def test_history_is_compacted_when_over_budget(project):
    (project / "big.py").write_text("z = 1\n" * 3000)
    responses = []
    for _ in range(4):
        responses.append(call("read_file", path="big.py", limit=1000))
    responses.append(text("done"))
    events: list[str] = []
    backend = Scripted(responses, window=6000)
    session = AgentSession(backend, "m", project, on_event=events.append)
    await session.send("read big.py many times")
    assert any("context compacted" in e for e in events)
    assert any(
        "elided during compaction" in (m.get("content") or "")
        for m in session.messages
        if m.get("role") == "tool"
    )


async def test_an_empty_reply_mid_task_is_nudged_once(project):
    backend = Scripted([call("read_file", path="app.py"), text(""), text("f returns 1")])
    session = AgentSession(backend, "m", project)
    assert await session.send("what does f return?") == "f returns 1"
    assert any("reply was empty" in (m.get("content") or "") for m in session.messages)


async def test_answer_naming_an_unchanged_file_is_challenged_once(project):
    (project / "main.py").write_text("from app import f\n", newline="\n")
    backend = Scripted(
        [
            call("edit_file", path="app.py", old_string="def f():", new_string="def g():"),
            text("Renamed f to g in app.py and main.py."),
            call("read_file", path="main.py"),
            call("edit_file", path="main.py", old_string="import f", new_string="import g"),
            text("Renamed f to g in app.py and main.py."),
        ]
    )
    session = AgentSession(backend, "m", project)
    await session.send("rename f to g everywhere")
    nudges = [m for m in session.messages if "mentions main.py" in (m.get("content") or "")]
    assert len(nudges) == 1
    assert (project / "main.py").read_text() == "from app import g\n"


async def test_announcing_a_next_step_instead_of_doing_it_is_nudged(project):
    backend = Scripted(
        [
            call("edit_file", path="app.py", old_string="return 1", new_string="return 2"),
            text("Updated app.py. Next, I need to update the tests."),
            text("Done: app.py now returns 2."),
        ]
    )
    session = AgentSession(backend, "m", project)
    assert await session.send("change f to return 2") == "Done: app.py now returns 2."
    assert any(
        "next step instead of doing it" in (m.get("content") or "") for m in session.messages
    )


async def test_edit_that_removes_a_name_reports_remaining_uses(project):
    (project / "util.py").write_text("def fetch():\n    return 1\n", newline="\n")
    (project / "main.py").write_text("from util import fetch\n\nprint(fetch())\n", newline="\n")
    backend = Scripted(
        [
            call("edit_file", path="util.py", old_string="def fetch():", new_string="def load():"),
            text("ok"),
            text("ok"),
        ]
    )
    session = AgentSession(backend, "m", project)
    await session.send("rename fetch to load")
    result = next(m for m in session.messages if m.get("name") == "edit_file")
    assert "'fetch' still appears at main.py:1, main.py:3" in result["content"]


async def test_create_file_on_an_existing_file_is_refused_without_prompting(project):
    asked: list[str] = []
    backend = Scripted(
        [call("create_file", path="app.py", content="x = 1\n"), text("ok"), text("ok")]
    )
    session = AgentSession(
        backend, "m", project, confirm_write=lambda p, c: asked.append(p) or True
    )
    await session.send("make app.py")
    refusal = next(m for m in session.messages if m.get("name") == "create_file")
    assert "already exists" in refusal["content"] and "edit_file" in refusal["content"]
    assert asked == []
    assert (project / "app.py").read_text() == "def f():\n    return 1\n"


async def test_double_escaped_file_content_is_written_with_real_newlines(project):
    backend = Scripted(
        [call("create_file", path="new.py", content="a = 1\\nb = 2\\n"), text("done")]
    )
    session = AgentSession(backend, "m", project)
    await session.send("create new.py")
    assert (project / "new.py").read_text() == "a = 1\nb = 2\n"


async def test_a_fix_task_ending_with_no_change_is_challenged_once(project):
    backend = Scripted(
        [
            text("Let's update the code and proceed."),
            call("edit_file", path="app.py", old_string="return 1", new_string="return 2"),
            text("Changed f to return 2."),
        ]
    )
    session = AgentSession(backend, "m", project)
    assert await session.send("fix f so it returns 2") == "Changed f to return 2."
    assert (project / "app.py").read_text() == "def f():\n    return 2\n"


async def test_a_verbatim_repeat_with_no_change_returns_the_earlier_result(project):
    backend = Scripted(
        [
            call("read_file", path="app.py"),
            call("read_file", path="app.py"),
            call("edit_file", path="app.py", old_string="return 1", new_string="return 2"),
            call("read_file", path="app.py"),
            text("done"),
        ]
    )
    session = AgentSession(backend, "m", project)
    await session.send("change f to return 2")
    results = [m["content"] for m in session.messages if m.get("name") == "read_file"]
    assert "You already made this exact call" in results[1]
    # After a file changed, the same call runs for real again.
    assert "return 2" in results[2] and "already made" not in results[2]
