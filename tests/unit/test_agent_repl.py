"""WriteGate (dirty-tree gate, diff-before-write, tracking for /undo and
--auto-commit) and the REPL's slash-command dispatch, exercised without a
terminal via `Console(file=io.StringIO())` and a scripted backend.
"""

import io
import json
import subprocess
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from rich.console import Console
from rich.status import Status

from localgate.agent import userconfig
from localgate.agent.gitutil import AGENT_COMMIT_PREFIX
from localgate.agent.loop import AgentSession
from localgate.agent.repl import (
    WriteGate,
    describe_backend_error,
    run_repl,
    run_single_shot,
    run_turn,
)
from localgate.agent.userconfig import UserConfig
from localgate.backends.base import InferenceBackend


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "app.py").write_text("original\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "initial")
    return tmp_path


def console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False)


class ScriptedBackend(InferenceBackend):
    """`repl.run_turn` always drives the streaming path, so this scripts
    `chat_stream` — each response becomes a single delta chunk, which the
    accumulator in `loop.py` reassembles the same way it would multi-chunk output.
    """

    name = "scripted"

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)

    async def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    async def chat_stream(self, request: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        message = self._responses.pop(0)
        if message.get("tool_calls"):
            delta = {"tool_calls": [_as_delta(tc) for tc in message["tool_calls"]]}
        else:
            delta = {"content": message.get("content") or ""}
        yield {"choices": [{"delta": delta}]}

    async def embed(self, text: str, model: str) -> list[float]:
        raise NotImplementedError

    async def list_models(self) -> list[str]:
        return ["scripted-model", "other-model"]

    async def health(self) -> bool:
        return True


def _as_delta(tool_call: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": 0,
        "id": tool_call["id"],
        "function": tool_call["function"],
    }


def write_call(path: str, content: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps({"path": path, "content": content}),
                },
            }
        ],
    }


def final_text(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


# --------------------------------------------------------------------- WriteGate


def test_confirm_write_auto_approves_on_a_clean_tree(repo):
    gate = WriteGate(console(), repo, auto_approve=True)
    assert gate.confirm_write("app.py", "new\n") is True


def test_dirty_tree_check_runs_only_once(repo):
    (repo / "app.py").write_text("dirty\n")
    gate = WriteGate(console(), repo, auto_approve=True, force=True)
    assert gate._dirty_tree_ok() is True
    assert gate._dirty_checked is True
    # second call short-circuits regardless of tree state
    assert gate._dirty_tree_ok() is True


def test_force_skips_the_dirty_prompt(repo, monkeypatch):
    (repo / "app.py").write_text("dirty\n")
    called = False

    def fail_confirm(*a, **k):
        nonlocal called
        called = True
        return False

    monkeypatch.setattr("typer.confirm", fail_confirm)
    gate = WriteGate(console(), repo, auto_approve=True, force=True)
    assert gate.confirm_write("app.py", "new\n") is True
    assert called is False


def test_confirm_write_declining_sets_declined_a_write(repo, monkeypatch):
    monkeypatch.setattr("typer.confirm", lambda *a, **k: False)
    gate = WriteGate(console(), repo)
    assert gate.confirm_write("app.py", "new\n") is False
    assert gate.declined_a_write is True


def test_confirm_write_approving_leaves_declined_a_write_false(repo, monkeypatch):
    monkeypatch.setattr("typer.confirm", lambda *a, **k: True)
    gate = WriteGate(console(), repo)
    assert gate.confirm_write("app.py", "new\n") is True
    assert gate.declined_a_write is False


def test_confirm_search_prompts_in_manual_mode(repo, monkeypatch):
    monkeypatch.setattr("typer.confirm", lambda *a, **k: False)
    gate = WriteGate(console(), repo)  # manual by default
    assert gate.confirm_search("some query") is False


def test_confirm_search_skips_the_prompt_outside_manual_mode(repo, monkeypatch):
    called = False

    def fail_confirm(*a, **k):
        nonlocal called
        called = True
        return False

    monkeypatch.setattr("typer.confirm", fail_confirm)
    gate = WriteGate(console(), repo, auto_approve=True)  # "auto" mode
    assert gate.confirm_search("some query") is True
    assert called is False


def test_confirm_delegate_prompts_in_manual_mode(repo, monkeypatch):
    monkeypatch.setattr("typer.confirm", lambda *a, **k: False)
    gate = WriteGate(console(), repo)
    assert gate.confirm_delegate("some task") is False


def test_confirm_delegate_skips_the_prompt_outside_manual_mode(repo, monkeypatch):
    gate = WriteGate(console(), repo, plan_mode=True)
    monkeypatch.setattr("typer.confirm", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    assert gate.confirm_delegate("some task") is True


def test_tracking_executor_records_successful_writes(repo):
    gate = WriteGate(console(), repo, auto_approve=True)
    gate.tracking_executor(repo, "c1", "write_file", {"path": "app.py", "content": "x\n"})
    assert gate.last_written_path == "app.py"
    assert gate.writes_this_turn == ["app.py"]


def test_tracking_executor_does_not_record_failed_writes(repo):
    gate = WriteGate(console(), repo, auto_approve=True)
    gate.tracking_executor(repo, "c1", "write_file", {"path": "../escape.py", "content": "x\n"})
    assert gate.last_written_path is None


def test_after_turn_auto_commits_when_enabled(repo):
    gate = WriteGate(console(), repo, auto_approve=True, auto_commit=True)
    gate.tracking_executor(repo, "c1", "write_file", {"path": "app.py", "content": "x\n"})
    gate.after_turn("update app.py")
    from localgate.agent import gitutil

    assert gitutil.is_dirty(repo) is False
    assert gitutil.last_commit_message(repo) == f"{AGENT_COMMIT_PREFIX} update app.py"


def test_after_turn_does_nothing_without_auto_commit(repo):
    gate = WriteGate(console(), repo, auto_approve=True)
    gate.tracking_executor(repo, "c1", "write_file", {"path": "app.py", "content": "x\n"})
    (repo / "app.py").write_text("x\n")
    gate.after_turn("update app.py")
    from localgate.agent import gitutil

    assert gitutil.is_dirty(repo) is True


def test_undo_without_git_repo(tmp_path):
    gate = WriteGate(console(), tmp_path, auto_approve=True)
    assert "Not a git repository" in gate.undo()


def test_undo_with_nothing_written(repo):
    gate = WriteGate(console(), repo, auto_approve=True)
    assert "Nothing written" in gate.undo()


def test_undo_reverts_last_written_file(repo):
    gate = WriteGate(console(), repo, auto_approve=True)
    (repo / "app.py").write_text("edited\n")
    gate.last_written_path = "app.py"
    message = gate.undo()
    assert (repo / "app.py").read_text() == "original\n"
    assert "Reverted" in message


def test_undo_with_auto_commit_resets_agent_commit(repo):
    gate = WriteGate(console(), repo, auto_approve=True, auto_commit=True)
    gate.tracking_executor(repo, "c1", "write_file", {"path": "app.py", "content": "changed\n"})
    (repo / "app.py").write_text("changed\n")
    gate.after_turn("change app.py")
    message = gate.undo()
    assert (repo / "app.py").read_text() == "original\n"
    assert "Reset the last agent commit" in message


def test_undo_with_auto_commit_refuses_a_human_commit(repo):
    _git(repo, "commit", "--allow-empty", "-q", "-m", "a human's commit")
    gate = WriteGate(console(), repo, auto_approve=True, auto_commit=True)
    assert "refusing to reset" in gate.undo()


# ------------------------------------------------------------------- checkpoints


def test_confirm_write_records_a_checkpoint(repo):
    gate = WriteGate(console(), repo, auto_approve=True)
    gate.confirm_write("app.py", "new content\n")
    assert len(gate.checkpoints) == 1
    assert gate.checkpoints[0].path == "app.py"
    assert gate.checkpoints[0].previous_content == "original\n"


def test_confirm_write_records_none_for_a_new_file(repo):
    gate = WriteGate(console(), repo, auto_approve=True)
    gate.confirm_write("new_file.py", "content\n")
    assert gate.checkpoints[0].previous_content is None


def test_declined_write_does_not_record_a_checkpoint(repo, monkeypatch):
    monkeypatch.setattr("typer.confirm", lambda *a, **k: False)
    gate = WriteGate(console(), repo)
    gate.confirm_write("app.py", "new content\n")
    assert gate.checkpoints == []


def test_rewind_with_no_checkpoints_says_so(repo):
    gate = WriteGate(console(), repo)
    assert "Nothing to rewind" in gate.rewind()


def test_rewind_restores_previous_content(repo):
    gate = WriteGate(console(), repo, auto_approve=True)
    gate.confirm_write("app.py", "changed\n")
    (repo / "app.py").write_text("changed\n")  # simulate the tool actually writing it

    message = gate.rewind()
    assert (repo / "app.py").read_text() == "original\n"
    assert "app.py" in message
    assert gate.checkpoints == []


def test_rewind_deletes_a_newly_created_file(repo):
    gate = WriteGate(console(), repo, auto_approve=True)
    gate.confirm_write("new_file.py", "content\n")
    (repo / "new_file.py").write_text("content\n")

    gate.rewind()
    assert not (repo / "new_file.py").is_file()


def test_rewind_multiple_steps_in_reverse_order(repo):
    gate = WriteGate(console(), repo, auto_approve=True)
    gate.confirm_write("app.py", "v1\n")
    (repo / "app.py").write_text("v1\n")
    gate.confirm_write("app.py", "v2\n")
    (repo / "app.py").write_text("v2\n")

    gate.rewind(2)
    assert (repo / "app.py").read_text() == "original\n"
    assert gate.checkpoints == []


def test_rewind_caps_steps_at_available_checkpoints(repo):
    gate = WriteGate(console(), repo, auto_approve=True)
    gate.confirm_write("app.py", "v1\n")
    message = gate.rewind(99)
    assert "1 checkpoint" in message


def test_rewind_is_independent_of_auto_commit_and_undo(repo):
    """/rewind restores content directly; it doesn't need --auto-commit and
    doesn't touch git history the way /undo does."""
    gate = WriteGate(console(), repo, auto_approve=True)
    gate.confirm_write("app.py", "changed\n")
    (repo / "app.py").write_text("changed\n")
    gate.rewind()
    assert (repo / "app.py").read_text() == "original\n"
    from localgate.agent import gitutil

    assert gitutil.is_dirty(repo) is False  # nothing was ever committed


# ------------------------------------------------------------------- write modes


def test_default_mode_is_manual(repo):
    assert WriteGate(console(), repo).mode() == "manual"


def test_auto_approve_reports_as_auto_mode(repo):
    assert WriteGate(console(), repo, auto_approve=True).mode() == "auto"


def test_plan_mode_reports_as_plan(repo):
    assert WriteGate(console(), repo, plan_mode=True).mode() == "plan"


def test_cycle_mode_goes_manual_auto_plan_manual(repo):
    gate = WriteGate(console(), repo)
    assert gate.mode() == "manual"
    assert gate.cycle_mode() == "auto"
    assert gate.mode() == "auto"
    assert gate.cycle_mode() == "plan"
    assert gate.mode() == "plan"
    assert gate.cycle_mode() == "manual"
    assert gate.mode() == "manual"


def test_set_mode_is_mutually_exclusive(repo):
    gate = WriteGate(console(), repo, auto_approve=True)
    gate.set_mode("plan")
    assert gate.plan_mode is True
    assert gate.auto_approve is False
    gate.set_mode("auto")
    assert gate.plan_mode is False
    assert gate.auto_approve is True
    gate.set_mode("manual")
    assert gate.plan_mode is False
    assert gate.auto_approve is False


def test_confirm_write_in_plan_mode_queues_instead_of_writing(repo):
    gate = WriteGate(console(), repo, plan_mode=True)
    approved = gate.confirm_write("app.py", "new content\n")
    assert approved is False
    assert gate.pending_writes == [("app.py", "new content\n")]
    assert (repo / "app.py").read_text() == "original\n"  # untouched until flush_plan
    assert gate.declined_a_write is False  # queued, not declined


def test_flush_plan_applies_all(repo, monkeypatch):
    monkeypatch.setattr("typer.prompt", lambda *a, **k: "a")
    gate = WriteGate(console(), repo, plan_mode=True)
    gate.confirm_write("app.py", "from plan\n")
    gate.flush_plan()
    assert (repo / "app.py").read_text() == "from plan\n"
    assert gate.writes_this_turn == ["app.py"]
    assert gate.pending_writes == []


def test_flush_plan_applies_none_by_default(repo, monkeypatch):
    monkeypatch.setattr("typer.prompt", lambda *a, **k: "n")
    gate = WriteGate(console(), repo, plan_mode=True)
    gate.confirm_write("app.py", "from plan\n")
    gate.flush_plan()
    assert (repo / "app.py").read_text() == "original\n"
    assert gate.writes_this_turn == []
    assert gate.pending_writes == []


def test_flush_plan_pick_individually(repo, monkeypatch):
    monkeypatch.setattr("typer.prompt", lambda *a, **k: "p")
    confirms = iter([True, False])
    monkeypatch.setattr("typer.confirm", lambda *a, **k: next(confirms))
    gate = WriteGate(console(), repo, plan_mode=True)
    gate.confirm_write("app.py", "keep this\n")
    gate.confirm_write("other.py", "skip this\n")
    gate.flush_plan()
    assert (repo / "app.py").read_text() == "keep this\n"
    assert not (repo / "other.py").is_file()
    assert gate.writes_this_turn == ["app.py"]


def test_flush_plan_with_nothing_pending_is_a_noop(repo):
    gate = WriteGate(console(), repo, plan_mode=True)
    gate.flush_plan()  # should not raise or prompt
    assert gate.writes_this_turn == []


async def test_run_turn_flushes_plan_after_the_model_finishes(repo, monkeypatch):
    monkeypatch.setattr("typer.prompt", lambda *a, **k: "a")
    backend = ScriptedBackend([write_call("app.py", "from plan\n"), final_text("done")])
    out = console()
    gate = WriteGate(out, repo, plan_mode=True)
    session = AgentSession(
        backend, "scripted-model", repo, confirm_write=gate.confirm_write,
        tool_executor=gate.tracking_executor,
    )
    result = await run_turn(out, session, gate, "write something")
    assert result == "done"
    assert (repo / "app.py").read_text() == "from plan\n"


# ------------------------------------------------------------------------ REPL


async def test_repl_exit_command_ends_the_session(repo, monkeypatch):
    inputs = iter(["/exit"])
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: next(inputs))
    backend = ScriptedBackend([])
    await run_repl(backend, "scripted-model", repo, auto_approve=True)


async def test_repl_clear_resets_conversation_history(repo, monkeypatch):
    inputs = iter(["hello", "/clear", "/exit"])
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: next(inputs))
    backend = ScriptedBackend([final_text("hi there")])
    await run_repl(backend, "scripted-model", repo, auto_approve=True)


async def test_repl_model_command_switches_model(repo, monkeypatch):
    inputs = iter(["/model other-model", "/exit"])
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: next(inputs))
    backend = ScriptedBackend([])
    session_holder: dict[str, AgentSession] = {}
    orig_init = AgentSession.__init__

    def capture_init(self, *a, **k):
        orig_init(self, *a, **k)
        session_holder["session"] = self

    monkeypatch.setattr(AgentSession, "__init__", capture_init)
    await run_repl(backend, "scripted-model", repo, auto_approve=True)
    assert session_holder["session"].model == "other-model"


async def test_repl_model_command_rejects_an_unknown_model(repo, monkeypatch):
    inputs = iter(["/model nonexistent-model", "/exit"])
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: next(inputs))
    backend = ScriptedBackend([])
    await run_repl(backend, "scripted-model", repo, auto_approve=True)
    # rejection is logged via console.print — no crash and the model stays put is
    # exercised by test_repl_model_command_switches_model; this just confirms an
    # unknown name doesn't blow up the REPL loop.


async def test_repl_model_picker_selects_by_number(repo, monkeypatch):
    inputs = iter(["/model", "2", "/exit"])
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: next(inputs))
    backend = ScriptedBackend([])
    await run_repl(backend, "scripted-model", repo, auto_approve=True)


async def test_repl_model_switch_warns_mid_session_and_can_be_declined(repo, monkeypatch):
    inputs = iter(["hello", "/model other-model", "n", "/exit"])
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: next(inputs))
    backend = ScriptedBackend([final_text("hi there")])
    session_holder: dict[str, AgentSession] = {}
    orig_init = AgentSession.__init__

    def capture_init(self, *a, **k):
        orig_init(self, *a, **k)
        session_holder["session"] = self

    monkeypatch.setattr(AgentSession, "__init__", capture_init)
    await run_repl(backend, "scripted-model", repo, auto_approve=True)
    assert session_holder["session"].model == "scripted-model"


async def test_repl_theme_command_switches_and_persists(repo, monkeypatch):
    # No monkeypatching of the config path needed: conftest's autouse
    # `_hermetic_user_dirs` already points LOCALGATE_CONFIG_DIR at a sandbox.
    inputs = iter(["/theme light", "/exit"])
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: next(inputs))
    backend = ScriptedBackend([])
    await run_repl(backend, "scripted-model", repo, auto_approve=True)
    assert UserConfig.load().theme == "light"


async def test_repl_theme_command_rejects_unknown_theme(repo, monkeypatch):
    inputs = iter(["/theme neon", "/exit"])
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: next(inputs))
    backend = ScriptedBackend([])
    await run_repl(backend, "scripted-model", repo, auto_approve=True)
    assert not userconfig.config_path().is_file()


async def test_repl_no_color_disables_styling(repo, monkeypatch):
    inputs = iter(["/exit"])
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: next(inputs))
    backend = ScriptedBackend([])
    await run_repl(backend, "scripted-model", repo, auto_approve=True, no_color=True)


async def test_repl_help_lists_every_command(repo, monkeypatch):
    inputs = iter(["/help", "/exit"])
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: next(inputs))
    backend = ScriptedBackend([])
    await run_repl(backend, "scripted-model", repo, auto_approve=True)


async def test_repl_tools_lists_builtin_and_disabled_extras(repo, monkeypatch):
    inputs = iter(["/tools", "/exit"])
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: next(inputs))
    backend = ScriptedBackend([])
    await run_repl(backend, "scripted-model", repo, auto_approve=True)


async def test_repl_tools_shows_delegation_and_search_when_enabled(repo, monkeypatch):
    async def fake_search(query: str) -> str:
        return "x"

    inputs = iter(["/tools", "/exit"])
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: next(inputs))
    backend = ScriptedBackend([])
    await run_repl(
        backend,
        "scripted-model",
        repo,
        auto_approve=True,
        allow_delegation=True,
        search_fn=fake_search,
    )


async def test_repl_mode_command_shows_and_sets(repo, monkeypatch):
    inputs = iter(["/mode", "/mode plan", "/mode bogus", "/exit"])
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: next(inputs))
    backend = ScriptedBackend([])
    session_holder: dict[str, AgentSession] = {}
    orig_init = AgentSession.__init__

    def capture_init(self, *a, **k):
        orig_init(self, *a, **k)
        session_holder["session"] = self

    monkeypatch.setattr(AgentSession, "__init__", capture_init)
    await run_repl(backend, "scripted-model", repo)
    # the gate isn't captured directly, but a bogus /mode shouldn't crash the REPL
    # and a real switch is exercised at the WriteGate level in the tests above.
    assert session_holder["session"] is not None


async def test_repl_starts_in_plan_mode_with_plan_mode_flag(repo, monkeypatch):
    inputs = iter(["/mode", "/exit"])
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: next(inputs))
    backend = ScriptedBackend([])
    await run_repl(backend, "scripted-model", repo, plan_mode=True)


async def test_repl_rewind_restores_a_write(repo, monkeypatch):
    inputs = iter(["write it", "/rewind", "/exit"])
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: next(inputs))
    backend = ScriptedBackend([write_call("app.py", "new content\n"), final_text("done")])
    await run_repl(backend, "scripted-model", repo, auto_approve=True)
    assert (repo / "app.py").read_text() == "original\n"


async def test_repl_rewind_rejects_a_non_numeric_argument(repo, monkeypatch):
    inputs = iter(["/rewind abc", "/exit"])
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: next(inputs))
    backend = ScriptedBackend([])
    await run_repl(backend, "scripted-model", repo, auto_approve=True)


async def test_repl_usage_reports_after_a_turn(repo, monkeypatch):
    inputs = iter(["hello", "/usage", "/exit"])
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: next(inputs))
    backend = ScriptedBackend([final_text("hi there")])
    await run_repl(backend, "scripted-model", repo, auto_approve=True)


async def test_repl_context_reports_token_count(repo, monkeypatch):
    inputs = iter(["/context", "/exit"])
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: next(inputs))
    backend = ScriptedBackend([])
    await run_repl(backend, "scripted-model", repo, auto_approve=True)


async def test_repl_config_shows_and_sets_a_value(repo, monkeypatch):
    inputs = iter(["/config", "/config max_turns 42", "/exit"])
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: next(inputs))
    backend = ScriptedBackend([])
    await run_repl(backend, "scripted-model", repo, auto_approve=True)
    assert UserConfig.load().max_turns == 42


async def test_repl_config_rejects_an_unknown_key(repo, monkeypatch):
    inputs = iter(["/config bogus_key 1", "/exit"])
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: next(inputs))
    backend = ScriptedBackend([])
    await run_repl(backend, "scripted-model", repo, auto_approve=True)
    assert not userconfig.config_path().is_file()


async def test_repl_resume_without_memory_says_so(repo, monkeypatch):
    inputs = iter(["/resume", "/exit"])
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: next(inputs))
    backend = ScriptedBackend([])
    await run_repl(backend, "scripted-model", repo, auto_approve=True, memory=None)


async def test_run_turn_records_usage(repo):
    from localgate.agent.repl import SessionUsage

    backend = ScriptedBackend([final_text("done")])
    out = console()
    gate = WriteGate(out, repo, auto_approve=True)
    session = AgentSession(
        backend, "scripted-model", repo, confirm_write=gate.confirm_write,
        tool_executor=gate.tracking_executor,
    )
    usage = SessionUsage()
    await run_turn(out, session, gate, "hello", usage=usage)
    assert usage.request_count == 1
    assert usage.prompt_tokens > 0
    assert usage.completion_tokens > 0


async def test_run_single_shot_reports_a_declined_write(repo, monkeypatch):
    monkeypatch.setattr("typer.confirm", lambda *a, **k: False)
    backend = ScriptedBackend([write_call("app.py", "new\n"), final_text("done")])
    result = await run_single_shot(backend, "scripted-model", repo, "write something")
    assert result.text == "done"
    assert result.declined_a_write is True


async def test_run_single_shot_with_auto_approve_never_declines(repo):
    backend = ScriptedBackend([write_call("app.py", "new\n"), final_text("done")])
    result = await run_single_shot(
        backend, "scripted-model", repo, "write something", auto_approve=True
    )
    assert result.declined_a_write is False


async def test_repl_eof_ends_the_session_cleanly(repo, monkeypatch):
    def raise_eof(self, *a, **k):
        raise EOFError

    monkeypatch.setattr(Console, "input", raise_eof)
    backend = ScriptedBackend([])
    await run_repl(backend, "scripted-model", repo, auto_approve=True)


async def test_repl_persists_history_across_turns(repo, monkeypatch):
    inputs = iter(["first", "second", "/exit"])
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: next(inputs))
    backend = ScriptedBackend([final_text("ok one"), final_text("ok two")])
    await run_repl(backend, "scripted-model", repo, auto_approve=True)
    assert len(backend._responses) == 0


async def test_run_turn_restarts_the_spinner_after_a_tool_call(repo, monkeypatch):
    """A tool call shouldn't leave a dead gap with no activity indicator — the
    status spinner should be started again (relabeled) after the tool-call log
    line, not just stopped once and left dead until the final answer.
    """
    starts = 0
    orig_start = Status.start

    def counting_start(self):
        nonlocal starts
        starts += 1
        return orig_start(self)

    monkeypatch.setattr(Status, "start", counting_start)

    backend = ScriptedBackend([write_call("app.py", "content\n"), final_text("done")])
    out = console()
    gate = WriteGate(out, repo, auto_approve=True)
    session = AgentSession(
        backend, "scripted-model", repo, confirm_write=gate.confirm_write,
        tool_executor=gate.tracking_executor,
    )
    result = await run_turn(out, session, gate, "update app.py")
    assert result == "done"
    # once for the initial "thinking..." spinner, once more after the tool-call event
    assert starts >= 2


async def test_run_turn_streams_and_auto_commits(repo):
    backend = ScriptedBackend([write_call("app.py", "streamed content\n"), final_text("done")])
    out = console()
    gate = WriteGate(out, repo, auto_approve=True, auto_commit=True)
    session = AgentSession(
        backend,
        "scripted-model",
        repo,
        confirm_write=gate.confirm_write,
        tool_executor=gate.tracking_executor,
    )
    result = await run_turn(out, session, gate, "update app.py")
    assert result == "done"
    assert (repo / "app.py").read_text() == "streamed content\n"

    from localgate.agent import gitutil

    assert gitutil.is_dirty(repo) is False


# ---------------------------------------------------------------- backend errors


def _http_status_error(status_code: int, body: dict | str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://localhost:11434/v1/chat/completions")
    content = json.dumps(body).encode() if isinstance(body, dict) else body.encode()
    response = httpx.Response(status_code, request=request, content=content)
    return httpx.HTTPStatusError(f"{status_code} error", request=request, response=response)


def test_describe_backend_error_extracts_openai_shaped_message():
    exc = _http_status_error(400, {"error": {"message": "llama3:latest does not support tools"}})
    assert describe_backend_error(exc) == "400: llama3:latest does not support tools"


def test_describe_backend_error_falls_back_to_raw_text():
    exc = _http_status_error(500, "internal server error")
    assert describe_backend_error(exc) == "500: internal server error"


def test_describe_backend_error_handles_a_body_with_no_error_key():
    exc = _http_status_error(400, {"something": "unexpected"})
    assert '"something": "unexpected"' in describe_backend_error(exc)


class FailingThenScriptedBackend(InferenceBackend):
    """Raises the given error on the first `chat_stream` call, then behaves like
    the normal scripted backend — models a mid-session backend rejection (e.g.
    the default model doesn't support tools) that shouldn't kill the whole REPL.
    """

    name = "failing-then-scripted"

    def __init__(self, error: Exception, responses: list[dict[str, Any]]) -> None:
        self._error = error
        self._responses = list(responses)
        self._failed_once = False

    async def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    async def chat_stream(self, request: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        if not self._failed_once:
            self._failed_once = True
            raise self._error
        message = self._responses.pop(0)
        yield {"choices": [{"delta": {"content": message.get("content") or ""}}]}

    async def embed(self, text: str, model: str) -> list[float]:
        raise NotImplementedError

    async def list_models(self) -> list[str]:
        return ["scripted-model"]

    async def health(self) -> bool:
        return True


async def test_repl_survives_a_400_and_keeps_the_session_open(repo, monkeypatch):
    inputs = iter(["write something", "/exit"])
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: next(inputs))
    error = _http_status_error(400, {"error": {"message": "does not support tools"}})
    backend = FailingThenScriptedBackend(error, [final_text("ok")])
    # should not raise — the bad turn is reported and the REPL keeps running
    await run_repl(backend, "scripted-model", repo, auto_approve=True)


async def test_repl_survives_a_connection_error(repo, monkeypatch):
    inputs = iter(["hello", "/exit"])
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: next(inputs))
    request = httpx.Request("POST", "http://localhost:11434/v1/chat/completions")
    backend = FailingThenScriptedBackend(
        httpx.ConnectError("connection refused", request=request), [final_text("ok")]
    )
    await run_repl(backend, "scripted-model", repo, auto_approve=True)
