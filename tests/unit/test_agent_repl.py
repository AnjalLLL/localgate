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

from localgate.agent.gitutil import AGENT_COMMIT_PREFIX
from localgate.agent.loop import AgentSession
from localgate.agent.repl import WriteGate, describe_backend_error, run_repl, run_turn
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
        return ["scripted-model"]

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
    await run_repl(backend, "scripted-model", repo, auto_approve=True)


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
