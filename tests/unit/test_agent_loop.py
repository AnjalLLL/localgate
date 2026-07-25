"""The agent loop's branching logic (tool call -> execute -> feed back -> repeat,
vs. plain text -> stop), exercised against a scripted fake backend rather than a
live tool-calling model.
"""

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from localgate.agent.loop import SYSTEM_PROMPT, AgentSession, AgentTurnLimitExceeded, run_agent
from localgate.backends.base import InferenceBackend


class ScriptedBackend(InferenceBackend):
    """Returns one scripted response per call to `chat`, in order."""

    name = "scripted"

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    async def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        message = self._responses.pop(0)
        return {"choices": [{"index": 0, "message": message, "finish_reason": "stop"}]}

    async def chat_stream(self, request: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        raise NotImplementedError

    async def embed(self, text: str, model: str) -> list[float]:
        raise NotImplementedError

    async def list_models(self) -> list[str]:
        return ["scripted-model"]

    async def health(self) -> bool:
        return True


def tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def final_text(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


@pytest.fixture
def project(tmp_path):
    (tmp_path / "app.py").write_text("old content\n")
    return tmp_path


async def test_plain_text_response_stops_immediately(project):
    """Short plain-text responses (under 80 chars) are treated as final answers."""
    backend = ScriptedBackend([final_text("nothing to do here")])
    result = await run_agent(backend, "scripted-model", project, "look around")
    assert result == "nothing to do here"
    assert len(backend.requests) == 1


async def test_single_tool_call_then_final_answer(project):
    backend = ScriptedBackend(
        [
            tool_call("c1", "read_file", {"path": "app.py"}),
            final_text("the file says 'old content'"),
        ]
    )
    result = await run_agent(backend, "scripted-model", project, "what's in app.py?")
    assert result == "the file says 'old content'"

    second_request = backend.requests[1]
    tool_message = next(m for m in second_request["messages"] if m["role"] == "tool")
    assert tool_message["content"] == "old content\n"
    assert tool_message["tool_call_id"] == "c1"


async def test_write_file_actually_writes_when_approved(project):
    backend = ScriptedBackend(
        [
            tool_call("c1", "write_file", {"path": "app.py", "content": "new content\n"}),
            final_text("updated app.py"),
        ]
    )
    result = await run_agent(
        backend, "scripted-model", project, "update app.py", confirm_write=lambda *_: True
    )
    assert result == "updated app.py"
    assert (project / "app.py").read_text() == "new content\n"


async def test_write_file_is_skipped_when_declined(project):
    backend = ScriptedBackend(
        [
            tool_call("c1", "write_file", {"path": "app.py", "content": "new content\n"}),
            final_text("ok, left it alone"),
        ]
    )
    result = await run_agent(
        backend, "scripted-model", project, "update app.py", confirm_write=lambda *_: False
    )
    assert result == "ok, left it alone"
    assert (project / "app.py").read_text() == "old content\n"  # unchanged

    second_request = backend.requests[1]
    tool_message = next(m for m in second_request["messages"] if m["role"] == "tool")
    assert "declined" in tool_message["content"]


async def test_multiple_tool_calls_in_one_turn_are_all_executed(project):
    (project / "b.py").write_text("b\n")
    backend = ScriptedBackend(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "app.py"}),
                        },
                    },
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "b.py"}),
                        },
                    },
                ],
            },
            final_text("read both"),
        ]
    )
    result = await run_agent(backend, "scripted-model", project, "read app.py and b.py")
    assert result == "read both"

    tool_messages = [m for m in backend.requests[1]["messages"] if m["role"] == "tool"]
    assert {m["tool_call_id"] for m in tool_messages} == {"c1", "c2"}


async def test_failed_tool_call_is_fed_back_instead_of_raising(project):
    backend = ScriptedBackend(
        [
            tool_call("c1", "read_file", {"path": "missing.py"}),
            final_text("that file doesn't exist"),
        ]
    )
    result = await run_agent(backend, "scripted-model", project, "read missing.py")
    assert result == "that file doesn't exist"

    tool_message = next(m for m in backend.requests[1]["messages"] if m["role"] == "tool")
    assert "No such file" in tool_message["content"]


async def test_exceeding_max_turns_raises(project):
    responses = [tool_call(f"c{i}", "list_directory", {}) for i in range(5)]
    backend = ScriptedBackend(responses)
    with pytest.raises(AgentTurnLimitExceeded):
        await run_agent(backend, "scripted-model", project, "loop forever", max_turns=5)


async def test_on_event_is_called_for_each_tool_call(project):
    backend = ScriptedBackend(
        [
            tool_call("c1", "read_file", {"path": "app.py"}),
            final_text("done"),
        ]
    )
    events: list[str] = []
    await run_agent(backend, "scripted-model", project, "read app.py", on_event=events.append)
    assert len(events) == 1
    assert "read_file" in events[0]


# ------------------------------------------- synthetic tool-call fallback (qwen2.5-coder)


async def test_raw_json_content_is_treated_as_a_tool_call(project):
    """The exact shape observed live from qwen2.5-coder:7b via Ollama's OpenAI-compat
    shim: no `tool_calls` field at all, just the call disguised as `content`.
    """
    backend = ScriptedBackend(
        [
            final_text('{"name": "read_file", "arguments": {"path": "app.py"}}'),
            final_text("the file says 'old content'"),
        ]
    )
    result = await run_agent(backend, "scripted-model", project, "what's in app.py?")
    assert result == "the file says 'old content'"

    second_request = backend.requests[1]
    tool_message = next(m for m in second_request["messages"] if m["role"] == "tool")
    assert tool_message["content"] == "old content\n"

    # the synthesized assistant message must look identical to a real tool-call
    # message once it's in history, or replaying it on a later turn would confuse
    # a model expecting the standard shape.
    assistant_message = next(m for m in second_request["messages"] if m.get("role") == "assistant")
    assert assistant_message["content"] is None
    assert assistant_message["tool_calls"][0]["function"]["name"] == "read_file"


async def test_markdown_fenced_json_is_treated_as_a_tool_call(project):
    backend = ScriptedBackend(
        [
            final_text('```json\n{"name": "read_file", "arguments": {"path": "app.py"}}\n```'),
            final_text("done"),
        ]
    )
    result = await run_agent(backend, "scripted-model", project, "read app.py")
    assert result == "done"
    tool_message = next(m for m in backend.requests[1]["messages"] if m["role"] == "tool")
    assert tool_message["content"] == "old content\n"


async def test_tool_call_xml_tag_is_treated_as_a_tool_call(project):
    tagged = '<tool_call>{"name": "read_file", "arguments": {"path": "app.py"}}</tool_call>'
    backend = ScriptedBackend([final_text(tagged), final_text("done")])
    result = await run_agent(backend, "scripted-model", project, "read app.py")
    assert result == "done"
    tool_message = next(m for m in backend.requests[1]["messages"] if m["role"] == "tool")
    assert tool_message["content"] == "old content\n"


async def test_a_different_xml_tag_name_is_also_treated_as_a_tool_call(project):
    """qwen2.5-coder was observed live wrapping the same call in `<tool_request>`
    on one run and `<tool_call>` on another, from an otherwise identical prompt —
    the tag name itself isn't reliable, so any single wrapping tag counts."""
    tagged = '<tool_request>{"name": "read_file", "arguments": {"path": "app.py"}}</tool_request>'
    backend = ScriptedBackend([final_text(tagged), final_text("done")])
    result = await run_agent(backend, "scripted-model", project, "read app.py")
    assert result == "done"
    tool_message = next(m for m in backend.requests[1]["messages"] if m["role"] == "tool")
    assert tool_message["content"] == "old content\n"


async def test_fenced_and_tagged_wrapping_together_is_treated_as_a_tool_call(project):
    doubly_wrapped = (
        '```json\n<tool_call>{"name": "read_file", "arguments": '
        '{"path": "app.py"}}</tool_call>\n```'
    )
    backend = ScriptedBackend([final_text(doubly_wrapped), final_text("done")])
    result = await run_agent(backend, "scripted-model", project, "read app.py")
    assert result == "done"
    tool_message = next(m for m in backend.requests[1]["messages"] if m["role"] == "tool")
    assert tool_message["content"] == "old content\n"


async def test_string_encoded_arguments_are_accepted(project):
    """Some models put the arguments through json.dumps twice — arguments arrives
    as a JSON *string*, not a nested object. Still a tool call, not a final answer.
    """
    backend = ScriptedBackend(
        [
            final_text('{"name": "read_file", "arguments": "{\\"path\\": \\"app.py\\"}"}'),
            final_text("done"),
        ]
    )
    result = await run_agent(backend, "scripted-model", project, "read app.py")
    assert result == "done"
    tool_message = next(m for m in backend.requests[1]["messages"] if m["role"] == "tool")
    assert tool_message["content"] == "old content\n"


async def test_write_confirmation_still_applies_to_a_synthetic_tool_call(project):
    """The whole point of routing synthetic calls through the same execution path:
    a disguised write_file still gets a diff/confirmation prompt, not a silent write.
    """
    backend = ScriptedBackend(
        [
            final_text(
                '{"name": "write_file", "arguments": '
                '{"path": "app.py", "content": "new content\\n"}}'
            ),
            final_text("ok, left it alone"),
        ]
    )
    result = await run_agent(
        backend, "scripted-model", project, "update app.py", confirm_write=lambda *_: False
    )
    assert result == "ok, left it alone"
    assert (project / "app.py").read_text() == "old content\n"  # write was declined, not silent


async def test_unrecognized_tool_name_is_not_treated_as_a_tool_call(project):
    """A name-shaped JSON object that isn't one of this session's actual tools —
    guards against a coincidentally tool-shaped final answer being misexecuted."""
    backend = ScriptedBackend(
        [final_text('{"name": "delete_everything", "arguments": {"path": "/"}}')]
    )
    result = await run_agent(backend, "scripted-model", project, "do something")
    assert result == '{"name": "delete_everything", "arguments": {"path": "/"}}'


async def test_plain_prose_final_answer_is_not_treated_as_a_tool_call(project):
    backend = ScriptedBackend([final_text("I looked at app.py and it defines add().")])
    result = await run_agent(backend, "scripted-model", project, "what does app.py do?")
    assert result == "I looked at app.py and it defines add()."


async def test_json_config_shown_as_a_final_answer_is_not_misread(project):
    """JSON content the model is *showing* the user (e.g. quoting a config file)
    must not be executed just because it happens to parse as an object — only a
    strict {name, arguments} shape referencing a real tool counts."""
    backend = ScriptedBackend([final_text('{"debug": true, "port": 8000}')])
    result = await run_agent(backend, "scripted-model", project, "what's in config.json?")
    assert result == '{"debug": true, "port": 8000}'


async def test_json_embedded_in_prose_is_extracted_as_a_tool_call(project):
    """Small models often mix prose with a JSON tool call — we extract and execute it."""
    backend = ScriptedBackend(
        [
            final_text('I\'ll read it: {"name": "read_file", "arguments": {"path": "app.py"}}'),
            final_text("the file has old content"),
        ]
    )
    result = await run_agent(backend, "scripted-model", project, "read app.py")
    assert result == "the file has old content"
    tool_message = next(m for m in backend.requests[1]["messages"] if m["role"] == "tool")
    assert tool_message["content"] == "old content\n"


async def test_fallback_never_triggers_for_real_structured_tool_calls(project):
    """A model that already emits proper `tool_calls` shouldn't have its content
    (which is typically None or empty on a tool-call turn) reinterpreted at all."""
    backend = ScriptedBackend(
        [
            tool_call("c1", "read_file", {"path": "app.py"}),
            final_text("done"),
        ]
    )
    events: list[str] = []
    result = await run_agent(
        backend, "scripted-model", project, "read app.py", on_event=events.append
    )
    assert result == "done"
    assert not any("parsed" in e for e in events)


async def test_on_event_notes_when_falling_back_to_synthetic_parsing(project):
    backend = ScriptedBackend(
        [
            final_text('{"name": "read_file", "arguments": {"path": "app.py"}}'),
            final_text("done"),
        ]
    )
    events: list[str] = []
    await run_agent(backend, "scripted-model", project, "read app.py", on_event=events.append)
    assert any("read_file" in e for e in events)


# --------------------------------------------------------------- delegate_task


async def test_delegate_task_is_not_offered_by_default(project):
    """A plain session's schema never includes delegate_task — the model
    literally cannot discover it exists unless the parent opts in."""
    backend = ScriptedBackend([final_text("done")])
    await run_agent(backend, "scripted-model", project, "do something")
    names = {t["function"]["name"] for t in backend.requests[0]["tools"]}
    assert "delegate_task" not in names


async def test_delegate_task_is_offered_when_allowed(project):
    backend = ScriptedBackend([final_text("done")])
    await run_agent(backend, "scripted-model", project, "do something", allow_delegation=True)
    names = {t["function"]["name"] for t in backend.requests[0]["tools"]}
    assert "delegate_task" in names


async def test_delegate_task_runs_a_sub_agent_and_returns_its_summary(project):
    backend = ScriptedBackend(
        [
            tool_call("c1", "delegate_task", {"task": "find every read_file call"}),
            # the sub-agent's own turn(s) come next, in call order:
            tool_call("c2", "read_file", {"path": "app.py"}),
            final_text("found one usage in app.py"),
            # then the parent's final turn:
            final_text("done — see sub-agent summary"),
        ]
    )
    result = await run_agent(
        backend, "scripted-model", project, "audit read_file usage", allow_delegation=True
    )
    assert result == "done — see sub-agent summary"

    # the parent's final request must carry the sub-agent's summary as the
    # delegate_task tool result, not the sub-agent's raw tool-call noise
    final_request = backend.requests[-1]
    tool_message = next(m for m in final_request["messages"] if m["role"] == "tool")
    assert tool_message["content"] == "found one usage in app.py"
    assert tool_message["tool_call_id"] == "c1"


async def test_delegate_task_defaults_to_read_only_tools(project):
    """Without an explicit allowed_tools, a sub-agent can't write — a write_file
    call it attempts should come back as 'unknown tool', not actually write."""
    backend = ScriptedBackend(
        [
            tool_call("c1", "delegate_task", {"task": "write something"}),
            tool_call("c2", "write_file", {"path": "app.py", "content": "hacked\n"}),
            final_text("sub-agent tried to write but couldn't"),
            final_text("done"),
        ]
    )
    await run_agent(backend, "scripted-model", project, "delegate a write", allow_delegation=True)
    assert (project / "app.py").read_text() == "old content\n"  # untouched


async def test_delegate_task_write_access_is_opt_in_and_still_confirmed(project):
    backend = ScriptedBackend(
        [
            tool_call(
                "c1",
                "delegate_task",
                {"task": "update app.py", "allowed_tools": ["write_file"]},
            ),
            tool_call("c2", "write_file", {"path": "app.py", "content": "new content\n"}),
            final_text("wrote it"),
            final_text("done"),
        ]
    )
    declined: list[str] = []

    def confirm_write(path: str, content: str) -> bool:
        declined.append(path)
        return False  # decline — proves the sub-agent's write went through the SAME gate

    await run_agent(
        backend,
        "scripted-model",
        project,
        "delegate a write",
        allow_delegation=True,
        confirm_write=confirm_write,
    )
    assert declined == ["app.py"]
    assert (project / "app.py").read_text() == "old content\n"  # declined, so unchanged


async def test_delegate_task_sub_agent_cannot_itself_delegate(project):
    """Depth cap: the sub-agent's own schema must not include delegate_task."""
    backend = ScriptedBackend(
        [
            tool_call("c1", "delegate_task", {"task": "nested task"}),
            final_text("sub-agent finished"),
            final_text("done"),
        ]
    )
    await run_agent(backend, "scripted-model", project, "delegate", allow_delegation=True)
    # the sub-agent's own request (index 1) is the one whose tools list to check
    sub_agent_request = backend.requests[1]
    names = {t["function"]["name"] for t in sub_agent_request["tools"]}
    assert "delegate_task" not in names


async def test_delegate_task_respects_a_custom_max_turns(project):
    responses = [tool_call("c1", "delegate_task", {"task": "loop", "max_turns": 2})]
    responses += [tool_call(f"sub{i}", "list_directory", {}) for i in range(2)]
    responses.append(final_text("done, sub-agent gave up"))
    backend = ScriptedBackend(responses)
    result = await run_agent(
        backend, "scripted-model", project, "delegate a looping task", allow_delegation=True
    )
    assert result == "done, sub-agent gave up"
    tool_message = next(m for m in backend.requests[-1]["messages"] if m["role"] == "tool")
    assert "Sub-agent stopped without finishing" in tool_message["content"]


async def test_delegate_task_requires_a_task_string(project):
    backend = ScriptedBackend(
        [
            tool_call("c1", "delegate_task", {}),
            final_text("done"),
        ]
    )
    await run_agent(backend, "scripted-model", project, "delegate nothing", allow_delegation=True)
    tool_message = next(m for m in backend.requests[-1]["messages"] if m["role"] == "tool")
    assert "requires a non-empty" in tool_message["content"]


async def test_delegating_does_not_mutate_the_shared_tool_schemas_list(project):
    """AgentSession.tool_schemas must be its own list — appending delegate_task
    for one session must not leak it into every other session's tool schema.
    """
    from localgate.agent.tools import TOOL_SCHEMAS

    before = len(TOOL_SCHEMAS)
    backend = ScriptedBackend([final_text("done")])
    await run_agent(backend, "scripted-model", project, "task", allow_delegation=True)
    assert len(TOOL_SCHEMAS) == before


# -------------------------------------------------------------------- web_search


async def test_web_search_is_not_offered_by_default(project):
    backend = ScriptedBackend([final_text("done")])
    await run_agent(backend, "scripted-model", project, "do something")
    names = {t["function"]["name"] for t in backend.requests[0]["tools"]}
    assert "web_search" not in names


async def test_web_search_is_offered_when_a_search_fn_is_configured(project):
    async def fake_search(query: str) -> str:
        return "fake results"

    backend = ScriptedBackend([final_text("done")])
    await run_agent(backend, "scripted-model", project, "do something", search_fn=fake_search)
    names = {t["function"]["name"] for t in backend.requests[0]["tools"]}
    assert "web_search" in names


async def test_web_search_calls_the_configured_search_fn(project):
    calls: list[str] = []

    async def fake_search(query: str) -> str:
        calls.append(query)
        return "- Example\n  a snippet\n  https://example.com"

    backend = ScriptedBackend(
        [
            tool_call("c1", "web_search", {"query": "localgate release notes"}),
            final_text("found it"),
        ]
    )
    result = await run_agent(
        backend, "scripted-model", project, "look it up", search_fn=fake_search
    )
    assert result == "found it"
    assert calls == ["localgate release notes"]

    tool_message = next(m for m in backend.requests[-1]["messages"] if m["role"] == "tool")
    assert "example.com" in tool_message["content"]


async def test_web_search_without_a_query_does_not_call_search_fn(project):
    calls: list[str] = []

    async def fake_search(query: str) -> str:
        calls.append(query)
        return "should not happen"

    backend = ScriptedBackend([tool_call("c1", "web_search", {}), final_text("done")])
    await run_agent(backend, "scripted-model", project, "look it up", search_fn=fake_search)
    assert calls == []


# ------------------------------------------------------------- system prompt guidance


def test_system_prompt_is_the_base_prompt_with_no_optional_tools(project):
    session = AgentSession(ScriptedBackend([]), "scripted-model", project)
    assert session.system_prompt().startswith(SYSTEM_PROMPT)
    assert f"Project directory: {project}" in session.system_prompt()
    assert session.messages[0]["content"].startswith(SYSTEM_PROMPT)


def test_system_prompt_adds_delegate_guidance_when_allowed(project):
    session = AgentSession(ScriptedBackend([]), "scripted-model", project, allow_delegation=True)
    prompt = session.system_prompt()
    assert prompt.startswith(SYSTEM_PROMPT)
    assert "delegate_task" in prompt
    assert "web_search" not in prompt


async def test_system_prompt_adds_search_guidance_when_configured(project):
    async def fake_search(query: str) -> str:
        return "x"

    session = AgentSession(ScriptedBackend([]), "scripted-model", project, search_fn=fake_search)
    prompt = session.system_prompt()
    assert "web_search" in prompt
    assert "delegate_task" not in prompt


async def test_system_prompt_adds_both_when_both_are_configured(project):
    async def fake_search(query: str) -> str:
        return "x"

    session = AgentSession(
        ScriptedBackend([]),
        "scripted-model",
        project,
        allow_delegation=True,
        search_fn=fake_search,
    )
    prompt = session.system_prompt()
    assert "delegate_task" in prompt
    assert "web_search" in prompt


def test_reset_rebuilds_the_dynamic_system_prompt(project):
    session = AgentSession(ScriptedBackend([]), "scripted-model", project, allow_delegation=True)
    session.messages.append({"role": "user", "content": "hi"})
    session.reset()
    assert session.messages == [{"role": "system", "content": session.system_prompt()}]


# ---------------------------------------------------------------- confirm_search/delegate


async def test_confirm_search_declining_skips_the_search(project):
    calls: list[str] = []

    async def fake_search(query: str) -> str:
        calls.append(query)
        return "should not happen"

    backend = ScriptedBackend(
        [tool_call("c1", "web_search", {"query": "x"}), final_text("ok, declined")]
    )
    result = await run_agent(
        backend,
        "scripted-model",
        project,
        "search something",
        search_fn=fake_search,
        confirm_search=lambda q: False,
    )
    assert result == "ok, declined"
    assert calls == []
    tool_message = next(m for m in backend.requests[-1]["messages"] if m["role"] == "tool")
    assert "declined" in tool_message["content"]


async def test_confirm_search_approving_runs_the_search(project):
    async def fake_search(query: str) -> str:
        return "results here"

    backend = ScriptedBackend([tool_call("c1", "web_search", {"query": "x"}), final_text("done")])
    result = await run_agent(
        backend,
        "scripted-model",
        project,
        "search something",
        search_fn=fake_search,
        confirm_search=lambda q: True,
    )
    assert result == "done"
    tool_message = next(m for m in backend.requests[-1]["messages"] if m["role"] == "tool")
    assert tool_message["content"] == "results here"


async def test_confirm_delegate_declining_skips_the_sub_agent(project):
    backend = ScriptedBackend(
        [tool_call("c1", "delegate_task", {"task": "explore"}), final_text("ok, declined")]
    )
    result = await run_agent(
        backend,
        "scripted-model",
        project,
        "delegate something",
        allow_delegation=True,
        confirm_delegate=lambda t: False,
    )
    assert result == "ok, declined"
    # only one chat() call happened at all — no sub-agent turn was ever run
    assert len(backend.requests) == 2
    tool_message = next(m for m in backend.requests[-1]["messages"] if m["role"] == "tool")
    assert "declined" in tool_message["content"]


async def test_confirm_delegate_approving_runs_the_sub_agent(project):
    backend = ScriptedBackend(
        [
            tool_call("c1", "delegate_task", {"task": "explore"}),
            final_text("sub-agent summary"),
            final_text("done"),
        ]
    )
    result = await run_agent(
        backend,
        "scripted-model",
        project,
        "delegate something",
        allow_delegation=True,
        confirm_delegate=lambda t: True,
    )
    assert result == "done"
    tool_message = next(m for m in backend.requests[-1]["messages"] if m["role"] == "tool")
    assert tool_message["content"] == "sub-agent summary"


async def test_multiple_code_blocks_extracted_as_multiple_writes(tmp_path):
    """When the model dumps multiple named code blocks, all are executed as writes."""
    multi_block_prose = (
        "Here are your files:\n\n"
        "### index.html\n"
        "```html\n"
        "<html><head><title>Test</title></head><body><h1>Hello World</h1>"
        "<p>This is a test page with enough content to pass the threshold.</p>"
        "</body></html>\n"
        "```\n\n"
        "### style.css\n"
        "```css\n"
        "body { margin: 0; padding: 20px; font-family: sans-serif; }\n"
        "h1 { color: navy; border-bottom: 2px solid navy; padding-bottom: 10px; }\n"
        "```\n\n"
        "Done: created index.html and style.css"
    )
    backend = ScriptedBackend(
        [
            {"role": "assistant", "content": multi_block_prose},
            {"role": "assistant", "content": "Done: created two files."},
        ]
    )
    events: list[str] = []
    session = AgentSession(
        backend, "m", tmp_path, on_event=events.append, confirm_write=lambda *_: True
    )
    await session.send("create a page")

    # Both files should have been written
    assert (tmp_path / "index.html").exists()
    assert (tmp_path / "style.css").exists()
    assert "<html>" in (tmp_path / "index.html").read_text()
    assert "sans-serif" in (tmp_path / "style.css").read_text()
