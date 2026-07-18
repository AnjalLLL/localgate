"""The agent loop's branching logic (tool call -> execute -> feed back -> repeat,
vs. plain text -> stop), exercised against a scripted fake backend rather than a
live tool-calling model.
"""

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from localgate.agent.loop import AgentTurnLimitExceeded, run_agent
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


async def test_json_embedded_in_prose_is_not_treated_as_a_tool_call(project):
    backend = ScriptedBackend(
        [final_text('I\'ll read it: {"name": "read_file", "arguments": {"path": "app.py"}}')]
    )
    result = await run_agent(backend, "scripted-model", project, "read app.py")
    assert "I'll read it" in result


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
    assert not any("isn't using structured tool calls" in e for e in events)


async def test_on_event_notes_when_falling_back_to_synthetic_parsing(project):
    backend = ScriptedBackend(
        [
            final_text('{"name": "read_file", "arguments": {"path": "app.py"}}'),
            final_text("done"),
        ]
    )
    events: list[str] = []
    await run_agent(backend, "scripted-model", project, "read app.py", on_event=events.append)
    assert any("isn't using structured tool calls" in e for e in events)
