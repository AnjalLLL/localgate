"""Session ledger and compaction: intent survives, tool-call pairs stay intact."""

import json

from localgate.agent.context import ContextManager, SessionLedger


def tool_exchange(i: int, size: int = 4000) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"c{i}",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps({"path": f"f{i}.py", "content": "x" * size}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": f"c{i}", "name": "write_file", "content": "y" * size},
    ]


def long_history(exchanges: int = 12) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": "system prompt"}]
    messages.append({"role": "user", "content": "Build the login page with OAuth, no passwords."})
    for i in range(exchanges):
        messages.extend(tool_exchange(i))
    messages.append({"role": "assistant", "content": "done"})
    messages.append({"role": "user", "content": "now add tests"})
    return messages


def assert_tool_pairs_intact(messages: list[dict]) -> None:
    pending: set[str] = set()
    for message in messages:
        for call in message.get("tool_calls") or []:
            pending.add(call["id"])
        if message.get("role") == "tool":
            assert message["tool_call_id"] in pending, "tool result without its call"


def test_ledger_records_goal_task_files_and_errors_deterministically():
    ledger = SessionLedger()
    ledger.observe_user("first goal")
    ledger.observe_user("second task")
    ledger.observe_tool("read_file", {"path": "a.py"}, True, "...")
    ledger.observe_tool("edit_file", {"path": "a.py"}, True, "...")
    ledger.observe_tool("create_file", {"path": "b.py"}, True, "...")
    ledger.observe_tool("delete_file", {"path": "c.py"}, True, "...")
    ledger.observe_tool("read_file", {"path": "z.py"}, False, "No such file: z.py")
    assert ledger.goal == "first goal" and ledger.current_task == "second task"
    assert ledger.files_changed == {"a.py": "modified", "b.py": "created", "c.py": "deleted"}
    assert ledger.files_read == ["a.py"]
    assert "No such file" in (ledger.last_error or "")
    restored = SessionLedger.from_dict(json.loads(json.dumps(ledger.to_dict())))
    assert restored == ledger


def test_model_updates_never_overwrite_goal_or_files():
    ledger = SessionLedger(goal="real goal", files_changed={"a.py": "modified"})
    ledger.merge_model_update(
        {"goal": "hijacked", "files_changed": {}, "decisions": ["use OAuth"], "notes": "n"}
    )
    assert ledger.goal == "real goal"
    assert ledger.files_changed == {"a.py": "modified"}
    assert ledger.decisions == ["use OAuth"]


async def test_compaction_elides_old_tool_results_first():
    manager = ContextManager(8000)
    messages = long_history()
    assert manager.needs_compaction(messages, None, SessionLedger())
    out, report = await manager.compact(messages, None, SessionLedger())
    assert report.changed and report.after < report.before
    assert report.elided_tool_results > 0
    assert out[-1] == messages[-1]
    assert_tool_pairs_intact(out)


async def test_folding_keeps_user_words_and_feeds_the_ledger():
    seen: list[str] = []

    async def updater(ledger, transcript, focus):
        seen.append(transcript)
        return {
            "decisions": ["OAuth only"],
            "constraints": [],
            "open_todos": ["tests"],
            "notes": "x",
        }

    manager = ContextManager(800, updater=updater)
    ledger = SessionLedger(goal="Build the login page with OAuth, no passwords.")
    out, report = await manager.compact(long_history(), None, ledger, force=True)
    assert report.folded_messages > 0
    assert seen and "write_file" in seen[0]
    joined = "\n".join(m.get("content") or "" for m in out)
    assert "Build the login page with OAuth, no passwords." in joined
    assert ledger.decisions == ["OAuth only"]
    assert_tool_pairs_intact(out)
    roles = [m["role"] for m in out]
    assert roles[0] == "system"


async def test_a_long_single_task_is_compacted_inside_its_own_tool_chain():
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "do it"}]
    for i in range(6):
        messages.extend(tool_exchange(i, size=6000))
    manager = ContextManager(4000, keep_recent=50)
    out, report = await manager.compact(messages, None, SessionLedger())
    assert report.elided_tool_results > 0
    # The latest exchange (what the model is waiting on) is untouched.
    assert out[-1]["content"] == messages[-1]["content"]
    assert_tool_pairs_intact(out)


async def test_under_budget_history_is_left_alone():
    manager = ContextManager(100_000)
    messages = long_history(2)
    out, report = await manager.compact(messages, None, SessionLedger())
    assert not report.changed and out == messages
