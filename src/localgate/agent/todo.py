"""The agent's visible to-do list (`todo_write`).

Writing the plan down and ticking items off keeps a small model on track through
a multi-step task far better than holding the plan in prose. The list is shown to
the user as it changes and mirrored into the session ledger's open to-dos, so it
survives context compaction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

STATUSES = ("pending", "in_progress", "done")
_MARKS = {"pending": "[ ]", "in_progress": "[~]", "done": "[x]"}

TODO_WRITE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "todo_write",
        "description": (
            "Record or update your step-by-step plan for a multi-step task. Send the whole "
            "list every time, marking each item pending, in_progress or done. Keep exactly "
            "one item in_progress while you work on it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {"type": "string", "enum": list(STATUSES)},
                        },
                        "required": ["content", "status"],
                    },
                }
            },
            "required": ["todos"],
        },
    },
}

TODO_GUIDANCE = (
    " For a task with three or more steps, call todo_write first with your plan, then "
    "update it as you finish each step."
)


@dataclass
class TodoList:
    items: list[tuple[str, str]] = field(default_factory=list)
    #: Called with the new list after every update — the REPL shows it.
    on_change: Callable[[TodoList], None] | None = None

    def pending(self) -> list[str]:
        return [content for content, status in self.items if status != "done"]

    def render(self) -> str:
        if not self.items:
            return "(no to-dos)"
        return "\n".join(f"{_MARKS[status]} {content}" for content, status in self.items)

    async def write(self, arguments: dict[str, Any]) -> str:
        raw = arguments.get("todos")
        if not isinstance(raw, list):
            return "todo_write needs a 'todos' list."
        items: list[tuple[str, str]] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            content = str(entry.get("content") or "").strip()
            status = str(entry.get("status") or "pending")
            if content:
                items.append((content[:200], status if status in STATUSES else "pending"))
        self.items = items[:30]
        if self.on_change is not None:
            self.on_change(self)
        return self.render()
