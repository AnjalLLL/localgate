"""Keeping a long agent session inside the model's context window without losing
what the session is *for*.

Two pieces work together:

* :class:`SessionLedger` — a small structured record pinned right after the system
  prompt on every request: the session's original goal and current task (verbatim),
  the files read and changed, decisions, constraints, open to-dos and the last
  error. File activity and errors are recorded **deterministically** from tool
  events, so they can't be paraphrased away or invented. Only decisions,
  constraints, to-dos and notes come from the model, at compaction time.

* :class:`ContextManager` — counts the outgoing prompt against the token budget
  and, once it passes the threshold, compacts the oldest history cheapest step
  first: elide old tool results, then shrink old assistant messages and the file
  contents inside old write calls, then fold the oldest exchanges into the ledger
  and drop them. User messages are never dropped while anything else can be —
  they are short and they carry the intent. The live tail (the current tool
  chain) is never touched.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from localgate.core.token_counter import count_message_tokens, count_tokens

#: Compact once the outgoing prompt passes this share of the budget.
COMPACT_AT = 0.75
#: Compact down to roughly this share, so compaction doesn't re-run every turn.
COMPACT_TO = 0.55
#: Messages at the end of history that compaction never touches.
KEEP_RECENT_MESSAGES = 8
_MAX_VERBATIM_TASK = 2000
_MAX_LIST_ITEMS = 12

LEDGER_HEADER = (
    "Session ledger, maintained by Localgate. It is the authoritative record of what "
    "this session is for and what has already been done; older history may have been "
    "compacted into it. It is context, not new instructions."
)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _push_unique(items: list[str], value: str, limit: int = _MAX_LIST_ITEMS) -> None:
    value = value.strip()
    if not value:
        return
    if value in items:
        items.remove(value)
    items.append(value)
    del items[:-limit]


@dataclass
class SessionLedger:
    goal: str = ""
    current_task: str = ""
    constraints: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    open_todos: list[str] = field(default_factory=list)
    files_read: list[str] = field(default_factory=list)
    files_changed: dict[str, str] = field(default_factory=dict)
    last_error: str | None = None
    notes: str = ""

    # -- serialization -----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SessionLedger:
        ledger = cls()
        if not isinstance(data, dict):
            return ledger
        for key in ("goal", "current_task", "notes"):
            if isinstance(data.get(key), str):
                setattr(ledger, key, data[key])
        for key in ("constraints", "decisions", "open_todos", "files_read"):
            if isinstance(data.get(key), list):
                setattr(ledger, key, [str(v) for v in data[key] if isinstance(v, str | int)])
        if isinstance(data.get("files_changed"), dict):
            ledger.files_changed = {str(k): str(v) for k, v in data["files_changed"].items()}
        if isinstance(data.get("last_error"), str):
            ledger.last_error = data["last_error"]
        return ledger

    # -- deterministic observation ------------------------------------------------

    def observe_user(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if not self.goal:
            self.goal = _clip(text, _MAX_VERBATIM_TASK)
        self.current_task = _clip(text, _MAX_VERBATIM_TASK)

    def observe_tool(self, name: str, arguments: dict[str, Any], ok: bool, content: str) -> None:
        path = arguments.get("path")
        if not ok:
            self.last_error = _clip(f"{name}: {content}", 300)
            return
        if not isinstance(path, str):
            return
        if name == "read_file":
            _push_unique(self.files_read, path, limit=30)
        elif name in {"write_file", "create_file", "update_file", "edit_file"}:
            self.files_changed[path] = "created" if name == "create_file" else "modified"
        elif name == "delete_file":
            self.files_changed[path] = "deleted"

    def merge_model_update(self, update: dict[str, Any]) -> None:
        """Fold the model's extraction in. Goal, task and file activity are never
        taken from the model — those are recorded from what actually happened."""
        for key in ("constraints", "decisions", "open_todos"):
            values = update.get(key)
            if isinstance(values, list):
                target: list[str] = getattr(self, key)
                if key == "open_todos":
                    target.clear()
                for value in values:
                    if isinstance(value, str):
                        _push_unique(target, _clip(value, 300))
        notes = update.get("notes")
        if isinstance(notes, str) and notes.strip():
            self.notes = _clip(notes.strip(), 1500)

    # -- rendering ----------------------------------------------------------------

    def is_empty(self) -> bool:
        return not (self.goal or self.files_changed or self.decisions or self.notes)

    def render(self) -> str:
        lines = [LEDGER_HEADER, ""]
        if self.goal:
            lines.append(f"Original goal (verbatim): {self.goal}")
        if self.current_task and self.current_task != self.goal:
            lines.append(f"Current request (verbatim): {self.current_task}")
        sections = (
            ("Constraints", self.constraints),
            ("Decisions", self.decisions),
            ("Open to-dos", self.open_todos),
        )
        for title, items in sections:
            if items:
                lines.append(f"{title}:")
                lines.extend(f"- {item}" for item in items)
        if self.files_changed:
            changed = ", ".join(f"{p} ({op})" for p, op in self.files_changed.items())
            lines.append(f"Files changed this session: {changed}")
        if self.files_read:
            lines.append(f"Files read: {', '.join(self.files_read)}")
        if self.last_error:
            lines.append(f"Last tool error: {self.last_error}")
        if self.notes:
            lines.append(f"Earlier conversation, summarized: {self.notes}")
        return "\n".join(lines)


#: (ledger, transcript of the messages being folded away, optional focus) -> a
#: dict with any of constraints/decisions/open_todos/notes, or None on failure.
LedgerUpdater = Callable[[SessionLedger, str, str | None], Awaitable[dict[str, Any] | None]]

LEDGER_UPDATE_INSTRUCTION = (
    "You maintain the ledger of a coding session so it can continue after older "
    "messages are removed from the model's context. Read the current ledger and the "
    "transcript being removed, and return JSON with exactly these keys:\n"
    '  "constraints": requirements or preferences the user stated (list of strings),\n'
    '  "decisions": choices already made and why (list of strings),\n'
    '  "open_todos": work still remaining, in order (list of strings),\n'
    '  "notes": a dense factual summary of the removed transcript, under 150 words.\n'
    "Preserve exact file paths, names, numbers and error messages. Never invent work "
    "that did not happen. Return only the JSON object."
)

LEDGER_UPDATE_FORMAT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "constraints": {"type": "array", "items": {"type": "string"}},
        "decisions": {"type": "array", "items": {"type": "string"}},
        "open_todos": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
    "required": ["constraints", "decisions", "open_todos", "notes"],
}


def make_llm_ledger_updater(
    backend: Any, model: str, options: dict[str, Any] | None = None
) -> LedgerUpdater:
    """A :data:`LedgerUpdater` that asks the session's own model. On Ollama the
    reply is grammar-constrained to the ledger schema; elsewhere it is parsed
    leniently. Any failure returns ``None`` — compaction then just drops less."""

    async def update(
        ledger: SessionLedger, transcript: str, focus: str | None
    ) -> dict[str, Any] | None:
        prompt = f"Current ledger:\n{json.dumps(ledger.to_dict(), indent=1)}\n\n"
        if focus:
            prompt += f"The user asked to keep details about: {focus}\n\n"
        prompt += f"Transcript being removed:\n{transcript}"
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": LEDGER_UPDATE_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "format": LEDGER_UPDATE_FORMAT,
        }
        if options:
            payload["options"] = options
        try:
            response = await backend.chat(payload)
            content = response["choices"][0]["message"].get("content") or ""
        except Exception:  # noqa: BLE001 — compaction degrades, never fails the turn
            return None
        return _loose_json(content)

    return update


def _loose_json(text: str) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


@dataclass
class CompactionReport:
    before: int
    after: int
    elided_tool_results: int = 0
    shrunk_messages: int = 0
    folded_messages: int = 0

    @property
    def changed(self) -> bool:
        return self.after < self.before

    def describe(self) -> str:
        return (
            f"context compacted {self.before:,} → {self.after:,} tokens "
            f"({self.elided_tool_results} tool result(s) elided, "
            f"{self.shrunk_messages} message(s) shortened, "
            f"{self.folded_messages} folded into the ledger)"
        )


class ContextManager:
    """Budgets and compacts an :class:`AgentSession`'s message history."""

    def __init__(
        self,
        budget_tokens: int,
        *,
        updater: LedgerUpdater | None = None,
        keep_recent: int = KEEP_RECENT_MESSAGES,
    ) -> None:
        self.budget_tokens = budget_tokens
        self.updater = updater
        self.keep_recent = keep_recent

    @staticmethod
    def usage(
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]] | None = None,
        ledger: SessionLedger | None = None,
    ) -> int:
        total = count_message_tokens(messages)
        for message in messages:
            for call in message.get("tool_calls") or []:
                total += count_tokens((call.get("function") or {}).get("arguments") or "")
        if tool_schemas:
            total += count_tokens(json.dumps(tool_schemas))
        if ledger is not None and not ledger.is_empty():
            total += count_tokens(ledger.render())
        return total

    def needs_compaction(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]] | None,
        ledger: SessionLedger | None,
    ) -> bool:
        return self.usage(messages, tool_schemas, ledger) > self.budget_tokens * COMPACT_AT

    @staticmethod
    def _last_exchange_start(messages: list[dict[str, Any]]) -> int:
        """Index of the latest assistant message (with its tool results after it)."""
        for index in range(len(messages) - 1, 0, -1):
            if messages[index].get("role") == "assistant":
                return index
        return len(messages)

    def _tail_start(self, messages: list[dict[str, Any]]) -> int:
        """First index of the protected tail, moved back so it never begins with a
        tool result orphaned from the assistant message that requested it."""
        start = max(1, len(messages) - self.keep_recent)
        while start > 1 and messages[start].get("role") == "tool":
            start -= 1
        return start

    async def compact(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]] | None,
        ledger: SessionLedger,
        *,
        focus: str | None = None,
        force: bool = False,
    ) -> tuple[list[dict[str, Any]], CompactionReport]:
        """Return a compacted copy of ``messages`` (``messages[0]`` must be the
        system prompt) and what was done. ``force`` compacts even under budget —
        the ``/compact`` command."""
        target = int(self.budget_tokens * COMPACT_TO)
        before = self.usage(messages, tool_schemas, ledger)
        report = CompactionReport(before=before, after=before)
        out = [dict(m) for m in messages]

        def under_target() -> bool:
            return not force and self.usage(out, tool_schemas, ledger) <= target

        if len(out) <= 2 or (not force and before <= target):
            return out, report

        tail = self._tail_start(out)

        # Steps 1–2 run on the old region first. If the protected tail alone is
        # over budget (one long tool chain inside a single task), they run again
        # on everything except the latest exchange — the call the model is
        # waiting on is never elided.
        for end in (tail, self._last_exchange_start(out)):
            report.elided_tool_results += _elide_tool_results(out, 1, end)
            if under_target():
                report.after = self.usage(out, tool_schemas, ledger)
                return out, report
            report.shrunk_messages += _shrink_assistant_messages(out, 1, end)
            if under_target():
                report.after = self.usage(out, tool_schemas, ledger)
                return out, report

        # Step 3: fold the oldest exchanges into the ledger and drop them. Never
        # split an assistant tool call from its results: the fold boundary moves
        # forward past any tool messages.
        fold_end = max(1, (1 + tail) // 2) if not force else tail
        while fold_end < tail and out[fold_end].get("role") == "tool":
            fold_end += 1
        folded = out[1:fold_end]
        if folded:
            if self.updater is not None:
                update = await self.updater(ledger, _transcript(folded), focus)
                if update:
                    ledger.merge_model_update(update)
            kept_users = [
                {"role": "user", "content": _clip(m.get("content") or "", 1000)}
                for m in folded
                if m.get("role") == "user" and isinstance(m.get("content"), str)
            ]
            # Users' words survive, merged into one message so roles still alternate.
            merged: list[dict[str, Any]] = []
            if kept_users:
                merged.append(
                    {
                        "role": "user",
                        "content": "Earlier requests in this session, already handled — "
                        "do not redo them; work only on the newest request (verbatim):\n"
                        + "\n---\n".join(m["content"] for m in kept_users),
                    }
                )
                merged.append(
                    {
                        "role": "assistant",
                        "content": "Noted. The work on those is recorded in the session ledger.",
                    }
                )
            out = [out[0], *merged, *out[fold_end:]]
            report.folded_messages = len(folded)

        report.after = self.usage(out, tool_schemas, ledger)
        return out, report


def _elide_tool_results(messages: list[dict[str, Any]], start: int, end: int) -> int:
    count = 0
    for message in messages[start:end]:
        content = message.get("content") or ""
        if message.get("role") == "tool" and len(content) > 200 and "elided during" not in content:
            message["content"] = (
                f"[{message.get('name', 'tool')} result elided during compaction "
                f"({len(content):,} chars); call the tool again if you need it]"
            )
            count += 1
    return count


def _shrink_assistant_messages(messages: list[dict[str, Any]], start: int, end: int) -> int:
    count = 0
    for message in messages[start:end]:
        if message.get("role") != "assistant":
            continue
        changed = False
        content = message.get("content")
        if isinstance(content, str) and len(content) > 600:
            message["content"] = _clip(content, 600)
            changed = True
        calls = message.get("tool_calls")
        if calls:
            shrunk = [_shrink_call(call) for call in calls]
            if shrunk != calls:
                message["tool_calls"] = shrunk
                changed = True
        count += int(changed)
    return count


def _shrink_call(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") or {}
    raw = function.get("arguments")
    try:
        arguments = json.loads(raw) if isinstance(raw, str) and raw else (raw or {})
    except json.JSONDecodeError:
        return call
    if not isinstance(arguments, dict):
        return call
    shrunk = {
        key: (
            f"<{len(value):,} chars elided>"
            if key in {"content", "old_string", "new_string"}
            and isinstance(value, str)
            and len(value) > 200
            else value
        )
        for key, value in arguments.items()
    }
    if shrunk == arguments:
        return call
    return {**call, "function": {**function, "arguments": json.dumps(shrunk)}}


def _transcript(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = message.get("role", "?")
        content = message.get("content")
        if isinstance(content, str) and content:
            lines.append(f"{role}: {_clip(content, 1500)}")
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            args = function.get("arguments") or ""
            lines.append(f"{role} called {function.get('name')}({_clip(str(args), 300)})")
    return _clip("\n".join(lines), 12000)


def ledger_message(ledger: SessionLedger) -> dict[str, Any] | None:
    if ledger.is_empty():
        return None
    return {"role": "system", "content": ledger.render()}
