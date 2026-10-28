"""Enhanced tool execution with timeouts and logging."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from localgate.agent.tools import (
    MUTATING_TOOL_NAMES,
    READ_ONLY_TOOL_NAMES,
    ToolCallResult,
    execute_tool_call,
)
from localgate.config import Settings

READ_ONLY_TOOLS = READ_ONLY_TOOL_NAMES
WRITE_TOOLS = MUTATING_TOOL_NAMES


def get_tool_timeout(tool_name: str, settings: Settings) -> float:
    """Get the appropriate timeout for a tool based on its type."""
    if tool_name in READ_ONLY_TOOLS:
        return settings.tool_timeout_read
    elif tool_name in WRITE_TOOLS:
        return settings.tool_timeout_write
    elif tool_name == "web_search":
        return settings.tool_timeout_search
    elif tool_name == "delegate_task":
        return settings.tool_timeout_delegate
    else:
        # Default to read timeout for unknown tools (MCP tools, etc.)
        return settings.tool_timeout_read


async def execute_tool_call_with_timeout(
    root: Path,
    tool_call_id: str,
    name: str,
    arguments: dict[str, Any],
    settings: Settings,
    executor: Callable[[Path, str, str, dict[str, Any]], ToolCallResult] | None = None,
) -> ToolCallResult:
    """Execute a tool call with timeout and return result.

    Wraps the synchronous execute_tool_call with asyncio timeout.
    If timeout occurs, returns an error ToolCallResult.
    """
    timeout = get_tool_timeout(name, settings)
    selected_executor = executor or execute_tool_call

    try:
        # Run synchronous tool execution in a thread pool to avoid blocking
        loop = asyncio.get_running_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, selected_executor, root, tool_call_id, name, arguments),
            timeout=timeout,
        )
        return result
    except asyncio.TimeoutError:
        error_msg = (
            f"Tool '{name}' timed out after {timeout}s. "
            "Try a different approach or break the task into smaller steps."
        )
        return ToolCallResult(
            tool_call_id=tool_call_id, name=name, content=error_msg, is_error=True, timed_out=True
        )


class ToolCallTracker:
    """Tracks tool calls for retry logic and stuck detection."""

    def __init__(self) -> None:
        self.history: list[tuple[str, bool, float]] = []  # (tool_name, success, timestamp)
        #: (tool_name, canonical arguments) per call, parallel to `history`.
        self.signatures: list[tuple[str, str]] = []

    def record(
        self, tool_name: str, success: bool, arguments: dict[str, Any] | None = None
    ) -> None:
        """Record a tool call result."""
        self.history.append((tool_name, success, time.time()))
        self.signatures.append((tool_name, json.dumps(arguments or {}, sort_keys=True)))

    def get_recent_failures(self, tool_name: str, window_seconds: float = 60.0) -> int:
        """Count recent failures for a specific tool within the time window."""
        cutoff = time.time() - window_seconds
        return sum(
            1
            for name, success, ts in self.history
            if name == tool_name and not success and ts >= cutoff
        )

    def get_last_n_calls(self, n: int) -> list[str]:
        """Get the last N tool names called."""
        return [name for name, _, _ in self.history[-n:]]

    def is_stuck_loop(self, threshold: int = 3) -> bool:
        """Whether the model keeps repeating itself without progress: the same call
        (tool *and* arguments) made ``threshold`` times recently, or the last
        ``threshold`` calls all failing. Reading three different files in a row
        is normal exploration, and successful writes are progress, not a loop.
        """
        recent = [sig for sig in self.signatures[-threshold * 2 :] if sig[0] not in WRITE_TOOLS]
        if any(recent.count(sig) >= threshold for sig in set(recent)):
            return True
        last = self.history[-threshold:]
        return len(last) == threshold and not any(success for _, success, _ in last)
