"""Enhanced tool execution with timeouts and logging."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from localgate.agent.tools import ToolCallResult, execute_tool_call
from localgate.config import Settings

READ_ONLY_TOOLS = {"read_file", "list_directory", "search_files", "git_status", "git_diff"}
WRITE_TOOLS = {"write_file"}


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
) -> ToolCallResult:
    """Execute a tool call with timeout and return result.

    Wraps the synchronous execute_tool_call with asyncio timeout.
    If timeout occurs, returns an error ToolCallResult.
    """
    timeout = get_tool_timeout(name, settings)

    try:
        # Run synchronous tool execution in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, execute_tool_call, root, tool_call_id, name, arguments),
            timeout=timeout,
        )
        return result
    except asyncio.TimeoutError:
        error_msg = (
            f"Tool '{name}' timed out after {timeout}s. "
            "Try a different approach or break the task into smaller steps."
        )
        return ToolCallResult(
            tool_call_id=tool_call_id, name=name, content=error_msg, is_error=True
        )


class ToolCallTracker:
    """Tracks tool calls for retry logic and stuck detection."""

    def __init__(self) -> None:
        self.history: list[tuple[str, bool, float]] = []  # (tool_name, success, timestamp)

    def record(self, tool_name: str, success: bool) -> None:
        """Record a tool call result."""
        self.history.append((tool_name, success, time.time()))

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
        """Detect if the same tool is being called repeatedly without progress.

        Returns True if the last 'threshold' calls are all the same read-only tool.
        """
        if len(self.history) < threshold:
            return False

        last_n = self.get_last_n_calls(threshold)
        if not last_n:
            return False

        # Check if all are the same tool and it's read-only
        first_tool = last_n[0]
        return all(tool == first_tool for tool in last_n) and first_tool in READ_ONLY_TOOLS
