"""Tests for tool execution with timeouts and logging."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from localgate.agent.tool_executor import (
    ToolCallTracker,
    execute_tool_call_with_timeout,
    get_tool_timeout,
)
from localgate.config import Settings


@pytest.fixture
def settings() -> Settings:
    """Create a test settings object."""
    return Settings(
        tool_timeout_read=0.1,  # Short timeout for tests
        tool_timeout_write=0.2,
        tool_timeout_search=0.3,
        tool_timeout_delegate=0.5,
        tool_logging_enabled=True,
    )


def test_get_tool_timeout_for_read_tools(settings: Settings) -> None:
    """Read-only tools should use the read timeout."""
    assert get_tool_timeout("read_file", settings) == 0.1
    assert get_tool_timeout("list_directory", settings) == 0.1
    assert get_tool_timeout("search_files", settings) == 0.1
    assert get_tool_timeout("git_status", settings) == 0.1
    assert get_tool_timeout("git_diff", settings) == 0.1


def test_get_tool_timeout_for_write_tools(settings: Settings) -> None:
    """Write tools should use the write timeout."""
    assert get_tool_timeout("write_file", settings) == 0.2


def test_get_tool_timeout_for_special_tools(settings: Settings) -> None:
    """Special tools should use their specific timeouts."""
    assert get_tool_timeout("web_search", settings) == 0.3
    assert get_tool_timeout("delegate_task", settings) == 0.5


def test_get_tool_timeout_for_unknown_tools(settings: Settings) -> None:
    """Unknown tools should default to read timeout."""
    assert get_tool_timeout("mcp__some__tool", settings) == 0.1


@pytest.mark.asyncio
async def test_execute_tool_call_with_timeout_success(tmp_path: Path, settings: Settings) -> None:
    """Successful tool execution should return the result."""
    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello, world!")

    result = await execute_tool_call_with_timeout(
        tmp_path, "call_1", "read_file", {"path": "test.txt"}, settings
    )

    assert not result.is_error
    assert "Hello, world!" in result.content


@pytest.mark.asyncio
async def test_execute_tool_call_with_timeout_timeout(tmp_path: Path, settings: Settings) -> None:
    """Tool execution that times out should return an error result."""

    def slow_tool(*_args, **_kwargs):
        """Simulate a slow tool that takes too long."""
        import time

        time.sleep(10)  # Block for 10 seconds
        return Mock(is_error=False, content="Should not reach here")

    # Mock execute_tool_call to hang
    with patch(
        "localgate.agent.tool_executor.execute_tool_call",
        side_effect=slow_tool,
    ):
        result = await execute_tool_call_with_timeout(
            tmp_path, "call_1", "read_file", {"path": "test.txt"}, settings
        )

        assert result.is_error
        assert "timed out" in result.content.lower()
        assert "0.1s" in result.content  # Should show the timeout value


def test_tool_call_tracker_records_calls() -> None:
    """Tracker should record tool calls correctly."""
    tracker = ToolCallTracker()

    tracker.record("read_file", True)
    tracker.record("write_file", True)
    tracker.record("read_file", False)

    assert tracker.get_recent_failures("read_file", window_seconds=60.0) == 1
    assert tracker.get_recent_failures("write_file", window_seconds=60.0) == 0


def test_tool_call_tracker_get_last_n_calls() -> None:
    """Tracker should return the last N tool names."""
    tracker = ToolCallTracker()

    tracker.record("read_file", True)
    tracker.record("list_directory", True)
    tracker.record("search_files", True)

    assert tracker.get_last_n_calls(2) == ["list_directory", "search_files"]
    assert tracker.get_last_n_calls(5) == ["read_file", "list_directory", "search_files"]


def test_tool_call_tracker_detects_stuck_loop() -> None:
    """Tracker should detect when the same read-only tool is called repeatedly."""
    tracker = ToolCallTracker()

    # Not stuck yet
    tracker.record("read_file", True)
    tracker.record("read_file", True)
    assert not tracker.is_stuck_loop(threshold=3)

    # Now stuck
    tracker.record("read_file", True)
    assert tracker.is_stuck_loop(threshold=3)


def test_tool_call_tracker_not_stuck_with_write_calls() -> None:
    """Tracker should not flag stuck loop if write operations are happening."""
    tracker = ToolCallTracker()

    tracker.record("write_file", True)
    tracker.record("write_file", True)
    tracker.record("write_file", True)

    # Write operations don't count as stuck
    assert not tracker.is_stuck_loop(threshold=3)


def test_tool_call_tracker_not_stuck_with_varied_calls() -> None:
    """Tracker should not flag stuck loop if different tools are being called."""
    tracker = ToolCallTracker()

    tracker.record("read_file", True)
    tracker.record("list_directory", True)
    tracker.record("search_files", True)

    # Different tools, not stuck
    assert not tracker.is_stuck_loop(threshold=3)
