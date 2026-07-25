"""MCP client support: config load/save, and a real end-to-end connection to a
subprocess MCP server (`tests/fixtures/fake_mcp_server.py`) rather than a
mocked `ClientSession` — this is the actual protocol, not a stand-in for it.
"""

import sys
from pathlib import Path

from localgate.agent.mcp import (
    McpRegistry,
    McpServerConfig,
    McpServerConnection,
    is_mcp_tool_name,
    load_mcp_servers,
    save_mcp_servers,
)

FAKE_SERVER = str(Path(__file__).parent.parent / "fixtures" / "fake_mcp_server.py")


def _config(name: str = "fake") -> McpServerConfig:
    return McpServerConfig(name=name, command=sys.executable, args=[FAKE_SERVER])


# ------------------------------------------------------------------- config file


def test_load_mcp_servers_missing_file_returns_empty(tmp_path):
    assert load_mcp_servers(tmp_path / "missing.json") == []


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "mcp_servers.json"
    servers = [
        McpServerConfig(name="a", command="cmd-a", args=["--x"], env={"K": "V"}),
        McpServerConfig(name="b", command="cmd-b", args=[]),
    ]
    save_mcp_servers(servers, path)
    assert load_mcp_servers(path) == servers


def test_load_mcp_servers_ignores_malformed_entries(tmp_path):
    path = tmp_path / "mcp_servers.json"
    path.write_text(
        '[{"name": "ok", "command": "x"}, {"missing": "fields"}, "not-a-dict"]',
        encoding="utf-8",
    )
    servers = load_mcp_servers(path)
    assert len(servers) == 1
    assert servers[0].name == "ok"


def test_load_mcp_servers_tolerates_invalid_json(tmp_path):
    path = tmp_path / "mcp_servers.json"
    path.write_text("not json", encoding="utf-8")
    assert load_mcp_servers(path) == []


def test_load_mcp_servers_tolerates_a_non_list_top_level(tmp_path):
    path = tmp_path / "mcp_servers.json"
    path.write_text('{"not": "a list"}', encoding="utf-8")
    assert load_mcp_servers(path) == []


# --------------------------------------------------------------- tool name helpers


def test_is_mcp_tool_name():
    assert is_mcp_tool_name("mcp__fake__echo") is True
    assert is_mcp_tool_name("read_file") is False
    assert is_mcp_tool_name("web_search") is False


# ---------------------------------------------------------- real subprocess server


async def test_connect_lists_qualified_tool_schemas():
    connection = McpServerConnection(_config())
    try:
        await connection.connect()
        names = {t["function"]["name"] for t in connection.tools}
        assert names == {"mcp__fake__echo", "mcp__fake__fail"}
        echo_schema = next(
            t for t in connection.tools if t["function"]["name"] == "mcp__fake__echo"
        )
        assert echo_schema["type"] == "function"
        assert "text" in echo_schema["function"]["parameters"]["properties"]
    finally:
        await connection.aclose()


async def test_call_tool_returns_text_result():
    connection = McpServerConnection(_config())
    try:
        await connection.connect()
        result = await connection.call_tool("echo", {"text": "hello"})
        assert result == "echo: hello"
    finally:
        await connection.aclose()


async def test_call_tool_reports_a_server_side_error():
    connection = McpServerConnection(_config())
    try:
        await connection.connect()
        result = await connection.call_tool("fail", {})
        assert "Error from MCP tool" in result
        assert "intentional failure" in result
    finally:
        await connection.aclose()


async def test_call_tool_before_connect_says_so():
    connection = McpServerConnection(_config())
    result = await connection.call_tool("echo", {"text": "x"})
    assert "not connected" in result


# ------------------------------------------------------------------------ registry


async def test_registry_connects_and_aggregates_tools():
    registry = McpRegistry()
    try:
        await registry.connect_all([_config("fake")])
        names = {t["function"]["name"] for t in registry.tool_schemas}
        assert names == {"mcp__fake__echo", "mcp__fake__fail"}
    finally:
        await registry.aclose()


async def test_registry_call_tool_routes_to_the_right_server():
    registry = McpRegistry()
    try:
        await registry.connect_all([_config("fake")])
        result = await registry.call_tool("mcp__fake__echo", {"text": "routed"})
        assert result == "echo: routed"
    finally:
        await registry.aclose()


async def test_registry_call_tool_unknown_server():
    registry = McpRegistry()
    result = await registry.call_tool("mcp__nope__echo", {})
    assert "Unknown MCP server" in result


async def test_registry_call_tool_malformed_name():
    registry = McpRegistry()
    result = await registry.call_tool("not_a_qualified_name", {})
    assert "Unknown tool" in result


async def test_registry_connect_all_is_best_effort_on_a_bad_server():
    registry = McpRegistry()
    errors: list[tuple[str, Exception]] = []
    bad_config = McpServerConfig(name="broken", command="/no/such/binary", args=[])
    try:
        await registry.connect_all(
            [bad_config, _config("fake")], on_error=lambda name, exc: errors.append((name, exc))
        )
        assert [name for name, _ in errors] == ["broken"]
        # the good server still connected despite the bad one failing
        names = {t["function"]["name"] for t in registry.tool_schemas}
        assert "mcp__fake__echo" in names
    finally:
        await registry.aclose()


async def test_registry_servers_lists_names_and_tools():
    registry = McpRegistry()
    try:
        await registry.connect_all([_config("fake")])
        servers = registry.servers()
        assert len(servers) == 1
        name, tool_names = servers[0]
        assert name == "fake"
        assert set(tool_names) == {"mcp__fake__echo", "mcp__fake__fail"}
    finally:
        await registry.aclose()
