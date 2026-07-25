"""MCP (Model Context Protocol) client support — connect to external MCP
servers over stdio and expose their tools to the coding agent alongside its
own built-in ones.

Deliberately the smallest useful slice, not a general MCP platform:

- **stdio transport only.** Every server config is a local subprocess command.
  HTTP/SSE-hosted servers are out of scope for now — the doc this was built
  from (`NEW_FEATURES.md`) itself calls MCP support the largest, least
  de-risked item on the backlog; stdio-only keeps that risk bounded.
- **Config-driven, off by default.** Servers are listed in
  `~/.config/localgate/mcp_servers.json`; no file (or an empty list) means no
  servers connect and no MCP tools are ever offered — the same
  silence-by-default property `websearch.py` and `delegate_task` follow.
- **Best-effort connection.** A server that fails to start or initialize is
  logged and skipped, not a reason to fail the whole `localgate code`
  invocation — one broken MCP server shouldn't take down the agent.

Tool names are exposed to the model as `mcp__<server>__<tool>`, mirroring the
convention MCP-integrated tools already use elsewhere, so two servers
offering a same-named tool don't collide.
"""

from __future__ import annotations

import json
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from localgate import paths


def mcp_servers_path() -> Path:
    """Where the server list lives — resolved at call time, see ``localgate.paths``."""
    return paths.config_dir() / "mcp_servers.json"


_TOOL_NAME_SEP = "__"
_TOOL_PREFIX = f"mcp{_TOOL_NAME_SEP}"


@dataclass(frozen=True)
class McpServerConfig:
    name: str
    command: str
    args: list[str]
    env: dict[str, str] | None = None


def load_mcp_servers(path: Path | None = None) -> list[McpServerConfig]:
    """Read the server list, tolerating a missing or malformed file — MCP
    support degrades to "no servers" rather than blocking `localgate code`.
    """
    path = path if path is not None else mcp_servers_path()
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []

    servers: list[McpServerConfig] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        command = entry.get("command")
        if not isinstance(name, str) or not isinstance(command, str):
            continue
        args = entry.get("args")
        args = [a for a in args if isinstance(a, str)] if isinstance(args, list) else []
        env = entry.get("env")
        env = env if isinstance(env, dict) else None
        servers.append(McpServerConfig(name=name, command=command, args=args, env=env))
    return servers


def save_mcp_servers(servers: list[McpServerConfig], path: Path | None = None) -> None:
    path = path if path is not None else mcp_servers_path()
    path.parent.mkdir(mode=paths.DIR_MODE, parents=True, exist_ok=True)
    payload = [
        {"name": s.name, "command": s.command, "args": s.args, "env": s.env} for s in servers
    ]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _qualified_name(server: str, tool: str) -> str:
    return f"{_TOOL_PREFIX}{server}{_TOOL_NAME_SEP}{tool}"


def is_mcp_tool_name(name: str) -> bool:
    return name.startswith(_TOOL_PREFIX)


def _split_qualified_name(name: str) -> tuple[str, str] | None:
    if not name.startswith(_TOOL_PREFIX):
        return None
    rest = name[len(_TOOL_PREFIX) :]
    server, sep, tool = rest.partition(_TOOL_NAME_SEP)
    return (server, tool) if sep else None


def _extract_text(result: Any) -> str:
    """Flatten a `CallToolResult`'s content blocks to text — image/audio/resource
    blocks are noted by type rather than silently dropped, so the model at
    least knows something came back that it can't read directly.
    """
    parts: list[str] = []
    for block in result.content or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
        else:
            parts.append(f"[unsupported MCP content block: {block.type}]")
    return "\n".join(parts) if parts else "(no content)"


class McpServerConnection:
    """One live stdio connection to one MCP server."""

    def __init__(self, config: McpServerConfig) -> None:
        self.config = config
        self.tools: list[dict[str, Any]] = []
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None

    async def connect(self) -> None:
        params = StdioServerParameters(
            command=self.config.command, args=self.config.args, env=self.config.env
        )
        read_stream, write_stream = await self._stack.enter_async_context(stdio_client(params))
        session = await self._stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        self._session = session

        listed = await session.list_tools()
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": _qualified_name(self.config.name, tool.name),
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
            for tool in listed.tools
        ]

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        if self._session is None:
            return f"MCP server {self.config.name!r} is not connected."
        result = await self._session.call_tool(tool_name, arguments)
        text = _extract_text(result)
        return f"Error from MCP tool: {text}" if result.isError else text

    async def aclose(self) -> None:
        await self._stack.aclose()


class McpRegistry:
    """Aggregates every connected server's tools into one OpenAI-shaped schema
    list, and routes a qualified tool call back to the right server.
    """

    def __init__(self) -> None:
        self._connections: dict[str, McpServerConnection] = {}

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        return [schema for conn in self._connections.values() for schema in conn.tools]

    async def connect_all(
        self, configs: list[McpServerConfig], *, on_error: Any = None
    ) -> None:
        """Connect every configured server, best-effort — a failure connecting
        to one server is reported via `on_error(name, exc)` (if given) and
        otherwise skipped, not raised.
        """
        for config in configs:
            connection = McpServerConnection(config)
            try:
                await connection.connect()
            except Exception as exc:  # noqa: BLE001 — one bad server shouldn't block startup
                if on_error is not None:
                    on_error(config.name, exc)
                await connection.aclose()
                continue
            self._connections[config.name] = connection

    async def call_tool(self, qualified_name: str, arguments: dict[str, Any]) -> str:
        split = _split_qualified_name(qualified_name)
        if split is None:
            return f"Unknown tool: {qualified_name}"
        server, tool_name = split
        connection = self._connections.get(server)
        if connection is None:
            return f"Unknown MCP server: {server}"
        return await connection.call_tool(tool_name, arguments)

    def servers(self) -> list[tuple[str, list[str]]]:
        """(server name, [tool names]) for every connected server — what `/mcp`
        in the REPL lists.
        """
        return [
            (name, [t["function"]["name"] for t in conn.tools])
            for name, conn in self._connections.items()
        ]

    async def aclose(self) -> None:
        for connection in self._connections.values():
            await connection.aclose()
        self._connections.clear()
