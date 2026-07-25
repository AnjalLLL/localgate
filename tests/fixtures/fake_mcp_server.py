"""A minimal MCP stdio server, spawned as a real subprocess by
`test_agent_mcp.py` — exercises `agent/mcp.py` against the actual protocol
instead of mocking `ClientSession`.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("fake-server")


@mcp.tool()
def echo(text: str) -> str:
    """Echo the given text back."""
    return f"echo: {text}"


@mcp.tool()
def fail() -> str:
    """Always raises, to exercise the isError path."""
    raise RuntimeError("intentional failure")


if __name__ == "__main__":
    mcp.run(transport="stdio")
