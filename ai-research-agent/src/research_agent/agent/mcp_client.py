"""MCP client: spawns the custom MCP server as a stdio subprocess and exposes its tools to the agent."""
from __future__ import annotations

import asyncio
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from research_agent.agent.models import ToolResult, ToolSpec

SRC_DIR = str(Path(__file__).resolve().parents[2])


class MCPToolClient:
    """Async context manager. Enter and exit it from the same task (anyio cancel-scope requirement)."""

    def __init__(self, settings, *, command: str | None = None, args: list[str] | None = None):
        self.settings = settings
        self.command = command or sys.executable
        self.args = args or ["-m", "research_agent.mcp_server.server"]
        self._stack: AsyncExitStack | None = None
        self._session = None

    def _server_env(self) -> dict[str, str]:
        # The MCP SDK only forwards a minimal env by default, so pass what the server needs explicitly.
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in (SRC_DIR, env.get("PYTHONPATH", "")) if p)
        env["DATA_DIR"] = str(self.settings.data_dir)
        env["PYTHONUNBUFFERED"] = "1"
        if self.settings.anthropic_api_key:
            env["ANTHROPIC_API_KEY"] = self.settings.anthropic_api_key
        env["CLAUDE_MODEL"] = self.settings.claude_model
        for key, value in (("CHROMA_HOST", self.settings.chroma_host), ("CHROMA_PORT", self.settings.chroma_port)):
            if value:
                env[key] = str(value)
        return env

    async def __aenter__(self) -> "MCPToolClient":
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._stack = AsyncExitStack()
        try:
            params = StdioServerParameters(command=self.command, args=self.args, env=self._server_env())
            read, write = await self._stack.enter_async_context(stdio_client(params))
            self._session = await self._stack.enter_async_context(ClientSession(read, write))
            await asyncio.wait_for(self._session.initialize(), timeout=60)
        except BaseException:
            await self._stack.aclose()
            raise
        return self

    async def __aexit__(self, *exc) -> None:
        if self._stack:
            await self._stack.aclose()
        self._stack = self._session = None

    async def list_tools(self) -> list[ToolSpec]:
        result = await self._session.list_tools()
        return [ToolSpec(t.name, t.description or "", t.inputSchema) for t in result.tools]

    async def call(self, name: str, arguments: dict) -> ToolResult:
        result = await asyncio.wait_for(self._session.call_tool(name, arguments),
                                        timeout=self.settings.tool_timeout_s)
        text = "\n".join(c.text for c in result.content if getattr(c, "type", None) == "text")
        return ToolResult(text=text, is_error=bool(getattr(result, "isError", False)))
