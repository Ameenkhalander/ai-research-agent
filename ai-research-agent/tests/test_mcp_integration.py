"""Spawns the real MCP server over stdio and talks to it with the real MCP client (no network/API key needed)."""
import asyncio
import json

import pytest

pytest.importorskip("mcp")

from research_agent.agent.mcp_client import MCPToolClient
from research_agent.config import Settings

EXPECTED = {"web_search", "read_webpage", "search_documents", "summarize_text", "calculator", "extract_information"}


def test_mcp_server_lists_tools_and_runs_calculator(tmp_path):
    async def scenario():
        async with MCPToolClient(Settings(data_dir=tmp_path)) as client:
            specs = await client.list_tools()
            ok = await client.call("calculator", {"expression": "12000*3/1000000 + 2500*15/1000000"})
            bad = await client.call("calculator", {"expression": "__import__('os')"})
            return specs, ok, bad

    specs, ok, bad = asyncio.run(asyncio.wait_for(scenario(), timeout=90))
    assert EXPECTED <= {s.name for s in specs}
    calc = next(s for s in specs if s.name == "calculator")
    assert "expression" in calc.input_schema["properties"]
    assert json.loads(ok.text)["result"] == "0.0735" and not ok.is_error
    assert bad.is_error
