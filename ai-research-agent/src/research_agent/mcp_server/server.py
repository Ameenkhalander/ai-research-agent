"""Custom MCP server exposing the research tools over stdio.

Run standalone:   python -m research_agent.mcp_server.server
Inspect it with:  npx @modelcontextprotocol/inspector python -m research_agent.mcp_server.server

IMPORTANT: with the stdio transport stdout carries the protocol, so never print() here - log to stderr.
"""

import json
import logging
import sys

from mcp.server.fastmcp import FastMCP

from research_agent.config import get_settings
from research_agent.tools.calculator import calculate
from research_agent.tools.web import read_webpage as _read_webpage
from research_agent.tools.web import search_web

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("research_agent.mcp")

mcp = FastMCP("research-agent-tools")

_store = None
_llm = None


def _get_store():
    global _store
    if _store is None:
        from research_agent.documents.vectorstore import build_store
        _store = build_store(get_settings())
    return _store


def _get_llm():
    global _llm
    if _llm is None:
        from research_agent.llm import LLMClient
        _llm = LLMClient(get_settings())
    return _llm


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """Search the public web (DuckDuckGo). Use for current events, public facts and anything not in the
    uploaded documents. Returns JSON: {"query", "results": [{"title", "url", "snippet"}]}."""
    results = search_web(query, max_results=max(1, min(int(max_results), 10)))
    return _json({"query": query, "results": results})


@mcp.tool()
def read_webpage(url: str) -> str:
    """Fetch a public http(s) web page and return its readable text. Use on a promising URL from web_search
    when the snippet is not enough. Returns JSON: {"url", "title", "text", "truncated"}."""
    return _json(_read_webpage(url))


@mcp.tool()
def search_documents(query: str, top_k: int = 5, document: str | None = None) -> str:
    """Semantic search over the user's uploaded PDF/TXT documents. Optionally restrict to one filename.
    Returns JSON: {"query", "results": [{"document", "chunk_index", "page", "score", "text"}]}."""
    results = _get_store().query(query, top_k=max(1, min(int(top_k), 15)), filename=document)
    return _json({"query": query, "results": results})


@mcp.tool()
async def summarize_text(text: str, max_words: int = 150, focus: str | None = None) -> str:
    """Summarise a block of text faithfully in at most max_words words, optionally focused on a topic."""
    from research_agent.tools.llm_tools import summarize
    return await summarize(_get_llm(), text, max_words=max_words, focus=focus)


@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly. Supports + - * / // % ** ^, parentheses, pi, e and the
    functions sqrt, log, log10, log2, exp, sin, cos, tan, abs, round, floor, ceil, min, max, sum, mean, median.
    Convert percentages first (15% of 200 -> 0.15*200). Returns JSON: {"expression", "result"}."""
    return _json(calculate(expression))


@mcp.tool()
async def extract_information(text: str, fields: list[str]) -> str:
    """Extract named fields (e.g. ["author", "publication date", "key figures"]) from text. Fields that are
    not stated in the text come back as null. Returns JSON with "extracted" and supporting "evidence"."""
    from research_agent.tools.llm_tools import extract_information as extract
    return _json(await extract(_get_llm(), text, fields))


def main() -> None:
    log.info("Starting research-agent MCP server (stdio)")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
