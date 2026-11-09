"""Registry that assigns stable [S#] labels to everything the agent reads, enabling citations."""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class Source:
    id: int
    kind: str  # "web" | "document"
    title: str
    locator: str  # URL for web sources, "filename#chunk" for documents
    content: str = ""

    @property
    def url(self) -> str | None:
        return self.locator if self.kind == "web" else None

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "title": self.title, "locator": self.locator,
                "url": self.url, "content": self.content}


def _try_json(raw: str):
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


class SourceRegistry:
    MAX_CONTENT = 6000

    def __init__(self) -> None:
        self._sources: list[Source] = []
        self._index: dict[tuple[str, str], int] = {}

    def add(self, kind: str, title: str, locator: str, content: str = "") -> int:
        key = (kind, locator)
        if key in self._index:
            src = self._sources[self._index[key] - 1]
            if content and content not in src.content and len(src.content) < self.MAX_CONTENT:
                src.content = (src.content + "\n" + content)[: self.MAX_CONTENT]
            return src.id
        sid = len(self._sources) + 1
        self._sources.append(Source(sid, kind, title or locator, locator, content[: self.MAX_CONTENT]))
        self._index[key] = sid
        return sid

    def get(self, sid: int) -> Source | None:
        return self._sources[sid - 1] if 1 <= sid <= len(self._sources) else None

    def as_list(self) -> list[dict]:
        return [s.to_dict() for s in self._sources]

    def __len__(self) -> int:
        return len(self._sources)

    def ingest_tool_output(self, tool_name: str, raw: str) -> str:
        """Register sources found in a tool result and return the text shown to the model, with [S#] labels."""
        data = _try_json(raw)
        if not isinstance(data, dict):
            return raw
        if tool_name == "web_search":
            results = data.get("results") or []
            if not results:
                return "No web results found. Try a differently worded query."
            lines = []
            for r in results:
                sid = self.add("web", r.get("title", ""), r.get("url", ""), r.get("snippet", ""))
                lines.append(f"[S{sid}] {r.get('title', '')} ({r.get('url', '')})\n    {r.get('snippet', '')}")
            return "\n".join(lines)
        if tool_name == "read_webpage":
            sid = self.add("web", data.get("title", ""), data.get("url", ""), data.get("text", ""))
            note = " [truncated]" if data.get("truncated") else ""
            return f"[S{sid}] {data.get('title', '')} ({data.get('url', '')}){note}\n{data.get('text', '')}"
        if tool_name == "search_documents":
            results = data.get("results") or []
            if not results:
                return ("No matching passages found in the uploaded documents "
                        "(there may be no documents indexed, or nothing relevant).")
            lines = []
            for r in results:
                page = f", p.{r['page']}" if r.get("page") else ""
                title = f"{r.get('document')}{page}"
                sid = self.add("document", title, f"{r.get('document')}#chunk{r.get('chunk_index')}", r.get("text", ""))
                lines.append(f"[S{sid}] {title} (relevance {r.get('score')})\n    {r.get('text', '')}")
            return "\n".join(lines)
        if tool_name == "calculator" and "result" in data:
            return f"Calculator: {data.get('expression')} = {data['result']}"
        return raw
