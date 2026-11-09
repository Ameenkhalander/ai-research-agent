"""Test doubles shared by the unit tests (no network, no API key)."""
from __future__ import annotations

import hashlib
import json
import math
import re
from types import SimpleNamespace

from research_agent.agent.models import ToolResult, ToolSpec


def text_response(text: str):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn")


def tool_response(name: str, args: dict, tool_id: str = "tu_1"):
    return SimpleNamespace(content=[SimpleNamespace(type="tool_use", id=tool_id, name=name, input=args)],
                           stop_reason="tool_use")


class FakeLLM:
    """Scripted stand-in for LLMClient. ``structured`` and ``completions`` are consumed in order."""
    model = "fake-model"

    def __init__(self, structured=None, completions=None):
        self._structured = list(structured or [])
        self._completions = list(completions or [])
        self.calls: list[tuple] = []

    async def structured(self, **kw):
        self.calls.append(("structured", kw["tool_name"], kw))
        item = self._structured.pop(0)
        return item(kw) if callable(item) else item

    async def complete(self, **kw):
        self.calls.append(("complete", kw.get("tool_choice"), kw))
        item = self._completions.pop(0)
        return item(kw) if callable(item) else item

    async def text(self, **kw):
        self.calls.append(("text", None, kw))
        return self._completions.pop(0)


class FakeTools:
    def __init__(self, handlers: dict):
        self.handlers = handlers
        self.calls: list[tuple] = []

    async def list_tools(self):
        return [ToolSpec(n, f"{n} tool.", {"type": "object", "properties": {}}) for n in self.handlers]

    async def call(self, name, arguments):
        self.calls.append((name, arguments))
        out = self.handlers[name](arguments)
        return out if isinstance(out, ToolResult) else ToolResult(out)


def hash_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic bag-of-words embedding (64 dims) so vector-store tests need no model download."""
    out = []
    for t in texts:
        v = [0.0] * 64
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / norm for x in v])
    return out


class FakeStore:
    """In-memory replacement for VectorStore used by the API tests."""

    def __init__(self):
        self.rows: list[dict] = []

    def add_chunks(self, doc_id, filename, chunks):
        for c in chunks:
            self.rows.append({"doc_id": doc_id, "document": filename, "chunk_index": c.index,
                              "page": c.page, "text": c.text})
        return len(chunks)

    def delete_document(self, doc_id):
        self.rows = [r for r in self.rows if r["doc_id"] != doc_id]

    def count(self):
        return len(self.rows)

    def query(self, text, top_k=5, filename=None):
        words = set(re.findall(r"[a-z0-9]+", text.lower()))
        scored = [(len(words & set(re.findall(r"[a-z0-9]+", r["text"].lower()))), r) for r in self.rows
                  if not filename or r["document"] == filename]
        scored.sort(key=lambda x: -x[0])
        return [{**{k: v for k, v in r.items() if k != "doc_id"}, "score": float(s)} for s, r in scored[:top_k]]


def web_search_json(*items) -> str:
    return json.dumps({"query": "q", "results": [{"title": t, "url": u, "snippet": s} for t, u, s in items]})
