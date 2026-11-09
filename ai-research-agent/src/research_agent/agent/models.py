"""Shared dataclasses and protocols for the agent layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict


@dataclass
class ToolResult:
    text: str
    is_error: bool = False


class ToolProvider(Protocol):
    async def list_tools(self) -> list[ToolSpec]: ...
    async def call(self, name: str, arguments: dict) -> ToolResult: ...


@dataclass
class AgentResult:
    question: str
    style: str
    plan: list[dict]
    steps: list[dict]
    report: dict
    markdown: str
    sources: list[dict]
    evidence: str
    duration_s: float
    extra: dict[str, Any] = field(default_factory=dict)
