"""Thin async wrapper around the Anthropic Messages API (Claude)."""
from __future__ import annotations

from typing import Any


class MissingAPIKeyError(RuntimeError):
    """Raised when ANTHROPIC_API_KEY is not configured."""


class LLMError(RuntimeError):
    pass


def text_of(response: Any) -> str:
    return "".join(b.text for b in response.content if getattr(b, "type", None) == "text").strip()


def tool_uses(response: Any) -> list[Any]:
    return [b for b in response.content if getattr(b, "type", None) == "tool_use"]


class LLMClient:
    def __init__(self, settings, client: Any | None = None):
        self.settings = settings
        self.model = settings.claude_model
        if client is None:
            if not settings.anthropic_api_key:
                raise MissingAPIKeyError(
                    "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic(api_key=settings.anthropic_api_key, max_retries=3, timeout=120.0)
        self._client = client

    async def complete(self, *, system: str, messages: list[dict], tools: list[dict] | None = None,
                       tool_choice: dict | None = None, max_tokens: int | None = None) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model, "max_tokens": max_tokens or self.settings.max_tokens,
            "system": system, "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        return await self._client.messages.create(**kwargs)

    async def text(self, *, system: str, prompt: str, max_tokens: int | None = None) -> str:
        resp = await self.complete(system=system, messages=[{"role": "user", "content": prompt}],
                                   max_tokens=max_tokens)
        return text_of(resp)

    async def structured(self, *, system: str, prompt: str, tool_name: str, description: str,
                         schema: dict, max_tokens: int | None = None) -> dict:
        """Get schema-conforming JSON by forcing a single tool call."""
        resp = await self.complete(
            system=system, messages=[{"role": "user", "content": prompt}],
            tools=[{"name": tool_name, "description": description, "input_schema": schema}],
            tool_choice={"type": "tool", "name": tool_name}, max_tokens=max_tokens,
        )
        for block in tool_uses(resp):
            if block.name == tool_name:
                return dict(block.input)
        raise LLMError("Model did not return structured output")
