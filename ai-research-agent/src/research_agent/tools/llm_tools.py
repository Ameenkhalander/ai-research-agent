"""LLM-backed tools: summarisation and information extraction."""
from __future__ import annotations

MAX_INPUT_CHARS = 60_000


async def summarize(llm, text: str, max_words: int = 150, focus: str | None = None) -> str:
    if not text or not text.strip():
        raise ValueError("Text must not be empty")
    max_words = max(20, min(int(max_words), 600))
    focus_line = f" Focus on: {focus}." if focus else ""
    return await llm.text(
        system=("You are a precise summariser. Summarise ONLY what the text says; never add outside facts, "
                "opinions or numbers that are not in the text."),
        prompt=f"Summarise the following text in at most {max_words} words.{focus_line}\n\n"
               f"<text>\n{text[:MAX_INPUT_CHARS]}\n</text>",
        max_tokens=1500,
    )


EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "field": {"type": "string"},
            "value": {"type": ["string", "null"], "description": "Extracted value or null if absent"},
            "evidence": {"type": ["string", "null"], "description": "Short verbatim quote supporting the value"},
        },
        "required": ["field", "value", "evidence"]}}},
    "required": ["items"],
}


async def extract_information(llm, text: str, fields: list[str]) -> dict:
    fields = [f.strip() for f in fields if isinstance(f, str) and f.strip()]
    if not fields:
        raise ValueError("At least one field to extract is required")
    if not text or not text.strip():
        raise ValueError("Text must not be empty")
    data = await llm.structured(
        system=("You extract structured information from text. Return null for any field the text does not "
                "state explicitly. Never guess."),
        prompt=f"Extract these fields: {', '.join(fields)}\n\n<text>\n{text[:MAX_INPUT_CHARS]}\n</text>",
        tool_name="submit_extraction", description="Submit the extracted fields", schema=EXTRACT_SCHEMA,
        max_tokens=1500,
    )
    by_field = {item.get("field"): item for item in data.get("items", [])}
    return {"extracted": {f: (by_field.get(f) or {}).get("value") for f in fields},
            "evidence": {f: (by_field.get(f) or {}).get("evidence") for f in fields}}
