"""Helpers for the [S1]-style citation markers used throughout the project."""
from __future__ import annotations

import re

_GROUP = re.compile(r"\[(S\d+(?:\s*[,;]\s*S\d+)*)\]")
_ID = re.compile(r"S(\d+)")


def extract_citation_ids(text: str) -> list[int]:
    """Return every cited source id in order of appearance (duplicates kept)."""
    ids: list[int] = []
    for group in _GROUP.findall(text or ""):
        ids.extend(int(i) for i in _ID.findall(group))
    return ids


def strip_citations(text: str) -> str:
    return _GROUP.sub("", text or "")
