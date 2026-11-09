"""Central configuration, loaded from environment variables / a local .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional at import time
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    try:
        return int(value) if value is not None else default
    except ValueError:
        raise ValueError(f"Environment variable {name} must be an integer, got {value!r}")


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None = None
    claude_model: str = "claude-sonnet-5-5"
    max_tokens: int = 4096
    data_dir: Path = PROJECT_ROOT / "storage"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 5
    max_tool_rounds: int = 4
    max_plan_tasks: int = 5
    tool_timeout_s: int = 90
    max_upload_mb: int = 20
    chroma_host: str | None = None
    chroma_port: int = 8000
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    backend_url: str = "http://127.0.0.1:8000"

    @property
    def has_api_key(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def sqlite_path(self) -> Path:
        return Path(self.data_dir) / "research_agent.db"

    @property
    def chroma_dir(self) -> Path:
        return Path(self.data_dir) / "chroma"

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        return cls(
            anthropic_api_key=_env("ANTHROPIC_API_KEY"),
            claude_model=_env("CLAUDE_MODEL", cls.claude_model),
            max_tokens=_env_int("MAX_TOKENS", cls.max_tokens),
            data_dir=Path(_env("DATA_DIR", str(PROJECT_ROOT / "storage"))).expanduser().resolve(),
            chunk_size=_env_int("CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_env_int("CHUNK_OVERLAP", cls.chunk_overlap),
            top_k=_env_int("TOP_K", cls.top_k),
            max_tool_rounds=_env_int("MAX_TOOL_ROUNDS", cls.max_tool_rounds),
            max_plan_tasks=_env_int("MAX_PLAN_TASKS", cls.max_plan_tasks),
            tool_timeout_s=_env_int("TOOL_TIMEOUT_S", cls.tool_timeout_s),
            max_upload_mb=_env_int("MAX_UPLOAD_MB", cls.max_upload_mb),
            chroma_host=_env("CHROMA_HOST"),
            chroma_port=_env_int("CHROMA_PORT", cls.chroma_port),
            api_host=_env("API_HOST", cls.api_host),
            api_port=_env_int("API_PORT", cls.api_port),
            backend_url=_env("BACKEND_URL", cls.backend_url),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
