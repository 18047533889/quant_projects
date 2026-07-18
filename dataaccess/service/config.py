"""Configuration for the optional HTTP read service."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = int(raw.replace("_", ""))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True)
class ServiceSettings:
    """Immutable service settings, resolved when the app is created."""

    api_key: str = ""
    production_mode: bool = False
    allow_open: bool = False
    max_rows: int = 500_000
    max_bytes: int = 512_000_000
    max_concurrency: int = 8
    request_id_header: str = "X-Request-ID"

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        return cls(
            api_key=os.environ.get("DATA_ACCESS_API_KEY", "").strip(),
            production_mode=_env_flag("QUANT_PRODUCTION_MODE"),
            allow_open=_env_flag("DATA_ACCESS_API_ALLOW_OPEN"),
            max_rows=_env_int("DATA_ACCESS_API_MAX_ROWS", 500_000),
            max_bytes=_env_int("DATA_ACCESS_API_MAX_BYTES", 512_000_000),
            max_concurrency=_env_int("DATA_ACCESS_API_MAX_CONCURRENCY", 8),
        )

    @property
    def auth_required(self) -> bool:
        return bool(self.api_key) or (self.production_mode and not self.allow_open)
