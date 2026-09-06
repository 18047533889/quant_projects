"""Configuration for the optional HTTP read service."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean true/false value")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = int(raw.replace("_", ""))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


# No deployable shared secret is embedded in source.  Production startup fails
# closed when DATA_ACCESS_API_KEY is absent and open mode was not explicitly set.
DEFAULT_API_KEY = ""


@dataclass(frozen=True)
class ServiceSettings:
    """Immutable service settings, resolved when the app is created."""

    api_key: str = DEFAULT_API_KEY
    production_mode: bool = False
    allow_open: bool = False
    max_rows: int = 500_000
    max_bytes: int = 512_000_000
    json_preview_max_rows: int = 10_000
    json_preview_max_bytes: int = 8_000_000
    max_concurrency: int = 8
    request_id_header: str = "X-Request-ID"

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        raw_key = os.environ.get("DATA_ACCESS_API_KEY")
        if raw_key is None:
            api_key = DEFAULT_API_KEY
        else:
            api_key = raw_key.strip()
        return cls(
            api_key=api_key,
            production_mode=_env_flag("QUANT_PRODUCTION_MODE"),
            allow_open=_env_flag("DATA_ACCESS_API_ALLOW_OPEN"),
            max_rows=_env_int("DATA_ACCESS_API_MAX_ROWS", 500_000),
            max_bytes=_env_int("DATA_ACCESS_API_MAX_BYTES", 512_000_000),
            json_preview_max_rows=_env_int("DATA_ACCESS_API_JSON_PREVIEW_MAX_ROWS", 10_000),
            json_preview_max_bytes=_env_int("DATA_ACCESS_API_JSON_PREVIEW_MAX_BYTES", 8_000_000),
            max_concurrency=_env_int("DATA_ACCESS_API_MAX_CONCURRENCY", 8),
        )

    @property
    def auth_required(self) -> bool:
        return bool(self.api_key) or (self.production_mode and not self.allow_open)

    def validate_resource_boundary(self) -> None:
        """Fail closed when process-local admission is deployed as multi-worker."""
        workers_raw = os.environ.get("DATA_ACCESS_WORKERS", "1").strip()
        try:
            workers = int(workers_raw)
        except ValueError as exc:
            raise ValueError("DATA_ACCESS_WORKERS must be a positive integer") from exc
        if workers <= 0:
            raise ValueError("DATA_ACCESS_WORKERS must be a positive integer")
        if self.production_mode and workers != 1:
            raise RuntimeError(
                "data_access service admission is process-local; production requires "
                "DATA_ACCESS_WORKERS=1"
            )
