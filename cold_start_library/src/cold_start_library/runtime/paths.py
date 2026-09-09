"""Resolve the single installed ``factor_engine`` package identity."""

from __future__ import annotations

import os
from pathlib import Path


def factor_engine_root() -> Path:
    import factor_engine

    canonical = Path(factor_engine.__file__).resolve().parent
    override = os.environ.get("FACTOR_ENGINE_ROOT")
    if override:
        requested = Path(override).expanduser().resolve()
        if requested != canonical:
            raise ValueError(
                "FACTOR_ENGINE_ROOT points to a different tree than the imported "
                "canonical factor_engine package"
            )
    return canonical


def ensure_factor_engine_importable() -> Path:
    try:
        from factor_engine import api  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise ImportError("canonical installed factor_engine.api is unavailable") from exc
    return factor_engine_root()
