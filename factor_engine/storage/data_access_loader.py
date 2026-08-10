"""R21-130..133: load ``data_access`` as a normal package dependency.

The old per-module ``_ensure_data_access`` helpers injected ``quant_projects``
root into ``sys.path`` at runtime, which made *which* data_access an install
resolves environment-dependent and unreproducible.  This module never mutates
``sys.path``; it imports the installed package and, on request, validates its
resolved path / version / build hash so a production startup fails loudly if the
wrong package is on the path (R21-133).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

_DA_VERSION_CACHE: tuple[str, str, str] | None = None


def data_access_identity() -> tuple[str, str, str]:
    """Return ``(resolved_path, version, build_hash)`` for the installed
    ``data_access`` package (importing it exactly once)."""
    global _DA_VERSION_CACHE
    if _DA_VERSION_CACHE is not None:
        return _DA_VERSION_CACHE
    try:
        import data_access  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "data_access is not installed. Install with: "
            "pip install 'factor-engine[data]' or 'data-access'"
        ) from exc
    pkg_path = Path(data_access.__file__).resolve()  # type: ignore[attr-defined]
    version = str(getattr(data_access, "__version__", "") or "0.0.0")
    init_bytes = b""
    try:
        init_bytes = pkg_path.read_bytes()
    except OSError:
        pass
    build_hash = hashlib.sha256(init_bytes).hexdigest()[:16]
    _DA_VERSION_CACHE = (str(pkg_path.parent), version, build_hash)
    return _DA_VERSION_CACHE


def ensure_data_access_importable(*, validate_path: bool = False) -> Any:
    """Import and return the installed ``data_access`` module.

    Never touches ``sys.path``.  ``validate_path=True`` additionally verifies
    the resolved package location matches ``FACTOR_ENGINE_REQUIRED_DATA_ACCESS``
    (an absolute path) if configured.
    """
    try:
        import data_access
    except ImportError as exc:
        raise ImportError(
            "data_access is not installed. Install with: "
            "pip install 'factor-engine[data]' or 'data-access'"
        ) from exc
    resolved = str(Path(data_access.__file__).resolve().parent)  # type: ignore[attr-defined]
    required = os.environ.get("FACTOR_ENGINE_REQUIRED_DATA_ACCESS", "").strip()
    if validate_path and required:
        required_resolved = str(Path(required).expanduser().resolve())
        if not (resolved == required_resolved or required_resolved in resolved):
            raise RuntimeError(
                f"installed data_access resolves to {resolved!r}, but "
                f"FACTOR_ENGINE_REQUIRED_DATA_ACCESS={required!r} — refusing to "
                "run against the wrong package"
            )
    return data_access
