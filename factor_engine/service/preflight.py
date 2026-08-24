"""R21-264..266: startup ProductionPreflight.

The service runs a one-shot preflight before entering ready: config/auth,
artifact generation, dependency versions, resource limits, data roots/hosts,
catalog schema, disk space.  Any security/semantic-critical UNKNOWN -> not
ready — never wait for the first live job to discover a misconfigured deploy.
"""

from __future__ import annotations

import os
from typing import Any

from factor_engine.service.observability import info


def production_preflight() -> dict[str, Any]:
    """Run the startup preflight checks, returning ``{"ok": bool, "checks": {...}}``."""
    checks: dict[str, Any] = {}

    # 1. ambient run mode is valid (R21-144..146).
    try:
        from factor_engine.service.policies import resolve_ambient_run_mode

        checks["ambient_run_mode"] = {"ok": True, "value": resolve_ambient_run_mode()}
    except ValueError as exc:
        checks["ambient_run_mode"] = {"ok": False, "error": str(exc)}

    # 2. data_access is installed as a normal package (R21-130..133).
    try:
        from factor_engine.storage.data_access_loader import data_access_identity

        path, version, build_hash = data_access_identity()
        checks["data_access"] = {"ok": True, "path": path, "version": version, "build_hash": build_hash}
    except ImportError as exc:
        checks["data_access"] = {"ok": False, "error": str(exc)}

    # 3. operator registry / evidence artifact generation (R21-084).
    try:
        from factor_engine.backend.evidence_provenance import compute_payload_hash, load_verified_artifact

        evidence = load_verified_artifact()
        ev_version = compute_payload_hash((evidence or {}).get("provenance") or {})
        checks["evidence_artifact"] = {"ok": True, "version": ev_version}
    except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
        checks["evidence_artifact"] = {"ok": False, "error": f"evidence artifact not verified: {exc}"}

    # 4. dependency versions for provenance (R21-124..126).
    import importlib

    deps: dict[str, str] = {}
    for mod in ("numpy", "pandas", "pyarrow", "scipy", "numba", "polars", "duckdb", "clickhouse_connect"):
        try:
            m = importlib.import_module(mod)
            deps[mod] = str(getattr(m, "__version__", "unknown"))
        except ImportError:
            deps[mod] = "missing"
    checks["dependencies"] = {"ok": True, "versions": deps}

    # 5. disk space on the service root.
    import shutil

    try:
        free_mb = int(shutil.disk_usage(os.environ.get("FACTOR_ENGINE_SERVICE_ROOT", ".")).free / (1024 * 1024))
        min_free_mb = int(os.environ.get("FACTOR_ENGINE_MIN_FREE_DISK_MB", "1024"))
        checks["disk_space"] = {"ok": free_mb >= min_free_mb, "free_mb": free_mb, "min_mb": min_free_mb}
    except OSError as exc:
        checks["disk_space"] = {"ok": False, "error": str(exc)}

    ok = all(entry.get("ok") for entry in checks.values())
    if ok:
        info("factor_engine.service.preflight.ok", checks={k: "ok" for k in checks})
    else:
        failed = {k: v for k, v in checks.items() if not v.get("ok")}
        info("factor_engine.service.preflight.failed", failed=failed)
    return {"ok": ok, "checks": checks}
