# -*- coding: utf-8 -*-
"""Versioned intermediate-factor registry with fail-closed production backfill."""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import yaml

_ACTIVE = threading.local()


def _factor_files(root: Path, factor_id: str) -> list[Path]:
    return sorted((root / "factors" / factor_id).glob("year=*/data.parquet"))


def _registry_path() -> Path | None:
    raw = os.environ.get("FACTOR_ENGINE_INTERMEDIATE_REGISTRY", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    default = Path(__file__).resolve().parent.parent / "config" / "intermediates.yaml"
    return default if default.is_file() else None


def load_intermediate_registry(
    path: str | Path | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve() if path is not None else _registry_path()
    if resolved is None:
        return None, {}
    if not resolved.is_file():
        raise FileNotFoundError(f"intermediate registry not found: {resolved}")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"intermediate registry must be a mapping: {resolved}")
    entries = payload.get("intermediates", payload)
    if not isinstance(entries, dict):
        raise ValueError("intermediate registry 'intermediates' must be a mapping")
    return resolved, entries


def _entry(entries: dict[str, Any], name: str, version: int) -> dict[str, Any] | None:
    raw = entries.get(name)
    if not isinstance(raw, dict):
        return None
    candidate = raw.get(str(version), raw.get(version))
    if candidate is None and raw.get("version") == version:
        candidate = raw
    return dict(candidate) if isinstance(candidate, dict) else None


def _observed_materialized_versions(files: list[Path]) -> set[str]:
    import pandas as pd

    seen: set[str] = set()
    for path in files:
        frame = pd.read_parquet(path, columns=["factor_version"])
        seen.update(str(x) for x in frame["factor_version"].dropna().unique())
        if len(seen) > 1:
            break
    return seen


def _validate_materialized_version(
    files: list[Path],
    expected: str | None,
    *,
    require_version_pin: bool = False,
) -> None:
    if not files:
        return
    if require_version_pin and not expected:
        raise RuntimeError(
            "production intermediate dependency requires expected_factor_version in registry"
        )
    if not expected:
        return
    seen = _observed_materialized_versions(files)
    if seen != {str(expected)}:
        raise RuntimeError(
            f"intermediate factor_version mismatch: expected={expected!r}, observed={sorted(seen)}"
        )


def intermediate_dependency_lineage(
    name: str,
    version: int,
    *,
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return stable dependency metadata for parent lineage/cache keys."""
    registry_file, entries = load_intermediate_registry(registry_path)
    entry = _entry(entries, str(name), int(version))
    if entry is None:
        raise RuntimeError(f"intermediate {name!r}@{version} is not registered")
    expected = entry.get("expected_factor_version")
    if not expected:
        raise RuntimeError(
            f"intermediate {name!r}@{version} lacks expected_factor_version"
        )
    return {
        "name": str(name),
        "version": int(version),
        "factor_id": str(entry.get("factor_id") or name),
        "expected_factor_version": str(expected),
        "registry": str(registry_file) if registry_file else None,
    }


def ensure_intermediate_materialized(
    name: str,
    version: int,
    *,
    lake_root: str | None = None,
    registry_path: str | Path | None = None,
    require_version_pin: bool = False,
) -> list[Path]:
    """Ensure ``intermediate(name, version)`` exists and matches the pinned hash.

    Production callers set ``require_version_pin=True``. In that mode the
    external numeric version is only an API label; the registry must map it to a
    concrete ``expected_factor_version`` content hash, and every parquet
    partition must carry exactly that hash.
    """
    name = str(name).strip()
    version = int(version)
    if not name or version <= 0:
        raise ValueError("intermediate name must be non-empty and version positive")

    from workspace_paths import default_factor_lake_root

    root = Path(lake_root or default_factor_lake_root())
    registry_file, entries = load_intermediate_registry(registry_path)
    entry = _entry(entries, name, version)
    if require_version_pin and entry is None:
        raise RuntimeError(
            f"production intermediate {name!r}@{version} must be registered"
        )
    factor_id = str((entry or {}).get("factor_id") or name)
    expected = (entry or {}).get("expected_factor_version")
    if require_version_pin and not expected:
        raise RuntimeError(
            f"production intermediate {name!r}@{version} requires expected_factor_version"
        )

    files = _factor_files(root, factor_id)
    if files:
        _validate_materialized_version(
            files,
            str(expected) if expected is not None else None,
            require_version_pin=require_version_pin,
        )
        return files

    if entry is None:
        where = str(registry_file) if registry_file else "<not configured>"
        raise RuntimeError(
            f"intermediate {name!r} version={version} is missing and has no backfill definition; "
            f"configure FACTOR_ENGINE_INTERMEDIATE_REGISTRY (current={where})"
        )
    config_value = entry.get("config")
    if not config_value:
        raise ValueError(f"intermediate {name!r} version={version} registry entry lacks config")
    base = registry_file.parent if registry_file is not None else Path.cwd()
    config_path = Path(str(config_value)).expanduser()
    if not config_path.is_absolute():
        config_path = (base / config_path).resolve()

    active: set[tuple[str, int]] = getattr(_ACTIVE, "keys", set())
    key = (name, version)
    if key in active:
        chain = " -> ".join(f"{n}@{v}" for n, v in [*sorted(active), key])
        raise RuntimeError(f"cyclic intermediate dependency detected: {chain}")
    _ACTIVE.keys = set(active) | {key}
    try:
        from factor_engine.runtime.engine import FactorEngine

        engine, factor, config = FactorEngine.from_config(config_path)
        if config.materialization is None:
            raise ValueError(
                f"intermediate backfill config {config_path} must define a materialization section"
            )
        # An intermediate that can be consumed by production must itself be
        # generated under production gates, regardless of a stale YAML run mode.
        engine.run_mode = "production"
        from factor_engine.runtime.config_runtime import resolve_materialize_kwargs

        opts = resolve_materialize_kwargs(config)
        kwargs = opts.to_engine_materialize_kwargs()
        kwargs["factor_id"] = factor_id
        kwargs["lake_root"] = str(root)
        kwargs["input_dq_check"] = True
        kwargs["input_dq_strict"] = True
        kwargs["auto_warmup"] = True
        kwargs["pit_enforce"] = True
        engine.materialize(factor, **kwargs)
    finally:
        _ACTIVE.keys = active

    files = _factor_files(root, factor_id)
    if not files:
        raise RuntimeError(
            f"intermediate backfill completed without materializing {factor_id!r} under {root}"
        )
    _validate_materialized_version(
        files,
        str(expected) if expected is not None else None,
        require_version_pin=require_version_pin,
    )
    return files
