"""Deterministic bootstrap for the embedded FactorEngine and DataAccess runtime.

The reconstructed AutoFactorEvaluation directory is a deployable unit.  By default
it must use the copies stored directly below this directory::

    AutoFactorEvaluation-RECONSTRUCT/
      factor_engine/
      data_access/

An external platform root can be selected explicitly for development, but an
accidental parent-directory fallback is never allowed in bundled mode.  This
prevents a deployment from silently running a different FactorEngine version
than the one reviewed with the evaluation system.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_ROOT / "embedded_platform_manifest.json"
_VALID_MODES = {"bundled", "external", "auto"}
_ACTIVE_LAYOUT: "PlatformLayout | None" = None


@dataclass(frozen=True)
class PlatformLayout:
    """Resolved runtime locations and provenance."""

    root: Path
    factor_engine_root: Path
    data_access_root: Path
    mode: str
    source_commit: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "root": str(self.root),
                "factor_engine_root": str(self.factor_engine_root),
                "data_access_root": str(self.data_access_root),
            }
        )
        return payload


def _platform_markers(root: Path) -> tuple[Path, Path]:
    return (
        root / "factor_engine" / "runtime" / "engine.py",
        root / "data_access" / "store.py",
    )


def _is_platform_root(root: Path) -> bool:
    return all(path.is_file() for path in _platform_markers(root))


def _manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        return {}
    try:
        raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def resolve_platform_root(mode: str | None = None) -> PlatformLayout:
    """Resolve the platform root without silently crossing deployment boundaries.

    Modes
    -----
    bundled
        Use only the copies under ``PROJECT_ROOT``. This is the default.
    external
        Require ``AUTOFAC_PLATFORM_ROOT`` or legacy ``QUANT_PROJECTS_ROOT``.
    auto
        Prefer an explicit external root, then bundled copies, then the monorepo
        parent. This mode is intended only for development and synchronization.
    """

    selected_mode = str(mode or os.environ.get("AUTOFAC_PLATFORM_MODE", "bundled")).strip().lower()
    if selected_mode not in _VALID_MODES:
        raise ValueError(
            f"AUTOFAC_PLATFORM_MODE must be one of {sorted(_VALID_MODES)}, got {selected_mode!r}"
        )

    explicit = (
        os.environ.get("AUTOFAC_PLATFORM_ROOT", "").strip()
        or os.environ.get("QUANT_PROJECTS_ROOT", "").strip()
    )
    candidates: list[tuple[str, Path]] = []
    if selected_mode == "bundled":
        candidates.append(("bundled", PROJECT_ROOT))
    elif selected_mode == "external":
        if not explicit:
            raise RuntimeError(
                "external mode requires AUTOFAC_PLATFORM_ROOT or QUANT_PROJECTS_ROOT"
            )
        candidates.append(("external", Path(explicit).expanduser()))
    else:
        if explicit:
            candidates.append(("external", Path(explicit).expanduser()))
        candidates.append(("bundled", PROJECT_ROOT))
        candidates.append(("monorepo", PROJECT_ROOT.parent))

    checked: list[str] = []
    for resolved_mode, candidate in candidates:
        root = candidate.resolve()
        checked.append(str(root))
        if not _is_platform_root(root):
            continue
        manifest = _manifest() if root == PROJECT_ROOT else {}
        return PlatformLayout(
            root=root,
            factor_engine_root=root / "factor_engine",
            data_access_root=root / "data_access",
            mode=resolved_mode,
            source_commit=str(manifest.get("source_commit") or "") or None,
        )

    markers = [str(path.relative_to(PROJECT_ROOT)) for path in _platform_markers(PROJECT_ROOT)]
    raise RuntimeError(
        "FactorEngine/DataAccess platform could not be resolved. "
        f"mode={selected_mode!r}, checked={checked}, required bundled markers={markers}. "
        "Run scripts/sync_embedded_platform.py inside the monorepo before packaging."
    )


def _prepend_sys_path(paths: list[Path]) -> None:
    normalized = [str(path.resolve()) for path in paths]
    for value in normalized:
        while value in sys.path:
            sys.path.remove(value)
    # FactorEngine intentionally exposes flat packages such as ``api`` and
    # ``runtime``. Its root must therefore precede the distribution root.
    sys.path[:0] = normalized


def activate_platform(*, mode: str | None = None, force: bool = False) -> PlatformLayout:
    """Activate the selected embedded platform exactly once per process."""

    global _ACTIVE_LAYOUT
    if _ACTIVE_LAYOUT is not None and not force and mode is None:
        return _ACTIVE_LAYOUT

    layout = resolve_platform_root(mode)
    _prepend_sys_path([layout.factor_engine_root, layout.root])

    # Preserve explicit user configuration. Otherwise pin DataAccess to the
    # registry shipped with this distribution rather than a working-directory
    # dependent file.
    os.environ.setdefault(
        "DATA_ACCESS_CONFIG",
        str(layout.data_access_root / "config" / "datasets.yaml"),
    )
    os.environ["QUANT_PROJECTS_ROOT"] = str(layout.root)
    os.environ["AUTOFAC_ACTIVE_PLATFORM_ROOT"] = str(layout.root)
    _ACTIVE_LAYOUT = layout
    return layout


def platform_diagnostics(*, import_runtime: bool = True) -> dict[str, Any]:
    """Return machine-readable standalone deployment diagnostics."""

    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: Any) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    try:
        layout = activate_platform()
        record("platform.resolve", True, layout.as_dict())
    except Exception as exc:  # pragma: no cover - exercised by deployment failures
        record("platform.resolve", False, f"{type(exc).__name__}: {exc}")
        return {"status": "FAIL", "project_root": str(PROJECT_ROOT), "checks": checks}

    manifest = _manifest()
    record("platform.manifest", bool(manifest), manifest or "missing or invalid manifest")
    for marker in _platform_markers(layout.root):
        record(f"file:{marker.name}", marker.is_file(), str(marker))

    config_path = Path(os.environ["DATA_ACCESS_CONFIG"])
    record("data_access.registry", config_path.is_file(), str(config_path))

    for package in ("pandas", "numpy", "pyarrow", "duckdb", "PyYAML"):
        try:
            version = importlib.metadata.version(package)
            record(f"dependency:{package}", True, version)
        except importlib.metadata.PackageNotFoundError:
            record(f"dependency:{package}", False, "not installed")

    if import_runtime:
        imports = {
            "data_access": "data_access",
            "factor_engine.dsl": "api.dsl_parser",
            "factor_engine.runtime": "runtime.engine",
            "factor_engine.backend": "backend.factory",
        }
        for name, module_name in imports.items():
            try:
                module = importlib.import_module(module_name)
                module_file = Path(getattr(module, "__file__", "")).resolve()
                inside = module_file.is_relative_to(layout.root)
                record(name, inside, str(module_file))
            except Exception as exc:
                record(name, False, f"{type(exc).__name__}: {exc}")

    status = "PASS" if all(item["ok"] for item in checks) else "FAIL"
    return {
        "status": status,
        "project_root": str(PROJECT_ROOT),
        "active_platform": layout.as_dict(),
        "checks": checks,
    }


__all__ = [
    "PlatformLayout",
    "PROJECT_ROOT",
    "activate_platform",
    "platform_diagnostics",
    "resolve_platform_root",
]
