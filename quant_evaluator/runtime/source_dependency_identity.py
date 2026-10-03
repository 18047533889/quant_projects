"""Strict source-content identities for the source-route code dependency closure.

Only already-loaded local Python package roots are inspected. No optional
runtime is imported, and the public digest contains component names and
relative-content digests, never filesystem roots.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from types import ModuleType

from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.runtime.source_identity import (
    ProcessSourceIdentity, SourceIdentityError, SourceIdentityReceipt,
)

SCHEMA_VERSION = "source_dependency_identity.v1"
COMPONENT_MODULES = (
    "quant_evaluator",
    "data_access",
    "factor_optimizer",
    "factor_preprocess",
)
DEFAULT_MAX_RECORDS = 40_000
DEFAULT_MAX_BYTES = 512 * 1024 * 1024


class SourceDependencyIdentityError(RuntimeError):
    """A required dependency root or bounded content identity is unavailable."""


@dataclass(frozen=True)
class SourceDependencyIdentityReceipt:
    """Aggregated, bounded identities and diagnostics for named local packages."""

    schema: str
    digest: str
    component_digests: tuple[tuple[str, str], ...]
    component_receipts: tuple[tuple[str, SourceIdentityReceipt], ...]
    total_stat_records: int
    total_source_size_bytes: int
    total_files_hashed: int
    total_bytes_hashed: int


def _package_root(module_name: str) -> Path:
    module = sys.modules.get(module_name)
    if not isinstance(module, ModuleType):
        raise SourceDependencyIdentityError(
            f"dependency component {module_name!r} is unavailable or not loaded")
    file_name = getattr(module, "__file__", None)
    package_path = getattr(module, "__path__", None)
    if (type(file_name) is not str or not file_name
            or package_path is None):
        raise SourceDependencyIdentityError(
            f"dependency component {module_name!r} has no concrete package root")
    try:
        paths = tuple(package_path)
        module_file = Path(file_name).resolve(strict=True)
        if (len(paths) != 1 or module_file.name != "__init__.py"
                or not module_file.is_file()):
            raise ValueError("namespace or non-package module layout")
        root = module_file.parent
        declared_path = Path(paths[0]).resolve(strict=True)
        if declared_path != root or not root.is_dir():
            raise ValueError("loaded package path differs from module file")
    except (OSError, TypeError, ValueError) as exc:
        raise SourceDependencyIdentityError(
            f"dependency component {module_name!r} package root is incomplete") from exc

    # A loaded submodule from outside the single package root would otherwise
    # execute code that the root scan does not identify.
    prefix = module_name + "."
    for name, loaded in tuple(sys.modules.items()):
        if name == module_name or not name.startswith(prefix) or loaded is None:
            continue
        loaded_file = getattr(loaded, "__file__", None)
        if type(loaded_file) is not str or not loaded_file:
            raise SourceDependencyIdentityError(
                f"loaded dependency submodule {name!r} has no source file")
        try:
            Path(loaded_file).resolve(strict=True).relative_to(root)
        except (OSError, ValueError) as exc:
            raise SourceDependencyIdentityError(
                f"loaded dependency submodule {name!r} escapes its package root") from exc
    return root


def capture_source_dependency_identity(
    *, max_records: int = DEFAULT_MAX_RECORDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> SourceDependencyIdentityReceipt:
    """Hash required local package sources under aggregate entry/byte limits.

    Budgets cover all components together: each scan receives only the
    remaining global allowance. ProcessSourceIdentity rehashes all Python
    files and rejects source races, symlink escapes, and incomplete trees.
    """
    if type(max_records) is not int or max_records <= 0:
        raise ValueError("max_records must be a positive integer")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")

    receipts: list[tuple[str, SourceIdentityReceipt]] = []
    total_records = total_source_bytes = total_files = total_hashed_bytes = 0
    for component in COMPONENT_MODULES:
        remaining_records = max_records - total_records
        remaining_bytes = max_bytes - total_source_bytes
        if remaining_records <= 0 or remaining_bytes <= 0:
            raise SourceDependencyIdentityError(
                "aggregate dependency identity budget exhausted")
        root = _package_root(component)
        try:
            receipt = ProcessSourceIdentity(
                root, strict_full_content=True,
                max_records=remaining_records, max_bytes=remaining_bytes,
            ).identify(strict_full_content=True)
        except (SourceIdentityError, OSError, ValueError) as exc:
            raise SourceDependencyIdentityError(
                f"dependency component {component!r} source identity scan failed") from exc
        if (receipt.strategy != "strict_full_content"
                or receipt.full_content_checked is not True
                or receipt.drifted is not False
                or receipt.stat_guard_verified is not True):
            raise SourceDependencyIdentityError(
                f"dependency component {component!r} source identity is incomplete")
        receipts.append((component, receipt))
        total_records += receipt.stat_records
        total_source_bytes += receipt.source_size_bytes
        total_files += receipt.files_hashed
        total_hashed_bytes += receipt.bytes_hashed
        if total_records > max_records or total_source_bytes > max_bytes:
            raise SourceDependencyIdentityError(
                "aggregate dependency identity budget exceeded")

    component_digests = tuple((name, receipt.digest) for name, receipt in receipts)
    digest = stable_content_hex(
        tag="SourceDependencyIdentity.v1",
        fields={"schema": SCHEMA_VERSION, "components": component_digests},
    )
    return SourceDependencyIdentityReceipt(
        schema=SCHEMA_VERSION, digest=digest,
        component_digests=component_digests,
        component_receipts=tuple(receipts),
        total_stat_records=total_records,
        total_source_size_bytes=total_source_bytes,
        total_files_hashed=total_files,
        total_bytes_hashed=total_hashed_bytes,
    )


__all__ = (
    "COMPONENT_MODULES", "DEFAULT_MAX_BYTES", "DEFAULT_MAX_RECORDS",
    "SCHEMA_VERSION", "SourceDependencyIdentityError",
    "SourceDependencyIdentityReceipt", "capture_source_dependency_identity",
)
