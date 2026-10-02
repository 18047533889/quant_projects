"""Benchmark-only hash of an explicitly declared source-file scope."""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import sys
from pathlib import Path

MAX_FILE_BYTES = 4 * 1024**2
MAX_TOTAL_BYTES = 64 * 1024**2
MAX_FILE_COUNT = 4096
SOURCE_DIRECTORIES = (
    "quant_evaluator/api", "quant_evaluator/adapters", "quant_evaluator/contracts",
    "quant_evaluator/kernels", "quant_evaluator/metrics", "quant_evaluator/runtime",
)
SOURCE_FILES = (
    "quant_evaluator/scripts/benchmark_real_cos_source_batch.py",
    "quant_evaluator/scripts/benchmark_real_cos_factor_tiles.py",
    "quant_evaluator/scripts/source_width_preparation.py",
    "quant_evaluator/scripts/f48_auto_references.py",
    "quant_evaluator/scripts/f48_benchmark_candidate.py",
    "quant_evaluator/scripts/source_tree_provenance.py",
)
LIMITATIONS = (
    "This receipt hashes only the declared current Python source-file scope.",
    "External DataAccess and Factor Optimizer dependencies are not attested.",
    "Already-imported in-memory code is not attested; this is not a transitive runtime closure.",
)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def capture_source_tree(root: str | Path, *, directories=SOURCE_DIRECTORIES,
                        files=SOURCE_FILES) -> dict:
    """Hash sorted relative paths and bytes without recording source locators."""
    base = Path(root).resolve()
    directories = tuple(directories)
    files = tuple(files)

    def checked_relative(value: str) -> Path:
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("source scope entries must be nonempty relative paths")
        return relative

    def reject_symlink_components(path: Path) -> None:
        current = base
        for part in path.relative_to(base).parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("source scope contains a symlink")

    paths: set[str] = set()
    for relative in directories:
        relative_path = checked_relative(relative)
        directory = base / relative_path
        reject_symlink_components(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"declared source directory is missing: {relative}")
        for current, dirnames, filenames in os.walk(directory, followlinks=False):
            current_path = Path(current)
            reject_symlink_components(current_path)
            for dirname in dirnames:
                if (current_path / dirname).is_symlink():
                    raise ValueError("source scope contains a symlink directory")
            for filename in filenames:
                candidate = current_path / filename
                if candidate.suffix == ".py":
                    reject_symlink_components(candidate)
                    if candidate.is_file():
                        paths.add(candidate.relative_to(base).as_posix())
                        if len(paths) > MAX_FILE_COUNT:
                            raise ValueError("source scope exceeds the file-count limit")
    for relative in files:
        relative_path = checked_relative(relative)
        path = base / relative_path
        reject_symlink_components(path)
        if not path.is_file():
            raise FileNotFoundError(f"declared source file is missing: {relative}")
        paths.add(relative_path.as_posix())
        if len(paths) > MAX_FILE_COUNT:
            raise ValueError("source scope exceeds the file-count limit")
    aggregate = hashlib.sha256()
    total_bytes = 0
    for relative in sorted(paths):
        path = base / relative
        reject_symlink_components(path)
        with path.open("rb") as stream:
            data = stream.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise ValueError("source file exceeds the per-file size limit")
        total_bytes += len(data)
        if total_bytes > MAX_TOTAL_BYTES:
            raise ValueError("source scope exceeds the total size limit")
        encoded_path = relative.encode("utf-8")
        aggregate.update(len(encoded_path).to_bytes(8, "big"))
        aggregate.update(encoded_path)
        aggregate.update(len(data).to_bytes(8, "big"))
        aggregate.update(data)
    return {
        "algorithm": "sha256_path_length_prefixed_v1",
        "aggregate_sha256": aggregate.hexdigest(),
        "file_count": len(paths),
        "scope": {"directories": list(directories), "files": list(files)},
        "versions": {
            "python": platform.python_version(),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "cupy": (getattr(sys.modules.get("cupy"), "__version__", None)
                     if "cupy" in sys.modules else None),
        },
        "limitations": list(LIMITATIONS),
    }


def finalize_source_tree(report: dict, before: dict, root: str | Path, *,
                         directories=SOURCE_DIRECTORIES, files=SOURCE_FILES) -> bool:
    """Attach a post-run receipt and prevent a complete result after source drift."""
    after = capture_source_tree(root, directories=directories, files=files)
    before_versions, after_versions = before["versions"], after["versions"]
    stable_versions = all(before_versions.get(key) == after_versions.get(key)
                          for key in ("python", "numpy", "pandas"))
    stable_preloaded_cupy = (before_versions.get("cupy") is None
                             or before_versions.get("cupy") == after_versions.get("cupy"))
    unchanged = (before["aggregate_sha256"] == after["aggregate_sha256"]
                 and before["scope"] == after["scope"] and stable_versions
                 and stable_preloaded_cupy)
    report["source_provenance"] = before
    report["source_provenance_verification"] = {
        "pass": unchanged,
        "status": "unchanged" if unchanged else "changed",
        "after_aggregate_sha256": after["aggregate_sha256"],
        "after_versions": after_versions,
    }
    if not unchanged and report.get("status") == "complete":
        report["status"] = "source_tree_changed"
    return unchanged
