#!/usr/bin/env python3
"""Synchronize reviewed FactorEngine and DataAccess code into this distribution.

Run from the monorepo before publishing only AutoFactorEvaluation-RECONSTRUCT::

    python AutoFactorEvaluation-RECONSTRUCT/scripts/sync_embedded_platform.py

The operation is deterministic, removes stale embedded files, excludes development
artifacts, validates required runtime markers, and writes a provenance manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT.parent
MANIFEST_PATH = PROJECT_ROOT / "embedded_platform_manifest.json"

FACTOR_ENGINE_EXCLUDES = {
    ".pytest_cache",
    "__pycache__",
    "tests",
    "benchmarks",
    "examples",
}
DATA_ACCESS_EXCLUDES = {
    ".pytest_cache",
    "__pycache__",
    "tests",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".ipynb"}
EXCLUDED_NAMES = {".pytest_full.txt", "cleaned_operators.zip"}


def _ignore(excluded_dirs: set[str]):
    def callback(_directory: str, names: list[str]) -> set[str]:
        ignored = set()
        for name in names:
            if name in excluded_dirs or name in EXCLUDED_NAMES:
                ignored.add(name)
                continue
            if Path(name).suffix in EXCLUDED_SUFFIXES:
                ignored.add(name)
        return ignored

    return callback


def _copy_atomic(source: Path, destination: Path, excluded_dirs: set[str]) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"source module missing: {source}")
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-sync-", dir=PROJECT_ROOT) as tmp:
        staged = Path(tmp) / destination.name
        shutil.copytree(source, staged, ignore=_ignore(excluded_dirs))
        backup = destination.with_name(f".{destination.name}.old")
        if backup.exists():
            shutil.rmtree(backup)
        if destination.exists():
            destination.rename(backup)
        staged.rename(destination)
        if backup.exists():
            shutil.rmtree(backup)


def _iter_files(root: Path) -> Iterable[Path]:
    yield from sorted(path for path in root.rglob("*") if path.is_file())


def _tree_digest(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    file_count = 0
    byte_count = 0
    for path in _iter_files(root):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        file_count += 1
        byte_count += len(payload)
    return digest.hexdigest(), file_count, byte_count


def _git_commit(source_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _verify_embedded() -> None:
    required = [
        PROJECT_ROOT / "factor_engine" / "runtime" / "engine.py",
        PROJECT_ROOT / "factor_engine" / "cleaned_operators" / "semantic_hardening.py",
        PROJECT_ROOT / "factor_engine" / "docs" / "dsl_allowlist.json",
        PROJECT_ROOT / "data_access" / "store.py",
        PROJECT_ROOT / "data_access" / "config" / "datasets.yaml",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"embedded platform validation failed; missing={missing}")


def _write_manifest(source_root: Path, source_commit: str | None) -> dict:
    modules = {}
    for name in ("factor_engine", "data_access"):
        digest, files, size_bytes = _tree_digest(PROJECT_ROOT / name)
        modules[name] = {
            "sha256": digest,
            "files": files,
            "size_bytes": size_bytes,
        }
    manifest = {
        "schema_version": 1,
        "distribution": "AutoFactorEvaluation-RECONSTRUCT",
        "source_repository": "18047533889/quant_projects",
        "source_root": str(source_root.resolve()),
        "source_commit": source_commit,
        "synced_at_utc": datetime.now(timezone.utc).isoformat(),
        "modules": modules,
        "exclusions": {
            "factor_engine": sorted(FACTOR_ENGINE_EXCLUDES | EXCLUDED_NAMES),
            "data_access": sorted(DATA_ACCESS_EXCLUDES),
            "suffixes": sorted(EXCLUDED_SUFFIXES),
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_manifest() -> dict:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"manifest missing: {MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    failures = []
    for name in ("factor_engine", "data_access"):
        expected = manifest.get("modules", {}).get(name, {})
        digest, files, size_bytes = _tree_digest(PROJECT_ROOT / name)
        actual = {"sha256": digest, "files": files, "size_bytes": size_bytes}
        if actual != expected:
            failures.append({"module": name, "expected": expected, "actual": actual})
    if failures:
        raise RuntimeError(f"embedded platform manifest mismatch: {failures}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--check", action="store_true", help="verify embedded hashes only")
    args = parser.parse_args()

    if args.check:
        manifest = verify_manifest()
        print(json.dumps({"status": "PASS", "manifest": manifest}, indent=2, ensure_ascii=False))
        return 0

    source_root = args.source_root.expanduser().resolve()
    source_commit = args.source_commit or os.environ.get("GITHUB_SHA") or _git_commit(source_root)
    _copy_atomic(source_root / "factor_engine", PROJECT_ROOT / "factor_engine", FACTOR_ENGINE_EXCLUDES)
    _copy_atomic(source_root / "data_access", PROJECT_ROOT / "data_access", DATA_ACCESS_EXCLUDES)
    _verify_embedded()
    manifest = _write_manifest(source_root, source_commit)
    print(json.dumps({"status": "PASS", "manifest": manifest}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
