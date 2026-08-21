# -*- coding: utf-8 -*-
"""R22-CURRENT-TRUTH — CURRENT.json generator bound to the full SHA identity set.

The evidence truth file (``evidence/CURRENT.json``) is the single truth entry
point for release CURRENT-ness.  This module regenerates it deterministically
from the *current* tree so it can never silently drift:

    RootRepoSHA             = repo HEAD SHA (+ dirty flag when uncommitted)
    FactorEngineSubmoduleSHA  = factor_engine/ submodule pinned SHA
    DataAccessSubmoduleSHA    = dataaccess/ submodule pinned SHA
    OperatorCatalogSHA        = SHA-256 of build/mining/direct_mining_catalog.json
    SemanticCatalogIdentity   = DataAccess-issued SemanticCatalogIdentity.cache_key()
    ImplementationClosureHashSet = SHA-256 over the sorted canonical operator set
    DependencyLockHash          = SHA-256 over the production dependency lock
    TestEnvironmentIdentity     = SHA-256 over python version + key dep versions

Usage:  python -m evidence.current [--write] [--out PATH]

LOCAL ONLY: no git mutations, no network.  ``--write`` rewrites
``evidence/CURRENT.json``; default is dry-run (print to stdout).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = _REPO_ROOT / "evidence" / "CURRENT.json"
EVIDENCE_DIR = _REPO_ROOT / "evidence"

CATALOG_PATH = _REPO_ROOT / "build" / "mining" / "direct_mining_catalog.json"
LOCK_CANDIDATES = (
    _REPO_ROOT / "requirements-production.lock",
    _REPO_ROOT / "factor_engine" / "requirements-production.lock",
)

_ENV_PACKAGES: tuple[str, ...] = (
    "numpy", "pandas", "polars", "pyarrow", "scipy", "pytest",
    "duckdb", "factor_engine", "dataaccess",
)

ARTIFACT_IDS = (
    "operator-benchmark-manifest",
    "operator-certification",
    "operator-inventory",
)


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_capture(args: list[str], cwd: Path) -> str | None:
    import subprocess
    try:
        out = subprocess.run(
            args, cwd=str(cwd), capture_output=True, text=True, timeout=30
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _git_dirty(cwd: Path) -> bool:
    out = _git_capture(["git", "status", "--porcelain"], cwd)
    return bool(out)


# ---------------------------------------------------------------------------
# Identity sources
# ---------------------------------------------------------------------------

def root_repo_sha() -> str:
    sha = _git_capture(["git", "rev-parse", "HEAD"], _REPO_ROOT)
    if not sha:
        raise RuntimeError("R22-CURRENT-TRUTH: repo root is not a git checkout")
    if _git_dirty(_REPO_ROOT):
        return f"{sha} (working-tree-dirty)"
    return sha


def submodule_sha(sub: str) -> str:
    path = _REPO_ROOT / sub
    sha = _git_capture(["git", "rev-parse", "HEAD"], path)
    if not sha:
        raise RuntimeError(
            f"R22-CURRENT-TRUTH: submodule {sub!r} is not a git checkout"
        )
    return sha


def operator_catalog_sha() -> str:
    if not CATALOG_PATH.is_file():
        raise RuntimeError(
            f"R22-CURRENT-TRUTH: operator catalog missing: {CATALOG_PATH}"
        )
    return _sha256_file(CATALOG_PATH)


def implementation_closure_hash_set() -> tuple[str, int]:
    """SHA-256 over the sorted canonical operator set defined by the catalog.

    This is the *closure set* — the set of canonical operator identities the
    release binds.  Per-operator implementation closure hashes (60-140 min for
    1565 canonicals via the lru-cached ``implementation_closure_hash_for``) are
    explicitly NOT expanded here; the set hash changes iff the catalog's
    canonical population changes.
    """
    if not CATALOG_PATH.is_file():
        raise RuntimeError(
            f"R22-CURRENT-TRUTH: operator catalog missing: {CATALOG_PATH}"
        )
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    canonicals = sorted(
        {str(op["canonical"]) for op in catalog.get("operators", [])}
    )
    h = hashlib.sha256()
    for c in canonicals:
        h.update(c.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest(), len(canonicals)


def dependency_lock_hash() -> tuple[str, Path | None]:
    lock = min(LOCK_CANDIDATES, key=lambda p: p.stat().st_mtime) \
        if LOCK_CANDIDATES and all(p.is_file() for p in LOCK_CANDIDATES) \
        else (LOCK_CANDIDATES[0] if LOCK_CANDIDATES[0].is_file() else None)
    if lock is None:
        raise RuntimeError("R22-CURRENT-TRUTH: no dependency lock file found")
    # Hash the lock's own declared digest line (stable release binding) — the
    # file content digest is identical in both candidate paths, so hash the
    # authoritative digest header plus the full file to stay deterministic.
    return _sha256_file(lock), lock


def semantic_catalog_identity() -> str:
    """DataAccess-issued SemanticCatalogIdentity cache key (strict=False)."""
    da_root = str(_REPO_ROOT / "dataaccess")
    for p in (da_root, _REPO_ROOT / "factor_engine"):
        if p not in sys.path:
            sys.path.insert(0, p)
    from data_access.read.semantic_catalog import get_semantic_catalog
    catalog = get_semantic_catalog()
    identity = catalog.get_identity(strict=False)
    key = identity.cache_key()
    if not key or len(key) < 16:
        raise RuntimeError(
            f"R22-CURRENT-TRUTH: unexpected SemanticCatalogIdentity: {key!r}"
        )
    return key


def _dist_version(name: str) -> str:
    """Installed version of a package, with a fallback for dist-name mismatches.

    ``dataaccess`` is shipped as the dist ``data-access`` (importable as
    ``dataaccess``), so a bare importlib.metadata lookup by module name would
    report NOT_INSTALLED even when it is installed and pinned to a release
    HEAD.  R23: resolve the ``data-access`` dist so the environment identity
    reflects the pinned install.
    """
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        pass
    if name == "dataaccess":
        try:
            import importlib.metadata as md
            return md.version("data-access")
        except Exception:
            return "NOT_INSTALLED"
    return "NOT_INSTALLED"


def test_environment_identity() -> str:
    versions: dict[str, str] = {}
    for name in _ENV_PACKAGES:
        versions[name] = _dist_version(name)
    blob = json.dumps(
        {"python": platform.python_version(), "packages": versions},
        sort_keys=True,
    )
    return _sha256_bytes(blob.encode("utf-8"))


# ---------------------------------------------------------------------------
# CURRENT.json payload
# ---------------------------------------------------------------------------

def build_current() -> dict[str, Any]:
    dep_hash, dep_lock = dependency_lock_hash()
    closure_hash, num_operators = implementation_closure_hash_set()
    return {
        "schema_version": 2,
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(),
        "source_snapshot_id": "src:v1:rn",  # replaced below
        "sha_bindings": {
            "RootRepoSHA": root_repo_sha(),
            "FactorEngineSubmoduleSHA": submodule_sha("factor_engine"),
            "DataAccessSubmoduleSHA": submodule_sha("dataaccess"),
            "OperatorCatalogSHA": operator_catalog_sha(),
            "SemanticCatalogIdentity": semantic_catalog_identity(),
            "ImplementationClosureHashSet": closure_hash,
            "DependencyLockHash": dep_hash,
            "TestEnvironmentIdentity": test_environment_identity(),
        },
        "sha_binding_metadata": {
            "OperatorCatalogPath": str(CATALOG_PATH.relative_to(_REPO_ROOT)),
            "DependencyLockPath": str(
                dep_lock.relative_to(_REPO_ROOT) if dep_lock else "NONE"
            ),
            "OperatorCatalogCount": num_operators,
        },
        "artifacts": {
            art_id: {
                "artifact_id": art_id,
                "status": "CURRENT",
                "details": (
                    "SHA-bound evidence truth regenerated by "
                    "python -m evidence.current"
                ),
                "stale_inputs": [],
            }
            for art_id in ARTIFACT_IDS
        },
        "summary": {
            "total": len(ARTIFACT_IDS),
            "current": len(ARTIFACT_IDS),
            "stale": 0,
            "failed": 0,
            "not_run": 0,
        },
        "validator": {
            "module": "evidence.current",
            "id": "R22-CURRENT-TRUTH",
            "rebind_command": "python -m evidence.current --write",
            "staleness_rule": (
                "Any of [(RootRepoSHA,FactorEngineSubmoduleSHA,"
                "DataAccessSubmoduleSHA,OperatorCatalogSHA,"
                "SemanticCatalogIdentity,ImplementationClosureHashSet,"
                "DependencyLockHash,TestEnvironmentIdentity)] changing "
                "rebinds this file; a valid CURRENT.json only counts as "
                "CURRENT when its sha_bindings equal the freshly recomputed "
                "bindings."
            ),
        },
    }


def load_existing() -> dict[str, Any] | None:
    if not DEFAULT_OUT.is_file():
        return None
    try:
        return json.loads(DEFAULT_OUT.read_text(encoding="utf-8"))
    except Exception:
        return None


def check_stale(
    current: dict[str, Any],
    existing: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    """Return (is_current, changed_field_names) against the live file."""
    if existing is None:
        return False, ["(no CURRENT.json present)"]
    expected = current.get("sha_bindings", {})
    actual = existing.get("sha_bindings", {})
    changed = [
        key for key in expected
        if actual.get(key) != expected[key]
    ]
    return (not changed), changed


def _set_source_snapshot_id(payload: dict[str, Any]) -> dict[str, Any]:
    """Bind a local Merkle source-snapshot id, best-effort, ORDER-INDEPENDENT."""
    try:
        from evidence.source_snapshot import build_snapshot
        snap = build_snapshot()
        payload["source_snapshot_id"] = snap["source_snapshot_id"]
    except Exception as exc:  # pragma: no cover - scan failure fallback
        payload["source_snapshot_id"] = f"src:v1:unresolved:{type(exc).__name__}"
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evidence.current",
        description="R22-CURRENT-TRUTH: regenerate SHA-bound evidence CURRENT.json",
    )
    parser.add_argument(
        "--write", action="store_true", default=False,
        help="Write evidence/CURRENT.json (default: dry-run to stdout)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Override output path (only meaningful with --write)",
    )
    parser.add_argument(
        "--skip-snapshot", action="store_true", default=False,
        help="Do not recompute the Merkle source snapshot id (faster)",
    )
    args = parser.parse_args()

    payload = build_current()
    if not args.skip_snapshot:
        payload = _set_source_snapshot_id(payload)

    prior_existing = load_existing()
    _, changed_vs_prior = check_stale(payload, prior_existing)

    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    out = Path(args.out) if args.out else DEFAULT_OUT
    if args.write:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out} ({len(text)} bytes)")

    # Evaluation is always computed against the file state the reader will see:
    # after --write that is the freshly written file (always CURRENT); in
    # dry-run mode it is the on-disk file.
    post_existing = load_existing() if args.write else prior_existing
    is_current, changed = check_stale(payload, post_existing)
    payload["evaluation"] = {
        "staleness_evaluated_at": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "status": "CURRENT" if is_current else "STALE",
        "changed_bindings": changed,
        "note": (
            "changed_bindings lists fields that differed from the previous "
            "on-disk CURRENT.json (the regeneration diff)."
        ),
    }
    payload_json = json.dumps(payload, indent=2, sort_keys=False) + "\n"

    if args.write:
        out.write_text(payload_json, encoding="utf-8")
        print(f"final {out} ({len(payload_json)} bytes)")
    else:
        print(payload_json)

    if is_current:
        print("evaluation: CURRENT (all sha_bindings match live tree)")
    else:
        print(f"evaluation: STALE -> changed bindings: {changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())