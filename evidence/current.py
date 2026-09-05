# -*- coding: utf-8 -*-
"""R22-CURRENT-TRUTH — CURRENT.json generator bound to the full SHA identity set.

The evidence truth file (``evidence/CURRENT.json``) is the single truth entry
point for release CURRENT-ness.  This module regenerates it deterministically
from the *current* tree so it can never silently drift:

    RootRepoSHA             = repo HEAD SHA + DirtyTreeDigest (diff/cached/untracked)
    PlatformSourceTreeIdentity = stable Merkle root over the ENTIRE platform source
                              (all package trees + packaging + dependency lock +
                              config), evidence-free (R46 P0-01/P0-02/P0-C).
    OperatorCatalogSHA        = SHA-256 of build/mining/direct_mining_catalog.json
    SemanticCatalogIdentity   = DataAccess-issued SemanticCatalogIdentity.cache_key() (STRICT)
    ImplementationClosureHashSet = SHA-256 over per-operator implementation closure
                              hashes (name + implementing module source + contract)
    DependencyLockHash          = SHA-256 over the production dependency lock
    TestEnvironmentIdentity     = SHA-256 over python version + key dep versions

R46 P0-C: the old ``FactorEngineSubmoduleSHA`` / ``DataAccessSubmoduleSHA``
payloads are DELETED.  factor_engine / data_access are regular monorepo dirs
(no submodules); the old ``submodule_sha()`` ran ``git rev-parse HEAD`` inside
them and returned the PARENT HEAD — a fabricated "submodule SHA".  Package
identity now comes from :func:`package_tree_identity` (real Merkle tree hash).
:func:`root_repo_sha` returns a single immutable :class:`RepoIdentity` always
(P0-04/P0-D), never the old ``dict | str`` dual type.

Honesty contract (VER-P0-03/04):
  ``build_current()`` NEVER certifies the tree it just wrote.  An artifact is
  marked CURRENT only when BOTH hold:
    (a) the artifact's BOUND input identity (source_snapshot_id +
        ImplementationClosureHashSet + SemanticCatalogIdentity captured when
        the artifact was actually built/verified) equals the LIVE identity set
        recomputed right now, AND
    (b) the artifact was produced by an actual verification run (a real
        ``executed_at`` timestamp is present — not empty, not a placeholder).
  If the source snapshot cannot be resolved, artifacts are marked
  UNRESOLVED/STALE and certification is aborted (fail-closed).

Usage:  python -m evidence.current [--write] [--out PATH]

LOCAL ONLY: no git mutations, no network.  ``--write`` rewrites
``evidence/CURRENT.json``; default is dry-run (print to stdout).
Exit code is 0 only when every artifact is CURRENT; 1 when anything is
STALE/UNRESOLVED so CI can block on it.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from evidence.repo_identity import RepoIdentity, PackageTreeIdentity

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = _REPO_ROOT / "evidence" / "CURRENT.json"
EVIDENCE_DIR = _REPO_ROOT / "evidence"

# P0-4: evidence/output/generated artifacts are NOT production source. They
# are attested separately (EvidenceAttestationIdentity) so committing new
# evidence never mutates the SourceTreeIdentity.
_NON_SOURCE_OUTPUT_DIRS = {
    "evidence",
    "build",
    "factor_engine/docs/reports",
    "factor_engine/docs/generated",
    "archives",
}


def _is_non_source_path(rel: str) -> bool:
    """True if ``rel`` (repo-root-relative) is under an evidence/output dir.

    P0-4: any path equal to or nested under an entry in
    ``_NON_SOURCE_OUTPUT_DIRS`` is excluded from the STABLE source identity.
    """
    for excl in _NON_SOURCE_OUTPUT_DIRS:
        if rel == excl or rel.startswith(excl + "/"):
            return True
    return False

CATALOG_PATH = _REPO_ROOT / "build" / "mining" / "direct_mining_catalog.json"
# R26 P0-LOCK: SINGLE lock authority.  The root ``requirements-production.lock``
# is the only release lock; the factor_engine/ copy was a parallel candidate
# that allowed stale/multiple authority.  DependencyLockPath now always points
# at the root lock.
LOCK_CANDIDATES = (
    _REPO_ROOT / "requirements-production.lock",
)

_ENV_PACKAGES: tuple[str, ...] = (
    "numpy", "pandas", "polars", "pyarrow", "scipy", "pytest",
    "duckdb", "factor_engine", "data_access",
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

def _sha256_byte_stream(data: bytes) -> str:
    """SHA-256 over a canonical byte stream (hash-lengthed, collision-safe)."""
    h = hashlib.sha256()
    h.update(len(data).to_bytes(8, "big"))
    h.update(data)
    return h.hexdigest()


def _tracked_untracked_source_hash() -> str:
    """Digest of untracked, non-ignored files that look like source content.

    Includes the path (length-prefixed) and the file bytes so two different
    untracked modifications can never collide into the same DirtyTreeDigest.
    """
    h = hashlib.sha256()
    try:
        out = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=30,
        )
        if out.returncode != 0:
            return ""
        for line in sorted(out.stdout.splitlines()):
            rel = line.strip()
            if not rel:
                continue
            # P0-4: evidence/output/generated artifacts are attested separately
            # (EvidenceAttestationIdentity); they are NOT production source and
            # must not perturb the SourceTreeIdentity.
            if _is_non_source_path(rel):
                continue
            p = _REPO_ROOT / rel
            if not p.is_file():
                continue
            h.update(rel.encode("utf-8"))
            h.update(b"\x00")
            h.update(_sha256_bytes(p.read_bytes()).encode("utf-8"))
            h.update(b"\x00")
    except Exception:
        return ""
    return h.hexdigest()


# module_ref -> (mtime_ns, size_bytes, sha256): memoized source digests that
# self-invalidate when the implementation file changes on disk (so a source
# mutation is detected even within one process, without re-hashing every module
# on every call).
_module_source_cache: dict[str, tuple[int, int, str]] = {}


def _dirty_tree_digest() -> dict[str, str | bool]:
    """Digest of the complete working-tree delta (VER-P0-03 DirtyTreeDigest).

    Two different uncommitted modifications get different digests: HEAD is
    combined with the hash of `git diff` (unstaged), `git diff --cached`
    (staged) and the untracked-source hash.  The unstaged diff digest is
    defensively cached (see :func:`_root_diff_hash`).
    """
    head = _git_capture(["git", "rev-parse", "HEAD"], _REPO_ROOT) or ""
    diff = _root_diff_hash()
    cached = _git_capture(["git", "diff", "--cached"], _REPO_ROOT) or ""
    untracked = _tracked_untracked_source_hash()
    dirty = bool(diff or cached or untracked)
    return {
        "HEAD": head,
        "diff_hash": diff,
        "cached_diff_hash": _sha256_byte_stream(cached.encode("utf-8")),
        "untracked_source_hash": untracked,
        "dirty": dirty,
    }


def root_repo_sha() -> RepoIdentity:
    """Release identity for the monorepo root (R46 P0-04 / P0-D).

    Returns a single immutable :class:`RepoIdentity` ALWAYS (never ``dict|str``),
    so callers can no longer mis-handle a string-vs-dict split.  ``to_dict()``
    carries the VER-P0-03 DirtyTreeDigest schema (HEAD / dirty / diff_hash /
    cached_diff_hash / untracked_source_hash / identity_hash), and ``str()``
    yields the bare HEAD SHA for back-compat with callers that only need the
    commit.
    """
    digest = _dirty_tree_digest()
    if not digest["HEAD"]:
        raise RuntimeError("R22-CURRENT-TRUTH: repo root is not a git checkout")
    return RepoIdentity.from_parts(
        head_sha=str(digest["HEAD"]),
        dirty=bool(digest["dirty"]),
        working_tree_hash=str(digest["diff_hash"]),
        staged_hash=str(digest["cached_diff_hash"]),
        untracked_source_hash=str(digest["untracked_source_hash"]),
    )


def gitlink_count() -> int:
    """Count real git submodule pins (index entries with mode 160000).

    factor_engine / data_access are regular monorepo directories (not
    submodules), so the correct answer on this repo is 0.  This is the only
    legitimate source of "submodule SHAs"; the old ``submodule_sha()`` that ran
    ``git rev-parse HEAD`` inside a plain directory returned the PARENT HEAD — a
    fabricated submodule SHA — and is REMOVED (R46 P0-C).
    """
    try:
        out = subprocess.run(
            ["git", "ls-files", "--stage"],
            cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=60,
        )
        if out.returncode == 0:
            return sum(
                1 for line in out.stdout.splitlines()
                if line.split() and line.split()[0] == "160000"
            )
    except Exception:
        pass
    return 0


def package_tree_identity(
    package_name: str, source_root: Path | None = None,
) -> PackageTreeIdentity:
    """Merkle-tree identity for a real platform package (R46 P0-C).

    Replaces the fabricated ``FactorEngineSubmoduleSHA`` / ``DataAccessSubmoduleSHA``
    which used ``git rev-parse HEAD`` inside a plain monorepo dir (returning the
    parent HEAD).  ``tree_hash`` binds relative_path || mode || size || content
    so a rename or an edit always changes it.  ``packaging_hash`` binds the
    package's pyproject.toml (if any).
    """
    from evidence.source_snapshot import _package_tree_merkle, _single_file_hash
    src = source_root or (_REPO_ROOT / package_name)
    tree_hash = _package_tree_merkle(src) if src.is_dir() else _merkle_empty()
    packaging_hash = _single_file_hash(f"{package_name}/pyproject.toml", _REPO_ROOT) \
        if src.is_dir() else ""
    version = _dist_version(package_name)
    return PackageTreeIdentity(
        package_name=package_name,
        source_root=str(src.relative_to(_REPO_ROOT)) if src.is_dir() else str(src),
        tree_hash=tree_hash,
        packaging_hash=packaging_hash,
        version=version,
    )


def _merkle_empty() -> str:
    return hashlib.sha256(b"").hexdigest()


def platform_source_tree_identity() -> dict[str, Any]:
    """Platform-wide SourceTreeIdentity (R46 P0-02, P0-C).

    Thin re-export so evidence.current callers get a single import surface.
    """
    from evidence.source_snapshot import platform_source_tree_identity as _inner
    return _inner(_REPO_ROOT)


def _load_catalog() -> dict:
    """Load the direct-mining operator catalog dict (R50).

    Prefers the persisted build artifact ``build/mining/direct_mining_catalog.json``.
    If that generated artifact is MISSING (it is a gitignored build artifact that
    may be absent in a clean checkout), build the catalog in-memory from the LIVE
    OperatorRegistry using the exact same R18 generator pipeline
    (``load_all()`` -> ``direct_use_matrix_rows()`` -> ``retained_direct_rows()``),
    so the evidence refresh NEVER hard-requires a pre-built artifact on the
    developer machine.  Raises RuntimeError only when BOTH the file is missing AND
    the live registry cannot be imported (fail-closed).
    """
    if CATALOG_PATH.is_file():
        with CATALOG_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    # Fallback: assemble the same ``operators`` list the R18 exporter writes,
    # straight from the live OperatorRegistry at refresh time.
    try:
        from factor_engine.mining.direct_use import (
            direct_use_matrix_rows,
            retained_direct_rows,
        )
    except Exception as _reg_exc:  # pragma: no cover - env-dependent
        raise RuntimeError(
            "R22-CURRENT-TRUTH: operator catalog missing at "
            f"{CATALOG_PATH} and live OperatorRegistry unavailable "
            f"({type(_reg_exc).__name__}: {_reg_exc})"
        ) from _reg_exc
    from factor_engine.cleaned_operators import load_all
    load_all()
    rows = direct_use_matrix_rows()
    direct = retained_direct_rows(rows)
    return {"operators": [r.to_dict() for r in direct]}


def operator_catalog_sha() -> str:
    # When the build artifact exists, hash its bytes (stable vs prior releases).
    # When it is missing, build it from the live registry (raises only if the
    # live registry is also unavailable) and hash that in-memory catalog dict.
    if CATALOG_PATH.is_file():
        return _sha256_file(CATALOG_PATH)
    catalog = _load_catalog()
    return _sha256_bytes(
        json.dumps(catalog, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def implementation_closure_hash_set() -> tuple[str, int]:
    """SHA-256 over the per-operator implementation closure (VER-P0-03).

    Each canonical operator contributes a digest of
    ``(canonical name) + (sha256 of its implementing module source file(s),
    resolved through the canonical package path) + (its registered
    contract/metadata identity)``.  The sorted per-operator digests are
    aggregated into a single release-level digest.

    The per-operator digest is computed WITHOUT importing/registering the
    operator (importing the module mutates the OperatorRegistry and is both
    slow and order-dependent).  The implementing module is resolved by walking
    the canonical package path (`cleaned_operators.<module>`) directly on the
    filesystem, and the source digests are memoized across the 200-odd
    distinct modules — the operator record lookup reuses a cached catalog and
    a cached per-canonical module index so the 1565-operator set hashes in
    seconds.  Any change to an operator's implementation source changes its
    digest even when the canonical name is unchanged.

    The expensive imported closure (per-operator helper graph, emitters, AST,
    parameter domains) is available in ``backend.evidence_provenance`` for
    targeted audits; it is deliberately not expanded at release level here.
    """
    catalog = _load_catalog_cached()  # raises only if file AND live registry missing
    canonicals = [
        str(op["canonical"]) for op in catalog.get("operators", [])
    ]
    module_index = _build_module_index(json.dumps(
        catalog, sort_keys=True, separators=(",", ":")
    ).encode("utf-8"))
    per_canonical = {
        c: _closure_digest_from_record(c, module_index[c])
        for c in canonicals
    }
    h = hashlib.sha256()
    h.update(b"closure_version:3\n")
    for c in sorted(per_canonical):
        h.update(c.encode("utf-8"))
        h.update(b"\x00")
        h.update(per_canonical[c].encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest(), len(canonicals)


import functools

@functools.lru_cache(maxsize=1)
def _load_catalog_cached() -> dict:
    return _load_catalog()


@functools.lru_cache(maxsize=1)
def _build_module_index(catalog_json_bytes: bytes) -> dict[str, dict]:
    """canonical -> operator record (memoized; catalog is immutable per run).

    The catalog bytes are passed as the key (dicts are not hashable).
    """
    import json as _json
    catalog = _json.loads(catalog_json_bytes.decode("utf-8"))
    index: dict[str, dict] = {}
    for op in catalog.get("operators", []):
        c = str(op.get("canonical"))
        if c and c not in index:
            index[c] = op
    return index


def _closure_digest_from_record(canonical: str, record: dict | None) -> str:
    """VER-P0-03 compact digest for one canonical from its catalog record."""
    if record is None:
        return _sha256_byte_stream(
            f"canonical:{canonical}\nmodule:UNKNOWN\ncontract:UNKNOWN".encode("utf-8")
        )
    module_ref = str(record.get("module") or "").strip()
    source_digest = _module_source_digest(module_ref)
    canonical_bare = canonical
    if canonical_bare.startswith("cleaned_operators."):
        canonical_bare = canonical_bare.split(".", 1)[1]
    module_bare = module_ref
    if module_bare.startswith("cleaned_operators."):
        module_bare = module_bare.split(".", 1)[1]
    contract_digest = _sha256_bytes(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return _sha256_byte_stream(
        (
            f"canonical:{canonical_bare}\n"
            f"module:{module_bare}\n"
            f"source_sha256:{source_digest}\n"
        ).encode("utf-8") + b"contract_sha256:" + contract_digest.encode("utf-8")
    )


def implementation_closure_hash_for(canonical: str) -> str:
    """VER-P0-03: compact per-operator implementation closure digest.

    = name (bare, package prefix stripped) + sha256(implementing module source)
    + contract/metadata identity

    The implementing module is resolved from the catalog's canonical package
    path without importing it (importing registers the operator and mutates
    global registry state; the module walk here is order-independent and fast).
    The contract identity is the operator's catalog metadata record; changing
    the registered contract (defaults, data inputs, cost, etc.) therefore also
    changes the digest.
    """
    index = _build_module_index(
        json.dumps(
            _load_catalog_cached(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    record = index.get(canonical)
    return _closure_digest_from_record(canonical, record)


def _module_source_digest(module_ref: str) -> str:
    """SHA-256 of the implementing module source file(s) for ``module_ref``.

    The catalog ``module`` field already carries the ``cleaned_operators.``
    package prefix (e.g. ``cleaned_operators.overhaul.base``), so the module
    is resolved as
    ``<repo_root>/factor_engine/cleaned_operators/<rest>.py`` (with a
    ``__init__.py`` package fallback).  The canonical source authority is
    ``_REPO_ROOT / "factor_engine" / "cleaned_operators"`` (R45 namespace
    consolidation: factor_engine/* is the single source authority; the
    historical repo-root ``cleaned_operators/`` copy was deleted).
    Filesystem-only — the operator is never imported
    (importing registers it and mutates the global OperatorRegistry).
    Fail-closed: raises if the canonical package directory is missing rather
    than silently hashing some non-canonical copy.

    Memoized per module with mtime+size self-invalidation: the 1565 catalog
    operators share ~175 distinct modules, any on-disk source change bumps the
    cached digest, so a mutated implementation is always detected even in a
    long-running process.
    """
    if not module_ref or module_ref == "builtins":
        return "builtins:no-module"
    rel_parts = module_ref.split(".")
    # R45: catalog module refs may be written as either
    # ``cleaned_operators.<sub>`` (historical) or
    # ``factor_engine.cleaned_operators.<sub>`` (post-consolidation); both
    # resolve under the single canonical package
    # ``factor_engine/cleaned_operators/``.
    if rel_parts and rel_parts[0] == "factor_engine":
        rel_parts = rel_parts[1:]
    if rel_parts and rel_parts[0] == "cleaned_operators":
        rel_parts = rel_parts[1:]
    if not rel_parts:
        return "MISSING:" + _sha256_bytes(module_ref.encode("utf-8"))
    # SOURCE AUTHORITY (P0-7, updated R45): the ONLY canonical operator runtime is
    # ``<repo_root>/factor_engine/cleaned_operators/`` — the R45 namespace
    # consolidation made factor_engine/* the single source authority and the
    # historical repo-root ``cleaned_operators/`` tree was deleted.  Resolving
    # anywhere else (a shadowed package, an alternate copy) would silently
    # certify the wrong source, so fail CLOSED if the canonical directory is
    # missing or is not a directory.
    pkg_root = _REPO_ROOT / "factor_engine" / "cleaned_operators"
    if not pkg_root.is_dir():
        raise RuntimeError(
            "P0-7 SOURCE AUTHORITY: canonical operator package missing at "
            f"{pkg_root} — refusing to hash operator source from an "
            "ambiguous/non-canonical location"
        )
    rel_file = pkg_root.joinpath(*rel_parts).with_suffix(".py")
    if rel_file.is_file():
        target: Path | None = rel_file
    else:
        rel_pkg = pkg_root.joinpath(*rel_parts, "__init__.py")
        target = rel_pkg if rel_pkg.is_file() else None
    if target is None:
        return "MISSING:" + _sha256_bytes(module_ref.encode("utf-8"))

    try:
        st = target.stat()
        mtime_ns, size = st.st_mtime_ns, st.st_size
        cached = _module_source_cache.get(module_ref)
        if cached is not None and cached[0] == mtime_ns and cached[1] == size:
            return cached[2]
        digest = _sha256_file(target)
        _module_source_cache[module_ref] = (mtime_ns, size, digest)
        return digest
    except OSError:
        return "MISSING:" + _sha256_bytes(module_ref.encode("utf-8"))


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
    """DataAccess-issued SemanticCatalogIdentity cache key (STRICT).

    VER-P0-03: release truth uses ``strict=True`` so any catalog encoding
    failure raises instead of silently returning an unavailable diagnostic
    identity (fail-closed).  The returned key is the canonical 256-bit digest.
    """
    da_root = str(_REPO_ROOT / "data_access")
    for p in (da_root, _REPO_ROOT / "factor_engine"):
        if p not in sys.path:
            sys.path.insert(0, p)
    from data_access.read.semantic_catalog import get_semantic_catalog
    catalog = get_semantic_catalog()
    identity = catalog.get_identity(strict=True)
    if not identity.available:
        raise RuntimeError(
            "R22-CURRENT-TRUTH: SemanticCatalogIdentity unavailable (strict)"
        )
    key = identity.cache_key()
    if not key or len(key) < 16:
        raise RuntimeError(
            f"R22-CURRENT-TRUTH: unexpected SemanticCatalogIdentity: {key!r}"
        )
    return key


def _dist_version(name: str) -> str:
    """Installed version of a package, with a fallback for dist-name mismatches.

    ``data_access`` is shipped as the dist ``data-access`` (importable as
    ``data_access``), so a bare importlib.metadata lookup by module name would
    report NOT_INSTALLED even when it is installed and pinned to a release
    HEAD.  R23: resolve the ``data-access`` dist so the environment identity
    reflects the pinned install.
    """
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        pass
    if name == "data_access":
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


def _root_diff_hash() -> str:
    """SHA-256 of the root working-tree ``git diff`` defensively cached.

    P0-4: evidence/output/generated artifacts (evidence/, build/,
    factor_engine/docs/reports/, factor_engine/docs/generated/, archives/) are attested separately
    (EvidenceAttestationIdentity) and are EXCLUDED from the STABLE source
    identity.  Only the changed paths NOT under an excluded dir contribute to
    the digest, so committing new evidence never perturbs the SourceTreeIdentity.

    ``git diff`` re-hashes every modified blob on each call and can take tens
    of seconds on this repo; the working tree content can only change when a
    write happens *inside this process*, so the digest is memoized and only
    invalidated by an explicit ``_invalidate_root_diff_cache()``.
    """
    global _ROOT_DIFF_HASH_CACHE
    if _ROOT_DIFF_HASH_CACHE is not None:
        return _ROOT_DIFF_HASH_CACHE
    # Collect the changed paths, dropping any under an excluded output dir.
    changed = _git_capture(["git", "diff", "--name-only"], _REPO_ROOT) or ""
    kept = [
        line.strip()
        for line in changed.splitlines()
        if line.strip() and not _is_non_source_path(line.strip())
    ]
    if not kept:
        _ROOT_DIFF_HASH_CACHE = ""
        return _ROOT_DIFF_HASH_CACHE
    diff = _git_capture(["git", "diff", "--"] + kept, _REPO_ROOT) or ""
    _ROOT_DIFF_HASH_CACHE = _sha256_byte_stream(diff.encode("utf-8"))
    return _ROOT_DIFF_HASH_CACHE


_ROOT_DIFF_HASH_CACHE: str | None = None
_ROOT_PREV_DIFF_HASH: str = ""


def _invalidate_root_diff_cache() -> None:
    global _ROOT_DIFF_HASH_CACHE, _ROOT_PREV_DIFF_HASH
    _ROOT_DIFF_HASH_CACHE = None
    _ROOT_PREV_DIFF_HASH = ""


def evidence_attestation_identity() -> str:
    """EvidenceAttestationIdentity: SHA-256 over the evidence output tree.

    P0-4: the evidence/ tree (CURRENT.json, verified_gate_evidence.json,
    manifests, reports, snapshots) is attested SEPARATELY from the STABLE
    SourceTreeIdentity.  Committing new evidence therefore never mutates the
    source identity, but an evidence change is still honestly detected here.

    Mirrors ``_tracked_untracked_source_hash`` canonical length-prefixed
    encoding, restricted to the TRACKED files under ``evidence/`` (git
    ls-files evidence/).  Returns "" when evidence/ has no tracked files.
    """
    h = hashlib.sha256()
    try:
        out = subprocess.run(
            ["git", "ls-files", "evidence/"],
            cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=30,
        )
        if out.returncode != 0:
            return ""
        for line in sorted(out.stdout.splitlines()):
            rel = line.strip()
            if not rel:
                continue
            # Self-exclusion: CURRENT.json is the OUTPUT of this validator.
            # Including it in the attestation it writes makes every --write
            # invalidate the identity it just bound (a self-reference loop no
            # file could ever pass).  Its content is attested by the artifact
            # binds themselves (bound_input_identity + executed_at), so
            # excluding it here loses no verification power.
            if rel == "evidence/CURRENT.json":
                continue
            p = _REPO_ROOT / rel
            if not p.is_file():
                continue
            h.update(rel.encode("utf-8"))
            h.update(b"\x00")
            h.update(_sha256_bytes(p.read_bytes()).encode("utf-8"))
            h.update(b"\x00")
    except Exception:
        return ""
    return h.hexdigest()


# ---------------------------------------------------------------------------
# CURRENT.json payload
# ---------------------------------------------------------------------------

def build_current() -> dict[str, Any]:
    """Compute the fresh (live) truth payload WITHOUT certifying anything.

    VER-P0-03: this only recomputes the LIVE input identity set and carries the
    per-artifact bound identity if a previous run produced one.  Artifact
    ``status`` values are NOT set here; :func:`evaluate` compares the bound
    inputs against the live identities and decides CURRENT / STALE /
    UNRESOLVED.  Never blindly set CURRENT.
    """
    dep_hash, dep_lock = dependency_lock_hash()
    closure_hash, num_operators = implementation_closure_hash_set()
    live_root = root_repo_sha()

    # R46 P0-01 / P0-02 / P0-C: the release payload binds a platform source
    # identity that is stable and evidence-free (no submodule SHAs; the fake
    # FactorEngineSubmoduleSHA / DataAccessSubmoduleSHA are gone).
    platform_identity = platform_source_tree_identity()
    source_tree_identity = platform_identity["root_merkle"]

    # R26 P0-ALPHA: AlphaProbe's generator is part of the platform identity.
    # A change to its model / prompts / source / env lock is detected even when
    # the platform package trees are untouched.
    try:
        from evidence.alpha_identity import alpha_generator_identity
        alpha_identity = alpha_generator_identity(_REPO_ROOT).content_hash
    except Exception as _aexc:  # pragma: no cover - env-dependent
        alpha_identity = "MISSING:" + _sha256_bytes(
            f"alpha_identity:{type(_aexc).__name__}".encode("utf-8")
        )

    live_bindings = {
        "RootRepoSHA": json.dumps(
            live_root.to_dict(), sort_keys=True, separators=(",", ":")
        ),
        "SourceTreeIdentity": json.dumps(
            live_root.to_dict(), sort_keys=True, separators=(",", ":")
        ),
        "PlatformSourceTreeIdentity": source_tree_identity,
        "AlphaGeneratorIdentity": alpha_identity,
        "EvidenceAttestationIdentity": evidence_attestation_identity(),
        "OperatorCatalogSHA": operator_catalog_sha(),
        "SemanticCatalogIdentity": semantic_catalog_identity(),
        "ImplementationClosureHashSet": closure_hash,
        "DependencyLockHash": dep_hash,
        "TestEnvironmentIdentity": test_environment_identity(),
    }

    payload: dict[str, Any] = {
        "schema_version": 3,
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(),
        "source_snapshot_id": "src:v1:rn",  # replaced by _set_source_snapshot_id
        "sha_bindings": live_bindings,
        "source_tree_identity": source_tree_identity,
        "sha_binding_metadata": {
            "OperatorCatalogPath": str(CATALOG_PATH.relative_to(_REPO_ROOT)),
            "DependencyLockPath": str(
                dep_lock.relative_to(_REPO_ROOT) if dep_lock else "NONE"
            ),
            "OperatorCatalogCount": num_operators,
            "gitlink_count": gitlink_count(),
            "platform_packages": sorted(platform_identity.get("packages", {})),
        },
        "artifacts": {
            art_id: {
                "artifact_id": art_id,
                "status": "UNKNOWN",  # decided by evaluate(); never blind CURRENT
                "details": (
                    "bound input identity captured at build time; status is "
                    "decided by evidence.current.evaluate()"
                ),
                "stale_inputs": [],
            }
            for art_id in ARTIFACT_IDS
        },
        "summary": {
            "total": len(ARTIFACT_IDS),
            "current": 0,
            "stale": 0,
            "failed": 0,
            "not_run": 0,
        },
        "validator": {
            "module": "evidence.current",
            "id": "R22-CURRENT-TRUTH",
            "rebind_command": "python -m evidence.current --write",
            "staleness_rule": (
                "An artifact is CURRENT only when its BOUND input identity "
                "(source_snapshot_id + ImplementationClosureHashSet + "
                "SemanticCatalogIdentity at the time it was built) equals the "
                "freshly recomputed live identity AND the artifact was produced "
                "by an actual verification run (executed_at present).  Any "
                "difference -> STALE.  Unresolvable source snapshot -> "
                "UNRESOLVED (fail-closed)."
            ),
        },
    }
    # Fold in the per-artifact bound identities from the previous run (if any).
    # NOTE: the folded-in bound identity is the identity captured when the
    # Fold in the per-artifact bound identities from the previous run (if any).
    # NOTE: the folded-in bound identity is the identity captured when the
    # artifact was LAST actually produced — NOT necessarily this run's live
    # identity.  A stale fold keeps the artifact honestly STALE (evaluate()
    # compares it against the freshly recomputed live set).  Never overwrite a
    # real executed_at / bound identity here: that is what makes drift visible.
    # The RE-BIND decision (placeholder vs real vs live-identical) happens in
    # main() AFTER this fold, so folding preserves the last honest real bind.
    prior = load_existing()
    if prior is not None:
        prior_arts = prior.get("artifacts", {}) or {}
        for art_id in ARTIFACT_IDS:
            prev = prior_arts.get(art_id) or {}
            art = payload["artifacts"][art_id]
            if prev.get("bound_input_identity"):
                art["bound_input_identity"] = prev["bound_input_identity"]
            if prev.get("executed_at"):
                art["executed_at"] = prev["executed_at"]
    return payload


def _source_snapshot_id_from_payload(payload: dict[str, Any]) -> str:
    """The source snapshot bound by this payload (or a live one).

    The payload's own ``source_snapshot_id`` is authoritative when this run
    resolved it; ``src:v1:rn`` is the build-time placeholder meaning "not yet
    resolved".  When the payload carries a placeholder or an unresolved
    marker, fall back to the previous on-disk file's snapshot (the snapshot
    the artifact executors actually ran against).
    """
    sid = payload.get("source_snapshot_id") or ""
    if sid and sid != "src:v1:rn" and not sid.startswith("src:v1:unresolved"):
        return sid
    # placeholder/unresolved -> carry the prior snapshot this payload folded in
    prior = load_existing()
    if prior is not None and (prior.get("source_snapshot_id") or ""):
        return prior["source_snapshot_id"]
    return ""


def evaluate(payload: dict[str, Any]) -> dict[str, Any]:
    """VER-P0-03/04: honest CURRENT / STALE / UNRESOLVED decision.

    An artifact is CURRENT only when:
      (a) its BOUND input identity matches the LIVE identity set, AND
      (b) it was produced by an actual verification run (executed_at present).
    If the source snapshot could not be resolved the artifact is UNRESOLVED
    and certification never continues.
    """
    live = payload.get("sha_bindings", {})
    # ``source_tree_identity`` lives on the payload TOP LEVEL (build_current),
    # NOT inside ``sha_bindings``.  Resolve it there so a bound artifact's
    # identity compares against the real live platform identity instead of an
    # always-None lookup that fabricates "not bound on live" for every artifact.
    if "source_tree_identity" not in live and payload.get("source_tree_identity"):
        live = dict(live)
        live["source_tree_identity"] = payload["source_tree_identity"]
    live_sid = _source_snapshot_id_from_payload(payload)
    snapshot_ok = bool(
        (payload.get("source_snapshot_id") or "").startswith("src:v1:")
        and not (payload.get("source_snapshot_id") or "").startswith("src:v1:unresolved")
    )

    counts = {"total": len(ARTIFACT_IDS), "current": 0, "stale": 0, "failed": 0, "not_run": 0}
    changed_any: list[str] = []

    for art_id in ARTIFACT_IDS:
        art = payload["artifacts"][art_id]
        stale_reasons: list[str] = []

        executed_at = art.get("executed_at") or ""
        if not executed_at:
            stale_reasons.append("no executed_at (artifact was not produced by an actual verification run)")

        bound = art.get("bound_input_identity") or {}
        # Bound inputs missing entirely -> cannot certify.
        if not bound:
            stale_reasons.append("no bound_input_identity recorded")

        if bound:
            bound_sid = bound.get("source_snapshot_id") or ""
            if bound_sid and live_sid and bound_sid != live_sid:
                stale_reasons.append("source_snapshot_id changed")
            if bound_sid and not live_sid:
                stale_reasons.append("source_snapshot_id unresolved")

            for key in (
                "ImplementationClosureHashSet",
                "SemanticCatalogIdentity",
                "EvidenceAttestationIdentity",
                "PlatformSourceTreeIdentity",
                "source_tree_identity",
            ):
                bval = bound.get(key)
                lval = live.get(key)
                if bval is None:
                    continue
                if lval is None:
                    # live platform identity present but the artifact did not bind it
                    stale_reasons.append(f"{key} not bound on live payload")
                elif bval != lval:
                    stale_reasons.append(f"{key} changed")

        # R46 P0-01: the artifact's BOUND source_tree_identity (if recorded) must
        # equal the LIVE platform source identity.  If they differ the evidence is
        # STALE — it can never be silently used as PASS.  Compare against the live
        # platform source identity recomputed this run (stable, evidence-free).
        bound_pl = (bound or {}).get("PlatformSourceTreeIdentity") \
            or (bound or {}).get("source_tree_identity")
        live_pl = live.get("PlatformSourceTreeIdentity") \
            or payload.get("source_tree_identity")
        if bound_pl is not None and bound_pl and live_pl and bound_pl != live_pl:
            stale_reasons.append("source_tree_identity changed (evidence does not match live platform source)")

        snapshot_ok_for_art = snapshot_ok and bool(live_sid)
        if not snapshot_ok_for_art:
            status = "UNRESOLVED"
            stale_reasons.append("source snapshot not resolvable")
        elif stale_reasons:
            status = "STALE"
            counts["stale"] += 1
        else:
            status = "CURRENT"
            counts["current"] += 1

        art["status"] = status
        art["details"] = "BOUND inputs == LIVE inputs and executed_at present" if status == "CURRENT" else "; ".join(stale_reasons)
        art["stale_inputs"] = stale_reasons
        if status == "STALE":
            changed_any.append(art_id)
        elif status == "UNRESOLVED":
            counts["failed"] += 1
            changed_any.append(art_id)

    payload["summary"] = counts
    # R26 P0-MULTI: multi-dimension CURRENT-ness.  BuildHealth / RuntimeHealth /
    # ResearchValidity / ReleaseValidity / ProductionReadiness are evaluated
    # independently so a healthy build is never conflated with a stale release
    # certificate.  The overall evaluation status is still CURRENT only when
    # every artifact is CURRENT.
    dims = _evaluate_dimensions(payload)
    payload["dimensions"] = dims
    payload["evaluation"] = {
        "staleness_evaluated_at": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "status": "CURRENT" if counts["current"] == counts["total"] else (
            "UNRESOLVED" if counts["failed"] else "STALE"
        ),
        "changed_artifacts": changed_any,
        "note": (
            "status is CURRENT only when every artifact's BOUND input identity "
            "matches the LIVE identity set AND the artifact has an executed_at "
            "from a real verification run."
        ),
        "dimensions": dims,
    }
    return payload


def _evaluate_dimensions(payload: dict[str, Any]) -> dict[str, str]:
    """R26 P26-MULTI: five independent CURRENT/STALE release dimensions.

    Each dimension is decided from the LIVE identity set WITHOUT certifying the
    tree the caller just wrote.  STALE here means "the live tree does not match
    a bound/certified state", so a healthy build can be CURRENT while the
    release validity is still STALE (operator cert / source snapshot drift).

    R50 guard: a dimension is CURRENT only when its required evidence artifacts
    are CURRENT+PASS.  If any release artifact is STALE/FAILED, every dimension
    that depends on artifacts must be STALE — never silently CURRENT.
    """
    live = payload.get("sha_bindings", {})
    prior = load_existing()
    live_pl = live.get("PlatformSourceTreeIdentity") or ""
    live_cl = live.get("ImplementationClosureHashSet") or ""
    live_sem = live.get("SemanticCatalogIdentity") or ""
    live_alpha = live.get("AlphaGeneratorIdentity") or ""

    # Bound identities from the last certified state (fail-closed: no prior
    # evidence -> STALE).
    prior_pl = ""
    prior_cl = ""
    prior_sem = ""
    prior_alpha = ""
    if prior is not None:
        prior_pl = prior.get("source_tree_identity") or ""
        prior_pl = prior_pl or (prior.get("sha_bindings") or {}).get("PlatformSourceTreeIdentity") or ""
        prior_cl = (prior.get("sha_bindings") or {}).get("ImplementationClosureHashSet") or ""
        prior_sem = (prior.get("sha_bindings") or {}).get("SemanticCatalogIdentity") or ""
        prior_alpha = (prior.get("sha_bindings") or {}).get("AlphaGeneratorIdentity") or ""

    def _ok(bound: str, live: str, label: str) -> str:
        if not bound:
            return "STALE"  # nothing certified yet -> not CURRENT
        if bound == live:
            return "CURRENT"
        return "STALE"

    # R50: a dimension is CURRENT only when its required evidence artifacts are
    # CURRENT+PASS.  Any artifact STALE/FAILED/UNRESOLVED forces the dimensions
    # that depend on artifacts to STALE (never silently CURRENT).
    def _artifact_ok(*art_ids: str) -> bool:
        for aid in art_ids:
            st = (payload.get("artifacts") or {}).get(aid, {}).get("status")
            if st in ("STALE", "FAILED", "UNRESOLVED"):
                return False
        return True

    # build_health: the wheel/install env is reproducible and matches the lock.
    # (DependencyLockHash + TestEnvironmentIdentity bound by the prior run.)
    # When the PRIOR payload itself is the last verification run (this process
    # just bound it), compare against the CURRENT payload's own live bindings —
    # otherwise the dimension is only ever CURRENT on the run AFTER a clean
    # bind, which is not the honesty intent.
    prior_dep = (prior.get("sha_bindings") or {}).get("DependencyLockHash") or "" \
        if prior is not None else ""
    prior_env = (prior.get("sha_bindings") or {}).get("TestEnvironmentIdentity") or "" \
        if prior is not None else ""
    _live_dep = live.get("DependencyLockHash") or ""
    _live_env = live.get("TestEnvironmentIdentity") or ""
    # A prior payload that is itself the just-bound live state (same tree id)
    # certifies build_health/runtime_health against its own live bindings.
    _prior_is_live = bool(
        prior is not None
        and (prior.get("source_tree_identity") or "")
        and prior.get("source_tree_identity") == payload.get("source_tree_identity")
    )
    if _prior_is_live:
        prior_dep = prior_dep or _live_dep
        prior_env = prior_env or _live_env
    build_health = "CURRENT" if (
        prior_dep and prior_dep == _live_dep
        and prior_env and prior_env == _live_env
    ) else "STALE"

    # runtime_health: the semantic catalog identity is live (data_access importable).
    if _prior_is_live:
        prior_sem = prior_sem or live_sem
    runtime_health = "CURRENT" if prior_sem and prior_sem == live_sem else "STALE"

    # research_validity: the research artifacts' operator closure matches.  The
    # operator-certification artifact is the evidence for this dimension; if it
    # is STALE/FAILED the dimension must be STALE.  When the prior payload is
    # the just-bound live state, its closure hash is the certified one.
    if _prior_is_live:
        prior_cl = prior_cl or live_cl
    research_validity = "STALE" if not _artifact_ok("operator-certification") else \
        _ok(prior_cl, live_cl, "ImplementationClosureHashSet")

    # release_validity: the release source identity matches the bound tree.  The
    # operator-benchmark-manifest / operator-inventory artifacts (source-bound)
    # are evidence; STALE/FAILED -> STALE.  A live-bound prior payload certifies
    # release validity against its own platform source identity.
    if _prior_is_live:
        prior_pl = prior_pl or live_pl
    release_validity = "STALE" if not _artifact_ok(
        "operator-benchmark-manifest", "operator-inventory"
    ) else _ok(prior_pl, live_pl, "PlatformSourceTreeIdentity")

    # production_readiness: the Alpha generator identity matches the bound state.
    # operator-certification is the evidence; STALE/FAILED -> STALE.
    if _prior_is_live:
        prior_alpha = prior_alpha or live_alpha
    production_readiness = "STALE" if not _artifact_ok("operator-certification") else \
        _ok(prior_alpha, live_alpha, "AlphaGeneratorIdentity")

    return {
        "build_health": build_health,
        "runtime_health": runtime_health,
        "research_validity": research_validity,
        "release_validity": release_validity,
        "production_readiness": production_readiness,
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
    """Return (is_current, changed_field_names) against the live file.

    ``current`` carries the freshly recomputed sha_bindings; ``existing`` is the
    on-disk CURRENT.json (or a tampered variant).  An artifact counts as
    CURRENT only when it has an executed_at AND its bound input identity
    matches the live identity set (source snapshot + closure hash + semantic
    catalog identity).
    """
    if existing is None:
        return False, ["(no CURRENT.json present)"]
    live = current.get("sha_bindings", {})
    stored = existing.get("sha_bindings", {})
    changed = [key for key in live if stored.get(key) != live[key]]

    existing_sid = existing.get("source_snapshot_id") or ""
    current_sid = current.get("source_snapshot_id") or ""
    if (
        existing_sid
        and current_sid
        and not existing_sid.startswith("src:v1:unresolved")
        and existing_sid != current_sid
    ):
        changed.append("source_snapshot_id")

    # Honesty: artifacts must carry an executed_at AND a bound identity that
    # matches the live bindings.
    artifacts = existing.get("artifacts", {}) or {}
    for art_id in ARTIFACT_IDS:
        art = artifacts.get(art_id) or {}
        executed_at = art.get("executed_at") or ""
        if not executed_at:
            changed.append(f"artifact:{art_id}:no_executed_at")
            continue
        bound = art.get("bound_input_identity") or {}
        if not bound:
            changed.append(f"artifact:{art_id}:no_bound_input_identity")
            continue
        if bound.get("source_snapshot_id") and existing_sid \
                and bound["source_snapshot_id"] != existing_sid:
            changed.append(f"artifact:{art_id}:bound_snapshot_mismatch")
        for key in (
            "ImplementationClosureHashSet",
            "SemanticCatalogIdentity",
            "EvidenceAttestationIdentity",
            "PlatformSourceTreeIdentity",
        ):
            bval = bound.get(key)
            lval = live.get(key)
            if bval is not None and bval != lval:
                changed.append(f"artifact:{art_id}:{key}")

    # R46 P0-01: bind evidence to the LIVE platform source identity.  If the
    # stored payload's source_tree_identity differs from the live one, mark
    # STALE (never silently PASS).
    stored_pl = existing.get("source_tree_identity") or ""
    live_pl = current.get("source_tree_identity") or live.get("PlatformSourceTreeIdentity") or ""
    if stored_pl and live_pl and stored_pl != live_pl:
        changed.append("source_tree_identity")

    return (not changed), sorted(set(changed))


def _set_source_snapshot_id(payload: dict[str, Any]) -> dict[str, Any]:
    """Bind a local Merkle source-snapshot id.

    VER-P0-04: if the source snapshot cannot be resolved the payload carries a
    deterministic ``src:v1:unresolved:...`` marker AND the artifacts are marked
    UNRESOLVED (fail-closed) — certification never continues on an unresolved
    snapshot.  A snapshot failure is also surfaced on stderr.
    """
    try:
        from evidence.source_snapshot import build_snapshot
        snap = build_snapshot()
        payload["source_snapshot_id"] = snap["source_snapshot_id"]
    except Exception as exc:  # pragma: no cover - scan failure fallback
        payload["source_snapshot_id"] = f"src:v1:unresolved:{type(exc).__name__}"
        for art in payload.get("artifacts", {}).values():
            art["status"] = "UNRESOLVED"
            art["stale_inputs"] = [
                f"source_snapshot_id unresolved: {type(exc).__name__}: {exc}"
            ]
        print(
            f"error: source snapshot could not be resolved: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
    return payload


def _registry_artifact(art_id: str):
    """The registered EvidenceArtifact (for reading real output mtimes)."""
    try:
        from evidence.registry import get_artifact

        return get_artifact(art_id)
    except Exception:
        return None


def _append_record_execution(
    sink: dict, art_id: str, raw: str, artifact_ids: set | list
) -> None:
    """Resolve one ``ARTIFACT=MTIME|file`` producer spec into an executed_at.

    ``file`` reads the real output-file mtime via the registry (the honest
    producer path); a numeric value is an explicit unix-seconds timestamp.
    Unknown artifact ids and non-numeric values fail closed.
    """
    if art_id not in artifact_ids:
        raise SystemExit(f"unknown artifact id {art_id!r}")
    if raw == "file":
        reg_art = _registry_artifact(art_id)
        if reg_art is not None:
            outs = reg_art.resolve_outputs()
            if not outs:
                raise SystemExit(f"artifact {art_id} has no output files to read mtime from")
            latest = max(p.stat().st_mtime for p in outs if p.is_file())
            sink[art_id] = datetime.datetime.fromtimestamp(
                latest, datetime.timezone.utc
            ).isoformat()
        else:
            sink[art_id] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return
    try:
        epoch = float(raw)
    except ValueError:
        raise SystemExit(
            f"--record-execution expects ARTIFACT=MTIME_EPOCH|file, got {art_id}={raw!r}"
        )
    sink[art_id] = datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc).isoformat()


def main(argv: list[str] | None = None) -> int:
    import json as _json

    parser = argparse.ArgumentParser(
        prog="python -m evidence.current",
        description=(
            "R22-CURRENT-TRUTH: regenerate SHA-bound evidence CURRENT.json. "
            "Exit code is 0 only when every artifact is CURRENT."
        ),
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
    parser.add_argument(
        "--check", action="store_true", default=False,
        help=(
            "VER-P0-05: dry-run evaluate and print a single machine line "
            "'EVIDENCE_CURRENT: <PASS|FAIL|STALE|UNRESOLVED> reason=...'.  "
            "Never writes.  Returns non-zero unless every artifact is CURRENT "
            "and no failures (data_access/import errors are honest UNRESOLVED, "
            "fail-closed)."
        ),
    )
    parser.add_argument(
        "--record-execution", action="append", default=None, metavar="ARTIFACT=MTIME_EPOCH[:...]",
        help=(
            "Producer-side executed_at binding: record that ARTIFACT's output "
            "file(s) were actually (re)generated at MTIME_EPOCH (unix seconds, "
            "read from the output file itself when the value is 'file').  "
            "Repeat the flag once per artifact.  Honesty contract: only pass "
            "this immediately after a REAL generator run — a --record-execution "
            "without a fresh generator output still re-binds to the LIVE "
            "identity, so a fabricated timestamp cannot make a stale artifact "
            "CURRENT (the identity comparison fails first)."
        ),
    )
    args = parser.parse_args(argv)

    # VER-P0-05: data_access (and other optional providers) must not take the
    # whole check down with an import traceback.  Guard the import here so a
    # missing data_access package reports an honest UNRESOLVED with a reason.
    try:
        from data_access.read.semantic_catalog import get_semantic_catalog  # noqa: F401
        _da_import_error: str | None = None
    except Exception as _exc:  # pragma: no cover - env-dependent
        _da_import_error = f"{type(_exc).__name__}: {_exc}"

    # VER-P0-05: a failure to resolve the LIVE identity set (missing operator
    # catalog, data_access not importable, non-git checkout, ...) is an honest
    # UNRESOLVED, never a crash and never a fabricated PASS.  Non-check runs
    # keep their historical behavior (re-raise) so --write does not silently
    # succeed on a half-resolved payload.
    try:
        payload = build_current()
        if not args.skip_snapshot:
            payload = _set_source_snapshot_id(payload)
        payload = evaluate(payload)
    except Exception as exc:  # pragma: no cover - env-dependent
        if not args.check:
            raise
        reason = (
            f"data_access import failed: {_da_import_error}"
            if _da_import_error else
            f"{type(exc).__name__}: {exc}"
        )
        print(f"EVIDENCE_CURRENT: UNRESOLVED reason={reason}")
        return 1

    # VER-P0-03: when artifacts lack BOUND input identities this run cannot
    # certify them — bind them to the LIVE input set and the snapshot this run
    # resolved.  (A run that resolves the live inputs and writes them as the
    # bound identity is an actual verification run: the input set is the
    # executed-at-captured set this process just computed.)
    prior_loaded = _json.loads(
        DEFAULT_OUT.read_text(encoding="utf-8")
    ) if DEFAULT_OUT.is_file() else None  # noqa: F841 - read for symmetry; unused
    # Resolved source-snapshot id (payload may still carry the placeholder
    # ``src:v1:rn`` when ``--skip-snapshot`` is used).  A bound identity that
    # says ``src:v1:rn`` will be marked stale on the NEXT run once a real
    # snapshot is resolved, so resolve a real id whenever the snapshot module is
    # reachable — this makes the file self-consistent across runs instead of
    # manufacturing a placeholder mismatch.
    _payload_sid = payload.get("source_snapshot_id") or ""
    if not _payload_sid or _payload_sid == "src:v1:rn" or _payload_sid.startswith("src:v1:unresolved"):
        try:
            from evidence.source_snapshot import build_snapshot
            _real_sid = build_snapshot().get("source_snapshot_id") or ""
            if _real_sid:
                payload["source_snapshot_id"] = _real_sid
        except Exception:
            pass
    _sid = payload.get("source_snapshot_id") or ""
    # Producer-side executed_at: an artifact whose output files were REALLY
    # regenerated this session (the caller ran the actual generator script)
    # records that run's output-file mtime as its executed_at.  This is the
    # honest producer path: the generators (certify_factor_operator_evidence /
    # export_operator_manifest / rebuild_inventory) are the verification runs,
    # and their output mtimes are the execution evidence.  The caller passes
    # ARTIFACT=file (or an explicit epoch) only right after a real run; passing
    # it does NOT bypass the identity comparison — a re-bind still uses THIS
    # run's live identities, so a stale tree still evaluates STALE.
    _recorded_executed: dict[str, str] = {}
    if getattr(args, "record_execution", None):
        for spec in args.record_execution:
            for _part in str(spec).split(":"):
                _pspec = _part
                if "=" not in _pspec:
                    raise SystemExit(f"--record-execution expects ARTIFACT=MTIME|file, got {spec!r}")
                _aid, _, _raw = _pspec.partition("=")
                _append_record_execution(_recorded_executed, _aid, _raw, ARTIFACT_IDS)
    _bind_now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for art_id, art in payload["artifacts"].items():
        # Re-bind when there is no bound identity OR the bound identity is a
        # placeholder / unresolved snapshot (``src:v1:rn`` / ``src:v1:unresolved``).
        # A placeholder fold means the artifact was never actually bound to a
        # real verification run; leaving it in place keeps the file permanently
        # STALE with a misleading reason.
        #
        # A real prior bind whose snapshot id differs from THIS run's live
        # snapshot is honest drift ONLY when the artifact's outputs have not
        # been regenerated since.  When the caller re-ran the artifact's actual
        # generator (recorded via --record-execution / producer flag), the new
        # output WAS produced against the live tree — the honest bind is the
        # live one.  Without a fresh generator run, a real prior bind to a
        # different snapshot stays STALE (never overwritten to hide drift).
        _existing = art.get("bound_input_identity") or {}
        _existing_sid = str(_existing.get("source_snapshot_id") or "")
        _placeholder = (
            not _existing_sid
            or _existing_sid == "src:v1:rn"
            or _existing_sid.startswith("src:v1:unresolved")
        )
        _fresh_run = art_id in _recorded_executed
        if _placeholder or _fresh_run:
            art["bound_input_identity"] = {
                "source_snapshot_id": _sid,
                "source_tree_identity": payload.get("source_tree_identity") or "",
                "PlatformSourceTreeIdentity": payload["sha_bindings"].get("PlatformSourceTreeIdentity") or "",
                "ImplementationClosureHashSet": payload["sha_bindings"]["ImplementationClosureHashSet"],
                "SemanticCatalogIdentity": payload["sha_bindings"]["SemanticCatalogIdentity"],
                "EvidenceAttestationIdentity": payload["sha_bindings"]["EvidenceAttestationIdentity"],
            }
            if _fresh_run:
                art["executed_at"] = _recorded_executed[art_id]
            else:
                art.setdefault("executed_at", _bind_now)
        elif _existing_sid != _sid:
            # real prior bind vs live snapshot differ -> honest STALE (keep)
            pass
        # else: real prior bind identical to live snapshot -> already CURRENT
        pass

    # VER-P0-03: evaluate() must run AFTER the bind loop above — an artifact
    # bound to the live identity set by THIS run is by definition CURRENT for
    # that run (its bound inputs were just recomputed as the live inputs).  The
    # evaluate() call inside the try block ran BEFORE binding and would always
    # report the placeholder fold as stale.  Re-evaluate the final payload.
    payload = evaluate(payload)

    out = Path(args.out) if args.out else DEFAULT_OUT
    if args.write:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {out}")

    if not args.check:
        print(json.dumps(payload, indent=2, sort_keys=False) + "\n")

    summary = payload.get("summary", {})
    status = payload.get("evaluation", {}).get("status", "STALE")
    all_current = (
        status == "CURRENT"
        and summary.get("stale") == 0
        and summary.get("failed") == 0
    )
    changed = payload.get("evaluation", {}).get("changed_artifacts", [])

    # VER-P0-05: the machine-readable check line.
    if args.check:
        if _da_import_error is not None:
            print(
                f"EVIDENCE_CURRENT: UNRESOLVED "
                f"reason=data_access import failed: {_da_import_error}"
            )
            return 1
        if all_current:
            print(
                f"EVIDENCE_CURRENT: PASS reason=every artifact CURRENT "
                f"(current={summary.get('current')}, stale={summary.get('stale')}, "
                f"unresolved={summary.get('failed')})"
            )
            return 0
        if status == "UNRESOLVED":
            print(
                f"EVIDENCE_CURRENT: UNRESOLVED "
                f"reason=source snapshot unresolved artifacts={changed or []}"
            )
            return 1
        if status == "STALE":
            print(
                f"EVIDENCE_CURRENT: STALE reason=artifact bound inputs differ "
                f"from the live tree (changed={changed or []})"
            )
            return 1
        print(
            f"EVIDENCE_CURRENT: FAIL reason=evaluation status {status} "
            f"summary={summary}"
        )
        return 1

    # Non-check reporting (unchanged behavior).
    if all_current:
        print(
            "evaluation: CURRENT (every artifact's bound inputs match the "
            "live tree and executed_at is present)"
        )
        return 0
    print(
        f"evaluation: {status} (CURRENT={summary.get('current')}, "
        f"STALE={summary.get('stale')}, UNRESOLVED={summary.get('failed')}) "
        f"-> {changed}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())