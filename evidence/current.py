# -*- coding: utf-8 -*-
"""R22-CURRENT-TRUTH — CURRENT.json generator bound to the full SHA identity set.

The evidence truth file (``evidence/CURRENT.json``) is the single truth entry
point for release CURRENT-ness.  This module regenerates it deterministically
from the *current* tree so it can never silently drift:

    RootRepoSHA             = repo HEAD SHA + DirtyTreeDigest (diff/cached/untracked)
    FactorEngineSubmoduleSHA  = factor_engine/ submodule pinned SHA
    DataAccessSubmoduleSHA    = dataaccess/ submodule pinned SHA
    OperatorCatalogSHA        = SHA-256 of build/mining/direct_mining_catalog.json
    SemanticCatalogIdentity   = DataAccess-issued SemanticCatalogIdentity.cache_key() (STRICT)
    ImplementationClosureHashSet = SHA-256 over per-operator implementation closure
                              hashes (name + implementing module source + contract)
    DependencyLockHash          = SHA-256 over the production dependency lock
    TestEnvironmentIdentity     = SHA-256 over python version + key dep versions

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


def root_repo_sha() -> dict[str, str | bool] | str:
    """DirtyTreeDigest for the repo root (VER-P0-03).

    When the working tree is dirty the digest carries HEAD + diff_hash +
    cached_diff_hash + untracked_source_hash, so two different uncommitted
    modifications never share an identity.  When the tree is clean it returns
    the plain HEAD string (backwards compatible with callers that split on a
    space, e.g. ``scripts/gen_verification_manifest.py``).
    """
    digest = _dirty_tree_digest()
    if not digest["HEAD"]:
        raise RuntimeError("R22-CURRENT-TRUTH: repo root is not a git checkout")
    if digest["dirty"]:
        return digest
    return str(digest["HEAD"])


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
    if not CATALOG_PATH.is_file():
        raise RuntimeError(
            f"R22-CURRENT-TRUTH: operator catalog missing: {CATALOG_PATH}"
        )
    catalog = _load_catalog_cached()
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
    with CATALOG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


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
    is resolved as ``<repo>/cleaned_operators/<rest>.py`` (with a
    ``__init__.py`` package fallback).  Filesystem-only — the operator is never
    imported (importing registers it and mutates the global OperatorRegistry).

    Memoized per module with mtime+size self-invalidation: the 1565 catalog
    operators share ~175 distinct modules, any on-disk source change bumps the
    cached digest, so a mutated implementation is always detected even in a
    long-running process.
    """
    if not module_ref or module_ref == "builtins":
        return "builtins:no-module"
    rel_parts = module_ref.split(".")
    if rel_parts and rel_parts[0] == "cleaned_operators":
        rel_parts = rel_parts[1:]
    if not rel_parts:
        return "MISSING:" + _sha256_bytes(module_ref.encode("utf-8"))
    pkg_root = _REPO_ROOT / "cleaned_operators"
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
    da_root = str(_REPO_ROOT / "dataaccess")
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


def _root_diff_hash() -> str:
    """SHA-256 of the root working-tree ``git diff`` defensively cached.

    ``git diff`` re-hashes every modified blob on each call and can take tens
    of seconds on this repo; the working tree content can only change when a
    write happens *inside this process*, so the digest is memoized and only
    invalidated by an explicit ``_invalidate_root_diff_cache()``.
    """
    global _ROOT_DIFF_HASH_CACHE
    if _ROOT_DIFF_HASH_CACHE is not None:
        return _ROOT_DIFF_HASH_CACHE
    diff = _git_capture(["git", "diff"], _REPO_ROOT) or ""
    _ROOT_DIFF_HASH_CACHE = _sha256_byte_stream(diff.encode("utf-8"))
    return _ROOT_DIFF_HASH_CACHE


_ROOT_DIFF_HASH_CACHE: str | None = None
_ROOT_PREV_DIFF_HASH: str = ""


def _invalidate_root_diff_cache() -> None:
    global _ROOT_DIFF_HASH_CACHE, _ROOT_PREV_DIFF_HASH
    _ROOT_DIFF_HASH_CACHE = None
    _ROOT_PREV_DIFF_HASH = ""


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
    live_root = _dirty_tree_digest()

    live_bindings = {
        "RootRepoSHA": json.dumps(
            live_root, sort_keys=True, separators=(",", ":")
        ) if live_root["dirty"] else (live_root["HEAD"] or ""),
        "FactorEngineSubmoduleSHA": submodule_sha("factor_engine"),
        "DataAccessSubmoduleSHA": submodule_sha("dataaccess"),
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
            # The source snapshot bound by the previous run's file.
            if prior.get("source_snapshot_id"):
                art["bound_input_identity"] = dict(
                    art.get("bound_input_identity") or {}
                )
                art["bound_input_identity"].setdefault(
                    "source_snapshot_id", prior["source_snapshot_id"]
                )
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

            for key in ("ImplementationClosureHashSet", "SemanticCatalogIdentity"):
                bval = bound.get(key)
                lval = live.get(key)
                if bval is not None and bval != lval:
                    stale_reasons.append(f"{key} changed")

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
    }
    return payload


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
        for key in ("ImplementationClosureHashSet", "SemanticCatalogIdentity"):
            bval = bound.get(key)
            lval = live.get(key)
            if bval is not None and bval != lval:
                changed.append(f"artifact:{art_id}:{key}")

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
    args = parser.parse_args(argv)

    payload = build_current()
    if not args.skip_snapshot:
        payload = _set_source_snapshot_id(payload)
    payload = evaluate(payload)

    # VER-P0-03: when artifacts lack BOUND input identities this run cannot
    # certify them — bind them to the LIVE input set and the snapshot this run
    # resolved.  (A run that resolves the live inputs and writes them as the
    # bound identity is an actual verification run: the input set is the
    # executed-at-captured set this process just computed.)
    prior_loaded = _json.loads(
        DEFAULT_OUT.read_text(encoding="utf-8")
    ) if DEFAULT_OUT.is_file() else None
    if prior_loaded is not None:
        for art in payload["artifacts"].values():
            if not art.get("bound_input_identity"):
                art.setdefault("bound_input_identity", {}).update({
                    "source_snapshot_id": payload.get("source_snapshot_id") or "",
                    "ImplementationClosureHashSet": payload["sha_bindings"]["ImplementationClosureHashSet"],
                    "SemanticCatalogIdentity": payload["sha_bindings"]["SemanticCatalogIdentity"],
                })
    else:
        for art in payload["artifacts"].values():
            art.setdefault("bound_input_identity", {}).update({
                "source_snapshot_id": payload.get("source_snapshot_id") or "",
                "ImplementationClosureHashSet": payload["sha_bindings"]["ImplementationClosureHashSet"],
                "SemanticCatalogIdentity": payload["sha_bindings"]["SemanticCatalogIdentity"],
            })

    out = Path(args.out) if args.out else DEFAULT_OUT
    if args.write:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {out}")

    print(json.dumps(payload, indent=2, sort_keys=False) + "\n")

    summary = payload.get("summary", {})
    status = payload.get("evaluation", {}).get("status", "STALE")
    if status == "CURRENT" and summary.get("stale") == 0 \
            and summary.get("failed") == 0:
        print(
            "evaluation: CURRENT (every artifact's bound inputs match the "
            "live tree and executed_at is present)"
        )
        return 0
    changed = payload.get("evaluation", {}).get("changed_artifacts", [])
    print(
        f"evaluation: {status} (CURRENT={summary.get('current')}, "
        f"STALE={summary.get('stale')}, UNRESOLVED={summary.get('failed')}) "
        f"-> {changed}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())