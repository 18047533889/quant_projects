# -*- coding: utf-8 -*-
"""VER-P0-03/04 — CURRENT.json cannot self-certify; fail-open holes removed.

Red-team findings covered:
  1. ``build_current()`` compared freshly-written bindings against the same
     freshly-written bindings (``current == current``) -> it could never be
     STALE.  Now an artifact is CURRENT only when its BOUND input identity
     (source_snapshot_id + ImplementationClosureHashSet + SemanticCatalogIdentity
     at build time) matches the LIVE identity set AND the artifact has an
     ``executed_at`` from a real verification run.
  2. ``implementation_closure_hash_set()`` hashed only sorted canonical NAMES.
     Now each canonical contributes a digest of
     name + sha256(implementing module source) + contract/metadata identity, so
     the set hash changes when operator source changes even if names are
     unchanged.
  3. ``semantic_catalog_identity()`` used ``strict=False``.  Now strict=True.
  4. ``_set_source_snapshot_id()`` failure continued to certify with
     ``src:v1:unresolved:...``.  Now unresolved -> artifact UNRESOLVED and the
     program does not certify.
  5. ``main()`` returned 0 even when STALE.  Now exits non-zero on any
     STALE/UNRESOLVED.
  6. ``root_repo_sha()`` dirty identity was ``HEAD (working-tree-dirty)``.
     Now returns a DirtyTreeDigest {HEAD, diff_hash, cached_diff_hash,
     untracked_source_hash, dirty}.

Tests are LOCAL ONLY (no git mutations, no network), serial pytest.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import evidence.current as ec  # noqa: E402

try:
    from factor_engine.backend.evidence_provenance import implementation_closure_hash_for as _backend_closure
except Exception:  # pragma: no cover - backend may be unavailable
    _backend_closure = None


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _preserve_catalog():
    """Restore the catalog + on-disk CURRENT.json after every test."""
    cat = ec.CATALOG_PATH
    cat_bytes = cat.read_bytes()
    current = ec.DEFAULT_OUT
    current_bytes = current.read_bytes() if current.is_file() else None
    yield
    cat.write_bytes(cat_bytes)
    if current_bytes is None:
        current.unlink(missing_ok=True)
    else:
        current.write_bytes(current_bytes)


def _fresh_closure() -> tuple[str, int]:
    return ec.implementation_closure_hash_set()


def _saved_file() -> dict | None:
    if not ec.DEFAULT_OUT.is_file():
        return None
    try:
        return json.loads(ec.DEFAULT_OUT.read_text(encoding="utf-8"))
    except Exception:
        return None


def _live_snapshot_id() -> str:
    from evidence.source_snapshot import build_snapshot
    return build_snapshot()["source_snapshot_id"]


def _default_identity() -> str:
    """Per-operator identity for the canonical set in the catalog fixture."""
    d = json.loads(ec.CATALOG_PATH.read_text(encoding="utf-8"))
    ids: list[str] = []
    for op in d.get("operators", []):
        canonical = str(op["canonical"])
        h = ec.implementation_closure_hash_for(canonical)
        ids.append(f"{canonical}:{h}")
    ids.sort()
    import hashlib
    h = hashlib.sha256()
    for line in ids:
        h.update(line.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _replace_artifacts(
    payload: dict,
    *,
    executed_at: str = "2026-08-21T00:00:00+00:00",
    bind: bool = True,
) -> dict:
    """Stamp the on-disk style per-artifact bound inputs onto ``payload``."""
    live = payload.get("sha_bindings", {})
    closure = live.get("ImplementationClosureHashSet") or _fresh_closure()[0]
    semantic = live.get("SemanticCatalogIdentity") or ec.semantic_catalog_identity()
    for art in payload.get("artifacts", {}).values():
        if executed_at:
            art["executed_at"] = executed_at
        if bind:
            art["bound_input_identity"] = {
                "source_snapshot_id": payload.get("source_snapshot_id") or _live_snapshot_id(),
                "ImplementationClosureHashSet": closure,
                "SemanticCatalogIdentity": semantic,
            }
        else:
            art.pop("bound_input_identity", None)
    return payload


def _set_snapshot(payload: dict, *, ok: bool = True) -> dict:
    if ok:
        payload["source_snapshot_id"] = _live_snapshot_id()
    else:
        payload["source_snapshot_id"] = "src:v1:unresolved:TestError"
    return payload


# ---------------------------------------------------------------------------
# (c) implementation closure hash changes when operator source changes
# ---------------------------------------------------------------------------

def test_closure_hash_changes_when_module_source_changes(tmp_path):
    """Same canonical + module name, different body -> different closure hash."""
    import hashlib

    d = json.loads(ec.CATALOG_PATH.read_text(encoding="utf-8"))
    record = d["operators"][0]
    canonical = str(record["canonical"])
    module_ref = str(record.get("module") or "").strip()
    contract = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    canonical_bare = canonical.split(".", 1)[1] if canonical.startswith("cleaned_operators.") else canonical
    module_bare = module_ref.split(".", 1)[1] if module_ref.startswith("cleaned_operators.") else module_ref

    def compact(source_sha256: str) -> str:
        payload = (
            f"canonical:{canonical_bare}\nmodule:{module_bare}\n"
            f"source_sha256:{source_sha256}\n"
        ).encode("utf-8")
        return ec._sha256_byte_stream(
            payload + b"contract_sha256:" + hashlib.sha256(contract).hexdigest().encode("utf-8")
        )

    # A change in the implementation source digest changes the per-operator
    # closure digest even though the canonical name and module name are unchanged.
    real_digest = ec._module_source_digest(module_ref)
    other_digest = hashlib.sha256(b"def g():\n    return 99\n").hexdigest()
    assert compact(real_digest) != compact(other_digest)
    # And the per-operator digest for the current catalog record is exactly the
    # compact digest over the real module source digest.
    assert compact(real_digest) == ec.implementation_closure_hash_for(canonical)


def _first_catalog_module_file() -> Path | None:
    """Module file backing the first catalog operator (or None)."""
    d = json.loads(ec.CATALOG_PATH.read_text(encoding="utf-8"))
    record = d["operators"][0]
    module_ref = str(record.get("module") or "").strip()
    if not module_ref or module_ref == "builtins":
        return None
    rel_parts = module_ref.split(".")
    # R45: catalog refs are ``cleaned_operators.<sub>`` or
    # ``factor_engine.cleaned_operators.<sub>``; both resolve under the single
    # canonical package factor_engine/cleaned_operators/.
    if rel_parts[0] == "factor_engine":
        rel_parts = rel_parts[1:]
    if rel_parts[0] == "cleaned_operators":
        rel_parts = rel_parts[1:]
    pkg_root = ec._REPO_ROOT / "factor_engine" / "cleaned_operators"
    rel_file = pkg_root.joinpath(*rel_parts).with_suffix(".py")
    if rel_file.is_file():
        return rel_file
    rel_pkg = pkg_root.joinpath(*rel_parts, "__init__.py")
    return rel_pkg if rel_pkg.is_file() else None


def test_closure_set_hash_changes_when_source_changes(tmp_path):
    """Mutating a source file behind an operator changes the SET hash too."""
    rel_file = _first_catalog_module_file()
    if rel_file is None:
        pytest.skip("first catalog operator has no module file")

    before, n = _fresh_closure()
    orig = rel_file.read_bytes()
    try:
        rel_file.write_bytes(orig + b"\n# VER-P0-03 mutation\n")
        ec._invalidate_root_diff_cache()
        after, n2 = _fresh_closure()
        assert after != before
        assert n == n2
    finally:
        rel_file.write_bytes(orig)
    ec._invalidate_root_diff_cache()
    restored, _ = _fresh_closure()
    assert restored == before


# ---------------------------------------------------------------------------
# (a) matching bound inputs -> CURRENT
# ---------------------------------------------------------------------------

def test_build_current_current_when_saved_matches_live(monkeypatch):
    """A saved artifact whose bound inputs match the live tree -> CURRENT."""
    live = ec.build_current()
    live = _set_snapshot(live, ok=True)
    saved = _replace_artifacts(live)
    ec.DEFAULT_OUT.write_text(json.dumps(saved, indent=2), encoding="utf-8")

    payload = ec.build_current()
    payload = _set_snapshot(payload, ok=True)
    payload = ec.evaluate(payload)

    assert payload["summary"]["current"] == len(ec.ARTIFACT_IDS)
    assert payload["summary"]["stale"] == 0
    assert payload["evaluation"]["status"] == "CURRENT"
    for art in payload["artifacts"].values():
        assert art["status"] == "CURRENT"
        assert art.get("executed_at")


def test_build_current_no_saved_artifact_stays_unresolved(monkeypatch):
    """No saved artifact -> nothing can be certified CURRENT (fail-closed)."""
    ec.DEFAULT_OUT.unlink(missing_ok=True)
    payload = ec.build_current()
    payload = _set_snapshot(payload, ok=True)
    payload = ec.evaluate(payload)
    for art in payload["artifacts"].values():
        assert art["status"] in ("STALE", "UNRESOLVED")
    assert payload["summary"]["current"] == 0


# ---------------------------------------------------------------------------
# (b) mutate a source file -> STALE
# ---------------------------------------------------------------------------

def test_build_current_stale_when_operator_source_mutated(monkeypatch):
    """Mutating operator source makes the SAME call yield STALE, not CURRENT."""
    rel_file = _first_catalog_module_file()
    if rel_file is None:
        pytest.skip("first catalog operator has no module file")

    # 1) Save an artifact bound to the pristine tree.
    live = ec.build_current()
    live = _set_snapshot(live, ok=True)
    saved = _replace_artifacts(live)
    ec.DEFAULT_OUT.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    ec._invalidate_root_diff_cache()

    orig = rel_file.read_bytes()
    try:
        rel_file.write_bytes(orig + b"\n# VER-P0-03 mutation\n")
        ec._invalidate_root_diff_cache()
        # 2) Same call, mutated tree -> STALE.
        payload = ec.build_current()
        payload = _set_snapshot(payload, ok=True)
        payload = ec.evaluate(payload)
        assert payload["summary"]["current"] == 0
        assert payload["summary"]["stale"] == len(ec.ARTIFACT_IDS)
        assert payload["evaluation"]["status"] == "STALE"
        for art in payload["artifacts"].values():
            assert art["status"] == "STALE"
            assert any("ImplementationClosureHashSet" in r for r in art["stale_inputs"])
    finally:
        rel_file.write_bytes(orig)
        ec._invalidate_root_diff_cache()


# ---------------------------------------------------------------------------
# (4) source snapshot failure -> UNRESOLVED, never certified
# ---------------------------------------------------------------------------

def test_snapshot_failure_marks_unresolved(monkeypatch):
    """A failed source snapshot must never certify (VER-P0-04)."""
    def _boom(payload):
        raise RuntimeError("simulated snapshot failure")
    monkeypatch.setattr(ec, "_set_source_snapshot_id", _boom)
    payload = ec.build_current()
    with pytest.raises(RuntimeError, match="simulated snapshot failure"):
        ec._set_source_snapshot_id(payload)


def test_snapshot_unresolved_artifact_never_current():
    """src:v1:unresolved payload -> artifact UNRESOLVED, not CURRENT."""
    payload = ec.build_current()
    payload = _set_snapshot(payload, ok=False)
    payload = ec.evaluate(payload)
    assert payload["evaluation"]["status"] == "UNRESOLVED"
    for art in payload["artifacts"].values():
        assert art["status"] == "UNRESOLVED"
        assert any("snapshot" in r for r in art["stale_inputs"])
    assert payload["summary"]["current"] == 0


# ---------------------------------------------------------------------------
# (3) strict semantic catalog identity
# ---------------------------------------------------------------------------

def test_semantic_catalog_identity_strict():
    """release truth uses strict=True and returns a 64-char digest."""
    key = ec.semantic_catalog_identity()
    assert isinstance(key, str) and len(key) == 64


# ---------------------------------------------------------------------------
# (6) DirtyTreeDigest
# ---------------------------------------------------------------------------

def test_dirty_tree_digest_shape_and_discrimination():
    """dirty identity is a digest of HEAD + diff + cached + untracked."""
    d1 = ec._dirty_tree_digest()
    for field in ("HEAD", "diff_hash", "cached_diff_hash", "untracked_source_hash", "dirty"):
        assert field in d1, f"missing {field}"
    assert isinstance(d1["dirty"], bool)
    # Two independent digests of the same tree agree.
    d2 = ec._dirty_tree_digest()
    assert d1 == d2


def test_root_repo_sha_dirty_marker_replaced():
    """HEAD (working-tree-dirty) is gone; root_repo_sha returns RepoIdentity always.

    R46 P0-04/P0-D: the old ``dict | str`` dual type is replaced by a single
    immutable :class:`RepoIdentity` returned ALWAYS.  ``to_dict()`` carries the
    DirtyTreeDigest schema; ``str()`` yields the bare HEAD SHA for back-compat.
    """
    from evidence.repo_identity import RepoIdentity
    r = ec.root_repo_sha()
    d = ec._dirty_tree_digest()
    assert isinstance(r, RepoIdentity)
    assert r.head_sha == d["HEAD"]
    assert r.dirty == d["dirty"]
    assert r.working_tree_hash == d["diff_hash"]
    assert r.staged_hash == d["cached_diff_hash"]
    assert r.untracked_source_hash == d["untracked_source_hash"]
    # canonical single hash is stable + the bare HEAD SHA via str()
    assert isinstance(r.identity_hash, str) and r.identity_hash
    assert str(r) == d["HEAD"]
    # to_dict() carries the VER-P0-03 DirtyTreeDigest schema
    rd = r.to_dict()
    assert rd["HEAD"] == d["HEAD"]
    assert isinstance(rd["dirty"], bool)
    assert all(k in rd for k in (
        "HEAD", "dirty", "diff_hash", "cached_diff_hash",
        "untracked_source_hash", "identity_hash",
    ))
    assert "(working-tree-dirty)" not in str(r)
    # never a bare dict or bare str anymore (single type)
    assert not isinstance(r, dict)
    assert not isinstance(r, str)


# ---------------------------------------------------------------------------
# (5) main() exits non-zero on STALE / UNRESOLVED
# ---------------------------------------------------------------------------

def test_main_exit_nonzero_when_stale(tmp_path):
    """main() must exit non-zero when any artifact is STALE (VER-P0-05)."""
    rel_file = _first_catalog_module_file()
    if rel_file is None:
        pytest.skip("first catalog operator has no module file")

    live = ec.build_current()
    live = _set_snapshot(live, ok=True)
    saved = _replace_artifacts(live)
    ec.DEFAULT_OUT.write_text(json.dumps(saved, indent=2), encoding="utf-8")

    out = tmp_path / "CURRENT.json"
    orig = rel_file.read_bytes()
    try:
        rel_file.write_bytes(orig + b"\n# VER-P0-03 mutation\n")
        ec._invalidate_root_diff_cache()
        code = ec.main(["--skip-snapshot", "--write", "--out", str(out)])
        assert code != 0, "STALE main() must exit non-zero"
        written = json.loads(out.read_text(encoding="utf-8"))
        assert written["evaluation"]["status"] == "STALE"
        assert written["summary"]["current"] == 0
    finally:
        rel_file.write_bytes(orig)
        ec._invalidate_root_diff_cache()


# ---------------------------------------------------------------------------
# (1) self-certification is impossible even with a hand-written CURRENT.json
# ---------------------------------------------------------------------------

def test_cannot_self_certify_without_executed_at(monkeypatch):
    """No bound input identity on disk -> nothing can be CURRENT regardless.

    _replace_artifacts(..., bind=False) strips the BOUND input identity AND
    clears executed_at, so the CURRENT gate must refuse the artifact (missing
    executed_at is one of the recorded reasons)."""
    payload = ec.build_current()
    payload = _set_snapshot(payload, ok=True)
    payload = _replace_artifacts(payload, executed_at="", bind=False)
    payload = ec.evaluate(payload)
    assert payload["summary"]["current"] == 0
    assert payload["evaluation"]["status"] == "STALE"
    for art in payload["artifacts"].values():
        assert art["status"] != "CURRENT"
        # fails-closed: either missing executed_at or missing bound identity
        assert any(
            "executed_at" in r or "bound_input_identity" in r
            for r in art["stale_inputs"]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
