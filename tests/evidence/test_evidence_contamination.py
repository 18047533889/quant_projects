# -*- coding: utf-8 -*-
"""R22-EVIDENCE-CONTAMINATION regression tests.

Production evidence discovery must NEVER recursively read test-certified /
fixture / synthetic subtrees.  ``evidence/r2/_test_certified/`` holds 264
PASS-looking artifacts (q_version 4.1, pykx 2.6.0, executed=true) that are
generated on demand by ``tests/q_backend/q_certified_compiler.py`` so the Q
region-executor tests can exercise the production compile path without a live q
runtime.  They are test fixtures, not production evidence.

Guard tests:
  1. production evidence discovery returns zero artifacts from _test_certified/
  2. the exclusion predicate covers _test_certified/ fixtures/ tests/ synthetic/
  3. explicit test-binding tree hashes (tests/backend_parity, golden dirs)
     are unaffected by the exclusion
  4. the source-snapshot recursive walk skips fixture subtrees
  5. a synthetic discovery helper returns zero _test_certified artifacts

Constraints:
  - LOCAL ONLY (no git, no network)
  - Serial pytest, no xdist; thread env vars set by the runner
"""
from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


def _repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(_REPO.resolve()))


@pytest.fixture(scope="module")
def _evidence_provenance():
    import backend.evidence_provenance as ep

    return ep


# ---------------------------------------------------------------------------
# Guard 1: production evidence discovery returns zero artifacts from
# _test_certified/
# ---------------------------------------------------------------------------


def test_production_evidence_discovery_excludes_test_certified(_evidence_provenance):
    """Whole-repo production discovery must never surface _test_certified files."""
    ep = _evidence_provenance
    files = ep._tracked_tree_files(ep.FE_ROOT, ("*.py", "*.json", "*.csv"))
    leaked = [p for p in files if "_test_certified" in p.parts]
    assert leaked == [], (
        f"production evidence discovery leaked {len(leaked)} _test_certified "
        f"artifacts: {[_repo_relative(p) for p in leaked[:5]]}"
    )


def test_tracked_tree_files_excludes_all_fixture_dirs(_evidence_provenance):
    """The recursive discovery must exclude _test_certified/fixtures/synthetic."""
    ep = _evidence_provenance
    files = ep._tracked_tree_files(ep.FE_ROOT, ("*.py", "*.json", "*.csv"))
    leaked = [
        p
        for p in files
        if any(part in ("_test_certified", "fixtures", "synthetic") for part in p.parts)
    ]
    assert leaked == [], (
        f"production discovery leaked fixture artifacts: "
        f"{[_repo_relative(p) for p in leaked[:5]]}"
    )


def test_test_certified_root_has_pass_looking_artifacts():
    """The guard is vacuous unless _test_certified actually holds PASS artifacts."""
    tc = _REPO / "evidence" / "r2" / "_test_certified"
    if not tc.is_dir():
        pytest.skip("_test_certified not present")
    jsons = list(tc.glob("*.json"))
    assert jsons, "expected _test_certified artifacts to exist for the guard"
    sample = jsons[0]
    import json

    payload = json.loads(sample.read_text(encoding="utf-8"))
    # The fixture signature the task calls out: q_version 4.1 / pykx 2.6.0 / executed=true
    assert payload.get("q_version") == "4.1"
    assert payload.get("pykx_version") == "2.6.0"
    assert payload.get("status") == "PASS"


# ---------------------------------------------------------------------------
# Guard 2: the exclusion predicate covers the required directory names
# ---------------------------------------------------------------------------


def test_evidence_path_excluded_covers_required_names(_evidence_provenance):
    """evidence_path_excluded must reject every disallowed fixture dir name."""
    ep = _evidence_provenance
    for rel in (
        "evidence/r2/_test_certified/_test_certified_add_compile.json",
        "evidence/r2/fixtures/artifact.json",
        "evidence/r2/synthetic/generated.json",
        "tests/backend_parity/evidence_case_registry.py",
        "tests/fixtures/golden/golden.csv",
    ):
        assert ep.evidence_path_excluded(_REPO / rel), rel

    for rel in (
        "evidence/r2/R21-EVIDENCE-REGISTRY.yaml",
        "backend/evidence_provenance.py",
        "factor_recipes/factor_recipe.py",
    ):
        assert not ep.evidence_path_excluded(_REPO / rel), rel


def test_evidence_path_excluded_is_root_relative(_evidence_provenance):
    """Explicit test-binding roots must keep working under exclusion."""
    ep = _evidence_provenance
    # A file under tests/backend_parity is allowed when hashing that root
    # directly (test_source_hash), but excluded in a whole-repo scan.
    p = _REPO / "tests" / "backend_parity" / "evidence_case_registry.py"
    assert not ep.evidence_path_excluded(p, root=_REPO / "tests" / "backend_parity")
    assert ep.evidence_path_excluded(p, root=_REPO)


# ---------------------------------------------------------------------------
# Guard 3: explicit test-binding tree hashes are unaffected
# ---------------------------------------------------------------------------


def test_explicit_test_binding_tree_hash_unaffected(_evidence_provenance):
    """_tree_hash(tests/backend_parity) still hashes that tree (no-op exclusion)."""
    ep = _evidence_provenance
    h = ep._tree_hash(_REPO / "tests" / "backend_parity", ("*.py",))
    assert h and len(h) == 16


def test_golden_data_hash_roots_unaffected(_evidence_provenance):
    """Golden-data binding roots remain hashable under the exclusion rule."""
    ep = _evidence_provenance
    for root in (
        _REPO / "tests" / "operator_golden",
        _REPO / "tests" / "fixtures" / "golden",
        _REPO / "tests" / "integration" / "fixtures" / "golden",
    ):
        if root.is_dir():
            h = ep._tree_hash(root, ("*.py", "*.csv", "*.json", "*.parquet"))
            assert h and len(h) == 16, root


# ---------------------------------------------------------------------------
# Guard 4: source snapshot recursive walk skips fixture subtrees
# ---------------------------------------------------------------------------


def test_source_snapshot_scan_skips_fixture_dirs():
    """evidence.source_snapshot must never descend into fixture/synthetic dirs."""
    from evidence.source_snapshot import EXCLUDE_DIRS, scan_source_tree

    assert "_test_certified" in EXCLUDE_DIRS
    assert "fixtures" in EXCLUDE_DIRS
    assert "synthetic" in EXCLUDE_DIRS

    entries = scan_source_tree(_REPO, scan_dirs=("evidence",))
    leaked = [
        e.rel_path
        for e in entries
        if any(part in ("_test_certified", "fixtures", "synthetic") for part in Path(e.rel_path).parts)
    ]
    assert leaked == [], f"source snapshot leaked fixture artifacts: {leaked[:5]}"


# ---------------------------------------------------------------------------
# Guard 5: a synthetic discovery helper returns zero _test_certified artifacts
# ---------------------------------------------------------------------------


def test_discovery_helper_returns_zero_test_certified(_evidence_provenance):
    """A repo-glob discovery over evidence/*.json returns zero _test_certified."""
    ep = _evidence_provenance
    # Simulate a production evidence loader that globs every JSON under the
    # evidence tree; the exclusion predicate must reject the fixture subtree.
    evidence_root = _REPO / "evidence"
    if not evidence_root.is_dir():
        pytest.skip("evidence dir not present")
    leaked = [
        p
        for p in evidence_root.rglob("*.json")
        if "_test_certified" in p.parts and not ep.evidence_path_excluded(p)
    ]
    assert leaked == [], (
        f"discovery helper leaked {len(leaked)} _test_certified artifacts: "
        f"{[_repo_relative(p) for p in leaked[:5]]}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
