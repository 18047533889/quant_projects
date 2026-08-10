# -*- coding: utf-8 -*-
"""R28 §一百五十八: evidence CI — evidence must be current (bound to HEAD),
cover every canonical, and every required artifact must be tracked & not ignored.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent.parent
_OUT = _REPO / "docs" / "evidence" / "r28"

REQUIRED_FILES = [
    "R28_OPERATOR_INVENTORY.csv",
    "R28_OPERATOR_INVENTORY.json",
    "R28_CANONICAL_TEST_COVERAGE.csv",
    "R28_CANONICAL_TEST_COVERAGE.json",
    "R28_PER_CANONICAL_TEST_RESULTS.csv",
    "R28_PER_CANONICAL_TEST_RESULTS.json",
    "R28_FORBIDDEN_OPERATOR_AUDIT.json",
    "R28_STATIC_LOOKAHEAD_SCAN.csv",
    "R28_STATIC_LOOKAHEAD_SCAN.json",
    "R28_MODEL_CAUSALITY_MATRIX.csv",
    "R28_MODEL_CAUSALITY_MATRIX.json",
    "R28_PYTEST_SUMMARY.json",
    "R28_ARTIFACT_MANIFEST.json",
    "R28_FINAL_ACCEPTANCE_REPORT.md",
    "R28_CANONICAL_SET_DIGEST.txt",
]


def _git(args):
    return subprocess.run(["git", "-C", str(_REPO), *args], capture_output=True, text=True)


def _head():
    return _git(["rev-parse", "HEAD"]).stdout.strip()


def test_r28_evidence_current_head():
    """The evidence manifest must be bound to the current code SHA.  The evidence
    commit itself only touches docs/evidence/r28, so HEAD == manifest_sha or
    HEAD is an evidence-only commit on top of manifest_sha."""
    manifest_path = _OUT / "R28_ARTIFACT_MANIFEST.json"
    if not manifest_path.exists():
        pytest.fail("R28_ARTIFACT_MANIFEST.json missing")
    manifest = json.loads(manifest_path.read_text())
    head = _head()
    manifest_sha = manifest.get("git_sha")
    if manifest_sha == head:
        return
    parent = _git(["rev-parse", "HEAD~1"]).stdout.strip()
    diff = _git(["diff", "--name-only", "HEAD~2", "HEAD"]).stdout.strip()
    evidence_only = all("/docs/evidence/r28/" in line for line in diff.splitlines())
    assert parent == manifest_sha and evidence_only, (
        f"evidence git_sha {manifest_sha} not current (HEAD {head}) (STALE — regenerate)"
    )


def test_r28_evidence_canonical_digest_current():
    digest_path = _OUT / "R28_CANONICAL_SET_DIGEST.txt"
    manifest_path = _OUT / "R28_ARTIFACT_MANIFEST.json"
    if not digest_path.exists() or not manifest_path.exists():
        pytest.fail("digest/manifest missing")
    current = ""
    for line in digest_path.read_text().splitlines():
        if line.startswith("canonical_set_digest="):
            current = line.split("=", 1)[1]
    manifest = json.loads(manifest_path.read_text())
    assert manifest.get("canonical_set_digest") == current, "evidence digest stale"


def test_r28_evidence_all_canonicals_present():
    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry

    load_all()
    registry = set(OperatorRegistry.list_canonical())
    coverage_path = _OUT / "R28_CANONICAL_TEST_COVERAGE.json"
    if not coverage_path.exists():
        pytest.fail("coverage evidence missing")
    covered = {row["canonical"] for row in json.loads(coverage_path.read_text())["rows"]}
    assert covered == registry, (
        f"evidence coverage ({len(covered)}) != registry ({len(registry)}); "
        f"missing={sorted(registry - covered)[:10]}"
    )


def test_r28_all_retained_have_test():
    coverage_path = _OUT / "R28_CANONICAL_TEST_COVERAGE.json"
    if not coverage_path.exists():
        pytest.fail("coverage evidence missing")
    rows = json.loads(coverage_path.read_text())["rows"]
    zero = [r["canonical"] for r in rows if r.get("test_count", 0) == 0]
    assert zero == [], f"retained canonicals with zero tests: {zero}"


def test_r28_all_required_files_tracked():
    for name in REQUIRED_FILES:
        p = _OUT / name
        assert p.exists(), f"missing artifact {name}"
        rel = p.relative_to(_REPO).as_posix()
        tracked = _git(["ls-files", "--error-unmatch", rel])
        assert tracked.returncode == 0, f"{name} not tracked in git"
        ignored = _git(["check-ignore", rel])
        assert ignored.returncode != 0, f"{name} is gitignored"


def test_r28_evidence_no_extra_deleted_canonicals():
    """Evidence must not reference canonicals that no longer exist."""
    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry

    load_all()
    registry = set(OperatorRegistry.list_canonical())
    inventory_path = _OUT / "R28_OPERATOR_INVENTORY.json"
    if not inventory_path.exists():
        pytest.fail("inventory missing")
    inventory = {row["canonical"] for row in json.loads(inventory_path.read_text())["operators"]}
    assert inventory <= registry, f"evidence references deleted canonicals: {sorted(inventory - registry)[:10]}"
