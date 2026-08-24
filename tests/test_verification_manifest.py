# -*- coding: utf-8 -*-
"""REL-P0-01 — VerificationManifest contract tests.

Asserts the release evidence manifest at evidence/VerificationManifest.json:
  1. contains all 15 required gates by name,
  2. no gate status is silently blank / empty,
  3. every required package has a non-empty source-tree hash,
  4. core snapshot fields are present and non-empty.

Re-generate the manifest with:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        /usr/bin/python3 scripts/gen_verification_manifest.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "evidence" / "VerificationManifest.json"

REQUIRED_GATES = (
    "UNIT",
    "NUMERICAL_ORACLE",
    "PROPERTY",
    "CROSS_PACKAGE_CORE",
    "CROSS_PACKAGE_OPTIONAL",
    "LEAKAGE",
    "PIT",
    "DETERMINISM",
    "SERIALIZATION",
    "CHECKPOINT_RESUME",
    "FRESH_WHEEL",
    "1K_SCALE",
    "10K_SCALE",
    "100K_SCALE",
    "ASHARE_SEMANTIC_CONTRACT_GOLDEN",
    "ASHARE_REAL_DATA_SHADOW",
)

VALID_STATUSES = {"PASS", "FAIL", "NOT_RUN", "BLOCKED", "NOT_APPLICABLE"}

REQUIRED_PACKAGES = (
    "factor_engine",
    "quant_evaluator",
    "factor_optimizer",
    "factor_assets",
    "factor_preprocess",
)


@pytest.fixture(scope="module")
def manifest() -> dict:
    if not MANIFEST_PATH.is_file():
        pytest.fail(
            f"evidence/VerificationManifest.json not found at {MANIFEST_PATH}. "
            "Run scripts/gen_verification_manifest.py first."
        )
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        pytest.fail(f"evidence/VerificationManifest.json is not valid JSON: {exc}")


def test_manifest_has_all_15_gates(manifest: dict) -> None:
    gates = manifest.get("gates")
    assert gates is not None, "manifest.gates is missing"
    assert isinstance(gates, list), "manifest.gates must be a list"
    names = [g.get("name") for g in gates]
    missing = [name for name in REQUIRED_GATES if name not in names]
    assert not missing, f"missing gates: {missing}"
    # also fail if any name is repeated (sloppy config)
    assert len(names) == len(set(names)), "duplicate gate names in manifest"


def test_no_gate_status_is_blank(manifest: dict) -> None:
    gates = manifest.get("gates", [])
    for g in gates:
        name = g.get("name")
        status = g.get("status")
        assert status is not None and str(status).strip() != "", (
            f"gate {name!r} has blank status"
        )
        assert str(status) in VALID_STATUSES, (
            f"gate {name!r} has invalid status {status!r}; "
            f"expected one of {sorted(VALID_STATUSES)}"
        )


@pytest.mark.parametrize("pkg", REQUIRED_PACKAGES)
def test_package_has_source_tree_hash(manifest: dict, pkg: str) -> None:
    entry = manifest.get("packages", {}).get(pkg)
    assert entry is not None, f"package {pkg!r} missing from manifest.packages"
    tree_hash = entry.get("tree_sha256")
    assert tree_hash is not None and str(tree_hash).strip(), (
        f"package {pkg!r} has no source-tree hash (tree_sha256)"
    )
    assert len(tree_hash) == 64, f"package {pkg!r} tree hash must be sha256 hex (64 chars)"


def test_snapshot_fields_present(manifest: dict) -> None:
    snap = manifest.get("source_snapshot")
    assert snap is not None, "manifest.source_snapshot missing"
    assert snap.get("git_repo") is True, "expected git repo at source_snapshot.git_repo"
    assert snap.get("git_sha"), "source_snapshot.git_sha missing/empty"
    assert "git_dirty" in snap, "source_snapshot.git_dirty missing"


def test_release_notes_capture_stale_artifact(manifest: dict) -> None:
    notes = manifest.get("release_notes", [])
    rel02 = [n for n in notes if n.get("id") == "REL-P0-02"]
    assert rel02, "expected REL-P0-02 release note about stale .pytest_full.txt"
    assert any(
        a.get("path") == ".pytest_full.txt" for a in rel02[0].get("artifacts", [])
    ), "REL-P0-02 note must reference the root .pytest_full.txt artifact"
