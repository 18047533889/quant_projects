# -*- coding: utf-8 -*-
"""Tests for FE-P0-023, FE-P0-024, FE-P0-038 evidence verification fixes.

FE-P0-023: Production hard verification must reject count-only legacy evidence.
FE-P0-024: Production evidence cannot reference artifacts without blob hashes.
FE-P0-038: Production evidence must reject "unknown" implementation/emitter/artifact hashes.
"""
from __future__ import annotations

import json

import pytest

import factor_engine.backend.factor_operator_evidence as foe
import factor_engine.backend.evidence_provenance as ep
import evidence.scm_manifest as sm


def _patch_heavy_deps(monkeypatch):
    """Skip heavy registry loads to focus on validation logic."""
    monkeypatch.setattr(foe, "_production_sets", lambda: (set(), set()))
    monkeypatch.setattr(ep, "evidence_artifact_valid", lambda *a, **k: True)
    monkeypatch.setattr(ep, "implementation_hashes_for", lambda *a, **k: {})


def _write_manifest(
    path,
    *,
    build: str,
    certified: str,
    ancestor: bool,
    changed: list[str] | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "build_commit_sha": build,
                "certified_commit_sha": certified,
                "certified_is_ancestor": ancestor,
                "changed_since_certified": list(changed or []),
                "generated_at": "2026-08-14T00:00:00+00:00",
                "source": "git",
            }
        ),
        encoding="utf-8",
    )


def _minimal_payload(certified: str = "aaaa1111") -> dict:
    return {
        "certification_mode": "inherited_runtime_audit",
        "certified_commit_sha": certified,
        "allowed_post_certification_blob_shas": {},
        "allowed_unhashed_artifact_paths": [],
        "audited_factor_canonicals": [],
        "audited_factor_implementation_hashes": {},
    }


# ---------------------------------------------------------------------------
# FE-P0-024: reject unhashed artifacts
# ---------------------------------------------------------------------------


def test_p0_024_unhashed_artifacts_rejected(tmp_path, monkeypatch):
    """FE-P0-024: production evidence must reject any unhashed artifact paths."""
    manifest_path = tmp_path / "scm_manifest.json"
    _write_manifest(manifest_path, build="aaaa1111", certified="aaaa1111", ancestor=True)
    _patch_heavy_deps(monkeypatch)
    monkeypatch.setattr(foe, "_git_available", lambda: False)
    monkeypatch.setattr(sm, "SCM_MANIFEST_PATH", manifest_path)

    payload = _minimal_payload()
    payload["allowed_unhashed_artifact_paths"] = [
        "factor_engine/backend/some_file.py",
        "factor_engine/cleaned_operators/another.py",
    ]

    errors = foe._inherited_validation_errors(payload)
    assert any("unhashed artifacts" in e for e in errors), errors
    assert any("some_file.py" in e for e in errors), errors
    assert any("another.py" in e for e in errors), errors


def test_p0_024_hashed_artifacts_accepted(tmp_path, monkeypatch):
    """FE-P0-024: properly hashed artifacts should pass."""
    manifest_path = tmp_path / "scm_manifest.json"
    _write_manifest(manifest_path, build="aaaa1111", certified="aaaa1111", ancestor=True)
    _patch_heavy_deps(monkeypatch)
    monkeypatch.setattr(foe, "_git_available", lambda: False)
    monkeypatch.setattr(sm, "SCM_MANIFEST_PATH", manifest_path)

    payload = _minimal_payload()
    payload["allowed_post_certification_blob_shas"] = {
        "factor_engine/backend/some_file.py": "abc123def456",
    }
    payload["allowed_unhashed_artifact_paths"] = []

    errors = foe._inherited_validation_errors(payload)
    assert not any("unhashed artifacts" in e for e in errors), errors


# ---------------------------------------------------------------------------
# FE-P0-023: reject count-only legacy evidence
# ---------------------------------------------------------------------------


def test_p0_023_count_only_legacy_rejected(tmp_path, monkeypatch):
    """FE-P0-023: production evidence must have exact canonical set, not just count."""
    manifest_path = tmp_path / "scm_manifest.json"
    _write_manifest(manifest_path, build="aaaa1111", certified="aaaa1111", ancestor=True)
    _patch_heavy_deps(monkeypatch)
    monkeypatch.setattr(foe, "_git_available", lambda: False)
    monkeypatch.setattr(sm, "SCM_MANIFEST_PATH", manifest_path)

    # Legacy payload with count but no exact canonical list
    payload = _minimal_payload()
    payload["audited_factor_canonicals"] = []  # Empty = legacy
    payload["audited_factor_target_count"] = 100
    payload["audited_nonprimitive_target_count"] = 50

    errors = foe._inherited_validation_errors(payload)
    assert any("exact canonical set" in e for e in errors), errors
    assert any("count-only" in e for e in errors), errors


def test_p0_023_exact_canonical_set_required(tmp_path, monkeypatch):
    """FE-P0-023: production evidence must have exact canonical list."""
    manifest_path = tmp_path / "scm_manifest.json"
    _write_manifest(manifest_path, build="aaaa1111", certified="aaaa1111", ancestor=True)
    monkeypatch.setattr(foe, "_production_sets", lambda: ({"op_a", "op_b"}, set()))
    monkeypatch.setattr(ep, "evidence_artifact_valid", lambda *a, **k: True)
    monkeypatch.setattr(ep, "implementation_hashes_for", lambda c: {"hash": "abc"})
    monkeypatch.setattr(foe, "_git_available", lambda: False)
    monkeypatch.setattr(sm, "SCM_MANIFEST_PATH", manifest_path)

    # Payload with exact canonical set
    payload = _minimal_payload()
    payload["audited_factor_canonicals"] = ["op_a", "op_b"]
    payload["audited_factor_implementation_hashes"] = {
        "op_a": {"hash": "abc"},
        "op_b": {"hash": "abc"},
    }

    errors = foe._inherited_validation_errors(payload)
    assert not any("count-only" in e for e in errors), errors


def test_p0_023_canonical_substitution_detected(tmp_path, monkeypatch):
    """FE-P0-023: same count but different canonicals must be detected."""
    manifest_path = tmp_path / "scm_manifest.json"
    _write_manifest(manifest_path, build="aaaa1111", certified="aaaa1111", ancestor=True)
    # Current has {op_a, op_b, op_c}
    monkeypatch.setattr(foe, "_production_sets", lambda: ({"op_a", "op_b", "op_c"}, set()))
    monkeypatch.setattr(ep, "evidence_artifact_valid", lambda *a, **k: True)
    monkeypatch.setattr(ep, "implementation_hashes_for", lambda c: {"hash": "xyz"})
    monkeypatch.setattr(foe, "_git_available", lambda: False)
    monkeypatch.setattr(sm, "SCM_MANIFEST_PATH", manifest_path)

    # Certified had {op_a, op_b, op_d} - same count (3) but different set
    payload = _minimal_payload()
    payload["audited_factor_canonicals"] = ["op_a", "op_b", "op_d"]
    payload["audited_factor_implementation_hashes"] = {
        "op_a": {"hash": "xyz"},
        "op_b": {"hash": "xyz"},
        "op_d": {"hash": "xyz"},
    }

    errors = foe._inherited_validation_errors(payload)
    assert any("target set changed" in e for e in errors), errors


def test_p0_023_missing_implementation_hashes_rejected(tmp_path, monkeypatch):
    """FE-P0-023: production evidence must have per-canonical implementation hashes."""
    manifest_path = tmp_path / "scm_manifest.json"
    _write_manifest(manifest_path, build="aaaa1111", certified="aaaa1111", ancestor=True)
    monkeypatch.setattr(foe, "_production_sets", lambda: ({"op_a"}, set()))
    monkeypatch.setattr(ep, "evidence_artifact_valid", lambda *a, **k: True)
    monkeypatch.setattr(ep, "implementation_hashes_for", lambda c: {"hash": "abc"})
    monkeypatch.setattr(foe, "_git_available", lambda: False)
    monkeypatch.setattr(sm, "SCM_MANIFEST_PATH", manifest_path)

    payload = _minimal_payload()
    payload["audited_factor_canonicals"] = ["op_a"]
    payload["audited_factor_implementation_hashes"] = {}  # Missing

    errors = foe._inherited_validation_errors(payload)
    assert any("implementation hashes" in e for e in errors), errors


# ---------------------------------------------------------------------------
# FE-P0-038: reject "unknown" hashes
# ---------------------------------------------------------------------------


def test_p0_038_unknown_blob_hash_rejected(tmp_path, monkeypatch):
    """FE-P0-038: production evidence must reject blob_sha="unknown"."""
    manifest_path = tmp_path / "scm_manifest.json"
    _write_manifest(manifest_path, build="aaaa1111", certified="aaaa1111", ancestor=True)
    _patch_heavy_deps(monkeypatch)
    monkeypatch.setattr(foe, "_git_available", lambda: False)
    monkeypatch.setattr(sm, "SCM_MANIFEST_PATH", manifest_path)

    payload = _minimal_payload()
    payload["allowed_post_certification_blob_shas"] = {
        "factor_engine/backend/some_file.py": "unknown",
    }

    errors = foe._inherited_validation_errors(payload)
    # Look for the specific unknown hash error (not blob mismatch)
    matching = [e for e in errors if "unknown/missing hash in allowed_post_certification_blob_shas" in e]
    # Exactly one error per bad hash, no duplicates
    assert len(matching) == 1, f"Expected exactly 1 error, got {len(matching)}: {matching}"


def test_p0_038_empty_blob_hash_rejected(tmp_path, monkeypatch):
    """FE-P0-038: production evidence must reject empty blob hash."""
    manifest_path = tmp_path / "scm_manifest.json"
    _write_manifest(manifest_path, build="aaaa1111", certified="aaaa1111", ancestor=True)
    _patch_heavy_deps(monkeypatch)
    monkeypatch.setattr(foe, "_git_available", lambda: False)
    monkeypatch.setattr(sm, "SCM_MANIFEST_PATH", manifest_path)

    payload = _minimal_payload()
    payload["allowed_post_certification_blob_shas"] = {
        "factor_engine/backend/some_file.py": "",
    }

    errors = foe._inherited_validation_errors(payload)
    # Look for the specific unknown/missing hash error (not blob mismatch)
    matching = [e for e in errors if "unknown/missing hash in allowed_post_certification_blob_shas" in e]
    # Exactly one error per bad hash
    assert len(matching) == 1, f"Expected exactly 1 error, got {len(matching)}: {matching}"


def test_p0_038_unknown_emitter_hash_rejected(tmp_path, monkeypatch):
    """FE-P0-038: production evidence must reject emitter_hash="unknown"."""
    manifest_path = tmp_path / "scm_manifest.json"
    _write_manifest(manifest_path, build="aaaa1111", certified="aaaa1111", ancestor=True)
    _patch_heavy_deps(monkeypatch)
    monkeypatch.setattr(foe, "_git_available", lambda: False)
    monkeypatch.setattr(sm, "SCM_MANIFEST_PATH", manifest_path)

    payload = _minimal_payload()
    payload["emitter_hashes"] = {
        "implementation_hash_polars_emitter": "unknown",
    }

    errors = foe._inherited_validation_errors(payload)
    matching = [e for e in errors if "unknown" in e.lower() and "emitter" in e.lower() and "polars_emitter" in e]
    # Exactly one error for the bad emitter hash
    assert len(matching) == 1, f"Expected exactly 1 error, got {len(matching)}: {matching}"
    assert "emitter_hashes" in matching[0]


def test_p0_038_valid_hashes_accepted(tmp_path, monkeypatch):
    """FE-P0-038: valid non-unknown hashes should pass."""
    manifest_path = tmp_path / "scm_manifest.json"
    _write_manifest(manifest_path, build="aaaa1111", certified="aaaa1111", ancestor=True)
    monkeypatch.setattr(foe, "_production_sets", lambda: ({"op_a"}, set()))
    monkeypatch.setattr(ep, "evidence_artifact_valid", lambda *a, **k: True)
    monkeypatch.setattr(ep, "implementation_hashes_for", lambda c: {"hash": "abc123"})
    monkeypatch.setattr(foe, "_git_available", lambda: False)
    monkeypatch.setattr(sm, "SCM_MANIFEST_PATH", manifest_path)

    payload = _minimal_payload()
    payload["audited_factor_canonicals"] = ["op_a"]
    payload["audited_factor_implementation_hashes"] = {"op_a": {"hash": "abc123"}}
    payload["allowed_post_certification_blob_shas"] = {
        "factor_engine/backend/some_file.py": "abc123def456",
    }
    payload["emitter_hashes"] = {
        "implementation_hash_polars_emitter": "valid_hash_123",
    }

    errors = foe._inherited_validation_errors(payload)
    # Should not have any unknown/missing hash errors
    assert not any("unknown" in e.lower() and ("hash" in e.lower() or "emitter" in e.lower()) for e in errors), errors


def test_p0_038_missing_emitter_hashes_rejected(tmp_path, monkeypatch):
    """FE-P0-038: production evidence must have emitter_hashes."""
    manifest_path = tmp_path / "scm_manifest.json"
    _write_manifest(manifest_path, build="aaaa1111", certified="aaaa1111", ancestor=True)
    _patch_heavy_deps(monkeypatch)
    monkeypatch.setattr(foe, "_git_available", lambda: False)
    monkeypatch.setattr(sm, "SCM_MANIFEST_PATH", manifest_path)

    payload = _minimal_payload()
    # Missing emitter_hashes entirely
    payload.pop("emitter_hashes", None)

    errors = foe._inherited_validation_errors(payload)
    matching = [e for e in errors if "emitter_hashes" in e and "missing" in e.lower()]
    assert len(matching) == 1, f"Expected exactly 1 error for missing emitter_hashes, got {len(matching)}: {matching}"


def test_p0_038_unknown_implementation_hash_in_factor_hashes_rejected(tmp_path, monkeypatch):
    """FE-P0-038: production evidence must reject unknown hash in audited_factor_implementation_hashes."""
    manifest_path = tmp_path / "scm_manifest.json"
    _write_manifest(manifest_path, build="aaaa1111", certified="aaaa1111", ancestor=True)
    monkeypatch.setattr(foe, "_production_sets", lambda: ({"op_a"}, set()))
    monkeypatch.setattr(ep, "evidence_artifact_valid", lambda *a, **k: True)
    monkeypatch.setattr(ep, "implementation_hashes_for", lambda c: {"hash": "abc123"})
    monkeypatch.setattr(foe, "_git_available", lambda: False)
    monkeypatch.setattr(sm, "SCM_MANIFEST_PATH", manifest_path)

    payload = _minimal_payload()
    payload["audited_factor_canonicals"] = ["op_a"]
    payload["audited_factor_implementation_hashes"] = {
        "op_a": {"hash": "unknown"}  # Bad hash in implementation hashes
    }
    payload["emitter_hashes"] = {"polars": "valid"}

    errors = foe._inherited_validation_errors(payload)
    matching = [e for e in errors if "audited_factor_implementation_hashes" in e and "op_a" in e and "unknown" in e.lower()]
    # Exactly one error for the unknown hash in implementation hashes
    assert len(matching) == 1, f"Expected exactly 1 error, got {len(matching)}: {matching}"


def test_p0_038_multiple_unknown_hashes_no_duplicates(tmp_path, monkeypatch):
    """FE-P0-038: multiple unknown hashes should produce exactly one error each, no duplicates."""
    manifest_path = tmp_path / "scm_manifest.json"
    _write_manifest(manifest_path, build="aaaa1111", certified="aaaa1111", ancestor=True)
    monkeypatch.setattr(foe, "_production_sets", lambda: ({"op_a"}, set()))
    monkeypatch.setattr(ep, "evidence_artifact_valid", lambda *a, **k: True)
    monkeypatch.setattr(ep, "implementation_hashes_for", lambda c: {"hash": "abc123"})
    monkeypatch.setattr(foe, "_git_available", lambda: False)
    monkeypatch.setattr(sm, "SCM_MANIFEST_PATH", manifest_path)

    payload = _minimal_payload()
    payload["audited_factor_canonicals"] = ["op_a"]
    payload["audited_factor_implementation_hashes"] = {
        "op_a": {"hash": "unknown"}
    }
    payload["allowed_post_certification_blob_shas"] = {
        "file1.py": "unknown",
        "file2.py": "",
    }
    payload["emitter_hashes"] = {
        "polars": "unknown",
        "duckdb": "",
    }

    errors = foe._inherited_validation_errors(payload)

    # Count errors by category
    impl_hash_errors = [e for e in errors if "audited_factor_implementation_hashes" in e and "unknown" in e.lower()]
    blob_errors = [e for e in errors if "allowed_post_certification_blob_shas" in e and ("unknown" in e.lower() or "missing" in e.lower())]
    emitter_errors = [e for e in errors if "emitter_hashes" in e and ("unknown" in e.lower() or "missing" in e.lower())]

    # Exactly one error per bad hash, no duplicates
    assert len(impl_hash_errors) == 1, f"Expected 1 impl hash error, got {len(impl_hash_errors)}: {impl_hash_errors}"
    assert len(blob_errors) == 2, f"Expected 2 blob errors, got {len(blob_errors)}: {blob_errors}"
    assert len(emitter_errors) == 2, f"Expected 2 emitter errors, got {len(emitter_errors)}: {emitter_errors}"


# ---------------------------------------------------------------------------
# Integration: all three fixes together
# ---------------------------------------------------------------------------


def test_all_three_fixes_enforced_together(tmp_path, monkeypatch):
    """Integration test: all three P0 fixes must be enforced together."""
    manifest_path = tmp_path / "scm_manifest.json"
    _write_manifest(manifest_path, build="aaaa1111", certified="aaaa1111", ancestor=True)
    _patch_heavy_deps(monkeypatch)
    monkeypatch.setattr(foe, "_git_available", lambda: False)
    monkeypatch.setattr(sm, "SCM_MANIFEST_PATH", manifest_path)

    # Payload violating all three
    payload = _minimal_payload()
    payload["allowed_unhashed_artifact_paths"] = ["factor_engine/bad.py"]  # P0-024
    payload["audited_factor_canonicals"] = []  # P0-023: legacy count-only
    payload["audited_factor_target_count"] = 10
    payload["emitter_hashes"] = {"polars_emitter": "unknown"}  # P0-038

    errors = foe._inherited_validation_errors(payload)

    # Must have errors for all three
    assert any("unhashed" in e for e in errors), "P0-024 not enforced"
    assert any("count-only" in e for e in errors), "P0-023 not enforced"
    assert any("unknown" in e.lower() and "emitter" in e.lower() for e in errors), "P0-038 not enforced"


def test_mutation_of_payload_does_not_bypass_checks(tmp_path, monkeypatch):
    """Ensure validation cannot be bypassed by payload mutation."""
    manifest_path = tmp_path / "scm_manifest.json"
    _write_manifest(manifest_path, build="aaaa1111", certified="aaaa1111", ancestor=True)
    _patch_heavy_deps(monkeypatch)
    monkeypatch.setattr(foe, "_git_available", lambda: False)
    monkeypatch.setattr(sm, "SCM_MANIFEST_PATH", manifest_path)

    payload = _minimal_payload()
    payload["allowed_unhashed_artifact_paths"] = ["bad.py"]

    # Call validation
    errors1 = foe._inherited_validation_errors(payload)
    assert any("unhashed" in e for e in errors1)

    # Mutate payload after validation
    payload["allowed_unhashed_artifact_paths"] = []

    # Call again - should still enforce on the current state
    errors2 = foe._inherited_validation_errors(payload)
    assert not any("unhashed" in e for e in errors2)
