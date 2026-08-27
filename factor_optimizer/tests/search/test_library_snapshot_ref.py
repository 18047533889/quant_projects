"""DLIB-FO-008: FO winner -> FA library-snapshot reference binding.

Covers:
- ``LibrarySnapshotRef`` contract (immutable, validated, content-hashed).
- ``TreatmentOptimizationResultArtifact`` binding the FA library snapshot the
  search operated against, so the winner is attributable to the exact library
  version and cannot silently drift onto a stale/unrelated snapshot.
- Round-trip serde (dict <-> artifact) preserving the new binding.
- Fail-closed behavior: ``require_library_snapshot_ref=True`` rejects an
  artifact without a snapshot ref; a bare-str ref is normalized to the
  canonical ``LibrarySnapshotRef`` form.
"""

import pytest

from factor_optimizer.contracts.library_snapshot_ref import LibrarySnapshotRef
from factor_optimizer.contracts.treatment_result import (
    TreatmentOptimizationResultArtifact,
)


def _snapshot(**overrides):
    defaults = dict(
        library_version_ref="lib-v42/abc",
        cluster_set_version_ref="csv-7/xyz",
        factor_set_version_ref="fs-9",
        snapshot_ref="snap-2026-08-01",
        universe_ref="uni-ashare",
    )
    defaults.update(overrides)
    return LibrarySnapshotRef(**defaults)


def _result(**overrides):
    defaults = dict(
        search_session_id="s1",
        source_factor_value_ref="sfv-1",
        raw_baseline_evidence_ref="raw-ev-1",
        factor_profile_ref="fp-1",
        treatment_search_space_ref="tss-1",
        transform_registry_snapshot_ref="trs-1",
        desirability_policy_ref="dp-1",
        winner_policy_ref="wp-1",
        split_plan_ref="sp-1",
        trial_ledger_ref="tl-1",
        all_trial_refs=("t1", "t2", "t3"),
        pareto_trial_refs=("t1", "t2"),
        multiplicity_ref="mult-1",
        selected_trial_ref="t2",
        uncertainty_evidence_ref="ue-1",
    )
    defaults.update(overrides)
    return TreatmentOptimizationResultArtifact(**defaults)


def test_library_snapshot_ref_contract_and_hash():
    snap = _snapshot()
    assert snap.library_version_ref == "lib-v42/abc"
    assert snap.cluster_set_version_ref == "csv-7/xyz"
    assert snap.content_hash
    # Same content -> same hash (order-independent via sort_keys).
    assert snap.content_hash == _snapshot().content_hash
    # Different ref -> different hash.
    assert snap.content_hash != _snapshot(library_version_ref="lib-v99").content_hash
    # Round-trip.
    restored = LibrarySnapshotRef.from_dict(snap.to_dict())
    assert restored.content_hash == snap.content_hash


def test_library_snapshot_ref_requires_library_version():
    with pytest.raises(ValueError, match="library_version_ref"):
        LibrarySnapshotRef(library_version_ref="")
    with pytest.raises(ValueError, match="must be a non-empty string"):
        LibrarySnapshotRef(library_version_ref=123)


def test_result_binds_library_snapshot_ref():
    snap = _snapshot()
    artifact = _result(library_snapshot_ref=snap)
    assert artifact.library_snapshot_ref is snap
    # The ref is part of the content hash.
    assert artifact.content_hash
    assert artifact.content_hash != _result().content_hash


def test_result_bare_str_ref_is_normalized_to_snapshot_ref():
    artifact = _result(library_snapshot_ref="lib-v42/abc")
    assert isinstance(artifact.library_snapshot_ref, LibrarySnapshotRef)
    assert artifact.library_snapshot_ref.library_version_ref == "lib-v42/abc"


def test_result_rejects_invalid_ref_type():
    with pytest.raises(TypeError, match="library_snapshot_ref"):
        _result(library_snapshot_ref=object())


def test_result_optional_by_default():
    artifact = _result()
    assert artifact.library_snapshot_ref is None


def test_result_fails_closed_when_required_ref_missing():
    with pytest.raises(ValueError, match="require_library_snapshot_ref"):
        _result(require_library_snapshot_ref=True)


def test_result_roundtrip_preserves_snapshot_binding():
    snap = _snapshot()
    artifact = _result(library_snapshot_ref=snap)
    restored = TreatmentOptimizationResultArtifact.from_dict(artifact.to_dict())
    assert isinstance(restored.library_snapshot_ref, LibrarySnapshotRef)
    assert restored.library_snapshot_ref.content_hash == snap.content_hash
    assert restored.content_hash == artifact.content_hash


def test_result_roundtrip_backward_compatible_without_snapshot():
    """Artifacts serialized before FO-008 lack the snapshot keys — must
    round-trip with the binding left None and no failure."""
    artifact = _result()
    data = artifact.to_dict()
    del data["library_snapshot_ref"]
    del data["require_library_snapshot_ref"]
    restored = TreatmentOptimizationResultArtifact.from_dict(data)
    assert restored.library_snapshot_ref is None
    assert restored.require_library_snapshot_ref is False


def test_result_tampering_with_snapshot_is_detected():
    """Changing the bound library ref after construction is a tamper — the
    content hash must detect it."""
    snap_a = _snapshot()
    snap_b = _snapshot(library_version_ref="lib-v99")
    artifact = _result(library_snapshot_ref=snap_a)
    object.__setattr__(artifact, "library_snapshot_ref", snap_b)
    with pytest.raises(ValueError, match="tampered"):
        artifact.verify()


def test_winner_to_library_linkage_integrity():
    """End-to-end: a winner result bound to a library snapshot carries a
    deterministic, attributable ref that FA can validate at assembly time
    (FA consumes the treatment artifact's content_hash as treatment_selection_ref;
    the FO library ref makes the snapshot attributable)."""
    snap = _snapshot(
        library_version_ref="lib-v42/abc",
        cluster_set_version_ref="csv-7/xyz",
        snapshot_ref="snap-2026-08-01",
    )
    artifact = _result(
        selected_trial_ref="t2",
        library_snapshot_ref=snap,
    )
    # The linkage ref is derived, stable, and attributable.
    linkage = artifact.library_snapshot_ref.to_dict()
    assert linkage["library_version_ref"] == "lib-v42/abc"
    assert linkage["cluster_set_version_ref"] == "csv-7/xyz"
    assert linkage["snapshot_ref"] == "snap-2026-08-01"
    # A different library version changes the linkage hash.
    assert snap.content_hash != _snapshot(library_version_ref="lib-v99").content_hash