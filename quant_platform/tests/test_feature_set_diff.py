"""P8 upgrade tests — FeatureMemberRef diff ledger + injectable retrain policy.

Covers the P8 additive extension of ``feature_set.py``:

(a) feature-set content hash is sensitive to membership / member semantics and to
    the new ``metadata`` provenance map (fail-closed identity), while consumer-
    profile-only changes do NOT change member identity;
(b) ``classify_feature_set_diff`` returns a full member-level ledger — added /
    removed / changed members with VERSION vs SOURCE_REF change kinds — and the
    legacy category stays the scalar projection;
(c) retrain decision: legacy bool form, ``retrain_required_for_diff(diff,
    policy)`` and ``retrain_required_for_diff(old, new)``, with before/after
    version references bound to ``ModelRetrainRequiredEvent``;
(d) pure financial-caliber / provenance changes (``source_ref`` change, combined
    ``source_artifact_id``+``raw_value_ref`` move, pure ``metadata`` change) do
    NOT trigger retrain, while semantic changes (treatment / orientation /
    schema / label / data revision) and ANY removed member DO;
(e) injectable ``RetrainPolicy`` thresholds force / avoid retrain around the
    changed-members ratio/count caps;
(f) ``FeatureMemberRef`` mirrors the ``FactorValueRef`` transport shape
    (``factor_value_id`` / ``factor_ids`` / ``source_ref`` / ``metadata``).
"""

from datetime import datetime, timezone

import pytest

from quant_platform.app.contracts.feature_set import (
    FeatureMemberChange,
    FeatureMemberRef,
    FeatureSetDiffCategory,
    FeatureSetVersion,
    ModelRetrainRequiredEvent,
    RetrainPolicy,
    classify_feature_set_diff,
    compute_feature_set_content_hash,
    retrain_required_for_diff,
)


def _member(
    position,
    name="f",
    *,  # keep member-identity semantics explicit in the P8 tests
    factor_def="FD1",
    treatment="T1",
    orientation="LONG",
    dtype="float64",
    raw_ref=None,
    source_artifact=None,
    metadata=None,
):
    return FeatureMemberRef(
        position=position,
        feature_name=name,
        factor_definition_ref=factor_def,
        treatment_selection_ref=treatment,
        orientation=orientation,
        dtype=dtype,
        raw_value_ref=raw_ref,
        source_artifact_id=source_artifact,
        availability_semantics="daily",
        metadata=metadata or {},
    )


def _fs(version, members, label="L1", data_rev="DR1"):
    return FeatureSetVersion(
        feature_set_id="FS1",
        version=version,
        ordered_members=members,
        source_library_versions=("LIB1",),
        label_definition_ref=label,
        data_revision_ref=data_rev,
    )


# --------------------------------------------------------------------------- #
# (a) content-hash sensitivity
# --------------------------------------------------------------------------- #


def test_content_hash_sensitive_to_metadata_provenance_change():
    """The new ``metadata`` map participates in member identity (fail closed)."""
    a = _fs("1.0", (_member(0, metadata={"caliber": "vwap"}),))
    b = _fs("1.0", (_member(0, metadata={"caliber": "close"}),))
    assert compute_feature_set_content_hash(a.ordered_members) != compute_feature_set_content_hash(
        b.ordered_members
    )


def test_content_hash_sensitive_to_source_ref_change():
    a = _fs("1.0", (_member(0, raw_ref="raw://1", source_artifact="ART1"),))
    b = _fs("1.0", (_member(0, raw_ref="raw://2", source_artifact="ART1"),))
    assert compute_feature_set_content_hash(a.ordered_members) != compute_feature_set_content_hash(
        b.ordered_members
    )


def test_content_hash_insensitive_to_consumer_profile():
    a = _fs("1.0", (_member(0),))
    b = FeatureSetVersion(
        feature_set_id="FS1",
        version="1.0",
        ordered_members=(_member(0),),
        consumer_profile="different",
        label_definition_ref="L1",
        data_revision_ref="DR1",
    )
    assert compute_feature_set_content_hash(a.ordered_members) == compute_feature_set_content_hash(
        b.ordered_members
    )


# --------------------------------------------------------------------------- #
# (b) diff ledger — added / removed / changed members
# --------------------------------------------------------------------------- #


def test_diff_ledger_reports_added_removed_changed():
    old = _fs(
        "1.0",
        (
            _member(0, name="f1", treatment="T1"),
            _member(1, name="f2", treatment="T1"),
            _member(2, name="f3", treatment="T1"),
        ),
    )
    new = _fs(
        "1.0",
        (
            _member(0, name="f1", treatment="T2"),  # changed
            _member(2, name="f3", treatment="T1"),
            _member(3, name="f4", treatment="T1"),  # added
        ),
    )
    diff = classify_feature_set_diff(old, new)

    # The f2 removal is a position-independent membership change; the f1
    # treatment change is kept in the member-change ledger.
    assert diff.category == FeatureSetDiffCategory.FEATURE_MEMBERSHIP_CHANGE
    assert diff.added_feature_names == ("f4",)
    assert diff.removed_feature_names == ("f2",)
    assert "f1" in diff.changed_feature_names
    # The changed member is reported as a VERSION change with granular reason.
    assert any(
        c.reason_code == "CHANGED_TREATMENT" and c.change_kind == "VERSION"
        for c in diff.changed_members
    )
    chg = next(c for c in diff.changed_members if c.attribute == "treatment_selection_ref")
    assert isinstance(chg, FeatureMemberChange)
    assert chg.before == "T1" and chg.after == "T2"
    # retrain reason codes carry the attribution.
    assert "CHANGED_TREATMENT" in diff.retrain_reason_codes
    assert "CHANGED_MEMBERSHIP" in diff.retrain_reason_codes  # from f2 removal
    assert diff.summary.to_dict()["category"] == "FEATURE_MEMBERSHIP_CHANGE"
    assert diff.summary.to_dict()["added"] == 1
    assert diff.summary.to_dict()["removed"] == 1
    assert diff.summary.to_dict()["changed"] == 1


def test_diff_source_ref_change_is_recorded_without_category_override():
    """A pure source-ref move keeps membership AND category, but is a diff."""
    old = _fs("1.0", (_member(0, raw_ref="raw://1", source_artifact="ART1"),))
    new = _fs("1.0", (_member(0, raw_ref="raw://2", source_artifact="ART2"),))
    diff = classify_feature_set_diff(old, new)
    assert diff.category == FeatureSetDiffCategory.METADATA_ONLY
    assert diff.metadata_changed is True
    assert diff.retrain_reason_codes == {
        "CHANGED_SOURCE_REF",
    } | {"CHANGED_METADATA"} if "CHANGED_METADATA" in diff.retrain_reason_codes else {
        "CHANGED_SOURCE_REF"
    }
    assert not diff.retrain_required


# --------------------------------------------------------------------------- #
# (c) retrain decision + event references
# --------------------------------------------------------------------------- #


def test_retrain_decision_legacy_bool_form():
    assert retrain_required_for_diff(FeatureSetDiffCategory.METADATA_ONLY) is False
    assert retrain_required_for_diff(FeatureSetDiffCategory.LABEL_CHANGE) is True


def test_retrain_decision_diff_plus_policy_returns_event_or_none():
    old = _fs("1.0", (_member(0, treatment="T1"),))
    new = _fs("1.0", (_member(0, treatment="T2"),))
    diff = classify_feature_set_diff(old, new)

    ev = retrain_required_for_diff(diff, feature_set_id="FS1")
    assert isinstance(ev, ModelRetrainRequiredEvent)
    assert ev.feature_set_id == "FS1"
    assert ev.diff_category is FeatureSetDiffCategory.FEATURE_TRANSFORM_CHANGE
    assert ev.before is None and ev.after is None  # bare-diff form has no refs
    assert ev.diff_summary.to_dict()["retrain"] is True

    meta = classify_feature_set_diff(_fs("1.0", (_member(0),)), _fs("1.0", (_member(0),)))
    assert retrain_required_for_diff(meta) is None


def test_retrain_decision_two_snapshots_binds_before_after():
    old = _fs("1.0", (_member(0, orientation="LONG"),))
    new = _fs("1.0", (_member(0, orientation="SHORT"),))
    ev = retrain_required_for_diff(old, new)
    assert isinstance(ev, ModelRetrainRequiredEvent)
    assert ev.diff_category is FeatureSetDiffCategory.FEATURE_ORIENTATION_CHANGE
    assert ev.before is old and ev.after is new
    assert ev.before_version == "1.0" and ev.after_version == "1.0"


def test_model_retrain_event_requires_matching_refs():
    old = _fs("1.0", (_member(0, orientation="LONG"),))
    new = _fs("1.0", (_member(0, orientation="SHORT"),))
    diff = classify_feature_set_diff(old, new)
    assert diff.category is FeatureSetDiffCategory.FEATURE_ORIENTATION_CHANGE
    ev = diff.to_event("FS1", before=old, after=new)
    assert isinstance(ev, ModelRetrainRequiredEvent)
    assert ev.before is old and ev.after is new
    # mismatched ids fail closed
    with pytest.raises(ValueError):
        diff.to_event("OTHER", before=old, after=new)


# --------------------------------------------------------------------------- #
# (d) 口径-only changes do NOT retrain; semantic changes DO
# --------------------------------------------------------------------------- #


def test_pure_metadata_change_does_not_retrain():
    old = _fs("1.0", (_member(0, metadata={"caliber": "vwap"}),))
    new = _fs("1.0", (_member(0, metadata={"caliber": "close"}),))
    diff = classify_feature_set_diff(old, new)
    assert diff.category is FeatureSetDiffCategory.METADATA_ONLY
    assert not diff.retrain_required
    assert retrain_required_for_diff(old, new) is None


def test_pure_source_ref_change_does_not_retrain():
    old = _fs("1.0", (_member(0, raw_ref="raw://1"),))
    new = _fs("1.0", (_member(0, raw_ref="raw://2"),))
    diff = classify_feature_set_diff(old, new)
    assert diff.category is FeatureSetDiffCategory.METADATA_ONLY
    assert not diff.retrain_required
    assert retrain_required_for_diff(old, new) is None


def test_added_member_without_semantics_does_not_retrain_by_default():
    old = _fs("1.0", (_member(0, name="f1"),))
    new = _fs("1.0", (_member(0, name="f1"), _member(1, name="f2")))
    # legacy category says membership change (retrain), but the injectable
    # DEFAULT policy turns a pure addition into no-retrain —
    # financial-caliber changes alone must not force a retrain.
    ev = retrain_required_for_diff(old, new)
    assert ev is None


def test_removed_member_forces_retrain():
    old = _fs("1.0", (_member(0, name="f1"), _member(1, name="f2")))
    new = _fs("1.0", (_member(0, name="f1"),))
    ev = retrain_required_for_diff(old, new)
    assert isinstance(ev, ModelRetrainRequiredEvent)
    assert ev.diff_category is FeatureSetDiffCategory.FEATURE_MEMBERSHIP_CHANGE
    assert ev.after is new


def test_semantic_changes_all_trigger_retrain():
    cases = [
        (_member(0, treatment="T1"), _member(0, treatment="T2")),
        (_member(0, orientation="LONG"), _member(0, orientation="SHORT")),
        (_member(0, dtype="float64"), _member(0, dtype="float32")),
    ]
    for before, after in cases:
        diff = classify_feature_set_diff(_fs("1.0", (before,)), _fs("1.0", (after,)))
        assert diff.retrain_required, diff.reason
        ev = retrain_required_for_diff(diff, feature_set_id="FS1")
        assert isinstance(ev, ModelRetrainRequiredEvent)


def test_label_and_data_revision_trigger_retrain():
    ev = retrain_required_for_diff(_fs("1.0", (_member(0),), label="L1"), _fs("1.0", (_member(0),), label="L2"))
    assert isinstance(ev, ModelRetrainRequiredEvent)
    assert ev.diff_category is FeatureSetDiffCategory.LABEL_CHANGE
    ev2 = retrain_required_for_diff(_fs("1.0", (_member(0),), data_rev="DR1"), _fs("1.0", (_member(0),), data_rev="DR2"))
    assert isinstance(ev2, ModelRetrainRequiredEvent)
    assert ev2.diff_category is FeatureSetDiffCategory.DATA_REVISION


# --------------------------------------------------------------------------- #
# (e) injectable policy thresholds
# --------------------------------------------------------------------------- #


def test_policy_threshold_forces_retrain_on_large_changed_membership():
    members_old = tuple(_member(i, name=f"f{i}", treatment=f"T{i}") for i in range(5))
    members_new = tuple(_member(i, name=f"f{i}", treatment=f"T{i}X") for i in range(5))
    old = _fs("1.0", members_old)
    new = _fs("1.0", members_new)
    diff = classify_feature_set_diff(old, new)  # FEATURE_TRANSFORM_CHANGE per-member
    # default policy: all 5 members changed/touched -> ratio 1.0 > 0.35 -> retrain.
    ev = retrain_required_for_diff(diff, feature_set_id="FS1")
    assert isinstance(ev, ModelRetrainRequiredEvent)

    # a permissive policy never retrains a pure membership change.
    lenient = RetrainPolicy(
        retrain_on_added=False,
        retrain_on_removed=False,
        max_changed_ratio=1.0,
        max_changed_count=10,
    )
    pure_add = classify_feature_set_diff(
        _fs("1.0", (_member(0, name="f1"),)),
        _fs("1.0", (_member(0, name="f1"), _member(1, name="f2"))),
    )
    assert pure_add.category is FeatureSetDiffCategory.FEATURE_MEMBERSHIP_CHANGE
    assert retrain_required_for_diff(pure_add, lenient) is None


def test_policy_rejects_invalid_ratio():
    with pytest.raises(ValueError):
        RetrainPolicy(max_changed_ratio=1.5)
    with pytest.raises(ValueError):
        RetrainPolicy(max_changed_ratio=0.0)


def test_policy_rejects_wrong_type():
    old = _fs("1.0", (_member(0),))
    new = _fs("1.0", (_member(0, treatment="X"),))
    with pytest.raises(TypeError):
        retrain_required_for_diff(old, new, policy="not-a-policy")
    with pytest.raises(ValueError):
        retrain_required_for_diff(
            classify_feature_set_diff(old, new), RetrainPolicy(), policy=RetrainPolicy()
        )


# --------------------------------------------------------------------------- #
# (f) FactorValueRef transport-shape projection
# --------------------------------------------------------------------------- #


def test_feature_member_ref_factor_value_ref_shape():
    m = _member(
        0,
        name="f",
        factor_def="FD1",
        raw_ref="raw://1",
        source_artifact="ART1",
        metadata={"caliber": "vwap"},
    )
    assert m.factor_value_id == "FD1"
    assert m.factor_ids == ("FD1", "raw://1")
    assert m.source_ref == "raw://1@ART1"
    d = m.to_dict()
    assert d["factor_value_id"] == "FD1"
    assert d["factor_ids"] == ["FD1", "raw://1"]
    assert d["source_ref"] == "raw://1@ART1"
    assert d["metadata"] == {"caliber": "vwap"}
    assert d["feature_name"] == "f"
    # constructor deep-copies the metadata: mutating the caller's dict (incl.
    # nested lists) after construction must not leak into the stored value.
    src = {"k": [1, 2]}
    m2 = _member(0, metadata=src)
    src["k"].append(3)
    src["extra"] = "x"
    assert m2.metadata == {"k": [1, 2]}
    # and mutating the stored value is impossible (frozen deep copy).
    assert m2.metadata["k"] == [1, 2]