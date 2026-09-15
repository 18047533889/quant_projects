from dataclasses import replace

import pytest

from quant_platform.app.contracts.feature_set import (
    FeatureMemberRef, FeatureSetVersion, RetrainPolicy,
    classify_feature_set_diff, retrain_required_for_diff,
)


@pytest.mark.parametrize("field", ["timing_ref", "treated_feature_ref", "availability_semantics"])
@pytest.mark.parametrize("evidence", [False, True])
def test_version_semantics_cannot_be_classified_away(field, evidence):
    member = FeatureMemberRef(0, "f", "fd")
    old = FeatureSetVersion("fs", "1", (member,))
    new = FeatureSetVersion("fs", "2", (replace(member, **{field: "new"}),),
                            source_library_versions=("lib2",) if evidence else ())
    diff = classify_feature_set_diff(old, new)
    assert "CHANGED_VERSION" in diff.retrain_reason_codes
    for event in (retrain_required_for_diff(old, new),
                  retrain_required_for_diff(diff, feature_set_id="fs")):
        assert event is not None
        assert event.diff_summary.retrain is True


@pytest.mark.parametrize("field", ["treatment_selection_ref", "orientation"])
def test_addition_does_not_waive_mandatory_semantic_change(field):
    member = FeatureMemberRef(0, "f", "fd")
    old = FeatureSetVersion("fs", "1", (member,))
    new = FeatureSetVersion("fs", "2", (
        replace(member, **{field: "new"}), FeatureMemberRef(1, "g", "fd2"),
    ))
    event = retrain_required_for_diff(old, new,
        policy=RetrainPolicy(max_changed_ratio=1.0, max_changed_count=10))
    assert event is not None
    assert event.diff_summary.retrain is True
