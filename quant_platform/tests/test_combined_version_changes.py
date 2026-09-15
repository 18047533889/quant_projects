"""Version-level invalidation must survive the legacy first-match category."""
from dataclasses import replace

import pytest

from quant_platform.app.contracts.feature_set import (
    FeatureMemberRef, FeatureSetVersion, RetrainPolicy,
    classify_feature_set_diff, retrain_required_for_diff,
)


@pytest.mark.parametrize("change", ["add", "schema", "treatment", "metadata"])
@pytest.mark.parametrize("field,code", [
    ("label_definition_ref", "CHANGED_LABEL"),
    ("data_revision_ref", "CHANGED_DATA_REVISION"),
])
def test_global_change_not_masked_by_member_change(change, field, code):
    member = FeatureMemberRef(0, "a", "fd", dtype="float64")
    before = FeatureSetVersion("fs", "1", (member,))
    members = {
        "add": (member, FeatureMemberRef(1, "b", "fd2")),
        "schema": (replace(member, dtype="float32"),),
        "treatment": (replace(member, treatment_selection_ref="t2"),),
        "metadata": (replace(member, metadata={"note": "new"}),),
    }[change]
    after = FeatureSetVersion("fs", "2", members, **{field: "new"})
    policy = RetrainPolicy(max_changed_ratio=1.0, max_changed_count=10)
    diff = classify_feature_set_diff(before, after)
    assert code in diff.retrain_reason_codes
    for event in (
        retrain_required_for_diff(before, after, policy=policy),
        retrain_required_for_diff(diff, policy, feature_set_id="fs"),
    ):
        assert event is not None
        assert code in event.retrain_reason_codes
        assert event.diff_summary.retrain


def test_simultaneous_label_and_revision_keep_both_reasons():
    before = FeatureSetVersion("fs", "1", (FeatureMemberRef(0, "a", "fd"),))
    after = FeatureSetVersion("fs", "2", before.ordered_members,
                              label_definition_ref="L2", data_revision_ref="D2")
    diff = classify_feature_set_diff(before, after)
    assert {"CHANGED_LABEL", "CHANGED_DATA_REVISION"} <= diff.retrain_reason_codes


def test_pure_add_still_does_not_force_retrain():
    member = FeatureMemberRef(0, "a", "fd")
    before = FeatureSetVersion("fs", "1", (member,))
    after = FeatureSetVersion("fs", "2", (member, FeatureMemberRef(1, "b", "fd2")))
    assert retrain_required_for_diff(before, after) is None
