from dataclasses import replace

import pytest

from quant_platform.app.contracts.feature_set import (
    FeatureMemberRef, FeatureSetVersion, classify_feature_set_diff,
    retrain_required_for_diff,
)


@pytest.mark.parametrize("global_field,global_code", [
    ("label_definition_ref", "CHANGED_LABEL"),
    ("data_revision_ref", "CHANGED_DATA_REVISION"),
])
@pytest.mark.parametrize("attr,value,member_code", [
    ("raw_value_ref", "raw2", "CHANGED_SOURCE_REF"),
    ("metadata", {"note": "new"}, "CHANGED_METADATA"),
    ("timing_ref", "timing2", "CHANGED_VERSION"),
])
def test_global_category_retains_member_audit(global_field, global_code, attr, value, member_code):
    member = FeatureMemberRef(0, "f", "fd")
    before = FeatureSetVersion("fs", "1", (member,))
    after = FeatureSetVersion("fs", "2", (replace(member, **{attr: value}),),
                              **{global_field: "new"})
    diff = classify_feature_set_diff(before, after)
    assert len(diff.changed_members) == 1
    assert {global_code, member_code} <= diff.retrain_reason_codes
    assert diff.changed_members[0].member == after.ordered_members[0]
    assert diff.summary.changed == 1
    event = retrain_required_for_diff(before, after)
    assert event is not None
    assert event.diff_summary.changed == 1
    assert {global_code, member_code} <= event.retrain_reason_codes
