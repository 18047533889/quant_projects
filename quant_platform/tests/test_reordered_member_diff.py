from dataclasses import replace

import pytest

from quant_platform.app.contracts.feature_set import (
    FeatureMemberRef, FeatureSetVersion, classify_feature_set_diff,
    retrain_required_for_diff,
)


@pytest.mark.parametrize("insert", [False, True])
@pytest.mark.parametrize("attr,value,code", [
    ("treatment_selection_ref", "new", "CHANGED_TREATMENT"),
    ("raw_value_ref", "new", "CHANGED_SOURCE_REF"),
    ("metadata", {"note": "new"}, "CHANGED_METADATA"),
])
def test_retained_member_changes_survive_movement(insert, attr, value, code):
    a = FeatureMemberRef(0, "a", "fd-a")
    b = FeatureMemberRef(1, "b", "fd-b")
    before = FeatureSetVersion("fs", "1", (a, b))
    changed = replace(a, position=1, **{attr: value})
    first = FeatureMemberRef(0, "c", "fd-c") if insert else replace(b, position=0)
    members = (first, changed, replace(b, position=2)) if insert else (first, changed)
    after = FeatureSetVersion("fs", "2", members)
    diff = classify_feature_set_diff(before, after)
    assert code in diff.retrain_reason_codes
    matching = [c for c in diff.changed_members if c.reason_code == code]
    assert len(matching) == 1
    assert matching[0].member.feature_name == "a"
    event = retrain_required_for_diff(before, after)
    assert (event is not None) == (attr == "treatment_selection_ref")


@pytest.mark.parametrize("side", ["old", "new"])
def test_duplicate_identity_cannot_be_silently_collapsed(side):
    a = FeatureMemberRef(0, "a", "fd-a")
    good = FeatureSetVersion("fs", "1", (a,))
    ambiguous = FeatureSetVersion("fs", "2", (a, replace(a, position=1)))
    pair = (ambiguous, good) if side == "old" else (good, ambiguous)
    with pytest.raises(ValueError, match="duplicate.*identity"):
        classify_feature_set_diff(*pair)


def test_reordering_alone_does_not_invent_member_changes():
    a, b = FeatureMemberRef(0, "a", "fd-a"), FeatureMemberRef(1, "b", "fd-b")
    before = FeatureSetVersion("fs", "1", (a, b))
    after = FeatureSetVersion("fs", "2", (replace(b, position=0), replace(a, position=1)))
    assert classify_feature_set_diff(before, after).changed_members == ()
