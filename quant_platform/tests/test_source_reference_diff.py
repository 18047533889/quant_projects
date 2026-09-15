"""Source identity is the pair of raw and artifact refs, not their projection."""
import pytest

from quant_platform.app.contracts.feature_set import (
    FeatureMemberRef, FeatureSetVersion, classify_feature_set_diff,
    retrain_required_for_diff,
)


@pytest.mark.parametrize("old_refs,new_refs", [
    (("raw", "artifact1"), ("raw", "artifact2")),
    (("raw", None), ("raw", "artifact")),
    (("raw", "artifact"), ("raw", None)),
    (("same", None), (None, "same")),
    (("a@b", "c"), ("a", "b@c")),
])
def test_source_components_are_not_masked(old_refs, new_refs):
    def version(refs):
        return FeatureSetVersion("fs", "1", (FeatureMemberRef(
            0, "f", "fd", raw_value_ref=refs[0], source_artifact_id=refs[1],
        ),))
    before, after = version(old_refs), version(new_refs)
    diff = classify_feature_set_diff(before, after)
    changes = [c for c in diff.changed_members if c.change_kind == "SOURCE_REF"]
    assert len(changes) == 1
    assert "CHANGED_SOURCE_REF" in diff.retrain_reason_codes
    assert diff.metadata_changed
    assert retrain_required_for_diff(before, after) is None


def test_unchanged_source_pair_has_no_false_change():
    before = FeatureSetVersion("fs", "1", (FeatureMemberRef(
        0, "f", "fd", raw_value_ref="raw", source_artifact_id="artifact",
    ),))
    assert classify_feature_set_diff(before, before).changed_members == ()
