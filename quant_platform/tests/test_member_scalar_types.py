import pytest

from quant_platform.app.contracts.feature_set import FeatureMemberRef


@pytest.mark.parametrize("value", [True, False, 0.5, float("nan"), float("inf"), "0", None])
def test_position_requires_integer(value):
    with pytest.raises(TypeError, match="position"):
        FeatureMemberRef(value, "f", "fd")


@pytest.mark.parametrize("name", ["feature_name", "factor_definition_ref",
    "raw_value_ref", "treatment_selection_ref", "treated_feature_ref",
    "orientation", "dtype", "channel", "timing_ref", "source_artifact_id",
    "availability_semantics"])
@pytest.mark.parametrize("value", [["mutable"], {"mutable": "value"}, 123, True])
def test_reference_fields_reject_non_strings(name, value):
    kwargs = {"position": 0, "feature_name": "f", "factor_definition_ref": "fd"}
    kwargs[name] = value
    with pytest.raises(TypeError, match=name):
        FeatureMemberRef(**kwargs)


def test_valid_optional_values_and_nonnegative_position():
    member = FeatureMemberRef(0, "f", "fd", raw_value_ref=None, channel="")
    assert member.position == 0
    with pytest.raises(ValueError, match="position"):
        FeatureMemberRef(-1, "f", "fd")
