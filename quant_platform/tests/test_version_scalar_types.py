import pytest

from quant_platform.app.contracts.feature_set import FeatureSetVersion


@pytest.mark.parametrize("name", ["feature_set_id", "version", "consumer_profile",
                                     "label_definition_ref", "data_revision_ref"])
@pytest.mark.parametrize("value", [["mutable"], {"key": "value"}, 1, True])
def test_version_fields_reject_non_strings(name, value):
    kwargs = {"feature_set_id": "fs", "version": "1", name: value}
    with pytest.raises(TypeError, match=name):
        FeatureSetVersion(**kwargs)


def test_valid_optional_empty_values_remain_supported():
    version = FeatureSetVersion("fs", "1", consumer_profile="",
                                label_definition_ref=None, data_revision_ref="")
    assert version.consumer_profile == ""
    assert version.label_definition_ref is None
