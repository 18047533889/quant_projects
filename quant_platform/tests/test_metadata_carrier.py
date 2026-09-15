import pytest

from quant_platform.app.contracts.feature_set import FeatureMemberRef


@pytest.mark.parametrize("value", [False, 0, "", [], (), [("source", "a"), ("source", "b")]])
def test_invalid_metadata_carrier_is_not_silently_normalized(value):
    with pytest.raises(TypeError, match="metadata"):
        FeatureMemberRef(0, "f", "fd", metadata=value)


def test_falsey_mapping_preserves_provenance():
    class FalseyDict(dict):
        def __bool__(self):
            return False
    member = FeatureMemberRef(0, "f", "fd", metadata=FalseyDict(source="actual"))
    assert member.metadata["source"] == "actual"


@pytest.mark.parametrize("value", [None, {}])
def test_legacy_empty_metadata_remains_supported(value):
    assert dict(FeatureMemberRef(0, "f", "fd", metadata=value).metadata) == {}
