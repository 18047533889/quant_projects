import json
from collections.abc import Mapping
from dataclasses import asdict

import pytest

from quant_platform.app.contracts.feature_set import (
    FeatureMemberRef, FeatureSetVersion, classify_feature_set_diff,
    retrain_required_for_diff,
)
from quant_platform.app.contracts.security import SecurityClassification


@pytest.mark.parametrize("classification", list(SecurityClassification))
def test_json_restore_preserves_security_type_and_identity(classification):
    member = FeatureMemberRef(0, "f", "fd", security_classification=classification)
    original = FeatureSetVersion("fs", "v1", (member,))
    payload = json.loads(json.dumps(asdict(original), default=lambda obj:
        dict(obj) if isinstance(obj, Mapping) else obj.value))
    payload["ordered_members"] = tuple(FeatureMemberRef(**m) for m in payload["ordered_members"])
    restored = FeatureSetVersion(**payload)
    assert restored.ordered_members[0].security_classification is classification
    assert restored == original
    assert restored.semantic_hash == original.semantic_hash
    assert classify_feature_set_diff(original, restored).changed_members == ()
    assert retrain_required_for_diff(original, restored) is None


@pytest.mark.parametrize("value", ["UNKNOWN", {}, [], True, 1])
def test_invalid_classification_is_rejected(value):
    with pytest.raises((TypeError, ValueError), match="security_classification"):
        FeatureMemberRef(0, "f", "fd", security_classification=value)
