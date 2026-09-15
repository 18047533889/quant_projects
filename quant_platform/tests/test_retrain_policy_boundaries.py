"""Reject malformed retrain policies before they alter model decisions."""

import pytest

from quant_platform.app.contracts.feature_set import RetrainPolicy


@pytest.mark.parametrize("field", ["retrain_on_added", "retrain_on_removed"])
@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, [], {}])
def test_flags_require_actual_booleans(field, value):
    with pytest.raises(TypeError, match=field):
        RetrainPolicy(**{field: value})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 0, -1, 1.01])
def test_ratio_requires_finite_valid_range(value):
    with pytest.raises(ValueError, match="max_changed_ratio"):
        RetrainPolicy(max_changed_ratio=value)


@pytest.mark.parametrize("value", [True, False, "0.35", None])
def test_ratio_rejects_wrong_types(value):
    with pytest.raises(TypeError, match="max_changed_ratio"):
        RetrainPolicy(max_changed_ratio=value)


@pytest.mark.parametrize("value", [True, False, 1.5, float("nan"), float("inf"), "2", None])
def test_count_requires_actual_integer(value):
    with pytest.raises(TypeError, match="max_changed_count"):
        RetrainPolicy(max_changed_count=value)


def test_count_rejects_negative():
    with pytest.raises(ValueError, match="max_changed_count"):
        RetrainPolicy(max_changed_count=-1)


@pytest.mark.parametrize("ratio", [0.001, 0.35, 1, 1.0])
def test_valid_policy_boundaries_remain_supported(ratio):
    policy = RetrainPolicy(retrain_on_added=False, retrain_on_removed=True,
                           max_changed_ratio=ratio, max_changed_count=0)
    assert policy.max_changed_ratio == ratio
    assert policy.max_changed_count == 0
