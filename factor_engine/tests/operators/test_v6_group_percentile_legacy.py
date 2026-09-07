import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.common.group import GroupPercentile


def _panel():
    return pd.DataFrame([[3.0, 1.0, 2.0]], index=pd.to_datetime(["2026-01-02"]), columns=list("ABC"))


def test_legacy_group_percentile_global_fallback_is_defined_and_deterministic():
    out = GroupPercentile().calculate(_panel(), group=None, p=0.5, fallback_policy="global")
    expected = pd.DataFrame([[0.0, 1.0, 0.0]], index=_panel().index, columns=_panel().columns)
    pd.testing.assert_frame_equal(out, expected)


def test_legacy_group_percentile_missing_group_policies_are_explicit():
    x = _panel()
    nan_out = GroupPercentile().calculate(x, group=None, fallback_policy="nan")
    assert nan_out.isna().all().all()
    pd.testing.assert_frame_equal(
        GroupPercentile().calculate(x, group=None, fallback_policy="keep_original"), x
    )
    with pytest.raises(ValueError, match="missing group"):
        GroupPercentile().calculate(x, group=None, fallback_policy="error")


def test_legacy_group_percentile_rejects_bad_policy_probability_and_axes():
    x = _panel()
    with pytest.raises(ValueError, match="fallback_policy"):
        GroupPercentile().calculate(x, fallback_policy="typo")
    for p in (True, np.nan, -0.1, 1.1):
        with pytest.raises(ValueError, match="p must"):
            GroupPercentile().calculate(x, p=p, fallback_policy="global")
    shifted = pd.DataFrame([["g", "g", "g"]], index=pd.to_datetime(["2026-01-03"]), columns=x.columns)
    with pytest.raises(ValueError):
        GroupPercentile().calculate(x, group=shifted)
