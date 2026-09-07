import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.common.time_series import TSZScore


def _oracle(x, *, window, min_periods, ddof, includes_current_bar,
            nan_policy="propagate", zero_std_policy="zero"):
    finite = x.replace([np.inf, -np.inf], np.nan)
    stats = finite if includes_current_bar else finite.shift(1)
    mean = stats.rolling(window, min_periods=min_periods).mean()
    std = stats.rolling(window, min_periods=min_periods).std(ddof=ddof)
    out = (finite - mean) / std
    if nan_policy == "propagate":
        bad = stats.isna().astype(int).rolling(window, min_periods=1).sum() > 0
        out = out.mask(bad)
    out = out.mask(std.eq(0), 0.0 if zero_std_policy == "zero" else np.nan)
    return out.where(finite.notna())


@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("includes_current_bar", [True, False])
def test_pandas_matches_independent_oracle(ddof, includes_current_bar):
    x = pd.DataFrame({"a": [1.0, 2.0, 3.0, 8.0, 5.0]})
    got = TSZScore().calculate(
        x, window=3, min_periods=3, ddof=ddof,
        includes_current_bar=includes_current_bar, nan_policy="ignore",
    )
    expected = _oracle(
        x, window=3, min_periods=3, ddof=ddof,
        includes_current_bar=includes_current_bar, nan_policy="ignore",
    )
    pd.testing.assert_frame_equal(got, expected)


def test_nan_inf_and_constant_policies():
    x = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.inf, 5.0]})
    propagated = TSZScore().calculate(x, window=3, min_periods=2, nan_policy="propagate")
    omitted = TSZScore().calculate(x, window=3, min_periods=2, nan_policy="ignore")
    assert pd.isna(propagated.iloc[2, 0])
    assert pd.notna(omitted.iloc[2, 0])
    constant = pd.DataFrame({"a": [7.0, 7.0, 7.0]})
    assert TSZScore().calculate(constant, window=3, min_periods=3).iloc[-1, 0] == 0
    assert pd.isna(TSZScore().calculate(
        constant, window=3, min_periods=3, zero_std_policy="nan"
    ).iloc[-1, 0])


@pytest.mark.parametrize("kwargs", [
    {"ddof": 2}, {"ddof": True}, {"includes_current_bar": 1},
    {"min_periods": 4, "window": 3}, {"nan_policy": "mystery"},
    {"zero_std_policy": "epsilon"},
])
def test_invalid_parameter_domains_reject(kwargs):
    with pytest.raises((ValueError, TypeError)):
        TSZScore().calculate(pd.DataFrame({"a": [1.0, 2.0, 3.0]}), **kwargs)


def test_polars_native_contains_no_python_rolling_callback():
    from pathlib import Path
    source = Path(__file__).resolve().parents[2].joinpath(
        "cleaned_operators/common/polars_ts_stats.py"
    ).read_text(encoding="utf-8").split("class TSZScoreNative", 1)[1].split(
        "@register_operator", 1
    )[0]
    assert "rolling_map" not in source
    assert "to_numpy" not in source
