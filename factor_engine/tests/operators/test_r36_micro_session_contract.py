"""Session returns must not invent prices or silently discard recipe parameters."""
import numpy as np
import pandas as pd
import pytest
from factor_engine.cleaned_operators.microstructure.session import (
    pct_change_by_session, rolling_by_session, session_key_from_index,
)
from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry


@pytest.mark.parametrize("tz", [None, "Asia/Hong_Kong", "America/New_York"])
@pytest.mark.parametrize("periods", [1, 2])
def test_returns_preserve_gaps_and_session_identity(tz, periods):
    index = pd.date_range("2026-03-07 23:55", periods=12, freq="min", tz=tz)
    values = pd.Series([100., 102., np.nan, 106., 108., 200., 202., 204., np.nan, 208., 210., 212.], index=index)
    keys = session_key_from_index(index)
    assert keys.index.equals(index)
    assert keys.dtype == index.normalize().dtype
    expected = pd.Series(np.nan, index=index)
    for pos in range(periods, len(values)):
        if index[pos].normalize() == index[pos-periods].normalize():
            expected.iloc[pos] = values.iloc[pos] / values.iloc[pos-periods] - 1.
    pd.testing.assert_series_equal(pct_change_by_session(values, periods), expected)


def test_returns_range_index_does_not_fill_missing_prices():
    values = pd.Series([100., np.nan, 120., 126.])
    expected = pd.Series([np.nan, np.nan, np.nan, .05])
    pd.testing.assert_series_equal(pct_change_by_session(values), expected)


@pytest.mark.parametrize("canonical", ["recipe_micro_vpin", "recipe_micro_trade_imbalance"])
def test_recipe_scalar_keywords_match_positional_and_oracle(canonical):
    ensure_cleaned_loaded()
    op = OperatorRegistry.get(canonical, backend="pandas_numpy", mode="any")
    index = pd.date_range("2026-01-05 09:30", periods=12, freq="min")
    close = pd.DataFrame({"A": [100., 102., 101., 105., 104., 107., 108., 110., 109., 113., 112., 116.]}, index=index)
    volume = pd.DataFrame({"A": np.arange(1., 13.)}, index=index)
    positional = op.calculate(close, volume, 3, 3)
    keyword = op.calculate(close, volume, window=3, min_periods=3)
    pd.testing.assert_frame_equal(keyword, positional)
    all_keyword = op.calculate(close=close, volume=volume, window=3, min_periods=3)
    pd.testing.assert_frame_equal(all_keyword, positional)
    with pytest.raises((TypeError, ValueError)):
        op.calculate(close, volume, 3, window=4)
    ret = close.A / close.A.shift(1) - 1
    magnitude = ret.abs() if canonical == "recipe_micro_vpin" else np.sign(ret)
    expected = ((magnitude * volume.A).rolling(3, min_periods=3).sum()
                / volume.A.rolling(3, min_periods=3).sum())
    np.testing.assert_allclose(keyword.A, expected, equal_nan=True)
