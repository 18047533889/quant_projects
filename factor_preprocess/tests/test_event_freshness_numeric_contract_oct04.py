import numpy as np
import pandas as pd
import pytest

from factor_preprocess.transforms.event_decay import event_decay
from factor_preprocess.transforms.treatment_variants import freshness_aware_fill


def _frame(values, assets=None, dates=None, index=None):
    n = len(values)
    return pd.DataFrame({
        "asset_id": assets if assets is not None else ["a"] * n,
        "date": dates if dates is not None else pd.date_range("2024-01-01", periods=n),
        "value": values,
    }, index=index)


def _event_decay_reference(values, halflife, min_periods):
    """Independent scalar oracle: lag, reset, warm up, then seeded EMA."""
    alpha = min(np.log(2.0) / halflife, 1.0)
    output = np.full(len(values), np.nan)
    y = np.nan
    count = 0
    for t in range(1, len(values)):
        x = values[t - 1]
        if not np.isfinite(x):
            y = np.nan
            count = 0
            continue
        count += 1
        if count == 1:
            y = x
        else:
            y = (1.0 - alpha) * y + alpha * x
        if count < min_periods:
            continue
        output[t] = y
    return output


@pytest.mark.parametrize("min_periods", [1, 2, 3])
def test_event_decay_matches_independent_warmup_and_reset_oracle(min_periods):
    raw = np.array([2.0, 4.0, 8.0, np.nan, 3.0, 9.0, 12.0, np.inf, 5.0, 7.0, 8.0])
    frame = _frame(raw)
    actual = event_decay(frame, halflife=2.0, min_periods=min_periods).to_numpy()
    expected = _event_decay_reference(raw, halflife=2.0, min_periods=min_periods)
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_freshness_current_inclusive_max_lag_and_infinite_values_are_missing():
    frame = _frame([2.0, np.nan, np.nan, np.inf, -np.inf, 8.0])
    actual = freshness_aware_fill(frame, max_lag=2, decay_halflife=2.0).to_numpy()
    expected = np.array([2.0, 2.0 * 2**-0.5, 1.0, np.nan, np.nan, 8.0])
    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("bad", [True, np.bool_(False), 0, -1, np.nan, np.inf, -np.inf])
def test_event_decay_rejects_invalid_halflife(bad):
    with pytest.raises(ValueError):
        event_decay(_frame([1.0, 2.0]), halflife=bad)


@pytest.mark.parametrize("bad", [True, np.bool_(True), 1.5, 0, -1, np.nan, np.inf])
def test_event_decay_rejects_invalid_min_periods(bad):
    with pytest.raises(ValueError):
        event_decay(_frame([1.0, 2.0]), halflife=2.0, min_periods=bad)


@pytest.mark.parametrize("bad", [True, np.bool_(True), 1.0, 1.5, 0, -1, np.nan, np.inf])
def test_freshness_rejects_invalid_max_lag(bad):
    with pytest.raises(ValueError):
        freshness_aware_fill(_frame([1.0, np.nan]), max_lag=bad)


@pytest.mark.parametrize("bad", [True, np.bool_(False), 0, -1, np.nan, np.inf, -np.inf])
def test_freshness_rejects_invalid_halflife(bad):
    with pytest.raises(ValueError):
        freshness_aware_fill(_frame([1.0]), decay_halflife=bad)


def test_event_decay_is_asset_local_prefix_causal_and_index_preserving():
    frame = _frame(
        [2.0, 4.0, 8.0, 16.0, 100.0, 200.0, 300.0, 400.0],
        assets=["a"] * 4 + ["b"] * 4,
        dates=list(pd.date_range("2024-01-01", periods=4)) * 2,
        index=[7, 7, 3, 3, 7, 7, 3, 3],
    )
    actual = event_decay(frame, halflife=2.0, min_periods=1)
    altered = frame.copy()
    altered.loc[altered.asset_id == "b", "value"] = [100.0, 200.0, 300.0, 9999.0]
    altered_out = event_decay(altered, halflife=2.0, min_periods=1)
    assert actual.index.equals(frame.index)
    np.testing.assert_allclose(actual.iloc[:7], altered_out.iloc[:7], equal_nan=True)
    assert np.isnan(actual.iloc[4])


def test_event_decay_alpha_clips_at_one_for_short_halflife():
    actual = event_decay(_frame([2.0, 9.0, 5.0]), halflife=0.25).to_numpy()
    np.testing.assert_allclose(actual, [np.nan, 2.0, 9.0], equal_nan=True)


def test_freshness_is_asset_local_prefix_causal_and_index_preserving():
    frame = _frame(
        [4.0, np.nan, np.nan, 20.0, np.nan, np.nan],
        assets=["a"] * 3 + ["b"] * 3,
        dates=list(pd.date_range("2024-01-01", periods=3)) * 2,
        index=[1, 1, 2, 1, 1, 2],
    )
    actual = freshness_aware_fill(frame, max_lag=2, decay_halflife=2.0)
    altered = frame.copy()
    altered.loc[altered.asset_id == "a", "value"] = [4.0, np.nan, 40.0]
    altered_out = freshness_aware_fill(altered, max_lag=2, decay_halflife=2.0)
    expected = np.array([4.0, 4.0 * 2**-0.5, 2.0, 20.0, 20.0 * 2**-0.5, 10.0])
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    np.testing.assert_allclose(actual.iloc[:2], altered_out.iloc[:2], equal_nan=True)
    assert actual.index.equals(frame.index)
