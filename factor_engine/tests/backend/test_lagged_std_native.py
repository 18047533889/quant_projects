"""Standalone correctness gates for the long-panel lagged std fast path."""
# Decimal is the mathematical oracle over exact binary64 inputs; FP rolling_std
# is deliberately not the oracle at magnitudes where intermediate squares fail.
from decimal import Decimal, localcontext
import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.long_smoothing import (
    _native_std_domain_safe,
    _lagged_rolling,
    lagged_std,
)


def _decimal_std(values, ddof):
    finite = [float(value) for value in values if np.isfinite(value)]
    n = len(finite)
    if n <= ddof:
        return np.nan
    with localcontext() as context:
        context.prec = 1600
        exact = [Decimal.from_float(value) for value in finite]
        origin = exact[0]
        centered = [value - origin for value in exact]
        mean = sum(centered, Decimal(0)) / Decimal(n)
        variance = sum(((value - mean) ** 2 for value in centered), Decimal(0))
        variance /= Decimal(n - ddof)
        try:
            return float(variance.sqrt())
        except OverflowError:
            return np.inf


def _decimal_lagged_std(frame, *, window, min_periods, ddof):
    ddof = int(ddof)
    output = np.full(len(frame), np.nan, dtype=np.float64)
    history = {}
    for row, (asset, value) in enumerate(zip(frame.asset_id, frame.value)):
        if pd.isna(asset):
            continue
        prior = history.setdefault(asset, [])[-window:]
        finite = [item for item in prior if np.isfinite(item)]
        if len(finite) >= max(1, min_periods) and len(finite) > ddof:
            output[row] = _decimal_std(finite, ddof)
        history[asset].append(float(value) if np.isfinite(value) else np.nan)
    return output


def _frame(values, assets=None):
    values = np.asarray(values, dtype=np.float64)
    if assets is None:
        assets = np.repeat("A", len(values))
    return pd.DataFrame({
        "asset_id": assets,
        "date": pd.Series(assets).groupby(assets, sort=False).cumcount().to_numpy(),
        "value": values,
    }, index=np.arange(len(values)) % 7)


def _assert_float_close(actual, expected, *, rtol=3e-14, atol=0.0):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=rtol, atol=atol)


@pytest.mark.parametrize("values", [
    [1.0, 4.0, -2.0, 5.0, 1.0, 9.0, -3.0, 4.0],
    [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 5.0],
])
@pytest.mark.parametrize("ddof", [-1, 0, 1, 2, 1.5])
@pytest.mark.parametrize("min_periods", [1, 3, None])
def test_safe_native_lagged_std_matches_decimal_oracle(values, ddof, min_periods):
    frame = _frame(values)
    window = 4
    minimum = window if min_periods is None else min_periods
    actual = lagged_std(frame, window=window, min_periods=min_periods, ddof=ddof)
    expected = _decimal_lagged_std(
        frame, window=window, min_periods=minimum, ddof=ddof
    )
    _assert_float_close(actual.to_numpy(), expected)


def test_interleaved_assets_and_nonfinite_rows_match_decimal_oracle():
    frame = pd.DataFrame({
        "asset_id": ["A", "B", "A", "B", "A", "B", "A", "B"],
        "date": [0, 0, 1, 1, 2, 2, 3, 3],
        "value": [1.0, 10.0, np.nan, 13.0, 4.0, np.inf, 6.0, 12.0],
    }, index=[3, 3, 1, 1, 3, 3, 1, 1])
    actual = lagged_std(frame, window=3, min_periods=1, ddof=0)
    expected = _decimal_lagged_std(frame, window=3, min_periods=1, ddof=0)
    _assert_float_close(actual.to_numpy(), expected)


@pytest.mark.parametrize("values", [
    [2.0, np.nan, 3.0],
    [np.nan, 2.0, np.nan, 3.0],
])
@pytest.mark.parametrize("ddof", [-1, 0, 1])
def test_window_one_single_finite_observation_uses_count_ddof_contract(values, ddof):
    frame = _frame(values)
    actual = lagged_std(frame, window=1, min_periods=1, ddof=ddof)
    expected = _decimal_lagged_std(frame, window=1, min_periods=1, ddof=ddof)
    _assert_float_close(actual.to_numpy(), expected)
    if ddof <= 0:
        assert np.all(actual.iloc[1:][np.isfinite(actual.iloc[1:])] == 0.0)



@pytest.mark.parametrize("values", [
    [1e308, -1e308, 1e308, -1e308],
    [1e308, 1e308, 0.0, 1e308, 1e308],
    [1e-300, -1e-300, 5e-301, -5e-301],
    [1e-320, -1e-320, 5e-321, -5e-321],
])
def test_risky_magnitude_fallback_matches_decimal_oracle(values):
    frame = _frame(values)
    assert not _native_std_domain_safe(
        frame.value.to_numpy(), np.isfinite(frame.value.to_numpy()),
        np.zeros(len(frame), dtype=np.int64), window=2,
    )
    actual = lagged_std(frame, window=2, min_periods=1, ddof=1)
    generic = _lagged_rolling(
        frame, window=2, min_periods=1, asset_col="asset_id", time_col="date",
        value_col="value", reducer="std", ddof=1,
    )
    expected = _decimal_lagged_std(frame, window=2, min_periods=1, ddof=1)
    _assert_float_close(actual.to_numpy(), generic.to_numpy())
    _assert_float_close(actual.to_numpy(), expected, rtol=5e-13)


def test_mixed_group_magnitudes_force_whole_frame_fallback():
    groups = np.array([0, 0, 1, 1], dtype=np.int64)
    for values in (
        np.array([1.0, 1.0, 1e-320, -1e-320], dtype=np.float64),
        np.array([1.0, 1.0, 1e308, -1e308], dtype=np.float64),
    ):
        assert not _native_std_domain_safe(
            values, np.ones(values.size, dtype=bool), groups, window=2
        )


def test_magnitude_risk_bypasses_native_std_collector(monkeypatch):
    from factor_engine.backend import native_long_rolling_moments

    frame = _frame([1e308, -1e308, 1e308, -1e308])

    def unexpected(*args, **kwargs):
        raise AssertionError("unsafe magnitude reached native std collector")

    monkeypatch.setattr(
        native_long_rolling_moments, "collect_lagged_moments", unexpected
    )
    result = lagged_std(frame, window=2, min_periods=1, ddof=1)
    generic = _lagged_rolling(
        frame, window=2, min_periods=1, asset_col="asset_id", time_col="date",
        value_col="value", reducer="std", ddof=1,
    )
    _assert_float_close(result.to_numpy(), generic.to_numpy())


def test_window_local_near_constant_drift_forces_fallback_but_constants_do_not():
    level = 1e12
    values = np.r_[np.linspace(0.0, 10.0, 100), np.full(10, level)]
    values[-1] = np.nextafter(level, np.inf)
    assert not _native_std_domain_safe(
        values, np.ones(values.size, dtype=bool), np.zeros(values.size, dtype=int),
        window=8,
    )
    assert _native_std_domain_safe(
        np.full(20, 3.25), np.ones(20, dtype=bool), np.zeros(20, dtype=int),
        window=8,
    )


@pytest.mark.parametrize("window", [32, 1250, 5000])
def test_long_sparse_impulse_variance_bound_uses_decimal_terminal_window(window):
    impulse = 1e-153
    values = np.r_[impulse, np.zeros(window, dtype=np.float64)]
    frame = _frame(values)
    assert not _native_std_domain_safe(
        frame.value.to_numpy(), np.isfinite(frame.value.to_numpy()),
        np.zeros(len(frame), dtype=np.int64), window=window,
    )
    actual = lagged_std(frame, window=window, min_periods=1, ddof=0)
    # At row `window`, the prior window contains one impulse and W-1 zeros.
    expected = _decimal_std([impulse] + [0.0] * (window - 1), ddof=0)
    assert np.isfinite(actual.iloc[window])
    assert actual.iloc[window] > 0.0
    _assert_float_close([actual.iloc[window]], [expected], rtol=5e-13)


def test_std_only_native_plan_collects_once_without_joins_or_mean():
    import polars as pl
    from factor_engine.backend.native_long_rolling_moments import collect_lagged_moments

    calls = []
    original = pl.LazyFrame.collect

    def counted(plan, *args, **kwargs):
        calls.append(plan.explain().upper())
        return original(plan, *args, **kwargs)

    pl.LazyFrame.collect = counted
    try:
        frame = _frame([0.0, 2.0, 1.0, 4.0, 3.0])
        groups = np.zeros(len(frame), dtype=np.int64)
        result = collect_lagged_moments(
            np.arange(len(frame)), groups, frame.value.to_numpy(),
            np.ones(len(frame)), window=3, min_periods=1, ddof=1,
            include_mean=False,
        )
    finally:
        pl.LazyFrame.collect = original
    assert result.columns == ["ts", "inst", "_std"]
    assert len(calls) == 1
    assert "JOIN" not in calls[0]
    assert "_MEAN_RAW" not in calls[0]
