"""Parity and boundary tests for the FE paired lagged z-score recipe."""
import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.long_smoothing import lagged_zscore
from factor_preprocess.transforms.rolling import rolling_zscore


def _same_ieee(actual, expected):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(actual) & np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("ddof", [0, 1, 2, 1.5, -1.5, True])
@pytest.mark.parametrize("min_periods", [0, 1, 3, None])
def test_paired_recipe_matches_fp_parameter_coercion(ddof, min_periods):
    rng = np.random.default_rng(90210)
    frame = pd.DataFrame({
        "asset_id": np.repeat(["A", "B", "C"], 40),
        "date": np.tile(np.arange(40), 3),
        "value": rng.normal(size=120),
    }, index=np.arange(120) % 17)
    expected = rolling_zscore(frame, window=8, min_periods=min_periods, ddof=ddof)
    actual = lagged_zscore(frame, window=8, min_periods=min_periods, ddof=ddof)
    _same_ieee(actual, expected)


def test_lagging_nonfinite_current_and_ieee_zero_division():
    frame = pd.DataFrame({
        "asset_id": ["A"] * 12,
        "date": list(range(12)),
        "value": [5.0, 5.0, 9.0, np.nan, 3.0, np.inf, -np.inf,
                  7.0, 7.0, 7.0, np.inf, 8.0],
    }, index=[4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1])
    expected = rolling_zscore(frame, window=3, min_periods=2, ddof=0)
    actual = lagged_zscore(frame, window=3, min_periods=2, ddof=0)
    _same_ieee(actual, expected)
    assert np.isnan(actual.iloc[1])
    assert np.isposinf(actual.iloc[2])


def test_null_categorical_ids_duplicate_dates_and_duplicate_index_are_positional():
    ids = pd.Categorical(["A", None, "A", "B", "A", None, "B", "A"],
                         categories=["A", "B", "unused"])
    frame = pd.DataFrame({
        "asset_id": ids,
        "date": [1, 1, 1, 1, 2, 2, 2, 3],
        "value": [2., 8., 4., 3., 5., 9., 7., 6.],
    }, index=[5, 5, 2, 2, 5, 5, 2, 2])
    expected = rolling_zscore(frame, window=2, min_periods=1, ddof=1)
    actual = lagged_zscore(frame, window=2, min_periods=1, ddof=1)
    _same_ieee(actual, expected)
    assert np.isnan(actual.iloc[1]) and np.isnan(actual.iloc[5])


def test_null_dates_for_observed_asset_are_rejected_and_null_asset_dates_ignored():
    frame = pd.DataFrame({
        "asset_id": ["A", None, "A"],
        "date": [1, None, 2],
        "value": [1., 2., 3.],
    })
    result = lagged_zscore(frame, window=2)
    assert np.isnan(result.iloc[1])
    bad = pd.DataFrame({"asset_id": ["A", "A"], "date": [2, 1], "value": [1., 2.]})
    with pytest.raises(ValueError, match="monotone"):
        lagged_zscore(bad, window=2)


def test_prefix_invariance_and_current_infinite_value():
    frame = pd.DataFrame({
        "asset_id": ["A"] * 8,
        "date": range(8),
        "value": [1., 2., 3., 4., 5., 6., 7., np.inf],
    })
    full = lagged_zscore(frame, window=4, min_periods=2, ddof=1)
    prefix = lagged_zscore(frame.iloc[:7], window=4, min_periods=2, ddof=1)
    _same_ieee(full.iloc[:7], prefix)
    assert np.isposinf(full.iloc[-1])


def test_native_recipe_performs_one_polars_collect(monkeypatch):
    import polars as pl

    original = pl.LazyFrame.collect
    calls = []

    def counted(plan, *args, **kwargs):
        calls.append(plan.explain())
        return original(plan, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "collect", counted)
    frame = pd.DataFrame({
        "asset_id": ["A"] * 20 + ["B"] * 20,
        "date": list(range(20)) * 2,
        "value": np.arange(40, dtype=float),
    })
    actual = lagged_zscore(frame, window=5, min_periods=3, ddof=1)
    expected = rolling_zscore(frame, window=5, min_periods=3, ddof=1)
    _same_ieee(actual, expected)
    assert len(calls) == 1
    assert "JOIN" not in calls[0].upper()
    assert "MAP_GROUPS" not in calls[0].upper()


@pytest.mark.parametrize("overrides", [
    {"window": True},
    {"window": 3.5},
    {"min_periods": True},
    {"min_periods": 2.5},
    {"min_periods": -1},
    {"min_periods": 5},
])
def test_invalid_window_and_min_periods_fail_with_value_error(overrides):
    frame = pd.DataFrame({
        "asset_id": ["A"] * 8,
        "date": range(8),
        "value": np.arange(8, dtype=float),
    })
    kwargs = {"window": 4, "min_periods": 2}
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        rolling_zscore(frame, **kwargs)
    with pytest.raises(ValueError):
        lagged_zscore(frame, **kwargs)


def test_long_constant_impulses_preserve_ieee_categories():
    levels_and_impulses = [
        (3.14, 100.0),
        (3.14, -100.0),
        (0.1, 9.0),
        (0.1, -9.0),
        (1e12 + 0.25, 1e12 + 16.25),
        (1e12 + 0.25, 1e12 - 15.75),
    ]
    values = []
    assets = []
    dates = []
    for asset, (level, impulse) in enumerate(levels_and_impulses):
        segment = np.r_[np.full(140, level), impulse, np.full(100, level)]
        values.extend(segment)
        assets.extend([asset] * len(segment))
        dates.extend(range(len(segment)))
    frame = pd.DataFrame({"asset_id": assets, "date": dates, "value": values})
    expected = rolling_zscore(frame, window=32, min_periods=4, ddof=1)
    actual = lagged_zscore(frame, window=32, min_periods=4, ddof=1)
    _same_ieee(actual, expected)


def test_high_offset_huge_finite_and_sparse_history_preserve_ieee():
    rng = np.random.default_rng(773)
    values_by_case = {
        "offset_1e12": 1e12 + np.resize(
            np.array([-1., 0., 1., 2., -2., 0.5, -0.5]), 300
        ),
        "offset_1e15": 1e15 + np.resize(
            np.array([-4., -2., 0., 1., 3., 5., -1.]), 300
        ),
        "huge_finite": np.resize(
            np.array([1e150, -1e150, 5e149, -5e149]), 300
        ),
        "sparse_nonfinite": np.resize(
            np.array([1., np.nan, np.inf, 4., -np.inf, 3., np.nan, 9.]), 300
        ),
        "near_overflow_alternating": np.resize(
            np.array([1e308, -1e308, 1e308, -1e308]), 300
        ),
        "subnormal_fluctuations": np.resize(
            np.array([1e-300, -1e-300, 5e-301, -5e-301]), 300
        ),
        "offset_random_noise": 1e12 + rng.normal(0.0, 0.25, 300),
    }
    for values in values_by_case.values():
        frame = pd.DataFrame({
            "asset_id": ["A"] * len(values),
            "date": np.arange(len(values)),
            "value": values,
        })
        expected = rolling_zscore(frame, window=32, min_periods=4, ddof=1)
        actual = lagged_zscore(frame, window=32, min_periods=4, ddof=1)
        _same_ieee(actual, expected)

