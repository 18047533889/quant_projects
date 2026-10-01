"""Parity and boundary tests for the FE paired lagged z-score recipe."""
from decimal import Decimal, localcontext

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


def _decimal_zscore(values, *, window, min_periods, ddof):
    """High-precision oracle over the exact input binary64 values."""
    values = np.asarray(values, dtype=np.float64)
    output = np.full(values.size, np.nan, dtype=np.float64)
    required = max(1, int(min_periods))
    ddof = int(ddof)
    for row, current in enumerate(values):
        history = [float(value) for value in values[max(0, row - window):row]
                   if np.isfinite(value)]
        if len(history) < required or len(history) <= ddof:
            continue
        if not np.isfinite(current):
            output[row] = current
            continue
        with localcontext() as context:
            context.prec = 1200
            exact = [Decimal.from_float(value) for value in history]
            mean = sum(exact, Decimal(0)) / Decimal(len(exact))
            squared = sum(((value - mean) ** 2 for value in exact), Decimal(0))
            variance = squared / Decimal(len(exact) - ddof)
            difference = Decimal.from_float(float(current)) - mean
            if variance == 0:
                output[row] = np.nan if difference == 0 else (
                    np.inf if difference > 0 else -np.inf
                )
            else:
                output[row] = float(difference / variance.sqrt())
    return output


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
    # Use the independent research reference here: the public entry now
    # delegates to FE too and would count a second, separate invocation.
    from factor_preprocess.transforms.rolling import _rolling_zscore_fp_research
    expected = _rolling_zscore_fp_research(frame, window=5, min_periods=3, ddof=1)
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
        expected = _decimal_zscore(values, window=32, min_periods=4, ddof=1)
        actual = lagged_zscore(frame, window=32, min_periods=4, ddof=1)
        _same_ieee(actual, expected)
        direct = rolling_zscore(frame, window=32, min_periods=4, ddof=1)
        _same_ieee(direct, expected)

def test_adapter_and_registry_zscore_match_decimal_high_offset():
    from factor_preprocess.adapters.fe_smoothing import execute_rolling_zscore
    from factor_preprocess.registry.transforms import get_default_registry

    values = 1e12 + np.resize(np.array([-1., 0., 1., 2., -2., 0.5, -0.5]), 300)
    frame = pd.DataFrame({"asset_id": ["A"] * len(values),
                          "date": np.arange(len(values)), "value": values})
    expected = _decimal_zscore(values, window=32, min_periods=4, ddof=1)
    _same_ieee(execute_rolling_zscore(
        frame, window=32, min_periods=4, ddof=1
    ), expected)
    _same_ieee(get_default_registry().get_execution("rolling_zscore")(
        frame, window=32, min_periods=4, ddof=1
    ), expected)


def test_public_and_fe_zscore_routes_are_bit_exactly_identical():
    from factor_preprocess.adapters.fe_smoothing import execute_rolling_zscore
    from factor_preprocess.registry.transforms import get_default_registry

    rng = np.random.default_rng(81)
    frame = pd.DataFrame({"asset_id": np.repeat(["A", "B"], 30),
                          "date": np.tile(np.arange(30), 2),
                          "value": rng.normal(size=60)})
    kwargs = dict(window=8, min_periods=4, ddof=1)
    expected = lagged_zscore(frame, **kwargs).to_numpy()
    routes = (
        rolling_zscore(frame, **kwargs),
        execute_rolling_zscore(frame, **kwargs),
        get_default_registry().get_execution("rolling_zscore")(frame, **kwargs),
    )
    for actual in routes:
        actual = actual.to_numpy()
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
        np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
        finite = np.isfinite(expected)
        np.testing.assert_array_equal(actual[finite], expected[finite])


def test_public_zscore_fails_closed_without_fe_but_registry_research_is_explicit(monkeypatch):
    from factor_preprocess.adapters import fe_smoothing
    from factor_preprocess.errors import GovernanceError
    from factor_preprocess.registry.transforms import get_default_registry
    from factor_preprocess.transforms.rolling import _rolling_zscore_fp_research

    frame = pd.DataFrame({"asset_id": ["A"] * 40, "date": range(40),
                          "value": 1e12 + np.resize([-1., 0., 1., 2.], 40)})
    monkeypatch.setattr(fe_smoothing, "get_fe_composite_executor", lambda *args: None)
    with pytest.raises(GovernanceError, match="no implicit FP-native fallback"):
        rolling_zscore(frame, window=8, min_periods=4, ddof=1)

    registry = get_default_registry()
    with pytest.raises(GovernanceError, match="no implicit FP-native fallback"):
        registry.get_execution("rolling_zscore")(frame, window=8, min_periods=4, ddof=1)
    research = registry.get_execution("rolling_zscore", allow_research=True)
    expected = _rolling_zscore_fp_research(frame, window=8, min_periods=4, ddof=1)
    _same_ieee(research(frame, window=8, min_periods=4, ddof=1), expected)

