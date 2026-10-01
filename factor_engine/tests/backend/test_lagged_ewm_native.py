"""Parity, input-contract and native-execution checks for lagged EWMA."""
import ast
import inspect
import math

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.long_smoothing import lagged_ewma
from factor_engine.backend.long_ewm import halflife_to_alpha


def _pandas_reference(frame, *, halflife, min_periods, asset_col="asset_id",
                      time_col="date", value_col="value"):
    output = np.full(len(frame), np.nan, dtype=np.float64)
    for positions in frame.groupby(
        asset_col, sort=False, observed=True
    ).indices.values():
        values = frame.iloc[positions][value_col].to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        values[~np.isfinite(values)] = np.nan
        result = pd.Series(values).shift(1).ewm(
            halflife=halflife, min_periods=min_periods, adjust=False
        ).mean()
        output[np.asarray(positions, dtype=np.int64)] = result.to_numpy()
    return pd.Series(output, index=frame.index, name=value_col)


def _assert_close(actual, expected):
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(
        np.asarray(actual)[finite], np.asarray(expected)[finite],
        rtol=2e-12, atol=2e-12,
    )


@pytest.mark.parametrize("halflife", [0.5, 2.5, 3.7, 60.0])
@pytest.mark.parametrize("min_periods", [0, 1, 3])
def test_lagged_ewma_matches_independent_pandas_reference(
    halflife, min_periods
):
    frame = pd.DataFrame({
        "asset_id": pd.Categorical(
            ["A", "B", "A", None, "A", "B", "A", "B", "A"],
            categories=["A", "B", "unused"],
        ),
        "date": [0, 0, 1, 0, 1, 1, 2, 2, 3],
        "value": [1.0, 10.0, np.nan, 999.0, 3.0, np.inf,
                  5.0, -np.inf, 9.0],
    }, index=[4, 4, 1, 1, 4, 4, 1, 1, 4])
    original = frame.copy(deep=True)
    actual = lagged_ewma(
        frame, halflife=halflife, min_periods=min_periods
    )
    expected = _pandas_reference(
        frame, halflife=halflife, min_periods=min_periods
    )
    assert actual.index.equals(frame.index)
    assert actual.name == "value"
    _assert_close(actual.to_numpy(), expected.to_numpy())
    pd.testing.assert_frame_equal(frame, original)


def test_lagged_ewma_rejects_unsorted_observed_assets_but_ignores_null_asset_dates():
    frame = pd.DataFrame({
        "asset_id": ["A", None, "A"],
        "date": [1, None, 2],
        "value": [1.0, 50.0, 2.0],
    })
    assert lagged_ewma(frame, halflife=2.5).isna().iloc[1]
    bad = pd.DataFrame({
        "asset_id": ["A", "A"], "date": [2, 1], "value": [1.0, 2.0]
    })
    with pytest.raises(ValueError, match="monotone"):
        lagged_ewma(bad, halflife=2.5)
    bad_date = pd.DataFrame({
        "asset_id": ["A", "A"], "date": [1, None], "value": [1.0, 2.0]
    })
    with pytest.raises(ValueError, match="non-null"):
        lagged_ewma(bad_date, halflife=2.5)


def test_lagged_ewma_is_causal_under_current_and_future_perturbation():
    frame = pd.DataFrame({
        "asset_id": ["A"] * 6, "date": range(6),
        "value": [1.0, 2.0, 4.0, 3.0, 5.0, 7.0],
    })
    base = lagged_ewma(frame, halflife=2.5, min_periods=1).to_numpy()
    changed = frame.copy()
    changed.loc[3:, "value"] = [100.0, -100.0, 200.0]
    altered = lagged_ewma(changed, halflife=2.5, min_periods=1).to_numpy()
    np.testing.assert_array_equal(base[:4], altered[:4])


def test_lagged_ewma_collects_one_polars_plan_without_python_group_kernels(monkeypatch):
    import polars as pl
    from factor_engine.backend import native_long_ewm

    plans = []
    original = pl.LazyFrame.collect

    def counted(plan, *args, **kwargs):
        plans.append(plan.explain().upper())
        return original(plan, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "collect", counted)
    frame = pd.DataFrame({
        "asset_id": ["A", "B", "A", "B", "A", "B"],
        "date": [0, 0, 1, 1, 2, 2],
        "value": [1.0, 10.0, 3.0, 12.0, 5.0, 14.0],
    }, index=[1, 1, 1, 1, 1, 1])
    result = lagged_ewma(frame, halflife=2.5, min_periods=1)
    assert len(result) == len(frame)
    assert len(plans) == 1
    assert "JOIN" not in plans[0]
    assert "MAP_GROUPS" not in plans[0]
    tree = ast.parse(inspect.getsource(native_long_ewm.collect_lagged_ewma))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert any(
        isinstance(node.func, ast.Attribute) and node.func.attr == "ewm_mean"
        for node in calls
    )
    called = {
        node.func.attr for node in calls if isinstance(node.func, ast.Attribute)
    }
    assert not called.intersection({"map_elements", "map_batches", "map_groups", "apply"})


@pytest.mark.parametrize("halflife", [0.5, 2.5, 3.7, 60.0])
def test_halflife_alpha_uses_stable_float_formula(halflife):
    expected = -math.expm1(-math.log(2.0) / halflife)
    assert halflife_to_alpha(halflife) == expected
    assert 0.0 < expected <= 1.0


def test_extreme_halflives_preserve_binary64_alpha_information():
    tiny = np.nextafter(0.0, 1.0)
    huge = np.finfo(np.float64).max
    assert halflife_to_alpha(tiny) == 1.0
    large_alpha = halflife_to_alpha(2.0 ** 53)
    assert large_alpha > 0.0
    assert halflife_to_alpha(huge) > 0.0


@pytest.mark.parametrize("halflife", [0, -1, np.nan, np.inf, -np.inf, True])
def test_lagged_ewma_rejects_invalid_halflife(halflife):
    frame = pd.DataFrame({
        "asset_id": ["A"], "date": [0], "value": [1.0]
    })
    with pytest.raises(ValueError, match="halflife"):
        lagged_ewma(frame, halflife=halflife)


@pytest.mark.parametrize("min_periods", [True, -1, 1.5, np.nan])
def test_lagged_ewma_rejects_invalid_min_periods(min_periods):
    frame = pd.DataFrame({
        "asset_id": ["A"], "date": [0], "value": [1.0]
    })
    with pytest.raises(ValueError, match="min_periods"):
        lagged_ewma(frame, halflife=2.5, min_periods=min_periods)


def test_lagged_ewma_rejects_missing_columns():
    frame = pd.DataFrame({"asset_id": ["A"], "date": [0]})
    with pytest.raises(ValueError, match="missing columns"):
        lagged_ewma(frame, halflife=2.5)
