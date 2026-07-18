# -*- coding: utf-8 -*-
"""Regression tests for audited edge-parameter and weighted semantics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def _load_registry() -> None:
    load_all()


def _pd(name):
    operator = OperatorRegistry.get(name, "pandas_numpy")
    assert operator is not None
    return operator


def _pl(name):
    operator = OperatorRegistry.get(name, "polars")
    assert operator is not None
    return operator


def test_cs_bucket_excludes_nan_and_infinity_in_both_backends():
    pl = pytest.importorskip("polars")
    values = pd.DataFrame([[1.0, 2.0, np.inf, -np.inf, np.nan]], columns=list("ABCDE"))
    pandas_result = _pd("cs_bucket").calculate(values, 4, True)
    polars_result = _pl("cs_bucket").calculate(pl.DataFrame(values), 4, True)
    assert pandas_result.loc[0, "A"] == 1.0
    assert pandas_result.loc[0, "B"] == 4.0
    assert pandas_result.loc[0, ["C", "D", "E"]].isna().all()
    np.testing.assert_allclose(
        polars_result.select(values.columns).to_numpy(),
        pandas_result.to_numpy(),
        equal_nan=True,
    )


def test_nth_value_rejects_fractional_n_and_min_periods():
    values = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        _pd("ts_nth_value").calculate(values, 3, 1.5)
    with pytest.raises(ValueError):
        _pd("ts_nth_value").calculate(values, 3, 1, 1.5)


def test_legacy_regression_rejects_future_or_fractional_lag():
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]})
    y = 2.0 * x + 1.0
    regression = OperatorRegistry.get("ts_regression")
    with pytest.raises(ValueError, match="future data"):
        regression.calculate(y, x, 4, -1, "slope")
    with pytest.raises(ValueError, match="integer"):
        regression.calculate(y, x, 4, 1.5, "slope")


def test_legacy_regression_no_intercept_r2_is_uncentered():
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]})
    y = pd.DataFrame({"A": [2.0, 3.0, 5.0, 8.0]})
    result = OperatorRegistry.get("ts_regression").calculate(
        y,
        x,
        4,
        0,
        "r2",
        min_periods=3,
        add_intercept=False,
    )
    xv, yv = x["A"].to_numpy(), y["A"].to_numpy()
    slope = float(np.dot(xv, yv) / np.dot(xv, xv))
    residual = yv - slope * xv
    expected = 1.0 - float(residual @ residual) / float(yv @ yv)
    assert result.iloc[-1, 0] == pytest.approx(expected)


@pytest.mark.parametrize("epsilon", [0.0, -1.0, np.nan, np.inf])
def test_div_or_null_rejects_invalid_epsilon(epsilon):
    pl = pytest.importorskip("polars")
    x = pd.DataFrame({"A": [1.0]})
    y = pd.DataFrame({"A": [2.0]})
    with pytest.raises(ValueError):
        _pd("div_or_null").calculate(x, y, epsilon)
    with pytest.raises(ValueError):
        _pl("div_or_null").calculate(pl.DataFrame(x), pl.DataFrame(y), epsilon)


def _weighted_frames():
    x = pd.DataFrame(
        [[1.0, 3.0, 100.0, 5.0]], columns=list("ABCD")
    )
    weight = pd.DataFrame(
        [[1.0, 3.0, 1.0, np.nan]], columns=list("ABCD")
    )
    group = pd.DataFrame(
        [["g", "g", pd.NA, "g"]], columns=list("ABCD")
    )
    return x, group, weight


@pytest.mark.parametrize(
    "canonical",
    [
        "cs_weighted_mean",
        "cs_weighted_demean",
        "cs_weighted_zscore",
    ],
)
def test_cross_sectional_weighted_native_polars_parity(canonical):
    pl = pytest.importorskip("polars")
    x, _, weight = _weighted_frames()
    pandas_result = _pd(canonical).calculate(x, weight)
    polars_result = _pl(canonical).calculate(
        pl.DataFrame(x), pl.DataFrame(weight)
    )
    np.testing.assert_allclose(
        polars_result.select(x.columns).to_numpy(),
        pandas_result.to_numpy(),
        equal_nan=True,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "canonical", ["group_weighted_mean", "group_weighted_zscore"]
)
def test_group_weighted_missing_labels_are_null_and_polars_matches(canonical):
    pl = pytest.importorskip("polars")
    x, group, weight = _weighted_frames()
    pandas_result = _pd(canonical).calculate(x, group, weight)
    polars_result = _pl(canonical).calculate(
        pl.DataFrame(x), pl.DataFrame(group), pl.DataFrame(weight)
    )
    assert np.isnan(pandas_result.loc[0, "C"])
    assert np.isnan(pandas_result.loc[0, "D"])
    np.testing.assert_allclose(
        polars_result.select(x.columns).to_numpy(),
        pandas_result.to_numpy(),
        equal_nan=True,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("canonical", ["cs_multi_resid", "cs_wls_resid"])
def test_cross_sectional_regression_rejects_fractional_min_obs(canonical):
    y = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], columns=list("ABCD"))
    x = pd.DataFrame([[1.0, 2.0, 4.0, 8.0]], columns=list("ABCD"))
    if canonical == "cs_multi_resid":
        args = (y, x)
    else:
        weight = pd.DataFrame([[1.0, 1.0, 1.0, 1.0]], columns=list("ABCD"))
        args = (y, x, weight)
    with pytest.raises(ValueError):
        _pd(canonical).calculate(*args, min_obs=3.5)
