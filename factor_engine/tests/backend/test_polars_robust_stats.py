# -*- coding: utf-8 -*-
"""Parity tests for native Polars robust-stat / conditional / risk operators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def _load_registry():
    load_all()


@pytest.fixture(scope="module")
def panels():
    rng = np.random.default_rng(991)
    index = pd.date_range("2022-01-01", periods=130, freq="D")
    x = pd.DataFrame(rng.normal(0.5, 2.0, (130, 2)), index=index, columns=["A", "B"])
    y = x * 0.8 + rng.normal(0, 0.5, (130, 2))
    condition = pd.DataFrame(rng.random((130, 2)) > 0.4, index=index, columns=["A", "B"]).astype(float)
    return x, y, condition


def _polars(frame: pd.DataFrame) -> pl.DataFrame:
    return pl.DataFrame({column: frame[column].to_numpy() for column in frame.columns})


def _assert_parity(name, args, kwargs, rtol=1e-8, atol=1e-8):
    pandas_op = OperatorRegistry.get(name, backend="pandas_numpy")
    polars_op = OperatorRegistry.get(name, backend="polars")
    assert pandas_op is not None, f"{name} missing pandas"
    assert polars_op is not None, f"{name} missing polars"
    pandas_out = pandas_op.calculate(*args, **kwargs)
    polars_out = polars_op.calculate(*[_polars(arg) for arg in args], **kwargs)
    assert list(pandas_out.columns) == list(polars_out.columns)
    for column in pandas_out.columns:
        np.testing.assert_allclose(
            pandas_out[column].to_numpy(),
            polars_out[column].to_numpy(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )


def test_native_polars_robust_stats_matches_pandas(panels):
    x, y, cond = panels
    cases = [
        ("ts_quantile_range", (x,), {"window": 20, "q_low": 0.25, "q_high": 0.75}),
        ("ts_robust_zscore", (x,), {"window": 20, "center": "median", "scale": "mad"}),
        ("ts_robust_zscore", (x,), {"window": 20, "center": "mean", "scale": "std", "clip": 3.0}),
        ("ts_trimmed_mean", (x,), {"window": 20, "trim_ratio": 0.1}),
        ("ts_abs_concentration", (x,), {"window": 20}),
        ("ts_abs_entropy", (x,), {"window": 20, "normalize": True}),
        ("ts_downside_deviation", (x,), {"window": 20, "target": 0.0}),
        ("ts_upside_deviation", (x,), {"window": 20, "target": 0.0}),
        ("ts_current_drawdown_duration", (x,), {"window": 20}),
        ("ts_time_under_water", (x,), {"window": 20}),
        ("ts_best_lag_corr", (y, x), {"window": 20, "max_lag": 3}),
        ("ts_price_delay", (x, x), {"window": 20, "max_lag": 3, "min_periods": 3}),
        ("ts_max_if", (x, cond), {"window": 20}),
        ("ts_min_if", (x, cond), {"window": 20}),
        ("ts_quantile_if", (x, cond), {"window": 20, "q": 0.5}),
        ("ts_corr_if", (x, y, cond), {"window": 20}),
        ("ts_beta_if", (y, x, cond), {"window": 20}),
        ("ts_regression_resid_if", (y, x, cond), {"window": 20}),
        ("ts_poly2_coeff", (x,), {"d": 20}),
        ("ts_poly2_resid", (y, x), {"d": 20}),
        ("lqtp_historical_cvar", (x,), {"window": 60, "q": 0.05}),
    ]
    for name, args, kwargs in cases:
        _assert_parity(name, args, kwargs)


def test_native_polars_robust_stats_nan_warmup_matches_pandas():
    rng = np.random.default_rng(21)
    index = pd.date_range("2023-01-01", periods=90, freq="D")
    mask = rng.random((90, 2)) < 0.12
    x = pd.DataFrame(rng.normal(0.5, 2.0, (90, 2)), index=index, columns=["A", "B"])
    x[mask] = np.nan
    y = x * 0.8 + rng.normal(0, 0.5, (90, 2))
    cond = pd.DataFrame(rng.random((90, 2)) > 0.4, index=index, columns=["A", "B"]).astype(float)
    cases = [
        ("ts_quantile_range", (x,), {"window": 10, "q_low": 0.25, "q_high": 0.75}),
        ("ts_robust_zscore", (x,), {"window": 10, "center": "median", "scale": "mad"}),
        ("ts_trimmed_mean", (x,), {"window": 10, "trim_ratio": 0.1}),
        ("ts_abs_concentration", (x,), {"window": 10}),
        ("ts_abs_entropy", (x,), {"window": 10, "normalize": True}),
        ("ts_downside_deviation", (x,), {"window": 10, "target": 0.0}),
        ("ts_upside_deviation", (x,), {"window": 10, "target": 0.0}),
        ("ts_current_drawdown_duration", (x,), {"window": 10}),
        ("ts_time_under_water", (x,), {"window": 10}),
        ("ts_best_lag_corr", (y, x), {"window": 10, "max_lag": 3}),
        ("ts_price_delay", (x, x), {"window": 10, "max_lag": 3, "min_periods": 3}),
        ("ts_max_if", (x, cond), {"window": 10}),
        ("ts_min_if", (x, cond), {"window": 10}),
        ("ts_quantile_if", (x, cond), {"window": 10, "q": 0.5}),
        ("ts_corr_if", (x, y, cond), {"window": 10}),
        ("ts_beta_if", (y, x, cond), {"window": 10}),
        ("ts_regression_resid_if", (y, x, cond), {"window": 10}),
        ("ts_poly2_coeff", (x,), {"d": 10}),
        ("ts_poly2_resid", (y, x), {"d": 10}),
        ("lqtp_historical_cvar", (x,), {"window": 30, "q": 0.05}),
    ]
    for name, args, kwargs in cases:
        _assert_parity(name, args, kwargs)
