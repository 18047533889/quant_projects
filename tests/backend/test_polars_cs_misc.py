# -*- coding: utf-8 -*-
"""Parity tests for limit/holder/normalizer/regression Polars backends."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def _load_registry():
    load_all()


@pytest.fixture(scope="module")
def panels():
    rng = np.random.default_rng(2025)
    index = pd.date_range("2022-01-01", periods=120, freq="D")
    close = pd.DataFrame(rng.uniform(5, 50, (120, 2)), index=index, columns=["A", "B"])
    upper = close * 1.1
    lower = close * 0.9
    listed = pd.DataFrame(np.ones((120, 2)), index=index, columns=["A", "B"])
    suspended = pd.DataFrame(rng.random((120, 2)) < 0.05, index=index, columns=["A", "B"]).astype(float)
    limit_up = pd.DataFrame(rng.random((120, 2)) < 0.03, index=index, columns=["A", "B"]).astype(float)
    limit_down = pd.DataFrame(rng.random((120, 2)) < 0.03, index=index, columns=["A", "B"]).astype(float)
    top_shares = pd.DataFrame(rng.uniform(1e6, 1e8, (120, 2)), index=index, columns=["A", "B"])
    total_shares = pd.DataFrame(rng.uniform(1e8, 1e9, (120, 2)), index=index, columns=["A", "B"])
    return close, upper, lower, listed, suspended, limit_up, limit_down, top_shares, total_shares


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


def test_native_polars_cs_misc_matches_pandas(panels):
    close, upper, lower, listed, suspended, lu, ld, top, total = panels
    cases = [
        ("limit_up_close", (close, upper), {"tick_tolerance": 0.005}),
        ("limit_down_close", (close, lower), {"tick_tolerance": 0.005}),
        ("tradable_state", (listed, suspended, lu, ld), {}),
        ("holder_concentration", (top, total), {}),
        ("unitize", (close,), {}),
        ("winsorize_mean", (close,), {"trim_pct": 0.1}),
        ("ts_regression_intercept", (close, close + 1.0), {"window": 20}),
        ("ts_regression_r2", (close, close + 1.0), {"window": 20}),
        ("ts_regression_resid", (close, close + 1.0), {"window": 20}),
    ]
    for name, args, kwargs in cases:
        _assert_parity(name, args, kwargs)


def test_native_polars_cs_misc_nan_warmup_matches_pandas():
    rng = np.random.default_rng(77)
    index = pd.date_range("2023-01-01", periods=80, freq="D")
    mask = rng.random((80, 2)) < 0.1
    close = pd.DataFrame(rng.uniform(5, 50, (80, 2)), index=index, columns=["A", "B"])
    close[mask] = np.nan
    upper = close * 1.1
    listed = pd.DataFrame(np.ones((80, 2)), index=index, columns=["A", "B"])
    suspended = pd.DataFrame(rng.random((80, 2)) < 0.05, index=index, columns=["A", "B"]).astype(float)
    top = pd.DataFrame(rng.uniform(1e6, 1e8, (80, 2)), index=index, columns=["A", "B"])
    total = pd.DataFrame(rng.uniform(1e8, 1e9, (80, 2)), index=index, columns=["A", "B"])
    cases = [
        ("limit_up_close", (close, upper), {"tick_tolerance": 0.005}),
        ("tradable_state", (listed, suspended, suspended, suspended), {}),
        ("holder_concentration", (top, total), {}),
        ("unitize", (close,), {}),
        ("winsorize_mean", (close,), {"trim_pct": 0.1}),
        ("ts_regression_intercept", (close, close + 1.0), {"window": 10}),
        ("ts_regression_r2", (close, close + 1.0), {"window": 10}),
        ("ts_regression_resid", (close, close + 1.0), {"window": 10}),
    ]
    for name, args, kwargs in cases:
        _assert_parity(name, args, kwargs)
