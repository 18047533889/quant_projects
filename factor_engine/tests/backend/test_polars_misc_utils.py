# -*- coding: utf-8 -*-
"""Parity tests for simple native Polars panel utilities."""
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
    rng = np.random.default_rng(2718)
    index = pd.date_range("2022-01-01", periods=120, freq="D")
    x = pd.DataFrame(rng.normal(0.5, 2.0, (120, 2)), index=index, columns=["A", "B"])
    pb = pd.DataFrame(rng.uniform(0.1, 5.0, (120, 2)), index=index, columns=["A", "B"])
    pe = pd.DataFrame(rng.uniform(1.0, 40.0, (120, 2)), index=index, columns=["A", "B"])
    return x, pb, pe


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


def test_native_polars_misc_utils_matches_pandas(panels):
    x, pb, pe = panels
    cases = [
        ("sqrt_abs", (x,), {}),
        ("book_to_price", (pb,), {}),
        ("earnings_yield", (pe,), {}),
        ("float_share_ratio", (pb, pe), {}),
        ("free_float_share_ratio", (pb, pe), {}),
        ("ts_ratio", (x,), {}),
        ("ts_sma_cn", (x,), {"n": 7, "m": 2}),
        ("ts_sma_cn", (x,), {"n": 10, "m": 3}),
        ("ts_positive_ratio", (x,), {"window": 20, "threshold": 0.5}),
        ("ts_negative_ratio", (x,), {"window": 20, "threshold": 0.0}),
        ("ts_zero_ratio", (x,), {"window": 20, "tolerance": 0.1}),
    ]
    for name, args, kwargs in cases:
        _assert_parity(name, args, kwargs)


def test_native_polars_misc_utils_nan_warmup_matches_pandas():
    rng = np.random.default_rng(55)
    index = pd.date_range("2023-01-01", periods=80, freq="D")
    mask = rng.random((80, 2)) < 0.12
    x = pd.DataFrame(rng.normal(0.5, 2.0, (80, 2)), index=index, columns=["A", "B"])
    x[mask] = np.nan
    pb = pd.DataFrame(rng.uniform(0.1, 5.0, (80, 2)), index=index, columns=["A", "B"])
    pb[mask] = np.nan
    cases = [
        ("sqrt_abs", (x,), {}),
        ("book_to_price", (pb,), {}),
        ("float_share_ratio", (pb, x.abs() + 1.0), {}),
        ("ts_ratio", (x,), {}),
        ("ts_sma_cn", (x,), {"n": 5, "m": 2}),
        ("ts_positive_ratio", (x,), {"window": 8, "threshold": 0.5}),
        ("ts_negative_ratio", (x,), {"window": 8, "threshold": 0.0}),
        ("ts_zero_ratio", (x,), {"window": 8, "tolerance": 0.1}),
    ]
    for name, args, kwargs in cases:
        _assert_parity(name, args, kwargs)
