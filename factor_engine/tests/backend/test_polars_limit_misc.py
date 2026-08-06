# -*- coding: utf-8 -*-
"""Parity tests for limit-state / benchmark / cash-flow / date Polars backends."""
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
    rng = np.random.default_rng(66)
    index = pd.date_range("2022-01-01", periods=120, freq="D")
    close = pd.DataFrame(rng.uniform(5, 50, (120, 2)), index=index, columns=["A", "B"])
    upper = close * 1.1
    lower = close * 0.9
    ret = pd.DataFrame(rng.normal(0, 0.01, (120, 2)), index=index, columns=["A", "B"])
    bmark = pd.DataFrame(rng.normal(0, 0.01, (120, 2)), index=index, columns=["A", "B"])
    op = pd.DataFrame(rng.uniform(5, 50, (120, 2)), index=index, columns=["A", "B"])
    ocf = pd.DataFrame(rng.normal(1, 2, (120, 2)), index=index, columns=["A", "B"])
    icf = pd.DataFrame(rng.normal(-1, 2, (120, 2)), index=index, columns=["A", "B"])
    fcf = pd.DataFrame(rng.normal(0, 1, (120, 2)), index=index, columns=["A", "B"])
    event = pd.DataFrame(rng.random((120, 2)) > 0.6, index=index, columns=["A", "B"]).astype(float)
    # date panels
    d1 = pd.DataFrame(np.array([index + pd.Timedelta(days=int(k)) for k in np.random.default_rng(1).integers(-10, 10, 2)]).T, index=index, columns=["A", "B"])
    d2 = pd.DataFrame(np.array([index + pd.Timedelta(days=int(k) + 30) for k in np.random.default_rng(1).integers(-10, 10, 2)]).T, index=index, columns=["A", "B"])
    return close, upper, lower, ret, bmark, op, ocf, icf, fcf, event, d1, d2


def _polars(frame: pd.DataFrame) -> pl.DataFrame:
    return pl.DataFrame({column: frame[column].tolist() for column in frame.columns})


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


def test_native_polars_limit_misc_matches_pandas(panels):
    close, upper, lower, ret, bmark, op, ocf, icf, fcf, event, d1, d2 = panels
    cases = [
        ("ashare_limit_distance", (close, upper), {}),
        ("ashare_limit_touch", (close, upper), {"tick_tolerance": 0.005}),
        ("ashare_limit_open_break", (op, upper), {}),
        ("ashare_limit_one_price", (close, upper, lower), {}),
        ("ashare_limit_failed", (close, upper), {"window": 20}),
        ("benchmark_excess_return", (ret, bmark), {}),
        ("benchmark_relative_price", (close, upper), {}),
        ("fin_applicability_mask", (ret,), {"threshold": 0.0}),
        ("cash_flow_lifecycle_stage", (ocf, icf, fcf), {}),
        ("event_decay_asof", (event,), {"half_life": 20.0}),
    ]
    for name, args, kwargs in cases:
        _assert_parity(name, args, kwargs)
