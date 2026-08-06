# -*- coding: utf-8 -*-
"""Parity tests for return-decomposition / state-event / max-buildup Polars."""
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
    rng = np.random.default_rng(5150)
    index = pd.date_range("2022-01-01", periods=140, freq="D")
    open_ = pd.DataFrame(rng.uniform(5, 50, (140, 2)), index=index, columns=["A", "B"])
    close = open_ + pd.DataFrame(rng.normal(0, 2.0, (140, 2)), index=index, columns=["A", "B"])
    pre_close = pd.DataFrame(rng.uniform(5, 50, (140, 2)), index=index, columns=["A", "B"])
    vwap = pd.DataFrame(rng.uniform(5, 50, (140, 2)), index=index, columns=["A", "B"])
    condition = pd.DataFrame(rng.random((140, 2)) > 0.5, index=index, columns=["A", "B"]).astype(float)
    return open_, close, pre_close, vwap, condition


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


def test_native_polars_state_event_matches_pandas(panels):
    o, c, pc, vwap, cond = panels
    cases = [
        ("open_close_return", (o, c), {}),
        ("open_to_vwap_return", (o, vwap), {}),
        ("overnight_return", (o, pc), {}),
        ("vwap_to_close_return", (vwap, c), {}),
        ("ts_max_buildup", (c,), {"d": 20}),
        ("ts_transition_count", (cond,), {"window": 20}),
        ("ts_time_since_change", (cond,), {"max_lookback": 50}),
        ("ts_time_since_change", (cond,), {}),
        ("ts_event_spacing_mean", (cond,), {"window": 60, "min_events": 2}),
        ("ts_event_spacing_cv", (cond,), {"window": 60, "min_events": 3}),
    ]
    for name, args, kwargs in cases:
        _assert_parity(name, args, kwargs)


def test_native_polars_state_event_nan_warmup_matches_pandas():
    rng = np.random.default_rng(9)
    index = pd.date_range("2023-01-01", periods=90, freq="D")
    mask = rng.random((90, 2)) < 0.12
    o = pd.DataFrame(rng.uniform(5, 50, (90, 2)), index=index, columns=["A", "B"])
    o[mask] = np.nan
    c = o + pd.DataFrame(rng.normal(0, 2.0, (90, 2)), index=index, columns=["A", "B"])
    pc = pd.DataFrame(rng.uniform(5, 50, (90, 2)), index=index, columns=["A", "B"])
    cond = pd.DataFrame(rng.random((90, 2)) > 0.5, index=index, columns=["A", "B"]).astype(float)
    cases = [
        ("open_close_return", (o, c), {}),
        ("overnight_return", (o, pc), {}),
        ("ts_max_buildup", (c,), {"d": 10}),
        ("ts_transition_count", (cond,), {"window": 10}),
        ("ts_time_since_change", (cond,), {"max_lookback": 30}),
        ("ts_event_spacing_mean", (cond,), {"window": 30, "min_events": 2}),
        ("ts_event_spacing_cv", (cond,), {"window": 30, "min_events": 3}),
    ]
    for name, args, kwargs in cases:
        _assert_parity(name, args, kwargs)
