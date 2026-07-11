# -*- coding: utf-8
"""Batch A / 低杠杆算子专用 edge 语义（volatility / count / cum / is_null）。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from backend.financial_semantics import volatility_annualization_factor
from cleaned_operators import load_all
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource

F = make_cleaned_call_factory


def _run(source, expr, backend: str):
    return FactorEngine(backend=build_backend(backend), data_source=source).run(
        Factor(name="t", expr=expr)
    )


def _series(run_out) -> pd.Series:
    return run_out["result"].sort_index()


def _triple_parity(source, expr, *, rtol=1e-5, atol=1e-5):
    pd_out = _series(_run(source, expr, "pandas"))
    pl_out = _series(_run(source, expr, "polars_long"))
    pd.testing.assert_series_equal(pd_out, pl_out, check_names=False, rtol=rtol, atol=atol)
    return pd_out


@pytest.fixture(scope="module")
def _loaded():
    load_all()


def test_volatility_uses_return_series_semantics(_loaded):
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-04"), "A"),
            (pd.Timestamp("2024-01-05"), "A"),
        ],
        names=["timestamp", "instrument"],
    )
    ret = pd.Series([0.01, 0.02, -0.01, 0.03], index=idx)
    src = InMemorySeriesSource(data={"ret": ret})
    out = _triple_parity(src, F("volatility")(col("ret"), 3))
    assert volatility_annualization_factor() == pytest.approx(252**0.5)
    assert not math.isnan(out.iloc[-1])


def test_volatility_constant_returns_zero_or_null(_loaded):
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(f"2024-01-{d:02d}"), "A") for d in range(2, 7)],
        names=["timestamp", "instrument"],
    )
    ret = pd.Series([0.01] * 5, index=idx)
    src = InMemorySeriesSource(data={"ret": ret})
    out = _triple_parity(src, F("volatility")(col("ret"), 3))
    tail = out.iloc[-1]
    assert tail == 0.0 or math.isnan(tail)


def test_volatility_window_not_full(_loaded):
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-03"), "A"),
        ],
        names=["timestamp", "instrument"],
    )
    ret = pd.Series([0.01, 0.02], index=idx)
    src = InMemorySeriesSource(data={"ret": ret})
    out = _triple_parity(src, F("volatility")(col("ret"), 5))
    assert math.isnan(out.iloc[0])


def test_volatility_multi_instrument_independent(_loaded):
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-04"), "A"),
            (pd.Timestamp("2024-01-02"), "B"),
            (pd.Timestamp("2024-01-03"), "B"),
            (pd.Timestamp("2024-01-04"), "B"),
        ],
        names=["timestamp", "instrument"],
    )
    ret = pd.Series([0.01, 0.02, 0.03, 0.10, -0.05, 0.02], index=idx)
    src = InMemorySeriesSource(data={"ret": ret})
    _triple_parity(src, F("volatility")(col("ret"), 2))


def test_count_null_not_counted(_loaded):
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-04"), "A"),
        ],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([1.0, np.nan, 3.0], index=idx)
    src = InMemorySeriesSource(data={"close": close})
    out = _triple_parity(src, F("count")(col("close")))
    assert out.iloc[0] == 1.0
    assert out.iloc[1] == 1.0
    assert out.iloc[2] == 2.0


def test_cum_max_current_null_outputs_null(_loaded):
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-04"), "A"),
        ],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([1.0, np.nan, 3.0], index=idx)
    src = InMemorySeriesSource(data={"close": close})
    out = _triple_parity(src, F("cum_max")(col("close")))
    assert out.iloc[0] == 1.0
    assert math.isnan(out.iloc[1])
    assert out.iloc[2] == 3.0


def test_expanding_sum_current_null_outputs_null(_loaded):
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-04"), "A"),
        ],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([1.0, np.nan, 2.0], index=idx)
    src = InMemorySeriesSource(data={"close": close})
    out = _triple_parity(src, F("expanding_sum")(col("close")))
    assert out.iloc[0] == 1.0
    assert math.isnan(out.iloc[1])
    assert out.iloc[2] == 3.0


def test_is_null_truth_table(_loaded):
    """``is_null`` 与 Pandas ``isna`` 对齐：NULL/NaN → 1。"""
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "A")],
        names=["timestamp", "instrument"],
    )
    for val, expected in (
        (np.nan, 1.0),
        (float("inf"), 0.0),
        (float("-inf"), 0.0),
        (1.0, 0.0),
    ):
        src = InMemorySeriesSource(data={"close": pd.Series([val], index=idx)})
        out = _triple_parity(src, F("is_null")(col("close")))
        assert out.iloc[0] == expected


def test_is_nan_truth_table(_loaded):
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "A")],
        names=["timestamp", "instrument"],
    )
    cases = [
        (np.nan, 1.0),
        (float("inf"), 0.0),
        (1.0, 0.0),
    ]
    for val, expected in cases:
        src = InMemorySeriesSource(data={"close": pd.Series([val], index=idx)})
        out = _triple_parity(src, F("is_nan")(col("close")))
        assert out.iloc[0] == expected
