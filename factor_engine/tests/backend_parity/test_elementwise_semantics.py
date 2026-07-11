# -*- coding: utf-8
"""元素级算子语义：max/min NULL、比较 NULL 传播、protected 域、fusion 路径。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.elementwise_semantics import (
    comparison_null_propagates,
    protected_div_null_preserved,
    protected_log_null_preserved,
)
from backend.factory import build_backend
from cleaned_operators import load_all
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource

F = make_cleaned_call_factory


@pytest.fixture(scope="module")
def _loaded():
    load_all()


def _sparse_src():
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-04"), "A"),
        ],
        names=["timestamp", "instrument"],
    )
    a = pd.Series([1.0, np.nan, 3.0], index=idx)
    b = pd.Series([2.0, 4.0, np.nan], index=idx)
    return InMemorySeriesSource(data={"a": a, "b": b, "close": a, "open": b})


def _run(src, expr, backend: str):
    return FactorEngine(backend=build_backend(backend), data_source=src).run(
        Factor(name="t", expr=expr)
    )["result"].sort_index()


def test_semantic_flags():
    assert comparison_null_propagates()
    assert protected_log_null_preserved()
    assert protected_div_null_preserved()


@pytest.mark.parametrize(
    "factory,builder",
    [
        ("maximum_col", lambda: F("maximum")(col("a"), col("b"))),
        ("minimum_col", lambda: F("minimum")(col("a"), col("b"))),
    ],
)
def test_max_min_null_is_null(_loaded, factory, builder):
    src = _sparse_src()
    expr = builder()
    pd_out = _run(src, expr, "pandas")
    long_out = _run(src, expr, "polars_long")
    assert pd.isna(pd_out.iloc[1])
    assert pd.isna(pd_out.iloc[2])
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False)


def test_max_min_ts_join_path_null_is_null(_loaded):
    """generic join 路径：ts_mean 子树一侧为 NULL 时 maximum 也应为 NULL。"""
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-04"), "A"),
        ],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([1.0, np.nan, np.nan], index=idx)
    open_ = pd.Series([np.nan, np.nan, 5.0], index=idx)
    src = InMemorySeriesSource(data={"close": close, "open": open_})
    expr = F("maximum")(F("ts_mean")(col("close"), 2), F("ts_mean")(col("open"), 2))
    pd_out = _run(src, expr, "pandas")
    long_out = _run(src, expr, "polars_long")
    assert pd.isna(pd_out.iloc[0])
    assert pd.isna(pd_out.iloc[1])
    assert pd.isna(pd_out.iloc[2])
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False)


@pytest.mark.parametrize("op", ["gt", "lt", "eq", "ge", "le", "ne"])
def test_compare_null_is_null(_loaded, op):
    src = _sparse_src()
    expr = F(op)(col("a"), col("b"))
    pd_out = _run(src, expr, "pandas")
    long_out = _run(src, expr, "polars_long")
    assert pd.isna(pd_out.iloc[1])
    assert pd.isna(pd_out.iloc[2])
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False)


def test_protected_log_null_stays_null(_loaded):
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "A"), (pd.Timestamp("2024-01-03"), "A")],
        names=["timestamp", "instrument"],
    )
    x = pd.Series([np.nan, 0.5], index=idx)
    src = InMemorySeriesSource(data={"x": x})
    expr = F("protected_log")(col("x"))
    pd_out = _run(src, expr, "pandas")
    long_out = _run(src, expr, "polars_long")
    assert pd.isna(pd_out.iloc[0])
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False, rtol=1e-6)


def test_protected_div_null_stays_null(_loaded):
    src = _sparse_src()
    expr = F("protected_div")(col("a"), col("b"))
    pd_out = _run(src, expr, "pandas")
    long_out = _run(src, expr, "polars_long")
    assert pd.isna(pd_out.iloc[1])
    assert pd.isna(pd_out.iloc[2])
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False)


def test_is_infinite_table(_loaded):
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-04"), "A"),
            (pd.Timestamp("2024-01-05"), "A"),
        ],
        names=["timestamp", "instrument"],
    )
    x = pd.Series([np.nan, float("inf"), float("-inf"), 1.0], index=idx)
    src = InMemorySeriesSource(data={"x": x})
    expr = F("is_infinite")(col("x"))
    long_out = _run(src, expr, "polars_long")
    assert long_out.iloc[0] == 0.0
    assert long_out.iloc[1] == 1.0
    assert long_out.iloc[2] == 1.0
    assert long_out.iloc[3] == 0.0


def test_is_finite_table(_loaded):
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-04"), "A"),
            (pd.Timestamp("2024-01-05"), "A"),
        ],
        names=["timestamp", "instrument"],
    )
    x = pd.Series([np.nan, float("inf"), 1.0, -2.5], index=idx)
    src = InMemorySeriesSource(data={"x": x})
    expr = F("is_finite")(col("x"))
    pd_out = _run(src, expr, "pandas")
    long_out = _run(src, expr, "polars_long")
    assert pd_out.iloc[0] == 0.0
    assert pd_out.iloc[1] == 0.0
    assert pd_out.iloc[2] == 1.0
    assert pd_out.iloc[3] == 1.0
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False)
