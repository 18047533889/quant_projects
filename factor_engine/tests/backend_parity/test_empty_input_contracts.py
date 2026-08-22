# -*- coding: utf-8
"""空输入 / 全 NULL 窗口 / 单 instrument 边界契约。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from cleaned_operators import load_all
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource

F = make_cleaned_call_factory


@pytest.fixture(scope="module")
def _loaded():
    load_all()


def _run(source, expr, backend: str):
    return FactorEngine(backend=build_backend(backend), data_source=source).run(
        Factor(name="t", expr=expr)
    )["result"].sort_index()


def _empty_src():
    idx = pd.MultiIndex.from_tuples([], names=["timestamp", "instrument"])
    x = pd.Series([], index=idx, dtype=float)
    return InMemorySeriesSource(data={"x": x, "close": x})


def _single_inst_all_null():
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "A"), (pd.Timestamp("2024-01-03"), "A")],
        names=["timestamp", "instrument"],
    )
    x = pd.Series([np.nan, np.nan], index=idx)
    return InMemorySeriesSource(data={"x": x, "close": x})


@pytest.mark.parametrize(
    "builder",
    [
        lambda: F("ts_mean")(col("x"), 2),
        lambda: F("rank")(col("x")),
        lambda: F("zscore")(col("x")),
        lambda: F("normalize")(col("x")),
        lambda: F("maximum")(col("x"), col("x")),
    ],
)
def test_empty_panel_does_not_crash(_loaded, builder):
    src = _empty_src()
    expr = builder()
    for backend in ("pandas", "polars_long"):
        out = _run(src, expr, backend)
        assert len(out) == 0


@pytest.mark.parametrize(
    "builder",
    [
        lambda: F("ts_mean")(col("x"), 2),
        lambda: F("zscore")(col("x")),
    ],
)
def test_all_null_window_stays_null_pandas_polars(_loaded, builder):
    src = _single_inst_all_null()
    expr = builder()
    pd_out = _run(src, expr, "pandas")
    long_out = _run(src, expr, "polars_long")
    assert pd_out.isna().all()
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False)
