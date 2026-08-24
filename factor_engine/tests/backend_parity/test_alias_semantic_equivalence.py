# -*- coding: utf-8
"""Alias / rewrite 路径必须与 canonical 实现语义等价。"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("polars")

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.cleaned_operators import load_all
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource

F = make_cleaned_call_factory


@pytest.fixture(scope="module")
def _loaded():
    load_all()


@pytest.fixture(scope="module")
def ts_source():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([10.0, 11.0, 10.5, 12.0, 20.0, 21.0, 20.5, 22.0], index=idx)
    open_ = pd.Series([9.5, 10.5, 10.0, 11.5, 19.0, 20.5, 20.0, 21.5], index=idx)
    ret = pd.Series([0.01, 0.02, -0.01, 0.03, 0.04, 0.01, 0.02, 0.05], index=idx)
    bm = pd.Series([0.005, 0.015, -0.005, 0.02, 0.03, 0.008, 0.015, 0.04], index=idx)
    return InMemorySeriesSource(data={"close": close, "open": open_, "ret": ret, "bm": bm})


def _run(source, expr, backend: str):
    return FactorEngine(backend=build_backend(backend), data_source=source).run(
        Factor(name="t", expr=expr)
    )["result"].sort_index()


@pytest.mark.parametrize("backend", ["pandas", "polars_long"])
def test_wma_equals_ts_decay_linear(_loaded, ts_source, backend):
    w = 3
    a = _run(ts_source, F("WMA")(col("close"), w), backend)
    b = _run(ts_source, F("ts_decay_linear")(col("close"), w), backend)
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("backend", ["pandas", "polars_long"])
def test_rolling_beta_equals_ts_beta(_loaded, ts_source, backend):
    w = 3
    a = _run(ts_source, F("rolling_beta")(col("ret"), col("bm"), w), backend)
    b = _run(ts_source, F("ts_beta")(col("ret"), col("bm"), w), backend)
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backend", ["pandas", "polars_long"])
def test_div_or_null_equals_protected_div(_loaded, ts_source, backend):
    a = _run(ts_source, F("div_or_null")(col("ret"), col("bm")), backend)
    b = _run(ts_source, F("protected_div")(col("ret"), col("bm")), backend)
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("backend", ["pandas", "polars_long"])
def test_if_else_equals_where(_loaded, ts_source, backend):
    cond = F("gt")(col("close"), col("open"))
    a = _run(ts_source, F("if_else")(cond, col("close"), col("open")), backend)
    b = _run(ts_source, F("where")(cond, col("close"), col("open")), backend)
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-6, atol=1e-6)
