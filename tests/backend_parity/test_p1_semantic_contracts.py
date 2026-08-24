# -*- coding: utf-8
"""P1 语义契约：coalesce n-ary、nan_to_num 参数、cum NULL、ts_rank min_periods、ts_ratio lowering。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skip(reason="legacy P1 aliases superseded by the canonical evidence registry")

pytest.importorskip("polars")

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.cleaned_operators import load_all
from factor_engine.planner.composite_lowering import lower_composite_operators
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


@pytest.fixture(scope="module")
def _loaded():
    load_all()


@pytest.fixture(scope="module")
def ts_source():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=8, freq="B"), ["A"]],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0], index=idx)
    alt = pd.Series([10.0, np.nan, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], index=idx)
    return InMemorySeriesSource(data={"close": close, "alt": alt})


def _run(source, expr, backend: str) -> pd.Series:
    out = FactorEngine(backend=build_backend(backend), data_source=source).run(
        Factor(name="t", expr=expr)
    )
    return out["result"].sort_index()


def test_coalesce_nary_polars_matches_pandas(_loaded, ts_source):
    expr = make_cleaned_call_factory("coalesce")(col("close"), col("alt"), 0.0)
    pd_out = _run(ts_source, expr, "pandas")
    pl_out = _run(ts_source, expr, "polars_long")
    pd.testing.assert_series_equal(pd_out, pl_out, check_names=False)


def test_nan_to_num_named_param_polars(_loaded, ts_source):
    expr = make_cleaned_call_factory("nan_to_num")(col("close"), num=7.0)
    pd_out = _run(ts_source, expr, "pandas")
    pl_out = _run(ts_source, expr, "polars_long")
    pd.testing.assert_series_equal(pd_out, pl_out, check_names=False)


def test_cum_sum_null_row_stays_null(_loaded, ts_source):
    expr = make_cleaned_call_factory("cum_sum")(col("close"))
    pd_out = _run(ts_source, expr, "pandas")
    pl_out = _run(ts_source, expr, "polars_long")
    assert pd.isna(pd_out.iloc[2])
    assert pd.isna(pl_out.iloc[2])
    pd.testing.assert_series_equal(pd_out, pl_out, check_names=False, rtol=1e-6, atol=1e-6)


def test_ts_rank_min_periods(_loaded, ts_source):
    w, mp = 4, 3
    expr = make_cleaned_call_factory("ts_rank")(col("close"), w, min_periods=mp)
    pd_out = _run(ts_source, expr, "pandas")
    pl_out = _run(ts_source, expr, "polars_long")
    assert pd.isna(pd_out.iloc[1])
    assert pd.isna(pl_out.iloc[1])
    pd.testing.assert_series_equal(pd_out, pl_out, check_names=False, rtol=1e-5, atol=1e-5)


def test_ts_ratio_composite_lowering():
    x = PlanNode(op="column", attrs={"name": "close"}, inputs=[])
    node = PlanNode(op="ts_ratio", inputs=[x], attrs={})
    lowered = lower_composite_operators(node)
    assert lowered.op == "safe_div_null"
    assert lowered.inputs[1].op == "ts_delay"


def test_cs_sum_all_null_cross_section(_loaded):
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=2, freq="B"), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([np.nan, np.nan, np.nan, np.nan], index=idx)
    src = InMemorySeriesSource(data={"close": close})
    expr = make_cleaned_call_factory("cs_sum")(col("close"))
    pd_out = _run(src, expr, "pandas")
    pl_out = _run(src, expr, "polars_long")
    assert pd_out.isna().all()
    assert pl_out.isna().all()
