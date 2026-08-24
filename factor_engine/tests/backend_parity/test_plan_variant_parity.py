# -*- coding: utf-8
"""同一算子在 direct / fused / nested 路径上须与 Pandas 一致。"""
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
    return InMemorySeriesSource(data={"close": close, "open": open_})


def _run(source, expr, backend: str):
    return FactorEngine(backend=build_backend(backend), data_source=source).run(
        Factor(name="t", expr=expr)
    )["result"].sort_index()


@pytest.mark.parametrize(
    "label,expr_builder",
    [
        ("direct_add", lambda: F("add")(col("close"), col("open"))),
        (
            "fused_ts_add",
            lambda: F("add")(F("ts_mean")(col("close"), 3), F("ts_std")(col("close"), 3)),
        ),
        (
            "nested_where",
            lambda: F("where")(
                F("gt")(col("close"), col("open")),
                F("ts_mean")(col("close"), 3),
                F("ts_mean")(col("open"), 3),
            ),
        ),
        (
            "fused_rank_ts_mean",
            lambda: F("rank_pct")(F("ts_mean")(col("close"), 3)),
        ),
        (
            "fused_zscore_ts_mean",
            lambda: F("zscore")(F("ts_mean")(col("close"), 3)),
        ),
    ],
)
def test_plan_variant_matches_pandas(_loaded, ts_source, label, expr_builder):
    expr = expr_builder()
    pd_out = _run(ts_source, expr, "pandas")
    long_out = _run(ts_source, expr, "polars_long")
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False, rtol=1e-5, atol=1e-5)
