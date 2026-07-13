# -*- coding: utf-8
"""Long-table Polars 表达式后端 parity 测试。"""
from __future__ import annotations

import os

import pandas as pd
import pytest

pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from backend.polars_expr_backend import POLARS_LONG_NATIVE, plan_is_polars_expr_capable
from cleaned_operators import load_all
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


@pytest.fixture(scope="module")
def source():
    load_all()
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-01"), "A"),
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-04"), "A"),
            (pd.Timestamp("2024-01-01"), "B"),
            (pd.Timestamp("2024-01-02"), "B"),
            (pd.Timestamp("2024-01-03"), "B"),
            (pd.Timestamp("2024-01-04"), "B"),
        ],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([10.0, 11.0, 10.5, 12.0, 20.0, 21.0, 20.5, 22.0], index=idx)
    ret = pd.Series([0.01, 0.02, -0.01, 0.03, 0.04, 0.01, 0.02, 0.05], index=idx)
    x = pd.Series([1.0, 2.0, 4.0, 8.0, 10.0, 20.0, 40.0, 80.0], index=idx)
    y = pd.Series([0.5, 1.0, 2.0, 4.0, 5.0, 10.0, 20.0, 40.0], index=idx)
    grp = pd.Series([1, 1, 1, 1, 2, 2, 2, 2], index=idx, dtype=float)
    return InMemorySeriesSource(data={"close": close, "ret": ret, "x": x, "y": y, "grp": grp})


def _run_expr(source, expr, *, use_polars_expr: bool):
    os.environ["FACTOR_ENGINE_DISABLE_BOTTLENECK"] = "1"
    if use_polars_expr:
        os.environ["FACTOR_ENGINE_POLARS_EXPR"] = "1"
    else:
        os.environ.pop("FACTOR_ENGINE_POLARS_EXPR", None)
    try:
        eng = FactorEngine(backend=build_backend("polars"), data_source=source)
        return eng.run(Factor(name="t", expr=expr))
    finally:
        os.environ.pop("FACTOR_ENGINE_DISABLE_BOTTLENECK", None)
        os.environ.pop("FACTOR_ENGINE_POLARS_EXPR", None)


def test_polars_expr_capable_subset_of_production_ops():
    from cleaned_operators.operator_policy import POLARS_PRODUCTION_SAFE

    # native expr 已实现但 production 门禁尚未同步的算子
    # native long path 已有；panel production parity 以 ``where`` 别名代表
    pending_production = {
        "bfill",
        "if_else",
        "is_infinite",
        "div_or_default",
        "log_fill_invalid",
        "div_or_null",
        # group agg native 已实现，production 门禁尚未并入
        "group_sum",
        "group_min",
        "group_max",
        "group_count",
    }
    native_ops = POLARS_LONG_NATIVE - {
        "column",
        "literal",
        "materialized_series",
        "plan_ref",
    }
    assert native_ops - pending_production <= POLARS_PRODUCTION_SAFE


@pytest.mark.parametrize(
    "factory_name,expr_builder",
    [
        ("ts_mean", lambda: make_cleaned_call_factory("ts_mean")(col("close"), 2)),
        ("rank_ts_mean", lambda: make_cleaned_call_factory("rank")(make_cleaned_call_factory("ts_mean")(col("close"), 2))),
        ("zscore", lambda: make_cleaned_call_factory("zscore")(col("close"))),
        ("rank_pct", lambda: make_cleaned_call_factory("rank_pct")(col("close"))),
        ("log_returns", lambda: make_cleaned_call_factory("log_returns")(col("close"))),
        ("volatility", lambda: make_cleaned_call_factory("volatility")(col("ret"), 2)),
        ("ts_zscore", lambda: make_cleaned_call_factory("ts_zscore")(col("close"), 2)),
        ("ts_corr", lambda: make_cleaned_call_factory("ts_corr")(col("x"), col("y"), 2)),
        ("winsorize", lambda: make_cleaned_call_factory("winsorize")(col("x"), 0.25)),
        ("coalesce", lambda: make_cleaned_call_factory("coalesce")(col("x"), col("y"))),
        ("where", lambda: make_cleaned_call_factory("where")(col("x"), col("x"), col("y"))),
        ("group_rank", lambda: make_cleaned_call_factory("group_rank")(col("x"), col("grp"))),
        ("group_zscore", lambda: make_cleaned_call_factory("group_zscore")(col("x"), col("grp"))),
        ("cs_quantile", lambda: make_cleaned_call_factory("cs_quantile")(col("x"), 0.5)),
        ("cum_sum", lambda: make_cleaned_call_factory("cum_sum")(col("x"))),
        ("gt", lambda: make_cleaned_call_factory("gt")(col("x"), col("y"))),
        ("ewm_mean", lambda: make_cleaned_call_factory("ewm_mean")(col("close"), 2)),
        ("ts_rank", lambda: make_cleaned_call_factory("ts_rank")(col("close"), 2)),
        ("maximum", lambda: make_cleaned_call_factory("maximum")(col("x"), col("y"))),
        ("cum_max", lambda: make_cleaned_call_factory("cum_max")(col("x"))),
        ("is_finite", lambda: make_cleaned_call_factory("is_finite")(col("x"))),
        ("fillna", lambda: make_cleaned_call_factory("fillna")(col("x"), 0)),
        ("group_winsorize", lambda: make_cleaned_call_factory("group_winsorize")(col("x"), col("grp"))),
        (
            "group_decay_linear",
            lambda: make_cleaned_call_factory("group_decay_linear")(col("x"), col("grp"), 5),
        ),
        ("cs_mad", lambda: make_cleaned_call_factory("cs_mad")(col("x"))),
        ("cs_mad_zscore", lambda: make_cleaned_call_factory("cs_mad_zscore")(col("x"))),
        ("c_mean", lambda: make_cleaned_call_factory("c_mean")(col("x"))),
        ("c_std", lambda: make_cleaned_call_factory("c_std")(col("x"))),
        ("ts_decay_linear", lambda: make_cleaned_call_factory("ts_decay_linear")(col("close"), 2)),
        ("WMA", lambda: make_cleaned_call_factory("WMA")(col("close"), 2)),
        ("ts_mad", lambda: make_cleaned_call_factory("ts_mad")(col("x"), 2)),
        ("ts_quantile", lambda: make_cleaned_call_factory("ts_quantile")(col("x"), 3, 0.5)),
        ("ts_product", lambda: make_cleaned_call_factory("ts_product")(col("x"), 2)),
        ("ts_skew", lambda: make_cleaned_call_factory("ts_skew")(col("x"), 3)),
        ("ts_argmax", lambda: make_cleaned_call_factory("ts_argmax")(col("x"), 2)),
        ("ts_regression", lambda: make_cleaned_call_factory("ts_regression")(col("x"), col("y"), 3)),
        ("Slope", lambda: make_cleaned_call_factory("Slope")(col("x"), 3)),
        ("log_abs", lambda: make_cleaned_call_factory("log_abs")(col("x"))),
        ("signed_log", lambda: make_cleaned_call_factory("signed_log")(col("x"))),
        ("ts_sharpe", lambda: make_cleaned_call_factory("ts_sharpe")(col("ret"), 2)),
        ("ts_autocorr", lambda: make_cleaned_call_factory("ts_autocorr")(col("ret"), 3)),
        ("cum_delta", lambda: make_cleaned_call_factory("cum_delta")(col("x"))),
        ("expanding_mean", lambda: make_cleaned_call_factory("expanding_mean")(col("x"))),
        ("expanding_std", lambda: make_cleaned_call_factory("expanding_std")(col("x"))),
        ("expanding_sum", lambda: make_cleaned_call_factory("expanding_sum")(col("x"))),
        ("count", lambda: make_cleaned_call_factory("count")(col("x"))),
        ("ewm_corr", lambda: make_cleaned_call_factory("ewm_corr")(col("x"), col("y"), 3)),
        ("ewm_cov", lambda: make_cleaned_call_factory("ewm_cov")(col("x"), col("y"), 3)),
        ("ts_ratio", lambda: make_cleaned_call_factory("ts_ratio")(col("close"))),
        ("ts_kurt", lambda: make_cleaned_call_factory("ts_kurt")(col("x"), 4)),
        ("ts_moment", lambda: make_cleaned_call_factory("ts_moment")(col("x"), 3, 2)),
        ("ts_max_buildup", lambda: make_cleaned_call_factory("ts_max_buildup")(col("x"), 2)),
        ("expanding_rank", lambda: make_cleaned_call_factory("expanding_rank")(col("x"))),
        ("quantile", lambda: make_cleaned_call_factory("quantile")(col("x"), 2)),
        ("fillna_interpolate", lambda: make_cleaned_call_factory("fillna_interpolate")(col("x"))),
        ("RSI_WILDER", lambda: make_cleaned_call_factory("RSI_WILDER")(col("close"), 2)),
    ],
)
def test_polars_expr_matches_polars_bridge(source, factory_name, expr_builder):
    expr = expr_builder()
    from runtime.engine import FactorEngine
    from backend.factory import build_backend
    from api.factor import Factor

    # compile plan for capability check
    eng = FactorEngine(backend=build_backend("pandas"), data_source=source)
    plan, _ = eng.compile(Factor(name="t", expr=expr))
    assert plan_is_polars_expr_capable(plan), factory_name

    base = _run_expr(source, expr, use_polars_expr=False)["result"].sort_index()
    fast_out = _run_expr(source, expr, use_polars_expr=True)
    fast = fast_out["result"].sort_index()
    assert fast_out.get("polars_expr") is True, factory_name
    pd.testing.assert_series_equal(base, fast, check_names=False, rtol=1e-6, atol=1e-6)


def test_polars_expr_fallback_on_unsupported_op(source):
    from api import MACD

    os.environ["FACTOR_ENGINE_POLARS_EXPR"] = "1"
    try:
        eng = FactorEngine(backend=build_backend("polars"), data_source=source)
        out = eng.run(Factor(name="t", expr=MACD(col("close"))))
        assert not out.get("polars_expr")
    finally:
        os.environ.pop("FACTOR_ENGINE_POLARS_EXPR", None)
