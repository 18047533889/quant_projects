# -*- coding: utf-8
"""polars_long / auto_long 在 CSE (plan_ref) 与 SQL partial (materialized_series) 下不 fallback。"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("polars")

from api import rank, ts_mean
from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.context import ExecutionContext
from backend.factory import build_backend
from backend.polars_long_backend import PolarsLongBackend
from planner.logical_plan import PlanNode
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


@pytest.fixture(scope="module")
def source():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([10.0, 11.0, 20.0, 21.0, 10.5, 12.0], index=idx)
    return InMemorySeriesSource(data={"close": close})


def test_polars_long_run_many_plan_ref_no_fallback(source):
    sub = ts_mean(col("close"), 2)
    f1 = Factor(name="a", expr=sub)
    f2 = Factor(name="b", expr=rank(sub))
    eng = FactorEngine(backend=build_backend("polars_long"), data_source=source)
    out = eng.run_many([f1, f2])
    assert len(out["dag"].shared_nodes) >= 1
    assert not out.get("production_pandas_fallbacks")
    pd_out = FactorEngine(backend=build_backend("pandas"), data_source=source).run_many([f1, f2])
    for name in ("a", "b"):
        pd.testing.assert_series_equal(
            pd_out["results"][name].sort_index(),
            out["results"][name].sort_index(),
            check_names=False,
            rtol=1e-6,
            atol=1e-6,
        )


def test_polars_long_materialized_series_no_fallback(source):
    """SQL partial 物化列应被 long emitter 直接消费，而非 fallback。"""
    mat = make_cleaned_call_factory("ts_mean")(col("close"), 2)
    pd_eng = FactorEngine(backend=build_backend("pandas"), data_source=source)
    mat_series = pd_eng.run(Factor(name="m", expr=mat))["result"]
    sid = "mat_ts_mean"
    root = PlanNode(
        op="add",
        attrs={},
        inputs=[
            PlanNode(op="materialized_series", attrs={"sid": sid}, inputs=[]),
            PlanNode(op="column", attrs={"name": "close"}, inputs=[]),
        ],
    )
    ctx = ExecutionContext(
        data_source=source,
        timestamp_col="timestamp",
        instrument_col="instrument",
        materialized_series={sid: mat_series},
    )
    backend = PolarsLongBackend()
    result = backend.execute(root, ctx)
    runtime = ctx.runtime_stats or {}
    assert runtime.get("used_polars_long_path") is True
    assert runtime.get("used_polars_long_native") is True
    assert not runtime.get("polars_long_fallback_reason")
    expected = (mat_series + source.data["close"]).sort_index()
    pd.testing.assert_series_equal(result.sort_index(), expected, check_names=False, rtol=1e-6, atol=1e-6)


def test_plan_ref_reuses_lazy_cache_no_pandas_roundtrip(source, monkeypatch):
    """CSE 共享子树应缓存 LazyFrame；``plan_ref`` 不得再 ``series_to_polars_long_lazy``。"""
    from unittest.mock import patch

    calls = {"count": 0}
    import backend.long_frame as lf_mod

    orig = lf_mod.series_to_polars_long_lazy

    def _counting(*args, **kwargs):
        calls["count"] += 1
        return orig(*args, **kwargs)

    sub = ts_mean(col("close"), 2)
    f1 = Factor(name="a", expr=sub)
    f2 = Factor(name="b", expr=rank(sub))
    eng = FactorEngine(backend=build_backend("polars_long"), data_source=source)
    with patch.object(lf_mod, "series_to_polars_long_lazy", side_effect=_counting):
        out = eng.run_many([f1, f2])
    assert len(out["dag"].shared_nodes) >= 1
    assert not out.get("polars_long_fallback_reason")
    # 共享子树 collect 前不应因 plan_ref 再次走 pandas long 转换
    assert calls["count"] == 0


def test_binary_column_fusion_from_wide_base(source):
    """两列二元 op 应直接宽表 expr，结果与 pandas 一致。"""
    close = source.data["close"]
    volume = pd.Series([100.0, 110.0, 200.0, 210.0, 150.0, 160.0], index=close.index)
    src = InMemorySeriesSource(data={"close": close, "volume": volume})
    expr = make_cleaned_call_factory("protected_div")(col("close"), col("volume"))
    pd_out = FactorEngine(backend=build_backend("pandas"), data_source=src).run(
        Factor(name="t", expr=expr)
    )
    long_out = FactorEngine(backend=build_backend("polars_long"), data_source=src).run(
        Factor(name="t", expr=expr)
    )
    assert long_out.get("used_polars_long_native") is True
    pd.testing.assert_series_equal(
        pd_out["result"].sort_index(),
        long_out["result"].sort_index(),
        check_names=False,
        rtol=1e-6,
        atol=1e-6,
    )


def test_polars_long_plan_ref_direct_no_fallback(source):
    sub = ts_mean(col("close"), 2)
    pd_eng = FactorEngine(backend=build_backend("pandas"), data_source=source)
    shared = pd_eng.run(Factor(name="s", expr=sub))["result"]
    sid = "shared_ts_mean"
    root = PlanNode(
        op="rank",
        attrs={},
        inputs=[PlanNode(op="plan_ref", attrs={"sid": sid}, inputs=[])],
    )
    ctx = ExecutionContext(
        data_source=source,
        timestamp_col="timestamp",
        instrument_col="instrument",
        shared_result_cache={sid: shared},
        shared_long_lazy_cache={},
    )
    backend = PolarsLongBackend()
    result = backend.execute(root, ctx)
    runtime = ctx.runtime_stats or {}
    assert runtime.get("used_polars_long_path") is True
    assert runtime.get("used_polars_long_native") is True
    expected = pd_eng.run(Factor(name="r", expr=rank(sub)))["result"].sort_index()
    pd.testing.assert_series_equal(result.sort_index(), expected, check_names=False, rtol=1e-6, atol=1e-6)
