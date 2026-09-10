# -*- coding: utf-8
"""polars_long / auto_long 在 CSE (plan_ref) 与 SQL partial (materialized_series) 下不 fallback。"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("polars")

from factor_engine.api import rank, ts_mean
from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.context import ExecutionContext
from factor_engine.backend.factory import build_backend
from factor_engine.backend.polars_long_backend import PolarsLongBackend
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.runtime.engine import FactorEngine
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
    import factor_engine.backend.long_frame as lf_mod

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


def test_run_many_shared_lazy_only_defers_collect(source, monkeypatch):
    """CSE 共享子树应 compile-only，不在 shared 阶段 collect。"""
    import polars as pl
    from contextvars import ContextVar
    from factor_engine.runtime import batch_service
    from factor_engine.runtime.buffer_ref import SourceWaveExecutor

    phase = ContextVar("collect_phase", default="root")
    collect_count = {"source": 0, "shared": 0, "root": 0}
    orig_collect = pl.LazyFrame.collect

    def in_phase(name, fn):
        def wrapped(*args, **kwargs):
            token = phase.set(name)
            try:
                return fn(*args, **kwargs)
            finally:
                phase.reset(token)
        return wrapped

    def counting_collect(self, *args, **kwargs):
        collect_count[phase.get()] += 1
        return orig_collect(self, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "collect", counting_collect)
    monkeypatch.setattr(SourceWaveExecutor, "execute_wave",
                        in_phase("source", SourceWaveExecutor.execute_wave))
    monkeypatch.setattr(batch_service, "_materialize_shared_subplan",
                        in_phase("shared", batch_service._materialize_shared_subplan))

    sub = ts_mean(col("close"), 2)
    f1 = Factor(name="a", expr=sub)
    f2 = Factor(name="b", expr=rank(sub))
    eng = FactorEngine(backend=build_backend("polars_long"), data_source=source)
    out = eng.run_many([f1, f2])
    assert len(out["dag"].shared_nodes) >= 1
    assert not out.get("production_pandas_fallbacks")
    # One bounded source-wave materialization is distinct from CSE collect.
    # Shared compilation stays lazy; each of the two roots collects once.
    assert collect_count == {"source": 1, "shared": 0, "root": 2}


def test_run_many_shared_lazy_only_skips_series_cache(source):
    """lazy-only 共享子树不应写入 ``shared_result_cache``。"""
    captured: dict[str, ExecutionContext] = {}

    sub = ts_mean(col("close"), 2)
    f1 = Factor(name="a", expr=sub)
    f2 = Factor(name="b", expr=rank(sub))
    eng = FactorEngine(backend=build_backend("polars_long"), data_source=source)
    orig_make = eng._make_context

    def hook(**kwargs):
        ctx = orig_make(**kwargs)
        captured["ctx"] = ctx
        return ctx

    eng._make_context = hook  # type: ignore[method-assign]
    out = eng.run_many([f1, f2])
    assert len(out["dag"].shared_nodes) >= 1
    ctx = captured["ctx"]
    ts_shared_sids = [
        sid for sid, sub in out["dag"].shared_nodes.items() if sub.op == "ts_mean"
    ]
    assert ts_shared_sids
    for sid in ts_shared_sids:
        assert sid not in (ctx.shared_result_cache or {})
        assert sid in (ctx.shared_long_lazy_cache or {})


def test_run_many_shared_lazy_records_hits(source):
    sub = ts_mean(col("close"), 2)
    f1 = Factor(name="a", expr=sub)
    f2 = Factor(name="b", expr=rank(sub))
    f3 = Factor(name="c", expr=make_cleaned_call_factory("zscore")(sub))
    eng = FactorEngine(backend=build_backend("polars_long"), data_source=source)
    out = eng.run_many([f1, f2, f3])
    summary = out.get("backend_path_summary") or {}
    paths = out.get("backend_paths") or {}
    hits = summary.get("shared_long_lazy_hits_total", 0)
    if hits < 1:
        hits = max(
            (int((p.get("backend_path_summary") or {}).get("shared_long_lazy_hits") or 0) for p in paths.values()),
            default=0,
        )
    assert hits >= 1


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
