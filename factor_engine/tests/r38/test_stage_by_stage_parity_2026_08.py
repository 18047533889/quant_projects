# -*- coding: utf-8 -*-
"""P0-020: physical stage-by-stage 执行基建 —— barrier 子树拆成真实可执行 stage。

验证 ``extract_stage_subplans`` 拆出的 reference-plan + stage 子计划，在 stage
逐棵执行并物化到 shared buffer 后，与「整 root 一次性执行」**数值等价**：
    - ts_rolling / cross-section 嵌套 barrier → 每个 stage 独立执行；
    - plan_ref 复用既有 CSE 解析机制；
    - ``stage-by-stage == whole-root``（parity，为 production 默认接线打地基）。
"""
from __future__ import annotations

import pandas as pd

from tests.helpers import InMemorySeriesSource


def _make_series(dates: list[str], assets: list[str]) -> pd.Series:
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(dates), assets],
        names=["timestamp", "instrument"],
    )
    vals = []
    for i, (d, a) in enumerate(idx):
        vals.append((i % 7) + 0.5 + (a[0] == "A"))
    return pd.Series(vals, index=idx)


def _engine():
    from factor_engine.runtime.engine import FactorEngine

    from factor_engine.backend.pandas_backend import PandasBackend

    dates = pd.bdate_range("2026-01-05", "2026-03-27").strftime("%Y-%m-%d").tolist()
    assets = ["A", "B", "C"]
    src = InMemorySeriesSource(data={"close": _make_series(dates, assets)})
    return FactorEngine(backend=PandasBackend(), data_source=src), dates, assets


def _assert_staged_equals_full(node, ctx):
    from factor_engine.planner.physical_lowerer import extract_stage_subplans

    from factor_engine.backend.context import ExecutionContext
    from factor_engine.runtime.buffer_store import GovernedBufferStore

    ref_plan, stages = extract_stage_subplans(node, factor_name="f")
    assert stages, "期望至少一个 barrier stage"
    store = getattr(ctx, "shared_buffers", None)
    if store is None:
        backing: dict = {}
        store = GovernedBufferStore(backing)
        ctx = ExecutionContext(
            data_source=ctx.data_source,
            shared_result_cache=backing,
            shared_buffers=store,
        )
    backend = ctx.backend if hasattr(ctx, "backend") else None
    from factor_engine.backend.pandas_backend import PandasBackend

    backend = backend or PandasBackend()
    # stage 逐棵执行（先叶后根：extract 顺序保证依赖已物化）。
    for sid, sub in stages:
        val = backend.execute(sub, ctx)
        assert val is not None, f"stage {sid} 执行结果为空"
        store.put(sid, val)
    staged = backend.execute(ref_plan, ctx)
    full = backend.execute(node, ctx)
    pd.testing.assert_series_equal(staged.sort_index(), full.sort_index(), check_names=False)
    return len(stages)


def test_stage_by_stage_nested_rolling_cs():
    from factor_engine.api import col, cs_rank, ts_mean

    engine, _dates, _assets = _engine()
    f = __import__("factor_engine.api.factor", fromlist=["Factor"]).Factor(
        name="nested", expr=cs_rank(ts_mean(col("close"), 10))
    )
    node, _analysis = engine.compile(f)
    ctx = engine._make_context(shared_result_cache={})
    n = _assert_staged_equals_full(node, ctx)
    assert n >= 2  # ts_mean 与 cs_rank 两个 stage


def test_stage_by_stage_elementwise_wrapper():
    from factor_engine.api import add, col, cs_rank, ts_mean

    engine, _dates, _assets = _engine()
    f = __import__("factor_engine.api.factor", fromlist=["Factor"]).Factor(
        name="wrap", expr=add(cs_rank(ts_mean(col("close"), 10)), col("close"))
    )
    node, _analysis = engine.compile(f)
    ctx = engine._make_context(shared_result_cache={})
    n = _assert_staged_equals_full(node, ctx)
    assert n >= 2


def test_stage_by_stage_plain_rolling_single_stage():
    from factor_engine.api import col, ts_mean

    engine, _dates, _assets = _engine()
    f = __import__("factor_engine.api.factor", fromlist=["Factor"]).Factor(
        name="plain", expr=ts_mean(col("close"), 5)
    )
    node, _analysis = engine.compile(f)
    ctx = engine._make_context(shared_result_cache={})
    n = _assert_staged_equals_full(node, ctx)
    assert n >= 1
