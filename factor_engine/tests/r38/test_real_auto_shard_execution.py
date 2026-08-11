# -*- coding: utf-8 -*-
"""R38 P0-001/002/003（§4）：真实 AutoShard 执行 —— full-run == sharded-run。

行为探针（§R38-P0-071）：整 task 真实执行 → 与「shard 逐片执行 + merge」结果
逐值一致：
    - asset shard：每片按仪器子集切片执行 → concat merge == full
    - time shard：每片按时间窗 + warmup overlap 切片执行 → 裁剪 + concat == full
    - cross-section：asset shard 被 gate 拒绝（§R38_CROSS_SECTION_ZERO_ASSET_SHARD）
    - shard 结果不在 merge 前全部常驻内存（SpooledShard 路径可触发）
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from tests.helpers import InMemorySeriesSource
from api import col, ts_mean, cs_rank
from backend.context import ExecutionContext
from backend.pandas_backend import PandasBackend
from runtime.auto_shard_planner import AutoShardPlanner
from runtime.resource_shape import ResourceShapeKey
from runtime.shard_execution_plan import SHARD_ASSET, SHARD_TIME
from runtime.shard_executor import ShardExecutor


def _make_series(dates: list[str], assets: list[str]) -> pd.Series:
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(dates), assets],
        names=["timestamp", "instrument"],
    )
    vals = []
    for i, (d, a) in enumerate(idx):
        vals.append((i % 7) + 0.5)
    return pd.Series(vals, index=idx)


def _task(task_id: str, op: str, *, peak_bytes: int, dim: str, time_range, instruments):
    from runtime.task_resource_contract import TaskResourceContract

    contract = TaskResourceContract(
        predicted_elapsed_ms=1.0,
        cpu_tokens=1,
        io_tokens=0,
        peak_memory_bytes=peak_bytes,
        output_bytes=peak_bytes,
        backend="pandas_numpy",
        backend_threads=1,
        shardable=True,
        shard_dimension=dim,
        estimate_basis="test",
    )
    return SimpleNamespace(
        task_id=task_id,
        op=op,
        task_type="ROOT",
        resource_contract=contract,
        time_range=time_range,
        instrument_scope=tuple(instruments),
        node_ref=None,
    )


def _execute_full(engine, f) -> pd.Series:
    out = engine.run(f)
    return out["result"]


def _build_and_run_shards(engine, f, task, *, time_range, instruments, lookback, dim):
    node, _analysis = engine.compile(f)
    planner = AutoShardPlanner(min_shards=2)
    plan = planner.build_shard_execution_plan(
        task,
        safe_envelope_bytes=10 * 1024**2,
        time_range=time_range,
        instrument_universe=instruments,
        lookback_bars=lookback,
    )
    assert plan is not None, f"plan expected for {task.task_id}"
    assert len(plan.shards) >= 2
    assert plan.merge_task_id == f"{task.task_id}:merge"
    ctx = engine._make_context(shared_result_cache={})
    ex = ShardExecutor()
    partials: dict[str, object] = {}
    for d in plan.shards:
        partials[d.shard_id] = ex.execute_shard(engine.backend, node, ctx, d)
    merged = ex.execute_merge(engine.backend, ctx, plan, partials)
    return merged, plan


# ---------------------------------------------------------------------------
# elementwise：asset shard parity
# ---------------------------------------------------------------------------


def test_asset_shard_parity_elementwise():
    dates = pd.bdate_range("2024-01-01", "2024-06-30").strftime("%Y-%m-%d").tolist()
    assets = [f"A{i:03d}" for i in range(60)]
    src = InMemorySeriesSource(data={"close": _make_series(dates, assets)})
    engine = __import__("runtime.engine", fromlist=["FactorEngine"]).FactorEngine(
        backend=PandasBackend(), data_source=src
    )
    f = __import__("api.factor", fromlist=["Factor"]).Factor(
        name="a", expr=ts_mean(col("close"), 3)
    )
    full = _execute_full(engine, f)
    task = _task("root:a", "ts_mean", peak_bytes=20 * 1024**2, dim="asset",
                 time_range=None, instruments=assets)
    merged, plan = _build_and_run_shards(
        engine, f, task,
        time_range=None, instruments=assets, lookback=10, dim="asset",
    )
    pd.testing.assert_series_equal(merged.sort_index(), full.sort_index(), check_names=False)


# ---------------------------------------------------------------------------
# ts_rolling：time shard + warmup overlap parity
# ---------------------------------------------------------------------------


def test_time_shard_rolling_overlap_parity():
    dates = pd.bdate_range("2023-10-01", "2024-06-30").strftime("%Y-%m-%d").tolist()
    assets = [f"A{i:03d}" for i in range(40)]
    src = InMemorySeriesSource(data={"close": _make_series(dates, assets)})
    engine = __import__("runtime.engine", fromlist=["FactorEngine"]).FactorEngine(
        backend=PandasBackend(), data_source=src
    )
    # window=20 的 rolling mean —— time shard 必须带 warmup overlap 才与 full 一致。
    f = __import__("api.factor", fromlist=["Factor"]).Factor(
        name="b", expr=ts_mean(col("close"), 20)
    )
    full = _execute_full(engine, f)
    task = _task("root:b", "ts_mean", peak_bytes=20 * 1024**2, dim="time",
                 time_range=("2023-10-01", "2024-06-30"), instruments=assets)
    merged, plan = _build_and_run_shards(
        engine, f, task,
        time_range=("2023-10-01", "2024-06-30"),
        instruments=assets, lookback=60, dim="time",
    )
    pd.testing.assert_series_equal(merged.sort_index(), full.sort_index(), check_names=False)


# ---------------------------------------------------------------------------
# cross-section：asset shard 必须被拒绝（§R38_CROSS_SECTION_ZERO_ASSET_SHARD）
# ---------------------------------------------------------------------------


def test_cross_section_asset_shard_rejected():
    dates = pd.bdate_range("2024-01-01", "2024-04-30").strftime("%Y-%m-%d").tolist()
    assets = [f"A{i:03d}" for i in range(30)]
    src = InMemorySeriesSource(data={"close": _make_series(dates, assets)})
    engine = __import__("runtime.engine", fromlist=["FactorEngine"]).FactorEngine(
        backend=PandasBackend(), data_source=src
    )
    f = __import__("api.factor", fromlist=["Factor"]).Factor(
        name="cs", expr=cs_rank(col("close"))
    )
    task = _task("root:cs", "cs_rank", peak_bytes=20 * 1024**2, dim="asset",
                 time_range=None, instruments=assets)
    planner = AutoShardPlanner()
    plan = planner.build_shard_execution_plan(
        task,
        safe_envelope_bytes=10 * 1024**2,
        time_range=("2024-01-01", "2024-04-30"),
        instrument_universe=assets,
        lookback_bars=0,
    )
    # cs 只能 time shard；asset shard 声明与语义冲突 → plan 无法按 asset 构造。
    if plan is not None:
        assert plan.dimension == "time", "cs must shard by time only"
    else:
        # 非法 asset 声明被拒绝（合法 None）
        pass


# ---------------------------------------------------------------------------
# time shard（cs 语义）：每日期完整 universe → parity
# ---------------------------------------------------------------------------


def test_cross_section_time_shard_parity():
    dates = pd.bdate_range("2024-01-01", "2024-04-30").strftime("%Y-%m-%d").tolist()
    assets = [f"A{i:03d}" for i in range(30)]
    src = InMemorySeriesSource(data={"close": _make_series(dates, assets)})
    engine = __import__("runtime.engine", fromlist=["FactorEngine"]).FactorEngine(
        backend=PandasBackend(), data_source=src
    )
    f = __import__("api.factor", fromlist=["Factor"]).Factor(
        name="cs", expr=cs_rank(col("close"))
    )
    full = _execute_full(engine, f)
    task = _task("root:cs", "cs_rank", peak_bytes=20 * 1024**2, dim="time",
                 time_range=("2024-01-01", "2024-04-30"), instruments=assets)
    merged, plan = _build_and_run_shards(
        engine, f, task,
        time_range=("2024-01-01", "2024-04-30"),
        instruments=assets, lookback=0, dim="time",
    )
    pd.testing.assert_series_equal(merged.sort_index(), full.sort_index(), check_names=False)


# ---------------------------------------------------------------------------
# spool：shard 结果不在 merge 前全部常驻（SpooledShard 路径）
# ---------------------------------------------------------------------------


def test_shard_spool_reload_roundtrip():
    from runtime.shard_executor import SPOOL_THRESHOLD_BYTES, SpooledShard

    dates = pd.bdate_range("2024-01-01", "2024-03-31").strftime("%Y-%m-%d").tolist()
    assets = [f"A{i:03d}" for i in range(50)]
    src = InMemorySeriesSource(data={"close": _make_series(dates, assets)})
    engine = __import__("runtime.engine", fromlist=["FactorEngine"]).FactorEngine(
        backend=PandasBackend(), data_source=src
    )
    f = __import__("api.factor", fromlist=["Factor"]).Factor(
        name="a", expr=ts_mean(col("close"), 5)
    )
    full = _execute_full(engine, f)
    ex = ShardExecutor()
    # 强制小阈值 → 触发 spool 路径。
    ref = ex.spool_or_keep(full, spool_threshold=1)
    assert isinstance(ref, SpooledShard), "result above threshold must spool"
    reloaded = ex.load_partial(ref)
    pd.testing.assert_series_equal(reloaded.sort_index(), full.sort_index(), check_names=False)


# ---------------------------------------------------------------------------
# shape signature：不同分片 → 不同签名（OOM replan 的 no-same-shape 基础）
# ---------------------------------------------------------------------------


def test_shape_signature_changes_with_shard_count():
    from runtime.shard_execution_plan import shape_signature

    s2 = shape_signature(task_id="t", dimension="time", shard_count=2, per_shard_peak_bytes=100)
    s4 = shape_signature(task_id="t", dimension="time", shard_count=4, per_shard_peak_bytes=50)
    assert s2 != s4
