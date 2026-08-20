# -*- coding: utf-8 -*-
"""R27-155/256: 每种合法 shard 的 sharded-compute + merge 必须等价 full compute。"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from api import ts_mean
from api.columns import col
from api.factor import Factor
from backend.pandas_backend import PandasBackend
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


def _engine() -> FactorEngine:
    dates = pd.bdate_range("2024-01-02", periods=60)
    idx = pd.MultiIndex.from_product([dates, ["A", "B", "C"]], names=["timestamp", "instrument"])
    close = pd.Series([float(i % 97) for i in range(len(idx))], index=idx)
    return FactorEngine(data_source=InMemorySeriesSource({"close": close}),
                        backend=PandasBackend())


def test_time_shard_equivalent_full():
    # 纯 instrument-separable 时序因子（ts_mean）→ time shard 合法。
    engine = _engine()
    factor = Factor(name="f", expr=ts_mean(col("close"), 5))

    # full：一次跑完。
    full = engine.run(factor)["result"]

    # shard by time_month：按月切（2024-01/02/03），scoped run + concat merge。
    # （等价于 materialize_sharded(time_month) 的 sharded-compute + merge。）
    shards = []
    for month in ("2024-01", "2024-02", "2024-03"):
        import copy

        scoped = copy.copy(engine)
        scoped.data_source = copy.copy(engine.data_source)
        scoped.data_source.start_date = f"{month}-01"
        scoped.data_source.end_date = f"{month}-28"
        r = scoped.run(factor)["result"]
        shards.append(r)
    merged = pd.concat(shards)
    merged = merged[~merged.index.duplicated(keep="last")]
    # full 与 sharded-merge 对齐到相同索引后逐元素比较（NaN mask + 数值）。
    common = full.index.intersection(merged.index)
    assert len(common) > 0
    a = full.loc[common]
    b = merged.loc[common]
    assert (a.isna() == b.isna()).all()
    assert a.fillna(0.0).equals(b.fillna(0.0))
    # warmup 边界：full 用整段历史，shard 首月缺前置 → 首月早期 NaN 差异允许，
    # 但 2024-02/03 的完整窗口必须一致。
    assert len(full) >= len(merged)


def test_asset_shard_unsafe_for_cross_section():
    # R27-084/245：含截面算子（rank）的计划禁止 asset shard。
    from runtime.adaptive_sharding import classify_shard_legality

    class RankNode:
        op = "rank"
        inputs = ()

    legality = classify_shard_legality(RankNode())
    assert legality.asset_shard_safe is False
    assert legality.best_dimension == "time"
