# -*- coding: utf-8 -*-
"""R27-140..142/205: materialize_many_fast / plan_many_fast 端到端。"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from factor_engine.api import rank, ts_mean, ts_std
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


@pytest.fixture(scope="module")
def engine():
    dates = pd.bdate_range("2024-01-02", periods=12)
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["timestamp", "instrument"])
    close = pd.Series([float(i) for i in range(len(idx))], index=idx)
    vol = pd.Series([float(i * 2) for i in range(len(idx))], index=idx)
    return FactorEngine(data_source=InMemorySeriesSource({"close": close, "volume": vol}),
                        backend=PandasBackend())


def _factors():
    return [
        Factor(name="f_mean5", expr=ts_mean(col("close"), 5)),
        Factor(name="f_std20", expr=ts_std(col("close"), 3)),
        Factor(name="f_rank", expr=rank(col("close"))),
    ]


def test_plan_many_fast_dry_run(engine):
    # R27-142/191：dry-run 只读 metadata，不读大数据。
    plan = engine.plan_many_fast(_factors(), enable_cse=True)
    assert plan["factors"] == 3
    assert plan["unique_dag_tasks"] >= 3
    assert plan["read_waves"]["wave_count"] >= 1
    assert plan["live_memory_headroom"] >= 0
    assert plan["recommended_concurrency"] >= 1
    assert "physical_dag" in plan


def test_materialize_many_fast_matches_serial(engine):
    # R27-154/257：fast 路径与 serial run_many 数值/NaN/索引一致。
    from factor_engine.runtime.batch_service import execute_run_many

    factors = _factors()
    # serial reference（无 CSE，逐因子）。
    ref = {}
    for f in factors:
        out = engine.run(f)
        ref[f.name] = out["result"]
    os.environ["FACTOR_ENGINE_HYBRID_FORCE"] = "thread"
    try:
        fast = engine.materialize_many_fast(
            factors, result_policy="sink", storage_format="long",
            writer_queue_bytes=1 << 20,
        )
    finally:
        os.environ.pop("FACTOR_ENGINE_HYBRID_FORCE", None)
    res = fast["results"]
    assert set(res.keys()) == set(ref.keys())
    for name in ref:
        a = ref[name]
        b = res[name]
        assert list(a.index) == list(b.index), f"{name} index mismatch"
        assert (a.isna() == b.isna()).all(), f"{name} NaN mask mismatch"
        assert a.fillna(0.0).equals(b.fillna(0.0)), f"{name} value mismatch"
    assert fast["done"] >= 3
    assert fast["explanations"], "R27-143: scheduler must explain decisions"
