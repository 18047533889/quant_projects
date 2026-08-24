"""Phase 23：事件驱动增量物化 + OperatorCost。"""

from __future__ import annotations

import pandas as pd

from factor_engine.api import rank, ts_mean
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.operator_cost import estimate_plan_cost, get_operator_cost
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.backend.routing import resolve_execution_tier, select_operator_backend
from factor_engine.runtime.engine import FactorEngine
from factor_engine.runtime.incremental_scheduler import DataEvent, execute_incremental_updates_from_event
from factor_engine.storage.materializer import ParquetMaterializer
from tests.helpers import InMemorySeriesSource


def _close_panel(n_days: int = 8) -> pd.Series:
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    idx = pd.MultiIndex.from_product([dates, ["AAA"]], names=["timestamp", "instrument"])
    return pd.Series([float(i + 1) for i in range(len(idx))], index=idx)


def test_materialize_incremental_from_event_dry_run(tmp_path):
    close = _close_panel(5)
    factor = Factor(name="mom", expr=rank(ts_mean(col("close"), 2)))
    fid = "mom_evt"
    lake = tmp_path / "lake"
    eng = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data={"close": close}),
    )
    eng.materialize(
        factor,
        factor_id=fid,
        lake_root=lake,
        expression='rank(ts_mean(col("close"), 2))',
    )
    mat = ParquetMaterializer(lake_root=lake)
    mat.catalog.record_factor_dependency(
        fid,
        referenced_columns=["close"],
        lookback=2,
        source_dataset="mock_ds",
        frequency="1d",
    )

    out = eng.materialize_incremental_from_event(
        DataEvent(dataset="mock_ds", column="close", updated_date="2024-01-08"),
        lake_root=lake,
        dry_run=True,
    )
    assert out["factor_count"] == 1
    assert out["dry_run"] is True
    assert "mom_evt" in {p["factor_id"] for p in out["plans"]}


def test_materialize_incremental_from_event_executes(tmp_path):
    close_old = _close_panel(5)
    close_new = _close_panel(8)
    factor = Factor(name="delay1", expr=col("close"))
    fid = "delay_evt"
    lake = tmp_path / "lake"

    eng = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data={"close": close_old}),
    )
    eng.materialize(
        factor,
        factor_id=fid,
        lake_root=lake,
        expression='col("close")',
    )
    ParquetMaterializer(lake_root=lake).catalog.record_factor_dependency(
        fid,
        referenced_columns=["close"],
        lookback=0,
        source_dataset="mock_ds",
        frequency="1d",
    )

    eng2 = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data={"close": close_new}),
    )
    out = eng2.materialize_incremental_from_event(
        DataEvent(dataset="mock_ds", column="close", updated_date="2024-01-10"),
        lake_root=lake,
        lookback_extra=0,
        recompute_tail_bars=0,
        dry_run=False,
    )
    assert out["succeeded"] == 1
    assert fid in out["materializations"]


def test_operator_cost_and_routing():
    mean_cost = get_operator_cost("ts_mean")
    assert mean_cost.tier == 0
    assert mean_cost.supports_incremental is True
    assert select_operator_backend(mean_cost) == "bottleneck"
    assert resolve_execution_tier("ts_rank", window=600) >= 1

    from factor_engine.planner.logical_plan import PlanNode

    plan = PlanNode(op="ts_mean", attrs={"window": 5}, inputs=[PlanNode(op="col", attrs={"name": "close"})])
    summary = estimate_plan_cost(plan)
    assert summary["node_count"] == 2
    assert summary["max_tier"] == 0


def test_run_many_includes_plan_costs():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02"]), ["A"]],
        names=["timestamp", "instrument"],
    )
    src = InMemorySeriesSource(data={"close": pd.Series([1.0, 2.0], index=idx)})
    eng = FactorEngine(backend=PandasBackend(), data_source=src)
    f1 = Factor(name="a", expr=ts_mean(col("close"), 2))
    f2 = Factor(name="b", expr=rank(col("close")))
    out = eng.run_many([f1, f2])
    assert "plan_costs" in out
    assert set(out["plan_costs"]) == {"a", "b"}
