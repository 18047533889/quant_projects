from __future__ import annotations

from collections import Counter
import gc
from dataclasses import dataclass, field, replace
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
import pytest

from factor_engine.api import add, ts_min
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.expr.literal import Literal
from factor_engine.runtime.buffer_store import GovernedBufferStore
from factor_engine.runtime.engine import FactorEngine
from factor_engine.runtime.perf_config import PerfConfig
from factor_engine.runtime.resource_broker import ResourceBroker
from tests.helpers import InMemorySeriesSource


@dataclass
class _CountingSource(InMemorySeriesSource):
    load_column_calls: Counter = field(default_factory=Counter, init=False)
    load_columns_calls: Counter = field(default_factory=Counter, init=False)

    def load_column(self, name: str) -> Any:
        self.load_column_calls[name] += 1
        return super().load_column(name)

    def load_columns(self, names: list[str]) -> dict[str, Any]:
        for name in names:
            self.load_columns_calls[name] += 1
        return super().load_columns(names)


def _close_panel() -> pd.Series:
    index = pd.MultiIndex.from_product(
        [pd.date_range("2025-01-01", periods=24), ["A", "B", "C"]],
        names=["timestamp", "instrument"],
    )
    values = np.arange(len(index), dtype=np.float64)
    return pd.Series(values + np.sin(values / 7.0), index=index, name="close")


def _coherent_broker(monkeypatch) -> ResourceBroker:
    broker = ResourceBroker(
        hard_memory_limit=2 * 1024**3,
        cpu_slots=4,
        min_host_reserve_gb=0,
        min_host_reserve_fraction=0,
    )
    original = broker._refresh

    def coherent(*, force=False):
        snapshot = original(force=force)
        hard = broker.hard_memory_limit
        return replace(
            snapshot,
            hard_memory_limit=hard,
            cgroup_memory_current=0,
            host_mem_available=hard,
            process_rss=0,
            worker_rss=0,
            process_family_rss=0,
            process_family_pss=0,
            host_mem_available_known=True,
        )

    monkeypatch.setattr(broker, "_refresh", coherent)
    decision = broker._conservative_decision()
    monkeypatch.setattr(broker, "resource_decision", lambda **_kwargs: decision)
    return broker


@pytest.mark.parametrize(
    ("native_fusion", "max_workers", "root_count"),
    [(False, 2, 5), (True, None, 5), (True, None, 513)],
)
def test_real_adaptive_batch_reuses_one_ts_min_and_releases_resources(
    monkeypatch, native_fusion, max_workers, root_count,
):
    close = _close_panel()
    source = _CountingSource(data={"close": close})
    backend = PandasBackend()
    broker = _coherent_broker(monkeypatch)
    engine = FactorEngine(backend=backend, data_source=source)
    engine.resource_broker = broker

    shared = ts_min(col("close"), 10)
    factors = [
        Factor(name=f"shared_min_{ordinal}", expr=add(shared, Literal(float(ordinal))))
        for ordinal in range(root_count)
    ]

    # Independent, non-CSE execution supplies the numerical reference before
    # installing counters on the batch backend.
    reference_engine = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data={"close": close.copy()}),
    )
    expected_base = reference_engine.run(
        Factor(name="reference_min", expr=ts_min(col("close"), 10))
    )["result"]

    eval_calls = Counter()
    plan_ref_reads = Counter()
    guard = Lock()
    original_eval = backend._eval
    original_get_ref = GovernedBufferStore.get_ref

    def counted_eval(node, ctx):
        if node.op == "ts_min":
            with guard:
                eval_calls["ts_min"] += 1
        return original_eval(node, ctx)

    def counted_get_ref(store, key):
        value = original_get_ref(store, key)
        if value is not None:
            with guard:
                plan_ref_reads[key] += 1
        return value

    monkeypatch.setattr(backend, "_eval", counted_eval)
    monkeypatch.setattr(GovernedBufferStore, "get_ref", counted_get_ref)

    sink_results: dict[str, pd.Series] = {}

    def sink(name: str, value: pd.Series) -> bool:
        sink_results[name] = value
        return True

    worker_override = {} if max_workers is None else {"n_jobs": max_workers}
    output = engine.run_many_parallel(
        factors,
        perf=PerfConfig(max_workers=max_workers, native_fusion=native_fusion),
        enable_cse=True,
        result_policy="sink",
        sink=sink,
        **worker_override,
    )

    assert output["executor"] == "adaptive_batch_scheduler"
    assert output["scheduler_stats"]["auto_execution_mode"] == "ADAPTIVE_DAG"
    assert len(output["dag"].shared_nodes) >= 1
    assert output["results"] == {}
    assert set(sink_results) == {factor.name for factor in factors}
    assert eval_calls["ts_min"] == 1
    ts_min_sid = next(
        sid for sid, node in output["dag"].shared_nodes.items()
        if node.op == "ts_min"
    )
    assert plan_ref_reads[ts_min_sid] == len(factors)
    assert all(count >= 1 for count in plan_ref_reads.values())
    assert source.load_columns_calls["close"] == 1
    assert source.load_column_calls["close"] == 0

    for ordinal, factor in enumerate(factors):
        pd.testing.assert_series_equal(
            sink_results[factor.name],
            expected_base + float(ordinal),
            check_names=False,
        )

    assert output["cse_cache_budget"]["accounted_bytes"] == 0
    sink_results.clear()
    del expected_base, engine, source, backend, close
    gc.collect()
    broker_state = broker.summary()
    cse_lease_ids = [
        lease_id for lease_id in broker._memory_leases
        if "cse:" in str(lease_id)
    ]
    assert cse_lease_ids == []
    assert broker_state["running_tasks"] == 0
    assert broker_state["cpu"]["in_use"] == 0
    assert broker_state["io"]["in_use"] == 0



def test_nested_shared_refcounts_are_root_reachable_and_valid():
    from types import SimpleNamespace

    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.runtime.batch_service import (
        _setup_cse_refcounts,
        validate_cse_refcount_integrity,
    )

    ref_leaf = PlanNode("plan_ref", attrs={"sid": "leaf"})
    ref_parent = PlanNode("plan_ref", attrs={"sid": "parent"})
    shared = {
        "leaf": PlanNode("column", attrs={"name": "close"}),
        "parent": PlanNode("ts_min", inputs=[ref_leaf], attrs={"window": 10}),
    }
    dag = SimpleNamespace(
        roots=[SimpleNamespace(root=ref_parent)],
        shared_nodes=shared,
    )
    ctx = SimpleNamespace(run_mode="production")
    _setup_cse_refcounts(ctx, dag.roots, shared_nodes=shared)
    report = validate_cse_refcount_integrity(dag, ctx)
    assert report["ok"] is True
    assert ctx._cse_refcounts == {"parent": 1, "leaf": 1}


def test_unreachable_shared_cycle_fails_closed():
    from types import SimpleNamespace

    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.runtime.batch_service import (
        _setup_cse_refcounts,
        validate_cse_refcount_integrity,
    )

    shared = {
        "live": PlanNode("column", attrs={"name": "close"}),
        "cycle_a": PlanNode("plan_ref", attrs={"sid": "cycle_b"}),
        "cycle_b": PlanNode("plan_ref", attrs={"sid": "cycle_a"}),
    }
    dag = SimpleNamespace(
        roots=[SimpleNamespace(root=PlanNode("plan_ref", attrs={"sid": "live"}))],
        shared_nodes=shared,
    )
    ctx = SimpleNamespace(run_mode="production")
    _setup_cse_refcounts(ctx, dag.roots, shared_nodes=shared)
    with pytest.raises(RuntimeError, match="orphaned|cycle"):
        validate_cse_refcount_integrity(dag, ctx)



def test_shared_reachability_handles_1200_node_chain_without_recursion():
    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.runtime.batch_service import _cse_shared_reachability

    shared = {"s0": PlanNode("column", attrs={"name": "close"})}
    for ordinal in range(1, 1200):
        shared[f"s{ordinal}"] = PlanNode(
            "plan_ref", attrs={"sid": f"s{ordinal - 1}"}
        )
    root = PlanNode("plan_ref", attrs={"sid": "s1199"})

    reachable, dangling, cycles = _cse_shared_reachability([root], shared)

    assert len(reachable) == 1200
    assert dangling == set()
    assert cycles == set()

