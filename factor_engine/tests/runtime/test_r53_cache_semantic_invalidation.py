"""run_many cache invalidation across semantic and source snapshot changes."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from benchmarks.benchmark_run_many_streaming_20260906 import Source
from factor_engine.api import col, ts_mean
from factor_engine.api.factor import Factor
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.engine import FactorEngine
from factor_engine.runtime.perf_config import PerfConfig
from factor_engine.runtime.resource_broker import ResourceBroker
from factor_engine.storage.cache import CacheManager
from factor_engine.storage.sources.data_access_source import ApprovedSnapshotMismatch


class VersionedSource(Source):
    dataset = "r53-cache"
    generation_id = "snapshot-1"

    def refresh_snapshot(self, force=False):
        return self.generation_id


def _engine(monkeypatch):
    from factor_engine.planner import physical_lowerer

    load_all()
    monkeypatch.setattr(physical_lowerer, "_get_adaptive_dag_width_limit", lambda: 2)
    monkeypatch.setattr(physical_lowerer, "_get_adaptive_chunk_size", lambda: 2)
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=6), ["A"]],
        names=["timestamp", "instrument"],
    )
    source = VersionedSource(pd.Series(np.arange(6.0), index=index))
    backend = PandasBackend()
    calls = {"ts_mean": 0}
    original = backend._registry._kernels["ts_mean"]

    def counted(node, ctx):
        calls["ts_mean"] += 1
        return original(node, ctx)

    monkeypatch.setitem(backend._registry._kernels, "ts_mean", counted)
    engine = FactorEngine(backend, source)
    broker = ResourceBroker(
        hard_memory_limit=2 * 1024**3, cpu_slots=2,
        min_host_reserve_gb=0, min_host_reserve_fraction=0,
    )
    hard = broker.hard_memory_limit
    snapshot = replace(
        broker.snapshot(), hard_memory_limit=hard, cgroup_memory_current=0,
        host_mem_available=hard, process_rss=0, worker_rss=0,
        process_family_rss=0, process_family_pss=0,
        host_mem_available_known=True,
    )
    broker._refresh = lambda force=False: snapshot
    engine.resource_broker = broker
    engine.cache = CacheManager(data_scope="r53-owned", budget_bytes=1024**2)
    return engine, source, calls


def _run(engine):
    shared = ts_mean(col("close"), 2)
    seen = {}
    summary = engine.run_many(
        [Factor("mean", shared), Factor("shifted", shared + 1)],
        result_policy="sink", sink=lambda name, value: seen.__setitem__(name, value),
        perf=PerfConfig(max_workers=1, native_fusion=False, result_budget_bytes=1024**2),
    )
    return summary, seen


def _run_stream(engine, sink):
    shared = ts_mean(col("close"), 2)
    return engine.run_many(
        [Factor(f"f{i}", shared + i) for i in range(5)],
        result_policy="sink", sink=sink,
        perf=PerfConfig(max_workers=1, native_fusion=False, result_budget_bytes=1024**2),
    )


def test_run_many_persistent_cache_invalidates_on_operator_semantic_change(monkeypatch):
    engine, _, calls = _engine(monkeypatch)
    _, first = _run(engine)
    first_calls = calls["ts_mean"]
    row = OperatorRegistry._catalog["ts_mean"]
    changed = {**row, "semantic_version": "999.0"}
    catalog = {**OperatorRegistry._catalog, "ts_mean": changed}
    monkeypatch.setattr(OperatorRegistry, "_catalog", catalog)

    _, second = _run(engine)

    assert calls["ts_mean"] > first_calls
    pd.testing.assert_series_equal(second["mean"], first["mean"])


def test_run_many_persistent_cache_invalidates_on_source_snapshot_change(monkeypatch):
    engine, source, calls = _engine(monkeypatch)
    _, first = _run(engine)
    first_calls = calls["ts_mean"]
    source.values = source.values + 1000.0
    source.generation_id = "snapshot-2"

    _, second = _run(engine)

    assert calls["ts_mean"] > first_calls
    assert float(second["mean"].iloc[-1]) == pytest.approx(1004.5)
    assert not second["mean"].equals(first["mean"])


def test_run_many_persistent_cache_reuses_same_source_snapshot(monkeypatch):
    engine, _, calls = _engine(monkeypatch)
    _, first = _run(engine)
    first_calls = calls["ts_mean"]

    _, second = _run(engine)

    assert calls["ts_mean"] == first_calls
    pd.testing.assert_series_equal(second["mean"], first["mean"])


def test_shared_cache_isolates_same_generation_across_datasets(monkeypatch):
    engine_a, source_a, calls = _engine(monkeypatch)
    shared_cache = engine_a.cache
    source_b = VersionedSource(source_a.values + 1000.0)
    source_b.dataset = "r53-cache-other"
    engine_b = FactorEngine(engine_a.backend, source_b, cache=shared_cache)
    engine_b.resource_broker = engine_a.resource_broker

    _run(engine_a)
    calls_after_a = calls["ts_mean"]
    _, first_b = _run(engine_b)
    calls_after_b = calls["ts_mean"]
    _, second_b = _run(engine_b)

    assert calls_after_b > calls_after_a
    assert calls["ts_mean"] == calls_after_b
    assert float(first_b["mean"].iloc[-1]) == pytest.approx(1004.5)
    pd.testing.assert_series_equal(second_b["mean"], first_b["mean"])


@pytest.mark.parametrize(
    "error",
    [ApprovedSnapshotMismatch("approved snapshot changed"), PermissionError("denied")],
)
@pytest.mark.parametrize("cache_present", [True, False])
def test_refresh_failure_propagates_before_sink(monkeypatch, error, cache_present):
    engine, source, _ = _engine(monkeypatch)
    if not cache_present:
        engine.cache = None
    seen = {}

    def fail_refresh(force=False):
        raise error

    source.refresh_snapshot = fail_refresh
    with pytest.raises(type(error), match=str(error)):
        _run_stream(engine, lambda name, value: seen.__setitem__(name, value))
    assert seen == {}


def test_existing_cache_snapshot_change_rejects_before_next_wave_sink(monkeypatch):
    engine, source, _ = _engine(monkeypatch)
    source.refreshes = 0
    source.change_on_refresh = 3

    def changing_refresh(force=False):
        source.refreshes += 1
        if source.refreshes == source.change_on_refresh:
            source.generation_id = "snapshot-2"
            source.values = source.values + 1000.0
        return source.generation_id

    source.refresh_snapshot = changing_refresh
    seen = {}
    with pytest.raises(RuntimeError, match="source identity changed between waves"):
        _run_stream(engine, lambda name, value: seen.__setitem__(name, value))
    assert set(seen) == {"f0", "f1"}
    assert source.refreshes == 3


def test_unknown_snapshot_disables_caller_cache_without_mutating_engine(monkeypatch):
    engine, source, calls = _engine(monkeypatch)
    cache = engine.cache
    source.generation_id = None

    _run(engine)
    first_calls = calls["ts_mean"]
    _run(engine)

    assert calls["ts_mean"] > first_calls
    assert engine.cache is cache


def test_data_access_source_refresh_rebinds_helper_scope(monkeypatch):
    from factor_engine.runtime.batch_service import _scope_engine_cache_for_source
    from factor_engine.storage.sources import data_access_source as module
    from factor_engine.storage.sources.data_access_source import DataAccessSource

    class Store:
        partition_version = "pv1"

        def manifest_version(self, dataset, **params):
            return {
                "dataset": dataset,
                "has_manifest": True,
                "fresh": True,
                "dataset_version": "dv1",
                "partition_version": self.partition_version,
                "file_count": 1,
            }

    store = Store()
    monkeypatch.setattr(module, "_get_store", lambda: store)
    source = DataAccessSource(dataset="r53-data-access", params={})
    source._data_snapshot_id = "snap-1"
    source._snapshot_ttl_seconds = 0
    engine = FactorEngine(
        PandasBackend(), source,
        cache=CacheManager(data_scope="caller-owned", budget_bytes=1024**2),
    )
    try:
        scoped_1, scope_1 = _scope_engine_cache_for_source(engine)
        store.partition_version = "pv2"
        scoped_2, scope_2 = _scope_engine_cache_for_source(engine)

        assert scope_1 != scope_2
        assert scoped_1.cache.data_scope != scoped_2.cache.data_scope
        assert engine.cache.data_scope == "caller-owned"
    finally:
        source.close()


def test_failed_second_run_releases_cross_wave_budget(monkeypatch):
    engine, source, _ = _engine(monkeypatch)
    engine.cache = None
    source.generation_id = "snapshot-2"

    def fail(*_):
        raise RuntimeError("r53 sink failed")

    shared = ts_mean(col("close"), 2)
    with pytest.raises(RuntimeError, match="r53 sink failed"):
        engine.run_many(
            [Factor(f"f{i}", shared + i) for i in range(5)],
            result_policy="sink", sink=fail,
            perf=PerfConfig(max_workers=1, native_fusion=False, result_budget_bytes=1024**2),
        )
    assert not any(
        lease_id.startswith("run-many-stream-cross-wave:")
        for lease_id in engine.resource_broker._memory_leases
    )
    assert engine.cache is None
