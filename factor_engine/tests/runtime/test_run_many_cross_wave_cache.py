from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from benchmarks.benchmark_run_many_streaming_20260906 import Source
from factor_engine.api import col, ts_mean
from factor_engine.api.factor import Factor, FactorExecutionScopeHint
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.runtime.engine import FactorEngine
from factor_engine.runtime.perf_config import PerfConfig
from factor_engine.runtime.resource_broker import ResourceBroker


class StableSource(Source):
    dataset = "bounded-test"
    generation_id = "snapshot-1"

    def __init__(self, values, *, change_on_refresh=None):
        super().__init__(values)
        self.refreshes = 0
        self.change_on_refresh = change_on_refresh

    def refresh_snapshot(self, force=False):
        self.refreshes += 1
        if self.refreshes == self.change_on_refresh:
            self.generation_id = "snapshot-2"
            self.values = self.values + 1000.0
        return self.generation_id


def _broker():
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
    return broker


def _engine(monkeypatch, *, source=None):
    from factor_engine.planner import physical_lowerer

    monkeypatch.setattr(physical_lowerer, "_get_adaptive_dag_width_limit", lambda: 2)
    monkeypatch.setattr(physical_lowerer, "_get_adaptive_chunk_size", lambda: 2)
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=6), ["A"]],
        names=["timestamp", "instrument"],
    )
    source = source or StableSource(pd.Series(np.arange(6.0), index=index))
    backend = PandasBackend()
    calls = {"ts_mean": 0}
    original = backend._registry._kernels["ts_mean"]

    def counted(node, ctx):
        calls["ts_mean"] += 1
        return original(node, ctx)

    monkeypatch.setitem(backend._registry._kernels, "ts_mean", counted)
    engine = FactorEngine(backend, source)
    engine.resource_broker = _broker()
    return engine, calls


def _run(engine, sink, *, count=5, max_workers=1):
    shared = ts_mean(col("close"), 2)
    return engine.run_many(
        [Factor(f"f{i}", shared + i) for i in range(count)],
        result_policy="sink", sink=sink,
        perf=PerfConfig(
            max_workers=max_workers, native_fusion=False, result_budget_bytes=1024**2,
        ),
    )


def test_bounded_cross_wave_cache_executes_shared_kernel_once_and_releases(monkeypatch):
    engine, calls = _engine(monkeypatch)
    seen = {}
    out = _run(engine, lambda name, value: seen.__setitem__(name, value))
    assert out["completed_waves"] == 3
    assert out["cross_wave_cache_mode"] == "transient_bounded_l2"
    assert out["cost_ledger"]["cross_wave_cache_enabled"] is True
    assert out["cost_ledger"]["cross_wave_value_reuse"] is None
    assert calls["ts_mean"] == 1
    assert not any(
        lease_id.startswith("run-many-stream-cross-wave:")
        for lease_id in engine.resource_broker._memory_leases
    )
    assert engine.cache is None
    expected = engine.data_source.values.rolling(2, min_periods=1).mean()
    for i in range(5):
        pd.testing.assert_series_equal(seen[f"f{i}"], expected + i, check_names=False)


def test_cross_wave_cache_preserves_parallel_wave_cse_and_values(monkeypatch):
    engine, calls = _engine(monkeypatch)
    seen = {}
    out = _run(
        engine, lambda name, value: seen.__setitem__(name, value), max_workers=2,
    )
    assert out["completed_waves"] == 3
    assert calls["ts_mean"] == 1
    expected = engine.data_source.values.rolling(2, min_periods=1).mean()
    for i in range(5):
        pd.testing.assert_series_equal(seen[f"f{i}"], expected + i, check_names=False)


def test_mixed_execution_scopes_disable_transient_cross_wave_cache(monkeypatch):
    engine, calls = _engine(monkeypatch)
    shared = ts_mean(col("close"), 2)
    factors = [
        Factor(
            f"f{i}", shared + i,
            semantic_identity=FactorExecutionScopeHint(
                decision_time_policy="close" if i % 2 else "open",
            ),
        )
        for i in range(5)
    ]
    out = engine.run_many(
        factors, result_policy="sink", sink=lambda *args: None,
        perf=PerfConfig(max_workers=2, native_fusion=False, result_budget_bytes=1024**2),
    )
    assert out["cross_wave_cache_mode"] == "mixed_execution_scopes"
    assert out["cost_ledger"]["cross_wave_cache_enabled"] is False
    assert out["cost_ledger"]["cross_wave_value_reuse"] is None
    assert calls["ts_mean"] >= 2


def test_snapshot_change_aborts_and_releases_before_mixing_waves(monkeypatch):
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=6), ["A"]],
        names=["timestamp", "instrument"],
    )
    source = StableSource(
        pd.Series(np.arange(6.0), index=index), change_on_refresh=3,
    )
    engine, _ = _engine(monkeypatch, source=source)
    seen = {}
    with pytest.raises(RuntimeError, match="source identity changed between waves"):
        _run(engine, lambda name, value: seen.__setitem__(name, value))
    original = pd.Series(np.arange(6.0), index=index).rolling(2, min_periods=1).mean()
    assert set(seen) == {"f0", "f1"}
    pd.testing.assert_series_equal(seen["f0"], original, check_names=False)
    assert not any(
        lease_id.startswith("run-many-stream-cross-wave:")
        for lease_id in engine.resource_broker._memory_leases
    )


def test_sink_failure_releases_transient_cache_lease(monkeypatch):
    engine, _ = _engine(monkeypatch)

    def fail(name, value):
        raise RuntimeError("sink exploded")

    with pytest.raises(RuntimeError, match="sink exploded"):
        _run(engine, fail)
    assert not any(
        lease_id.startswith("run-many-stream-cross-wave:")
        for lease_id in engine.resource_broker._memory_leases
    )


def test_existing_engine_cache_is_not_cleared_or_replaced(monkeypatch):
    from factor_engine.storage.cache import CacheManager

    engine, _ = _engine(monkeypatch)
    cache = CacheManager(data_scope="owned", budget_bytes=1024**2)
    cache.set("sentinel", pd.Series([1.0]))
    engine.cache = cache
    out = _run(engine, lambda *args: None)
    assert out["cross_wave_cache_mode"] == "existing_engine_cache"
    assert cache.get("sentinel") is not None
    assert engine.cache is cache


def test_zero_broker_cse_budget_falls_back_to_per_wave(monkeypatch):
    engine, calls = _engine(monkeypatch)
    monkeypatch.setattr(engine.resource_broker, "current_cse_budget", lambda: 0)
    out = _run(engine, lambda *args: None)
    assert out["cross_wave_cache_mode"] == "no_cse_budget"
    assert out["cost_ledger"]["cross_wave_value_reuse"] is False
    assert calls["ts_mean"] == 3


def test_dataset_name_without_snapshot_proof_does_not_enable_reuse(monkeypatch):
    engine, calls = _engine(monkeypatch)
    engine.data_source.generation_id = None
    out = _run(engine, lambda *args: None)
    assert out["cross_wave_cache_mode"] == "no_snapshot_identity"
    assert out["cost_ledger"]["cross_wave_value_reuse"] is False
    assert calls["ts_mean"] == 3


def test_cache_setup_failure_releases_acquired_lease(monkeypatch):
    from factor_engine.runtime.streaming_batch_service import _bounded_stream_cache
    from factor_engine.storage import cache as cache_module

    engine, _ = _engine(monkeypatch)

    def fail_cache(*args, **kwargs):
        raise RuntimeError("cache setup exploded")

    monkeypatch.setattr(cache_module, "CacheManager", fail_cache)
    execution_engine, cache, lease, mode = _bounded_stream_cache(engine, {})
    assert execution_engine is engine
    assert cache is lease is None
    assert mode == "cache_setup_unavailable"
    assert not any(
        lease_id.startswith("run-many-stream-cross-wave:")
        for lease_id in engine.resource_broker._memory_leases
    )


def test_writer_start_failure_releases_transient_lease(monkeypatch):
    from factor_engine.runtime.streaming_result_sink import StreamingResultSink

    engine, _ = _engine(monkeypatch)
    monkeypatch.setattr(
        StreamingResultSink, "start",
        lambda self: (_ for _ in ()).throw(RuntimeError("writer start exploded")),
    )
    with pytest.raises(RuntimeError, match="writer start exploded"):
        _run(engine, lambda *args: None)
    assert not any(
        lease_id.startswith("run-many-stream-cross-wave:")
        for lease_id in engine.resource_broker._memory_leases
    )


@pytest.mark.parametrize("definitions", [{}, None, "cycle"])
def test_transient_plan_key_invalid_refs_fail_closed_without_cache_io(definitions):
    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.planner.plan_hash import plan_cache_key

    class SpyCache:
        def __init__(self):
            self.calls = []

        def get(self, key):
            self.calls.append(("get", key))

        def set(self, key, value):
            self.calls.append(("set", key))

    if definitions == "cycle":
        definitions = {"x": PlanNode("plan_ref", attrs={"sid": "x"})}
    cache = SpyCache()
    cache.plan_key_shared_nodes = definitions
    ref = PlanNode("plan_ref", attrs={"sid": "x"})
    node = PlanNode("add", inputs=(ref, PlanNode("literal", attrs={"value": 1})))
    ctx = SimpleNamespace(
        cache=cache, shared_result_cache={"x": 2}, shared_buffers=None,
    )
    assert PandasBackend()._eval(node, ctx) == 3
    outer_key = plan_cache_key(node)
    assert all(key != outer_key for _, key in cache.calls)


def test_persistent_cache_keeps_original_plan_key_path(monkeypatch, tmp_path):
    import factor_engine.backend.pandas_backend as pandas_module
    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.storage.cache import PersistentPlanCache

    node = PlanNode("literal", attrs={"value": 7})
    cache = PersistentPlanCache(tmp_path, data_scope="stable", budget_bytes=1024)
    seen = []
    original = pandas_module.plan_cache_key

    def record(candidate):
        seen.append(candidate)
        return original(candidate)

    monkeypatch.setattr(pandas_module, "plan_cache_key", record)
    ctx = SimpleNamespace(cache=cache)
    assert PandasBackend()._eval(node, ctx) == 7
    assert seen == [node]
    assert not hasattr(cache, "plan_key_shared_nodes")


def test_manifest_token_change_aborts_even_when_data_snapshot_id_is_unchanged(monkeypatch):
    class ManifestSource(StableSource):
        data_snapshot_id = "data-id-unchanged"
        manifest = None

        @property
        def snapshot_token(self):
            return self.manifest

        def refresh_snapshot(self, force=False):
            self.refreshes += 1
            self.manifest = "manifest-2" if self.refreshes >= 3 else "manifest-1"
            if self.refreshes == 3:
                self.values = self.values + 1000.0
            # DataAccess may return a coarser ID than its authoritative token.
            return self.data_snapshot_id

    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=6), ["A"]],
        names=["timestamp", "instrument"],
    )
    source = ManifestSource(pd.Series(np.arange(6.0), index=index))
    engine, _ = _engine(monkeypatch, source=source)
    seen = {}
    assert source.snapshot_token is None
    with pytest.raises(RuntimeError, match="source identity changed between waves"):
        _run(engine, lambda name, value: seen.__setitem__(name, value))
    assert set(seen) == {"f0", "f1"}
    expected = pd.Series(np.arange(6.0), index=index).rolling(2, min_periods=1).mean()
    pd.testing.assert_series_equal(seen["f0"], expected, check_names=False)
    assert source.data_snapshot_id == "data-id-unchanged"
    assert not any(
        key.startswith("run-many-stream-cross-wave:")
        for key in engine.resource_broker._memory_leases
    )
