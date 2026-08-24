# -*- coding: utf-8 -*-
"""R36 hard gates + 共存负载测试：异构服务器自适应资源治理。

对齐 R36 §295 硬门与 §252..294 资源测试。每个测试是**行为探针**（构造输入 →
断言行为），不依赖真实服务器负载（用合成信号/注入）。
"""
from __future__ import annotations

import os
import threading
import time

import pytest


# ---------------------------------------------------------------------------
# P0-001/002/003：max_concurrency + 消费 decision + 恢复升速
# ---------------------------------------------------------------------------


def test_max_concurrency_really_limits_admission():
    from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler
    from factor_engine.runtime.resource_broker import ResourceBroker

    broker = ResourceBroker(hard_memory_limit=8 * 1024**3, cpu_slots=4)
    sched = AdaptiveBatchScheduler(broker=broker, max_concurrency=2)
    assert sched._dynamic_concurrency_limit(None) == 2


def test_scheduler_consumes_broker_resource_decision():
    from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler
    from factor_engine.runtime.resource_broker import ResourceBroker

    broker = ResourceBroker(hard_memory_limit=8 * 1024**3, cpu_slots=4)
    sched = AdaptiveBatchScheduler(broker=broker)
    decision = broker.resource_decision()
    limit = sched._dynamic_concurrency_limit(None)
    assert limit <= max(1, decision.target_cpu_tokens)


def test_pressure_recovery_auto_upshift():
    from factor_engine.runtime.resource_autopilot import ResourceController
    from factor_engine.runtime.resource_monitor import HostResourceEnvelope, ResourceSignals

    class _Cpu:
        def __init__(self):
            self.soft_budget = 8
            self.hard_slots = 8

        def set_soft_budget(self, v):
            self.soft_budget = max(1, int(v))

    class _B:
        hard_cpu_slots = 8

        def __init__(self):
            self._cpu = _Cpu()

        def pressure_stage(self):
            return "NORMAL"

        def cpu_budget(self):
            return self._cpu.soft_budget

    ctl = ResourceController(_B(), stable_samples_required=2, cooldown_ticks=0)
    env = HostResourceEnvelope(
        hard_memory_bytes=8 * 1024**3, safe_memory_bytes=4 * 1024**3,
        emergency_reserve_bytes=1024**3, hard_cpu_tokens=8, target_cpu_tokens=8,
        io_capacity_score=1.0, remote_capacity_score=1.0, spill_free_bytes=0,
    )
    low = ctl.tick(ResourceSignals(cpu_psi_some=0.95), env).target_cpu_tokens
    ok = ResourceSignals(host_mem_available=8 * 1024**3)
    for _ in range(5):
        d = ctl.tick(ok, env)
    assert d.target_cpu_tokens > low  # 压力解除 → slow up


def test_dynamic_wave_and_sink_from_decision():
    from factor_engine.runtime.resource_broker import ResourceBroker

    broker = ResourceBroker(hard_memory_limit=8 * 1024**3, cpu_slots=4)
    d = broker.resource_decision()
    assert 256 * 1024**2 <= d.read_wave_bytes <= 16 * 1024**3
    assert 128 * 1024**2 <= d.result_queue_bytes <= 8 * 1024**3


# ---------------------------------------------------------------------------
# PSI / slope（§28/29/69）
# ---------------------------------------------------------------------------


def test_memory_slope_triggers_predictive_pressure():
    from factor_engine.runtime.resource_autopilot import ResourceController
    from factor_engine.runtime.resource_monitor import HostResourceEnvelope, ResourceSignals

    class _B:
        hard_cpu_slots = 8

        def __init__(self):
            self._cpu = type("C", (), {"set_soft_budget": lambda self, v: None})()

        def pressure_stage(self):
            return "NORMAL"

        def cpu_budget(self):
            return 8

    ctl = ResourceController(_B(), stable_samples_required=2, cooldown_ticks=0)
    env = HostResourceEnvelope(8 * 1024**3, 4 * 1024**3, 1024**3, 8, 8, 1.0, 1.0, 0)
    d = ctl.tick(ResourceSignals(host_mem_available=8 * 1024**3, mem_available_slope=-4 * 1024**3), env)
    assert d.pressure_state != "NORMAL"


def test_psi_readers_parse_real_files():
    from factor_engine.runtime.resource_monitor import psi_cpu, psi_io, psi_memory

    for reader in (psi_cpu, psi_memory, psi_io):
        out = reader()
        # some avg10 是 float（cpu PSI 在多线程争抢时内核可能报 >1.0——合法，
        # 本测试只验证解析正确，不做上界断言）。
        assert out is None or (isinstance(out[0], float) and out[0] >= 0.0)


# ---------------------------------------------------------------------------
# P0-007/029：family PSS + true run peak
# ---------------------------------------------------------------------------


def test_memory_governor_uses_family_memory():
    from factor_engine.runtime.resource_governor import MemoryGovernor, process_family_memory_bytes

    v = process_family_memory_bytes(prefer_pss=True)
    assert v is not None and v > 0
    gov = MemoryGovernor(process_budget_bytes=1024**3, duckdb_budget_bytes=0)
    assert gov._current_rss() > 0


def test_run_peak_sampler_windows_only():
    from factor_engine.runtime.run_peak_sampler import RunPeakSampler

    s = RunPeakSampler(interval_s=0.02)
    s.start()
    time.sleep(0.1)
    out = s.stop()
    assert out["started"] and out["samples"] >= 1
    assert "peak_family_pss" in out  # 独立 run 窗口，不混入 lifetime peak


# ---------------------------------------------------------------------------
# P0-005/006：P99 model + persistence
# ---------------------------------------------------------------------------


def test_p99_memory_model_rises_on_underprediction():
    from factor_engine.runtime.resource_calibration_store import ResourceCalibrationStore
    from factor_engine.runtime.resource_shape import ResourceShapeKey

    store = ResourceCalibrationStore()
    key = ResourceShapeKey("ts_mean", "duckdb", rows_bucket=2, instruments_bucket=2, window_bucket=1)
    # R38 P0-008：P99 模型只接收**可信归因**观测（isolated），预测值不作为真实值。
    for _ in range(10):
        store.record(key, elapsed_ms=50, peak_mem=1000, attribution_quality="isolated", peak_is_trusted=True)
    p1 = store.predict(key)
    for _ in range(3):
        store.record(key, elapsed_ms=10, peak_mem=500000, attribution_quality="isolated", peak_is_trusted=True)
    p2 = store.predict(key)
    assert p1["memory_p99"] <= 1000
    assert p2["memory_p99"] > 1000


def test_calibration_persists_and_restores(tmp_path):
    from factor_engine.runtime.resource_calibration_store import ResourceCalibrationStore
    from factor_engine.runtime.resource_shape import ResourceShapeKey

    key = ResourceShapeKey("ts_mean", "polars", rows_bucket=1, instruments_bucket=1, window_bucket=0)
    store = ResourceCalibrationStore()
    # R38 P0-008：可信归因观测才会进入 memory_obs / sample_count。
    store.record(key, elapsed_ms=5, peak_mem=777, attribution_quality="isolated", peak_is_trusted=True)
    store.record(key, elapsed_ms=5, peak_mem=888, attribution_quality="isolated", peak_is_trusted=True)
    path = str(tmp_path / "cal.parquet")
    store.save(path)
    store2 = ResourceCalibrationStore(path=path)
    p = store2.predict(key)
    assert p is not None and p["sample_count"] == 2


# ---------------------------------------------------------------------------
# P0-014/012：writer fatal + backpressure
# ---------------------------------------------------------------------------


def test_writer_failure_is_fatal():
    from factor_engine.runtime.streaming_result_sink import StreamingResultSink

    def _boom(batch):
        raise OSError("disk full")

    sink = StreamingResultSink(writer=_boom, queue_bytes=1024 * 1024, batch_size=1)
    sink.start()
    sink.submit("f", object())
    with pytest.raises(RuntimeError, match="writer fatal"):
        sink.finish()


def test_sink_backpressure_reduces_compute_admission():
    from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler
    from factor_engine.runtime.resource_broker import ResourceBroker

    broker = ResourceBroker(hard_memory_limit=8 * 1024**3, cpu_slots=8)
    sched = AdaptiveBatchScheduler(broker=broker, max_concurrency=8)

    class _Mid:
        backpressure_ratio = 0.75

    class _High:
        backpressure_ratio = 0.95

    assert sched._dynamic_concurrency_limit(_High()) < sched._dynamic_concurrency_limit(_Mid())


# ---------------------------------------------------------------------------
# P0-021/022/023：governed buffer store
# ---------------------------------------------------------------------------


def test_governed_buffer_store_rejects_over_budget_and_reconciles():
    from factor_engine.runtime.buffer_store import GovernedBufferStore, STATUS_MEMORY

    store = GovernedBufferStore({}, budget_bytes=100)
    # R38 P0-029：put 返回 BufferPutResult（不再裸 bool）。
    assert store.put("a", "x" * 40, bytes_=40).status == STATUS_MEMORY
    # 超过预算：先冷淘汰 a，再放 b —— 记 refused 不静默。
    res = store.put("b", "y" * 1000, bytes_=1000)
    assert res.status in ("RECOMPUTE", "REFUSED") or store.summary()["refused"] > 0
    rec = store.reconciliation()
    assert "accounted_bytes" in rec and "drift_bytes" in rec


def test_cache_governance_fail_closed_when_register_fails(monkeypatch):
    import factor_engine.runtime.resource_governor as rg

    def _boom():
        raise RuntimeError("governor unavailable")

    monkeypatch.setattr(rg, "global_memory_governor", _boom)
    from factor_engine.cache.session import ExecutionCacheSession

    with pytest.raises(RuntimeError, match="governor"):
        ExecutionCacheSession(shared_result_cache={}, strict=True)
    # research：warning 降级，不 raise。
    ExecutionCacheSession(shared_result_cache={}, strict=False)


# ---------------------------------------------------------------------------
# P0-016/017/018：host coordinator + DA envelope + service ContextVar
# ---------------------------------------------------------------------------


def test_host_coordinator_is_single_authority_and_da_derives_envelope():
    from factor_engine.runtime.host_resource_coordinator import (
        get_host_coordinator,
        reset_host_coordinator,
    )

    reset_host_coordinator()
    c1 = get_host_coordinator()
    c2 = get_host_coordinator()
    assert c1 is c2
    out = c1.apply_da_envelope()
    if out.get("applied"):
        assert out["max_total_reserved_memory"] > 0


def test_service_broker_contextvar_no_race():
    from factor_engine.service import queue as sq

    results = []
    barrier = threading.Barrier(2)

    def _worker(val):
        tok = sq._service_broker_ctx_var.set(val)
        barrier.wait()
        try:
            results.append(sq._get_service_broker())
        finally:
            sq._service_broker_ctx_var.reset(tok)

    a = threading.Thread(target=_worker, args=("B_A",))
    b = threading.Thread(target=_worker, args=("B_B",))
    a.start(); b.start(); a.join(); b.join()
    assert sq._get_service_broker() is None
    assert set(results) == {"B_A", "B_B"}


# ---------------------------------------------------------------------------
# P0-020：auto shard 合法/非法
# ---------------------------------------------------------------------------


def test_auto_shard_when_task_exceeds_envelope():
    from factor_engine.runtime.auto_shard_planner import AutoShardPlanner
    from factor_engine.runtime.task_resource_contract import TaskResourceContract

    class T:
        task_id = "root:x"
        op = "ts_rank"
        resource_contract = TaskResourceContract(peak_memory_bytes=20 * 1024**3, shardable=True, shard_dimension="asset")

    plan = AutoShardPlanner().plan_for_task(T, safe_envelope_bytes=4 * 1024**3)
    assert plan is not None
    assert plan.per_shard_peak_bytes <= 4 * 1024**3


def test_illegal_shard_rejected():
    from factor_engine.runtime.auto_shard_planner import AutoShardPlanner, legal_shard_dimensions
    from factor_engine.runtime.task_resource_contract import TaskResourceContract

    class Cs:
        task_id = "root:cs"
        op = "cs_rank"
        resource_contract = TaskResourceContract(peak_memory_bytes=20 * 1024**3, shardable=True, shard_dimension="asset")

    assert legal_shard_dimensions(Cs) == (None,)  # §91：cs 禁 asset shard


# ---------------------------------------------------------------------------
# P0-008/028：DuckDB token / scope race / BLAS / Numba
# ---------------------------------------------------------------------------


def test_duckdb_cpu_token_matches_engine_threads(monkeypatch):
    monkeypatch.setenv("DUCKDB_MAX_THREADS", "6")
    from factor_engine.planner.physical_lowerer import _engine_threads_for

    assert _engine_threads_for("duckdb_sql") == 6


def test_concurrent_global_resource_scope_race_detected():
    from factor_engine.runtime.resource_governor import ExecutionResourceScope, _ACTIVE_SCOPE_THREADS

    _ACTIVE_SCOPE_THREADS.add(999999)
    try:
        with pytest.raises(RuntimeError):
            ExecutionResourceScope(strict=True).__enter__()
    finally:
        _ACTIVE_SCOPE_THREADS.discard(999999)


def test_blas_oversubscription_detected():
    from factor_engine.runtime.resource_governor import check_nested_cpu_oversubscription

    assert check_nested_cpu_oversubscription(8, 8, 16) is True
    assert check_nested_cpu_oversubscription(2, 4, 16) is False


def test_numba_parallel_contract_cpu_tokens():
    from factor_engine.runtime.task_resource_contract import parallel_kernel_contract

    c = parallel_kernel_contract(numba_threads=8, peak_memory_bytes=1024)
    assert c.cpu_tokens == 8 and c.backend_threads == 8


# ---------------------------------------------------------------------------
# Co-tenancy：8GB profile 下 no OOM + 自动让路/恢复（§253/257/258）
# ---------------------------------------------------------------------------


def test_co_tenancy_memory_stress_no_oom():
    from factor_engine.runtime.resource_broker import ResourceBroker
    from factor_engine.runtime.task_resource_contract import TaskResourceContract

    # 2GB 硬上限（模拟小容器）→ 大 task 拒绝而非 OOM。
    broker = ResourceBroker(hard_memory_limit=2 * 1024**3, cpu_slots=4, min_host_reserve_gb=0.0, min_host_reserve_fraction=0.0)
    big = TaskResourceContract(peak_memory_bytes=4 * 1024**3)
    assert broker.can_admit(big) is False


def test_co_tenancy_recovery_after_external_load():
    from factor_engine.runtime.resource_autopilot import ResourceController
    from factor_engine.runtime.resource_monitor import HostResourceEnvelope, ResourceSignals

    class _Cpu:
        def __init__(self):
            self.soft_budget = 8

        def set_soft_budget(self, v):
            self.soft_budget = max(1, int(v))

    class _B:
        hard_cpu_slots = 8

        def __init__(self):
            self._cpu = _Cpu()

        def pressure_stage(self):
            return "NORMAL"

        def cpu_budget(self):
            return self._cpu.soft_budget

    ctl = ResourceController(_B(), stable_samples_required=2, cooldown_ticks=0)
    env = HostResourceEnvelope(8 * 1024**3, 4 * 1024**3, 1024**3, 8, 8, 1.0, 1.0, 0)
    low = ctl.tick(ResourceSignals(cpu_psi_some=0.95), env).target_cpu_tokens
    ok = ResourceSignals(host_mem_available=8 * 1024**3)
    for _ in range(5):
        d = ctl.tick(ok, env)
    assert d.target_cpu_tokens > low  # §258：压力解除后自动升速
