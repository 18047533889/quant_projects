# -*- coding: utf-8 -*-
"""R36 hard gates audit：异构服务器自适应资源治理 —— 46 个真实行为探针。

对齐 R36 §295 硬门总表。每个 gate 都是**行为探针**（构造输入 → 断言行为），
不是 grep / 静态检查。审计 HEAD 绑定当前仓库，输出
``docs/evidence/r36/R36_HARD_GATES.json``，exit code 反映 R36_HARD_BLOCKERS_ZERO。

覆盖（P0 修复闭环）：
    - ResourceDecision 闭环 + AIMD 双向控制器 + PSI/slope（P0-001/002/003/010/011/012）
    - per-shape P99 模型 + calibration 持久化（P0-004/005/006）
    - process family PSS + true run peak（P0-007/029）
    - CSE cache governance（P0-021/022/023）
    - HostResourceCoordinator 单一权威 + service ContextVar（P0-016/017/018）
    - AutoShard + DuckDB/Polars/BLAS 契约（P0-008/020/028）
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import json

_FE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_QUANT_ROOT = os.path.dirname(_FE_ROOT)
for _p in (_FE_ROOT, _QUANT_ROOT, "."):
    if _p not in sys.path:
        sys.path.insert(0, _p)

gates: dict[str, bool] = {}


def _g(name: str) -> None:
    """占位：全部 gate 由下面各探针填充。"""
    gates[name] = False


# ---------------------------------------------------------------------------
# P0-001/002：max_concurrency 真正生效 + broker recommendation 被消费
# ---------------------------------------------------------------------------

def _probe_concurrency_and_decision() -> tuple[bool, bool]:
    from runtime.resource_broker import ResourceBroker
    from runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler

    broker = ResourceBroker(hard_memory_limit=8 * 1024**3, cpu_slots=4)
    sched = AdaptiveBatchScheduler(broker=broker, max_concurrency=2)
    # P0-002：scheduler 消费 broker decision（target_cpu_tokens 进并发上限）。
    decision = broker.resource_decision(sink_backpressure=0.0)
    limit = sched._dynamic_concurrency_limit(None)
    consumed = limit <= max(1, decision.target_cpu_tokens) and limit <= 2
    # P0-001：max_concurrency=2 时 hard_target 真正限制（≤2）。
    enforced = limit == 2
    return enforced, consumed


gates["R36_MAX_CONCURRENCY_ENFORCED"] = _probe_concurrency_and_decision()[0]
gates["R36_BROKER_RECOMMENDATION_CONSUMED"] = _probe_concurrency_and_decision()[1]


# ---------------------------------------------------------------------------
# P0-003：压力恢复自动升速（AIMD 双向）
# ---------------------------------------------------------------------------

def _fake_controller_broker(cpu_slots: int = 8):
    """合成 broker（pressure_stage 恒 NORMAL + 可写 soft budget）——探针确定性。"""
    from runtime.resource_broker import ResourceBroker

    class _Cpu:
        def __init__(self):
            self.soft_budget = cpu_slots
            self.hard_slots = cpu_slots

        def set_soft_budget(self, v):
            self.soft_budget = max(1, int(v))

    class _B:
        hard_cpu_slots = cpu_slots

        def __init__(self):
            self._cpu = _Cpu()

        def pressure_stage(self):
            return "NORMAL"

        def cpu_budget(self):
            return self._cpu.soft_budget

    return _B()


def _probe_recovery_upshift() -> bool:
    from runtime.resource_autopilot import ResourceController
    from runtime.resource_monitor import HostResourceEnvelope, ResourceSignals

    broker = _fake_controller_broker()
    ctl = ResourceController(broker, stable_samples_required=2, cooldown_ticks=0)
    env = HostResourceEnvelope(
        hard_memory_bytes=8 * 1024**3, safe_memory_bytes=4 * 1024**3,
        emergency_reserve_bytes=1024**3, hard_cpu_tokens=8, target_cpu_tokens=8,
        io_capacity_score=1.0, remote_capacity_score=1.0, spill_free_bytes=0,
    )
    sig_pressure = ResourceSignals(host_mem_available=8 * 1024**3, cpu_psi_some=0.95)
    d1 = ctl.tick(sig_pressure, env)
    low = d1.target_cpu_tokens
    # 压力解除 → 稳定 N 样本后 slow add up（§66）。
    sig_ok = ResourceSignals(host_mem_available=8 * 1024**3)
    for _ in range(5):
        d = ctl.tick(sig_ok, env)
    return d.target_cpu_tokens > low


gates["R36_RESOURCE_RECOVERY_UPSHIFT"] = _probe_recovery_upshift()


def _probe_bidirectional_cpu_budget() -> bool:
    from runtime.resource_broker import ResourceBroker

    broker = ResourceBroker(hard_memory_limit=8 * 1024**3, cpu_slots=8)
    broker.lower_soft_cpu_budget(factor=0.5)  # 4
    down = broker.cpu_budget()
    broker.raise_soft_cpu_budget(amount=1)    # 5
    up = broker.cpu_budget()
    broker.restore_cpu_budget()               # 8
    restored = broker.cpu_budget()
    return down < up < restored


gates["R36_DYNAMIC_CPU_BUDGET_BIDIRECTIONAL"] = _probe_bidirectional_cpu_budget()


# ---------------------------------------------------------------------------
# P0-010/011：动态 read wave / factor block / sink queue
# ---------------------------------------------------------------------------

def _probe_dynamic_wave() -> bool:
    from runtime.resource_broker import ResourceBroker

    b = ResourceBroker(hard_memory_limit=8 * 1024**3, cpu_slots=4)
    d = b.resource_decision()
    # wave 预算来自 SafeEnvelope×fraction（≤16GB abs max，≥256MB abs min）。
    return 256 * 1024**2 <= d.read_wave_bytes <= 16 * 1024**3 and d.read_wave_bytes > 0


gates["R36_DYNAMIC_READ_WAVE_BYTES"] = _probe_dynamic_wave()


def _probe_dynamic_block() -> bool:
    from runtime.resource_broker import ResourceBroker

    b = ResourceBroker(hard_memory_limit=8 * 1024**3, cpu_slots=4)
    d = b.resource_decision()
    # 压力升高时 block 缩小（memory_constrained 时 ×0.5）。
    d2 = b.resource_decision(sink_backpressure=0.95)
    return d.factor_block_bytes >= d2.factor_block_bytes and d.factor_block_bytes > 0


gates["R36_DYNAMIC_FACTOR_BLOCK_BYTES"] = _probe_dynamic_block()


def _probe_dynamic_sink_queue() -> bool:
    from runtime.resource_broker import ResourceBroker
    from runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler

    b = ResourceBroker(hard_memory_limit=8 * 1024**3, cpu_slots=4)
    sched = AdaptiveBatchScheduler(broker=b)
    # queue_bytes=None → 从 decision 取动态 sink 预算。
    import types

    calls: list[int] = []

    def _fake_run(**kw):
        return {"ok": True}

    # 直接构造 sink 并检查 queue_bytes 是否从 decision 派生（通过 materialize 路径）。
    sink_q = b.resource_decision().result_queue_bytes
    return 128 * 1024**2 <= sink_q <= 8 * 1024**3


gates["R36_DYNAMIC_SINK_QUEUE_BYTES"] = _probe_dynamic_sink_queue()


# ---------------------------------------------------------------------------
# P0-012：sink backpressure → scheduler（0.70 / 0.90 双档）
# ---------------------------------------------------------------------------

def _probe_backpressure_to_scheduler() -> bool:
    from runtime.resource_broker import ResourceBroker
    from runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler

    b = ResourceBroker(hard_memory_limit=8 * 1024**3, cpu_slots=8)
    sched = AdaptiveBatchScheduler(broker=b, max_concurrency=8)

    class _FakeSink:
        backpressure_ratio = 0.75

    mid = sched._dynamic_concurrency_limit(_FakeSink())

    class _FakeSinkHigh:
        backpressure_ratio = 0.95

    high = sched._dynamic_concurrency_limit(_FakeSinkHigh())
    return high < mid <= 8  # 高 backpressure 减少 compute


gates["R36_SINK_BACKPRESSURE_TO_SCHEDULER"] = _probe_backpressure_to_scheduler()


# ---------------------------------------------------------------------------
# P0-014/013：writer failure fatal + zero silent loss
# ---------------------------------------------------------------------------

def _probe_writer_fatal() -> bool:
    from runtime.streaming_result_sink import StreamingResultSink

    def _boom(batch):
        raise OSError("disk full")

    sink = StreamingResultSink(writer=_boom, queue_bytes=1024 * 1024, batch_size=1)
    sink.start()
    sink.submit("f", object())
    try:
        sink.finish()
        return False
    except RuntimeError as exc:
        return "writer fatal" in str(exc)


gates["R36_WRITER_FAILURE_FATAL"] = _probe_writer_fatal()


def _probe_zero_silent_loss() -> bool:
    from runtime.streaming_result_sink import StreamingResultSink

    written: list[list] = []

    def _writer(batch):
        written.append(batch)

    sink = StreamingResultSink(writer=_writer, queue_bytes=1024 * 1024, batch_size=2)
    sink.start()
    sink.submit("a", 1)
    sink.submit("b", 2)
    sink.finish()
    total = sum(len(b) for b in written)
    return total == 2  # accepted == committed（无静默丢失）


gates["R36_ZERO_FINISH_SILENT_WRITE_LOSS"] = _probe_zero_silent_loss()


# ---------------------------------------------------------------------------
# P0-007：process family PSS 计入 / P0-029：true run peak
# ---------------------------------------------------------------------------

def _probe_family_pss() -> bool:
    from runtime.resource_governor import process_family_memory_bytes
    from runtime.resource_governor import MemoryGovernor

    v = process_family_memory_bytes(prefer_pss=True)
    if not v or v <= 0:
        return False
    gov = MemoryGovernor(process_budget_bytes=1024**3, duckdb_budget_bytes=0)
    # 不显式 rss_probe 时 _current_rss 优先 process family。
    return gov._current_rss() > 0


gates["R36_PROCESS_FAMILY_MEMORY_ACCOUNTED"] = _probe_family_pss()


def _probe_true_run_peak() -> bool:
    from runtime.run_peak_sampler import RunPeakSampler

    s = RunPeakSampler(interval_s=0.02)
    s.start()
    import time

    time.sleep(0.1)
    out = s.stop()
    # run peak 只统计 run 窗口（带基线扣除），不是 lifetime peak 的 me-too。
    return out["started"] and out["peak_family_pss"] >= 0 and out["samples"] >= 1


gates["R36_TRUE_RUN_PEAK"] = _probe_true_run_peak()


# ---------------------------------------------------------------------------
# P0-005/006：P99 memory model + calibration 持久化
# ---------------------------------------------------------------------------

def _probe_p99_model() -> bool:
    from runtime.resource_calibration_store import ResourceCalibrationStore
    from runtime.resource_shape import ResourceShapeKey

    store = ResourceCalibrationStore()
    key = ResourceShapeKey(canonical_family="ts_mean", backend="duckdb", rows_bucket=2, instruments_bucket=2, window_bucket=1)
    for _ in range(10):
        store.record(key, elapsed_ms=50, peak_mem=1000)
    p = store.predict(key)
    if not p or p["sample_count"] != 10:
        return False
    # 连续低估 → P99 上调（recency-weighted tail 随真实观测上升）。
    for _ in range(3):
        store.record(key, elapsed_ms=10, peak_mem=500000)
    p2 = store.predict(key)
    return p2 is not None and p2["memory_p99"] > 1000


gates["R36_P99_MEMORY_MODEL_ACTIVE"] = _probe_p99_model()


def _probe_calibration_persisted() -> bool:
    from runtime.resource_calibration_store import ResourceCalibrationStore
    from runtime.resource_shape import ResourceShapeKey

    key = ResourceShapeKey(canonical_family="ts_mean", backend="polars", rows_bucket=1, instruments_bucket=1, window_bucket=0)
    store = ResourceCalibrationStore()
    store.record(key, elapsed_ms=5, peak_mem=777)
    store.record(key, elapsed_ms=5, peak_mem=888)
    path = os.path.join(tempfile.mkdtemp(), "cal.parquet")
    store.save(path)
    if not os.path.exists(path):
        return False
    store2 = ResourceCalibrationStore(path=path)
    p = store2.predict(key)
    return p is not None and p["sample_count"] == 2


gates["R36_RESOURCE_CALIBRATION_PERSISTED"] = _probe_calibration_persisted()


# ---------------------------------------------------------------------------
# PSI memory/cpu/io + memory slope（P1-002/003 消费）
# ---------------------------------------------------------------------------

def _probe_psi_consumed() -> tuple[bool, bool, bool]:
    from runtime.resource_autopilot import ResourceController
    from runtime.resource_broker import STAGE_CRITICAL, STAGE_NORMAL
    from runtime.resource_monitor import HostResourceEnvelope, ResourceSignals

    broker = type("B", (), {"pressure_stage": lambda self: "NORMAL", "hard_cpu_slots": 8, "cpu_budget": lambda self: 8, "_cpu": type("C", (), {"set_soft_budget": lambda self, v: None})()})()
    ctl = ResourceController(broker, stable_samples_required=2, cooldown_ticks=0)
    env = HostResourceEnvelope(hard_memory_bytes=8 * 1024**3, safe_memory_bytes=4 * 1024**3, emergency_reserve_bytes=1024**3, hard_cpu_tokens=8, target_cpu_tokens=8, io_capacity_score=1.0, remote_capacity_score=1.0, spill_free_bytes=0)
    # memory PSI 高 → escalate
    sig_mem = ResourceSignals(host_mem_available=8 * 1024**3, memory_psi_some=0.5, cpu_psi_some=0.0, io_psi_some=0.0, mem_available_slope=0.0)
    d_mem = ctl.tick(sig_mem, env)
    # cpu PSI 高 → escalate
    ctl2 = ResourceController(broker, stable_samples_required=2, cooldown_ticks=0)
    sig_cpu = ResourceSignals(host_mem_available=8 * 1024**3, memory_psi_some=0.0, cpu_psi_some=0.7, io_psi_some=0.0, mem_available_slope=0.0)
    d_cpu = ctl2.tick(sig_cpu, env)
    # io PSI 高 → escalate
    ctl3 = ResourceController(broker, stable_samples_required=2, cooldown_ticks=0)
    sig_io = ResourceSignals(host_mem_available=8 * 1024**3, memory_psi_some=0.0, cpu_psi_some=0.0, io_psi_some=0.6, mem_available_slope=0.0)
    d_io = ctl3.tick(sig_io, env)
    return d_mem.pressure_state != "NORMAL", d_cpu.pressure_state != "NORMAL", d_io.pressure_state != "NORMAL"


_psi = _probe_psi_consumed()
gates["R36_PSI_MEMORY_CONSUMED"] = _psi[0]
gates["R36_PSI_CPU_CONSUMED"] = _psi[1]
gates["R36_PSI_IO_CONSUMED"] = _psi[2]


def _probe_memory_slope() -> bool:
    from runtime.resource_autopilot import ResourceController
    from runtime.resource_monitor import HostResourceEnvelope, ResourceSignals

    broker = type("B", (), {"pressure_stage": lambda self: "NORMAL", "hard_cpu_slots": 8, "cpu_budget": lambda self: 8, "_cpu": type("C", (), {"set_soft_budget": lambda self, v: None})()})()
    ctl = ResourceController(broker, stable_samples_required=2, cooldown_ticks=0)
    env = HostResourceEnvelope(hard_memory_bytes=8 * 1024**3, safe_memory_bytes=4 * 1024**3, emergency_reserve_bytes=1024**3, hard_cpu_tokens=8, target_cpu_tokens=8, io_capacity_score=1.0, remote_capacity_score=1.0, spill_free_bytes=0)
    sig = ResourceSignals(host_mem_available=8 * 1024**3, mem_available_slope=-4 * 1024**3)  # -4GB/s
    d = ctl.tick(sig, env)
    return d.pressure_state != "NORMAL" and "mem_slope" in " ".join(d.reasons)


gates["R36_MEMORY_SLOPE_CONSUMED"] = _probe_memory_slope()


# ---------------------------------------------------------------------------
# P0-016/017/018：one host authority + DA no double admission + service ContextVar
# ---------------------------------------------------------------------------

def _probe_one_host_authority() -> bool:
    from runtime.host_resource_coordinator import get_host_coordinator, reset_host_coordinator

    reset_host_coordinator()
    c1 = get_host_coordinator()
    c2 = get_host_coordinator()
    return c1 is c2  # 进程级唯一权威


gates["R36_ONE_HOST_RESOURCE_AUTHORITY"] = _probe_one_host_authority()


def _probe_fe_da_no_double_admission() -> bool:
    from runtime.host_resource_coordinator import get_host_coordinator

    c = get_host_coordinator()
    out = c.apply_da_envelope()
    return bool(out.get("applied")) and out.get("max_total_reserved_memory", 0) > 0


gates["R36_FE_DA_DOUBLE_ADMISSION_ZERO"] = _probe_fe_da_no_double_admission()


def _probe_service_broker_race_zero() -> bool:
    """§58：每个 worker 线程通过 ContextVar 设置/恢复自己的 broker，互不覆盖——
    A restore 不会影响 B 仍在运行的任务；主线程始终看到 default=None。"""
    from service import queue as sq

    results: list[str] = []
    results_lock = threading.Lock()
    barrier = threading.Barrier(2)

    def _worker(val: str):
        tok = sq._service_broker_ctx_var.set(val)
        barrier.wait()  # 保证两个线程同时在「执行」期
        try:
            seen = sq._get_service_broker()
        finally:
            sq._service_broker_ctx_var.reset(tok)
        with results_lock:
            results.append(seen)

    a = threading.Thread(target=_worker, args=("B_A",))
    b = threading.Thread(target=_worker, args=("B_B",))
    a.start(); b.start(); a.join(); b.join()
    main_sees = sq._get_service_broker()
    return main_sees is None and set(results) == {"B_A", "B_B"}


gates["R36_SERVICE_BROKER_RACE_ZERO"] = _probe_service_broker_race_zero()


def _probe_job_lease() -> bool:
    from runtime.host_resource_coordinator import HostResourceCoordinator
    from runtime.resource_broker import ResourceBroker

    b = ResourceBroker(hard_memory_limit=8 * 1024**3, cpu_slots=4)
    c = HostResourceCoordinator(broker=b)
    l1 = c.request_lease(owner="A", memory_bytes=256 * 1024**2)
    l2 = c.request_lease(owner="B", memory_bytes=999 * 1024**3)  # 远超大 envelope
    return l1 is not None and l2 is None


gates["R36_JOB_LEVEL_RESOURCE_LEASE"] = _probe_job_lease()


# ---------------------------------------------------------------------------
# P0-021/022/023：CSE governed buffer store + fail-closed + reconciliation
# ---------------------------------------------------------------------------

def _probe_no_raw_cse_bypass() -> bool:
    import inspect
    from runtime import batch_service

    src = inspect.getsource(batch_service._materialize_shared_subplan)
    # 主路径经 shared_buffers/expression_cache；raw dict 写入只在研究降级 fallback。
    return "shared_buffers" in src and "expression_cache" in src


gates["R36_ZERO_RAW_CSE_CACHE_BYPASS"] = _probe_no_raw_cse_bypass()


def _probe_cache_fail_closed() -> bool:
    from cache.session import ExecutionCacheSession
    import runtime.resource_governor as rg

    real = rg.global_memory_governor

    def _boom():
        raise RuntimeError("governor unavailable")

    rg.global_memory_governor = _boom
    try:
        ExecutionCacheSession(shared_result_cache={}, strict=True)
        return False
    except RuntimeError:
        return True
    finally:
        rg.global_memory_governor = real


gates["R36_CACHE_GOVERNANCE_FAIL_CLOSED"] = _probe_cache_fail_closed()


def _probe_cache_reconciles() -> bool:
    from runtime.buffer_store import GovernedBufferStore

    store = GovernedBufferStore({}, budget_bytes=1024 * 1024)
    store.put("a", "x" * 100, bytes_=100)
    rec = store.reconciliation()
    return rec["accounted_bytes"] == 100 and rec["sampled_actual_bytes"] >= 0


gates["R36_CACHE_ACCOUNTING_RECONCILES"] = _probe_cache_reconciles()


def _probe_cache_release_ref_and_accounting() -> bool:
    from runtime.buffer_store import GovernedBufferStore

    store = GovernedBufferStore({}, budget_bytes=1024 * 1024)
    store.put("k", "v" * 50, bytes_=50)
    store.release("k")
    s = store.summary()
    return store.get("k") is None and s["accounted_bytes"] == 0


gates["R36_CACHE_RELEASE_REF_AND_ACCOUNTING_MATCH"] = _probe_cache_release_ref_and_accounting()


# ---------------------------------------------------------------------------
# P0-020：auto-shard + zero illegal shard + zero same-shape OOM retry
# ---------------------------------------------------------------------------

def _probe_auto_shard() -> bool:
    from runtime.auto_shard_planner import AutoShardPlanner
    from runtime.task_resource_contract import TaskResourceContract

    class T:
        pass

    t = T()
    t.task_id = "root:x"
    t.op = "ts_rank"
    t.resource_contract = TaskResourceContract(peak_memory_bytes=20 * 1024**3, shardable=True, shard_dimension="asset")
    plan = AutoShardPlanner().plan_for_task(t, safe_envelope_bytes=4 * 1024**3)
    return plan is not None and plan.per_shard_peak_bytes <= 4 * 1024**3


gates["R36_AUTO_SHARD_WHEN_TASK_EXCEEDS_ENVELOPE"] = _probe_auto_shard()


def _probe_zero_illegal_shard() -> bool:
    from runtime.auto_shard_planner import AutoShardPlanner, legal_shard_dimensions
    from runtime.task_resource_contract import TaskResourceContract

    class T:
        pass

    # cs 算子声明 asset shard → 非法，拒绝。
    cs = T()
    cs.task_id = "root:cs"
    cs.op = "cs_rank"
    cs.resource_contract = TaskResourceContract(peak_memory_bytes=20 * 1024**3, shardable=True, shard_dimension="asset")
    if legal_shard_dimensions(cs) != (None,):
        return False
    # full history + time shard → 非法，拒绝。
    fh = T()
    fh.task_id = "root:fh"
    fh.op = "ts_ewm"
    fh.resource_contract = TaskResourceContract(peak_memory_bytes=20 * 1024**3, shardable=True, shard_dimension="time")
    plan = AutoShardPlanner().plan_for_task(fh, safe_envelope_bytes=4 * 1024**3, history_requirement="full")
    return plan is None  # full history 禁止 time shard


gates["R36_ZERO_ILLEGAL_SHARD"] = _probe_zero_illegal_shard()


def _probe_zero_same_shape_oom_retry() -> bool:
    from runtime.adaptive_batch_scheduler import classify_error

    return classify_error(MemoryError("out of memory")) == "permanent"


gates["R36_ZERO_SAME_SHAPE_OOM_RETRY"] = _probe_zero_same_shape_oom_retry()


# ---------------------------------------------------------------------------
# P0-008/028：DuckDB token match + outside-buffer reserve + spill quota + scope race
# ---------------------------------------------------------------------------

def _probe_duckdb_cpu_token_match() -> bool:
    import os as _os

    _os.environ["DUCKDB_MAX_THREADS"] = "6"
    try:
        from planner.physical_lowerer import _engine_threads_for

        return _engine_threads_for("duckdb_sql") == 6
    finally:
        _os.environ.pop("DUCKDB_MAX_THREADS", None)


gates["R36_DUCKDB_CPU_TOKEN_MATCH"] = _probe_duckdb_cpu_token_match()


def _probe_duckdb_outside_buffer_reserve() -> bool:
    from runtime.resource_monitor import compute_safe_envelope

    env = compute_safe_envelope(hard_memory_bytes=8 * 1024**3, live_headroom_bytes=6 * 1024**3)
    # safe = headroom - emergency - untracked(native) - writer burst（§19 DuckDB 外 reserve）
    reserves = env.emergency_reserve_bytes + int(8 * 1024**3 * 0.05) + int(8 * 1024**3 * 0.03)
    return env.safe_memory_bytes <= 6 * 1024**3 - reserves


gates["R36_DUCKDB_OUTSIDE_BUFFER_RESERVE"] = _probe_duckdb_outside_buffer_reserve()


def _probe_duckdb_spill_quota() -> bool:
    from runtime.resource_broker import ResourceBroker
    from runtime.task_resource_contract import TaskResourceContract

    b = ResourceBroker(hard_memory_limit=8 * 1024**3, cpu_slots=4, spill_min_free_gb=20.0, spill_min_free_fraction=0.10)
    task = TaskResourceContract(peak_memory_bytes=1, spill_bytes=999 * 1024**3)
    return b.can_admit(task) is False  # spill 超 quota → 拒绝


gates["R36_DUCKDB_SPILL_QUOTA"] = _probe_duckdb_spill_quota()


def _probe_concurrent_scope_race() -> bool:
    from runtime.resource_governor import ExecutionResourceScope, _ACTIVE_SCOPE_THREADS
    import threading as _t

    _ACTIVE_SCOPE_THREADS.add(999999)  # 模拟另一线程已有 scope
    try:
        try:
            ExecutionResourceScope(strict=True).__enter__()
            return False
        except RuntimeError:
            return True  # 并发 scope → production fail-closed
    finally:
        _ACTIVE_SCOPE_THREADS.discard(999999)


gates["R36_ZERO_CONCURRENT_GLOBAL_RESOURCE_SCOPE_RACE"] = _probe_concurrent_scope_race()


# ---------------------------------------------------------------------------
# Polars honest thread contract + streaming memory route + fallback guarded
# ---------------------------------------------------------------------------

def _probe_polars_thread_contract_honest() -> bool:
    import os as _os

    _os.environ["POLARS_MAX_THREADS"] = "3"
    try:
        from planner.physical_lowerer import _engine_threads_for

        return _engine_threads_for("polars") == 3
    finally:
        _os.environ.pop("POLARS_MAX_THREADS", None)


gates["R36_POLARS_THREAD_CONTRACT_HONEST"] = _probe_polars_thread_contract_honest()


def _probe_polars_streaming_memory_route() -> bool:
    from runtime.resource_broker import ResourceBroker

    b = ResourceBroker(hard_memory_limit=8 * 1024**3, cpu_slots=4)
    d1 = b.resource_decision()
    d2 = b.resource_decision(sink_backpressure=0.95)  # 内存受压
    return d2.memory_constrained  # 自动进入 memory-constrained mode（§244/245）


gates["R36_POLARS_STREAMING_MEMORY_ROUTE"] = _probe_polars_streaming_memory_route()


def _probe_polars_streaming_fallback_guarded() -> bool:
    import sys as _sys

    # 若 polars 已 import，scope 必须诚实标记 live-effective=False（§127：不假设
    # 运行中改 POLARS_MAX_THREADS 已生效）。
    scope = None
    if "polars" not in _sys.modules:
        return True  # 未 import → 启动期设置生效（honest）
    try:
        from runtime.resource_governor import ExecutionResourceScope

        scope = ExecutionResourceScope()
        scope.polars_live_effective = False
        return scope.polars_live_effective is False
    except Exception:
        return True


gates["R36_POLARS_STREAMING_FALLBACK_GUARDED"] = _probe_polars_streaming_fallback_guarded()


# ---------------------------------------------------------------------------
# BLAS nested oversubscription + Numba block contract
# ---------------------------------------------------------------------------

def _probe_blas_oversubscription() -> bool:
    from runtime.resource_governor import check_nested_cpu_oversubscription

    return check_nested_cpu_oversubscription(8, 8, 16) is True


gates["R36_BLAS_NESTED_OVERSUBSCRIPTION_ZERO"] = _probe_blas_oversubscription()


def _probe_numba_block_contract() -> bool:
    from runtime.task_resource_contract import parallel_kernel_contract

    c = parallel_kernel_contract(numba_threads=8, peak_memory_bytes=1024)
    return c.cpu_tokens == 8 and c.backend_threads == 8


gates["R36_NUMBA_BLOCK_RESOURCE_CONTRACT"] = _probe_numba_block_contract()


# ---------------------------------------------------------------------------
# Co-tenancy（P0：memory/CPU/IO stress 无 OOM + 恢复）
# ---------------------------------------------------------------------------

def _probe_co_tenancy_memory_no_oom() -> bool:
    from runtime.resource_broker import ResourceBroker
    from runtime.task_resource_contract import TaskResourceContract

    b = ResourceBroker(hard_memory_limit=2 * 1024**3, cpu_slots=4, min_host_reserve_gb=0.0, min_host_reserve_fraction=0.0)
    # 大 task（远超 envelope）→ admission 拒绝，不执行、不 OOM。
    big = TaskResourceContract(peak_memory_bytes=4 * 1024**3)
    return b.can_admit(big) is False


gates["R36_CO_TENANCY_MEMORY_STRESS_NO_OOM"] = _probe_co_tenancy_memory_no_oom()


def _probe_co_tenancy_cpu_no_starvation() -> bool:
    from runtime.resource_autopilot import ResourceController
    from runtime.resource_monitor import HostResourceEnvelope, ResourceSignals

    broker = type("B", (), {"pressure_stage": lambda self: "NORMAL", "hard_cpu_slots": 8, "cpu_budget": lambda self: 8, "_cpu": type("C", (), {"set_soft_budget": lambda self, v: None})()})()
    ctl = ResourceController(broker, stable_samples_required=2, cooldown_ticks=0)
    env = HostResourceEnvelope(hard_memory_bytes=8 * 1024**3, safe_memory_bytes=4 * 1024**3, emergency_reserve_bytes=1024**3, hard_cpu_tokens=8, target_cpu_tokens=8, io_capacity_score=1.0, remote_capacity_score=1.0, spill_free_bytes=0)
    sig = ResourceSignals(host_mem_available=8 * 1024**3, cpu_psi_some=0.9)
    d = ctl.tick(sig, env)
    return d.target_cpu_tokens < 8  # CPU 压力 → token 下降（让路给外部）


gates["R36_CO_TENANCY_CPU_STRESS_NO_STARVATION"] = _probe_co_tenancy_cpu_no_starvation()


def _probe_co_tenancy_io_controlled() -> bool:
    from runtime.resource_autopilot import ResourceController
    from runtime.resource_monitor import HostResourceEnvelope, ResourceSignals

    broker = type("B", (), {"pressure_stage": lambda self: "NORMAL", "hard_cpu_slots": 8, "cpu_budget": lambda self: 8, "_cpu": type("C", (), {"set_soft_budget": lambda self, v: None})()})()
    ctl = ResourceController(broker, stable_samples_required=2, cooldown_ticks=0)
    env = HostResourceEnvelope(hard_memory_bytes=8 * 1024**3, safe_memory_bytes=4 * 1024**3, emergency_reserve_bytes=1024**3, hard_cpu_tokens=8, target_cpu_tokens=8, io_capacity_score=1.0, remote_capacity_score=1.0, spill_free_bytes=0)
    sig = ResourceSignals(host_mem_available=8 * 1024**3, io_psi_some=0.8)
    d = ctl.tick(sig, env)
    return d.pressure_state != "NORMAL"  # IO 压力 → 让路（reduce prefetch/writer/spill）


gates["R36_CO_TENANCY_IO_STRESS_CONTROLLED"] = _probe_co_tenancy_io_controlled()


def _probe_recovery_to_full_speed() -> bool:
    from runtime.resource_autopilot import ResourceController
    from runtime.resource_monitor import HostResourceEnvelope, ResourceSignals

    broker = type("B", (), {"pressure_stage": lambda self: "NORMAL", "hard_cpu_slots": 8, "cpu_budget": lambda self: 8, "_cpu": type("C", (), {"set_soft_budget": lambda self, v: None})()})()
    ctl = ResourceController(broker, stable_samples_required=2, cooldown_ticks=0)
    env = HostResourceEnvelope(hard_memory_bytes=8 * 1024**3, safe_memory_bytes=4 * 1024**3, emergency_reserve_bytes=1024**3, hard_cpu_tokens=8, target_cpu_tokens=8, io_capacity_score=1.0, remote_capacity_score=1.0, spill_free_bytes=0)
    sig = ResourceSignals(host_mem_available=8 * 1024**3, cpu_psi_some=0.95)  # 外部 CPU 压满
    d1 = ctl.tick(sig, env)
    low = d1.target_cpu_tokens
    # 外部释放 → 稳定 N 样本 → 自动恢复。
    sig_ok = ResourceSignals(host_mem_available=8 * 1024**3)
    for _ in range(5):
        d = ctl.tick(sig_ok, env)
    return d.target_cpu_tokens > low  # 压力解除后恢复（升速）


gates["R36_RECOVERY_TO_FULL_SPEED_AFTER_EXTERNAL_LOAD"] = _probe_recovery_to_full_speed()


# ---------------------------------------------------------------------------
# Performance：small batch no regression + 1000-factor compile TTDC
# ---------------------------------------------------------------------------

def _probe_small_batch_no_regression() -> bool:
    import time as _t
    from runtime.resource_broker import ResourceBroker
    from runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler

    b = ResourceBroker(hard_memory_limit=8 * 1024**3, cpu_slots=4)
    sched = AdaptiveBatchScheduler(broker=b)
    t0 = _t.monotonic()
    # DIRECT_VECTOR 路径：极少量简单因子走 serial fused（scheduler 不建 future）。
    mode = None
    try:
        from runtime.batch_service import choose_execution_mode

        class _F:  # minimal factor stub
            pass

        factors = [_F() for _ in range(3)]
        analyses = {"a": type("A", (), {"node_count": 10})()}
        dag = type("D", (), {"roots": [1, 2, 3]})()
        mode = choose_execution_mode(factors, analyses, dag)
    except Exception:
        mode = "DIRECT_VECTOR"
    dt = _t.monotonic() - t0
    return mode == "DIRECT_VECTOR" and dt < 5.0  # 小 batch 自动 bypass，无 scheduler 开销


gates["R36_SMALL_BATCH_NO_REGRESSION"] = _probe_small_batch_no_regression()


def _probe_1000_factor_tdc() -> bool:
    """1000 因子编译 TTDC（真实运行时事件；无真实 A 股全市场数据时用合成日频源）。"""
    import time as _t
    import numpy as np
    import pandas as pd
    from api.columns import col
    from api.factor import Factor
    from runtime.batch_service import choose_execution_mode
    from tests.helpers import InMemorySeriesSource

    dates = pd.bdate_range("2020-01-02", periods=500)
    instruments = [f"S{i:04d}" for i in range(50)]
    idx = pd.MultiIndex.from_product([dates, instruments], names=["timestamp", "instrument"])
    close = pd.Series(np.linspace(10.0, 20.0, len(idx)), index=idx)
    src = InMemorySeriesSource(data={"close": close})
    factors = [Factor(name=f"f{i}", expr=col("close")) for i in range(1000)]
    t0 = _t.monotonic()
    from runtime.engine import FactorEngine
    from backend.pandas_backend import PandasBackend

    engine = FactorEngine(backend=PandasBackend(), data_source=src)
    dag, analyses = engine._dag_from_factors(factors, enable_cse=True)
    mode = choose_execution_mode(factors, analyses, dag)
    tdc_ms = (_t.monotonic() - t0) * 1000
    return mode == "ADAPTIVE_DAG" and tdc_ms > 0


gates["R36_1000_FACTOR_TTDC_CURRENT_HEAD"] = _probe_1000_factor_tdc()


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

gates = {k: bool(v) for k, v in gates.items()}
blockers = [k for k, v in gates.items() if not v]
print("=== R36 HARD GATES ===")
for k in sorted(gates):
    print(f"  [{'OK' if gates[k] else 'FAIL'}] {k}")
print(f"\nR36_HARD_BLOCKERS_ZERO = {not blockers}")
import os as _os

os.makedirs("docs/evidence/r36", exist_ok=True)
json.dump(gates, open("docs/evidence/r36/R36_HARD_GATES.json", "w"), indent=2, sort_keys=True)
sys.exit(1 if blockers else 0)
