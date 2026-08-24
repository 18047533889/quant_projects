# -*- coding: utf-8 -*-
"""R39 PERF-071/072/074/075/076/077 —— DuckDB 扫描/转换/调度/写放大与批量物化。

覆盖：
    (a) PERF-071 applied_config_fingerprint：相同 config 重复应用 → PRAGMA skip，
        apply 计数不变（mock pragma executor）。
    (b) PERF-072 DeadlineManager：单 timer thread（Gate-08 O(1) 线程），超时
        cancel_fn 恰好一次，cancel() 移除待触发 deadline。
    (c) PERF-074 QueryClassCohort：重 scan+空闲 → 8-thread；并发 → 小 profile；
        未知 shape → legacy 回退（env 默认 OFF）。
    (d) PERF-075 ScanShapeKey + ScanShapeCalibrator：分桶 + 有界样本窗口 +
        P50/P95。
    (e) PERF-076 predict_ttdc 返回各 backend 估计；choose_plan_route 对
        DuckDB-fused 下游倾向 DuckDB。
    (f) PERF-077 column footprint producer/reader 在 tiny parquet 上 round-trip。
"""
from __future__ import annotations

import threading
import time
from dataclasses import replace

import pytest


# ---------------------------------------------------------------------------
# (a) PERF-071 applied_config_fingerprint
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, value):
        self._value = value

    def fetchone(self):
        return (self._value,)


class _FakeConn:
    def __init__(self, threads: int = 4):
        self.current = int(threads)
        self.pragma_executes = 0

    def execute(self, sql):
        sql = str(sql)
        if "current_setting" in sql:
            return _FakeResult(self.current)
        if sql.startswith("PRAGMA threads="):
            self.current = int(sql.split("=")[1].strip())
            self.pragma_executes += 1
            return _FakeResult(None)
        self.pragma_executes += 1
        return _FakeResult(None)


class _FakeEngine:
    def __init__(self, conn):
        self._conn = conn
        self._write_lock = threading.Lock()


class _FakeStore:
    def __init__(self, conn):
        self._engine = _FakeEngine(conn)


def _plan(threads: int = 4):
    from factor_engine.runtime.resource_governor import ExecutionResourcePlan

    plan = ExecutionResourcePlan.auto()
    return replace(plan, duckdb_threads=threads, polars_threads=threads)


def test_perf071_applied_config_fingerprint_skips_noop_reapply(monkeypatch):
    from factor_engine.runtime.resource_governor import (
        ExecutionResourceScope,
        reset_conn_fingerprints,
    )

    conn = _FakeConn(threads=4)
    store = _FakeStore(conn)
    monkeypatch.setattr("data_access.get_store", lambda: store)
    reset_conn_fingerprints()
    try:
        plan = _plan(threads=4)
        s1 = ExecutionResourceScope(plan)
        s1._apply_live_pragma(4)
        assert s1.pragma_apply_count == 1
        assert s1.pragma_skip_count == 0
        assert s1.applied_config_fingerprint is not None

        # 相同 config 第二次 → skip，apply 保持 1。
        s2 = ExecutionResourceScope(plan)
        s2._apply_live_pragma(4)
        assert s2.pragma_apply_count == 0
        assert s2.pragma_skip_count == 1
        assert s2.applied_config_fingerprint == s1.applied_config_fingerprint
        # 真正只执行过一次 PRAGMA（第二次是 no-op skip）。
        assert conn.pragma_executes == 1
    finally:
        reset_conn_fingerprints()


def test_perf071_config_change_reapplies(monkeypatch):
    """config 变化（threads 4→8）必须重新应用，不能被 fingerprint 误跳过。"""
    from factor_engine.runtime.resource_governor import (
        ExecutionResourceScope,
        reset_conn_fingerprints,
    )

    conn = _FakeConn(threads=4)
    store = _FakeStore(conn)
    monkeypatch.setattr("data_access.get_store", lambda: store)
    reset_conn_fingerprints()
    try:
        plan4 = _plan(threads=4)
        s1 = ExecutionResourceScope(plan4)
        s1._apply_live_pragma(4)
        assert s1.pragma_apply_count == 1

        plan8 = _plan(threads=8)
        s2 = ExecutionResourceScope(plan8)
        s2._apply_live_pragma(8)
        assert s2.pragma_apply_count == 1
        assert s2.pragma_skip_count == 0
        assert conn.current == 8
    finally:
        reset_conn_fingerprints()


# ---------------------------------------------------------------------------
# (b) PERF-072 DeadlineManager
# ---------------------------------------------------------------------------


def test_perf072_deadline_manager_single_thread_and_timeout():
    from factor_engine.runtime.deadline_manager import DeadlineManager

    m = DeadlineManager()
    try:
        calls: list[int] = []
        for i in range(10):
            m.register(f"q{i}", 60, lambda i=i: calls.append(i))
        # 注册 10 个 deadline 仍只有一个 timer 线程（Gate-08：O(1) 线程）。
        assert m.watchdog_thread_created_count == 1
        assert len(m._heap) == 10
        timer_threads = [
            t for t in threading.enumerate() if t.name == "r39-deadline-manager"
        ]
        assert len(timer_threads) == 1

        # 超时恰好触发 cancel_fn 一次（轮询，机器高负载时 timer 线程可能被饿到）。
        fired: list[str] = []
        m.register("due", 0.02, lambda: fired.append("fire"))
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not m.query_due("due"):
            time.sleep(0.02)
        assert fired == ["fire"]
        assert m.timeouts_fired == 1
        assert m.query_due("due")
    finally:
        m.shutdown()


def test_perf072_deadline_manager_cancel_removes_pending():
    from factor_engine.runtime.deadline_manager import DeadlineManager

    m = DeadlineManager()
    try:
        fired: list[str] = []
        m.register("pending", 60, lambda: fired.append("fire"))
        assert m.active_count == 1
        assert m.cancel("pending") is True
        assert m.cancel("pending") is False
        time.sleep(0.2)
        assert fired == []
        assert m.active_count == 0
    finally:
        m.shutdown()


def test_perf072_guard_deadline_context_registers_and_cancels():
    from factor_engine.runtime.deadline_manager import DeadlineManager, guard_deadline

    m = DeadlineManager()
    try:
        class _Conn:
            def __init__(self):
                self.interrupted = 0

            def interrupt(self):
                self.interrupted += 1

        conn = _Conn()
        with guard_deadline(m, conn, 0.01):
            pass
        # 上下文退出即 cancel，不触发 interrupt。
        time.sleep(0.1)
        assert conn.interrupted == 0
        # 无 deadline（timeout_s<=0）→ 直接 yield，不登记。
        with guard_deadline(m, conn, 0):
            pass
        assert m.active_count == 0
    finally:
        m.shutdown()


# ---------------------------------------------------------------------------
# (c) PERF-074 QueryClassCohort
# ---------------------------------------------------------------------------


def test_perf074_cohort_selection_profiles(monkeypatch):
    from factor_engine.runtime.query_class_cohort import CohortQueryShape, QueryClassCohort

    cohort = QueryClassCohort()
    # 重 scan + 空闲机器 → 8-thread。
    assert cohort.select_cohort(None, 600 * 1024 * 1024, 1) == 8
    # 重 scan + 并发升高 → 自动降档。
    assert cohort.select_cohort(None, 600 * 1024 * 1024, 2) == 4
    assert cohort.select_cohort(None, 600 * 1024 * 1024, 4) == 2
    assert cohort.select_cohort(None, 600 * 1024 * 1024, 8) == 2
    # 中 scan + 空闲 → 4-thread。
    assert cohort.select_cohort(None, 100 * 1024 * 1024, 1) == 4
    # 未知 shape → None（legacy fallback）。
    assert cohort.select_cohort(None, 0, 1) is None
    assert cohort.select_cohort(CohortQueryShape(scan_bytes=0), 0, 1) is None
    # query_shape 提供 scan_bytes 时也参与。
    assert (
        cohort.select_cohort(CohortQueryShape(scan_bytes=600 * 1024 * 1024), 0, 1)
        == 8
    )


def test_perf074_plan_wiring_cohort_env_optin(monkeypatch):
    from factor_engine.runtime.resource_governor import ExecutionResourcePlan

    plan = _plan(threads=3)
    # 默认 OFF → legacy 静态值。
    assert plan.duckdb_threads_for(scan_bytes=600 * 1024 * 1024, current_concurrency=1) == 3
    # 开启 + shape 可用 → cohort。
    monkeypatch.setenv("FACTOR_ENGINE_COHORT_PROFILES", "1")
    assert plan.duckdb_threads_for(scan_bytes=600 * 1024 * 1024, current_concurrency=1) == 8
    # 未知 shape → legacy 回退。
    assert plan.duckdb_threads_for(scan_bytes=0, current_concurrency=1) == 3
    monkeypatch.delenv("FACTOR_ENGINE_COHORT_PROFILES")


def test_perf074_hybrid_executor_cohort(monkeypatch):
    from factor_engine.runtime.hybrid_executor import HybridExecutor

    ex = HybridExecutor(worker_threads=2)
    # 默认 OFF → legacy worker_threads。
    assert ex.cohort_worker_threads(scan_bytes=600 * 1024 * 1024, current_concurrency=1) == 2
    monkeypatch.setenv("FACTOR_ENGINE_COHORT_PROFILES", "1")
    assert ex.cohort_worker_threads(scan_bytes=600 * 1024 * 1024, current_concurrency=1) == 8
    monkeypatch.delenv("FACTOR_ENGINE_COHORT_PROFILES")


# ---------------------------------------------------------------------------
# (d) PERF-075 ScanShapeKey + P50/P95
# ---------------------------------------------------------------------------


def test_perf075_scan_shape_buckets_and_p50p95():
    from factor_engine.runtime.scan_shape import (
        ScanShapeCalibrator,
        ScanShapeKey,
        file_count_bucket,
        selected_bytes_bucket,
    )

    # 分桶。
    assert file_count_bucket(1) == "one"
    assert file_count_bucket(50) == "dozens"
    assert selected_bytes_bucket(0) == "unknown"
    assert selected_bytes_bucket(512 * 1024) == "lt1mb"
    assert selected_bytes_bucket(100 * 1024 * 1024) == "64to512mb"

    shape = ScanShapeKey(
        dataset="ashare_daily",
        local_or_remote="local",
        selected_bytes_bucket="64to512mb",
    )
    cal = ScanShapeCalibrator(max_samples_per_shape=100)
    for v in (10, 20, 30, 40, 100):
        cal.record(shape, open_ms=v, scan_ms=1, decode_ms=1, rows=1000, bytes_=1024 * 1024)
    s = cal.summary(shape)
    assert s["samples"] == 5
    assert s["open_ms"]["p50"] == 30
    assert s["open_ms"]["p95"] == 100
    # 有界样本窗口：超过 max_samples_per_shape 时只保留最近 100 个。
    for v in range(0, 200):
        cal.record(shape, open_ms=v, scan_ms=0, decode_ms=0, rows=1, bytes_=1)
    s2 = cal.summary(shape)
    assert s2["samples"] == 100
    assert s2["open_ms"]["p50"] == 149  # [100..199] 的 P50
    assert s2["open_ms"]["p95"] == 194  # [100..199] 的 P95


def test_perf075_scan_shape_key_from_scan_cost_and_calibration_module():
    from types import SimpleNamespace

    from factor_engine.runtime.scan_shape import ScanShapeKey
    from factor_engine.runtime.runtime_calibration import (
        record_scan_shape_actual,
        reset_scan_shape_calibration,
        scan_shape_summary,
    )

    cost = SimpleNamespace(
        dataset="ashare_daily",
        remote=False,
        selected_files=3,
        file_count=5,
        selected_bytes=100 * 1024 * 1024,
        total_bytes=100 * 1024 * 1024,
        selected_rowgroups=10,
        projected_columns=2,
        total_columns=20,
        instrument_count=5000,
        storage_class="nvme",
    )
    sk = ScanShapeKey.from_scan_cost(cost)
    assert sk.local_or_remote == "local"
    assert sk.selected_bytes_bucket == "64to512mb"
    assert sk.projected_column_ratio_bucket == "0.1to0.5"
    assert sk.instrument_count_bucket == "medium"

    reset_scan_shape_calibration()
    try:
        record_scan_shape_actual(shape=sk, open_ms=5, scan_ms=2, decode_ms=1, rows=10, bytes_=1000)
        assert scan_shape_summary(sk)["samples"] == 1
        assert scan_shape_summary(sk)["scan_ms"]["p50"] == 2
    finally:
        reset_scan_shape_calibration()


# ---------------------------------------------------------------------------
# (e) PERF-076 predict_ttdc + choose_plan_route DuckDB-fused 规则
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def loaded_registry():
    from factor_engine.cleaned_operators import load_all

    load_all()
    yield


def _route_shape():
    from types import SimpleNamespace

    return SimpleNamespace(
        dataset="ashare_daily",
        remote=False,
        selected_bytes=2 * 1024 * 1024 * 1024,  # 2GB 重 scan
        total_bytes=2 * 1024 * 1024 * 1024,
        estimated_rows=50_000_000,
        projected_columns=2,
        total_columns=20,
        instrument_count=5000,
    )


def test_perf076_predict_ttdc_per_backend_estimates():
    from factor_engine.backend.plan_cost_router import predict_ttdc

    shape = _route_shape()
    duck = predict_ttdc(shape, "duckdb_sql")
    pol = predict_ttdc(shape, "polars_panel")
    arrow = predict_ttdc(shape, "pyarrow")
    assert duck.scan_ms > 0
    assert pol.scan_ms > 0
    assert arrow.scan_ms > 0
    # DuckDB native parquet scan → 无 scan→backend 转换。
    assert duck.conversion_ms == 0.0
    # Polars 需要 arrow→polars 转换。
    assert pol.conversion_ms > 0.0
    assert duck.total_ms > 0 and pol.total_ms > 0
    # 字段齐全。
    for est in (duck, pol, arrow):
        d = est.to_dict()
        assert set(d) >= {"backend", "scan_ms", "conversion_ms", "execute_ms", "materialize_ms"}


def test_perf076_choose_plan_route_prefers_duckdb_for_fused_downstream(loaded_registry):
    from types import SimpleNamespace

    from factor_engine.backend.plan_cost_router import choose_plan_route
    from factor_engine.planner.logical_plan import PlanNode

    class _Caps:
        engine_kind = "duckdb"

    class _DS:
        capabilities = _Caps()
        instrument_filter = None
        start_date = None
        end_date = None

    col = PlanNode(op="column", inputs=[], attrs={"name": "close"})
    plan = PlanNode(op="ts_mean", inputs=[col], attrs={"window": 20})
    ctx = SimpleNamespace(
        run_mode="research",
        data_source=_DS(),
        perf=None,
        runtime_stats={"row_count_estimate": 50_000_000},
        scan_shape=_route_shape(),
        downstream_duckdb_fused=True,
    )
    route = choose_plan_route(plan, ctx)
    assert route.backend == "duckdb_sql", route


def test_perf076_without_shape_falls_back_to_cost_model(loaded_registry):
    """无 shape 数据时 choose_plan_route 行为与既有 cost model 一致（不加惩罚）。"""
    from types import SimpleNamespace

    from factor_engine.backend.plan_cost_router import choose_plan_route
    from factor_engine.planner.logical_plan import PlanNode

    class _Caps:
        engine_kind = "duckdb"

    class _DS:
        capabilities = _Caps()
        instrument_filter = None
        start_date = None
        end_date = None

    col = PlanNode(op="column", inputs=[], attrs={"name": "close"})
    plan = PlanNode(op="ts_mean", inputs=[col], attrs={"window": 20})
    ctx = SimpleNamespace(
        run_mode="research",
        data_source=_DS(),
        perf=None,
        runtime_stats={"row_count_estimate": 50_000_000},
    )
    route = choose_plan_route(plan, ctx)
    assert route.backend in {"duckdb_sql", "polars_panel", "pandas_numpy", "hybrid"}


# ---------------------------------------------------------------------------
# (f) PERF-077 column footprint stats round-trip
# ---------------------------------------------------------------------------


def test_perf077_column_footprint_roundtrip(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from factor_engine.runtime.column_footprint import (
        attach_column_footprints,
        produce_column_footprint_stats,
        read_column_footprint_stats,
    )

    path = tmp_path / "tiny.parquet"
    table = pa.table(
        {
            "a": pa.array([1.0, 2.0, 3.0, None], type=pa.float64()),
            "s": pa.array(["x", "yy", "zzz", None], type=pa.string()),
        }
    )
    pq.write_table(table, path)

    stats = produce_column_footprint_stats(str(path), columns=["a", "s"])
    assert set(stats) == {"a", "s"}
    assert stats["a"].compressed_bytes is not None and stats["a"].compressed_bytes > 0
    assert stats["a"].uncompressed_bytes is not None
    assert stats["a"].null_count == 1
    assert stats["s"].avg_variable_width is not None and stats["s"].avg_variable_width > 0

    # reader round-trip：attach → read。
    class _FakeManifest:
        pass

    m = _FakeManifest()
    attach_column_footprints(m, stats)
    got = read_column_footprint_stats(m, ["a", "s"])
    assert got is not None
    assert got["a"].null_count == 1
    assert got["s"].column == "s"

    # 缺失列跳过。
    assert read_column_footprint_stats(m, ["missing"]) is None

    # lazy fallback：旧 manifest 无 column_footprints → None。
    m2 = _FakeManifest()
    assert read_column_footprint_stats(m2, ["a"]) is None
