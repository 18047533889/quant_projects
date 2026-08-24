# -*- coding: utf-8 -*-
"""R39-P1-PERF-081/082 + §1.2 — structured perf telemetry.

Covers:
  (a) PerfCounters: incr/add/snapshot, fixed slots, KeyError on undeclared names,
      snapshot never mutates counters.
  (b) ProductionExecutionCertificate built by choose_plan_route; O(1) validate
      passes on a matching backend event and fails on fallback / backend mismatch.
      HybridExecutor records the runtime backend event and validates O(1).
  (c) PerformanceRunSummary.from_counters maps counter names correctly.
  (d) render_r39_json writes a parseable JSON with required keys + environment.
  (e) capture_environment returns expected keys.
"""
from __future__ import annotations

import json

import pytest

from factor_engine.planner.logical_plan import PlanNode


@pytest.fixture(scope="module")
def loaded():
    from factor_engine.cleaned_operators import load_all

    load_all()
    yield


# ---------------------------------------------------------------------------
# (a) PerfCounters
# ---------------------------------------------------------------------------


def test_perf_counters_incr_add_snapshot_and_declared_slots():
    from factor_engine.runtime.perf_counters import COUNTER_SLOTS, PerfCounters

    c = PerfCounters()
    c.incr("future_count")
    c.incr("future_count", delta=2)
    c.add("factor_count", 5)
    c.add("legacy_union_prefetch_count", 1)

    snap = c.snapshot()
    assert snap["future_count"] == 3
    assert snap["factor_count"] == 5
    assert snap["legacy_union_prefetch_count"] == 1

    # every §1.2 / §28 KPI is a declared slot
    for name in (
        "future_count",
        "factor_count",
        "physical_query_count",
        "parquet_file_open_count",
        "parquet_file_write_count",
        "fsync_count",
        "sqlite_transaction_count",
        "full_factor_rescan_count",
        "watchdog_thread_created_count",
        "legacy_union_prefetch_count",
        "representation_transition_count",
        "matrix_join_count",
        "micro_batch_task_count",
    ):
        assert name in COUNTER_SLOTS
        assert name in snap


def test_perf_counters_undeclared_name_raises_keyerror():
    from factor_engine.runtime.perf_counters import PerfCounters

    c = PerfCounters()
    with pytest.raises(KeyError):
        c.incr("not_a_real_counter")
    with pytest.raises(KeyError):
        c.add("also_not_real", 1)


def test_perf_counters_snapshot_does_not_mutate():
    from factor_engine.runtime.perf_counters import PerfCounters

    c = PerfCounters()
    c.add("factor_count", 7)
    before = dict(c.snapshot())
    c.snapshot()
    assert c.snapshot() == before
    assert c.snapshot()["factor_count"] == 7


def test_perf_counters_global_singleton_and_reset():
    from factor_engine.runtime.perf_counters import get_global_counters, reset_global_counters

    reset_global_counters()
    g1 = get_global_counters()
    g1.incr("future_count")
    assert get_global_counters() is g1
    assert g1.snapshot()["future_count"] == 1
    reset_global_counters()
    assert get_global_counters() is not g1
    assert get_global_counters().snapshot()["future_count"] == 0


# ---------------------------------------------------------------------------
# (b) ProductionExecutionCertificate
# ---------------------------------------------------------------------------


def _simple_route_ctx():
    import pandas as pd

    from factor_engine.backend.context import ExecutionContext
    from factor_engine.runtime.perf_config import PerfConfig
    from tests.helpers import InMemorySeriesSource

    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=5), ["A"]],
        names=["timestamp", "instrument"],
    )
    source = InMemorySeriesSource(
        data={"close": pd.Series([1.0] * len(idx), index=idx)}
    )
    ctx = ExecutionContext(
        data_source=source,
        run_mode="research",
        perf=PerfConfig(operator_backend="auto"),
    )
    col = PlanNode(op="column", inputs=[], attrs={"name": "close"})
    plan = PlanNode(op="ts_mean", inputs=[col], attrs={"window": 3})
    return ctx, plan


def test_certificate_built_by_choose_plan_route_and_o1_validate(loaded):
    from factor_engine.backend.plan_cost_router import choose_plan_route
    from factor_engine.runtime.production_execution_certificate import normalize_backend, validate

    ctx, plan = _simple_route_ctx()
    route = choose_plan_route(plan, ctx)
    assert route.certificate is not None, (
        "choose_plan_route must attach a ProductionExecutionCertificate"
    )
    cert = route.certificate
    assert cert.bound_ops  # at least ts_mean
    assert cert.structural_hash
    assert cert.output_shape_hash
    # eligibility is executor-normalized (polars_long → polars)
    assert normalize_backend(route.backend) in cert.backend_eligibility

    # matching runtime backend event (no fallback) passes; raw router name is
    # normalized by validate, so either spelling matches an eligible backend
    matching = {"backend": route.backend, "execution_kind": "thread", "no_fallback": True}
    assert validate(cert, matching) is True

    # a runtime backend never eligible fails
    mismatch = {"backend": "clickhouse_sql", "execution_kind": "thread", "no_fallback": True}
    assert validate(cert, mismatch) is False

    # explicit fallback fails unless the fallback backend is allowed
    fallback = {"backend": route.backend, "execution_kind": "thread", "no_fallback": False}
    assert validate(cert, fallback) is False
    assert validate(cert, fallback, allowed_fallbacks=[route.backend]) is True

    # missing / None certificate fails closed
    assert validate(None, matching) is False
    assert validate(cert, None) is False


def test_certificate_hash_tamper_fails_closed():
    from dataclasses import replace

    from factor_engine.runtime.production_execution_certificate import ProductionExecutionCertificate

    cert = ProductionExecutionCertificate.build(
        structural_hash="abc",
        bound_ops=["ts_mean"],
        backend_eligibility=["pandas_numpy"],
        output_shape_hash="def",
    )
    assert cert.validate({"backend": "pandas_numpy", "no_fallback": True}) is True
    # a stale/tampered certificate hash must fail closed (O(1) integrity check)
    tampered = replace(cert, certificate_hash="deadbeef")
    assert tampered.validate({"backend": "pandas_numpy", "no_fallback": True}) is False


def test_hybrid_executor_records_event_and_validates_o1(loaded):
    from factor_engine.backend.plan_cost_router import choose_plan_route
    from factor_engine.runtime.hybrid_executor import HybridExecutor
    from factor_engine.runtime.resource_broker import ResourceBroker

    ctx, plan = _simple_route_ctx()
    route = choose_plan_route(plan, ctx)
    assert route.certificate is not None

    executor = HybridExecutor(
        broker=ResourceBroker(hard_memory_limit=8 * 1024**3, cpu_slots=4)
    )
    try:
        # thread-classified backend avoids spawning a process pool
        event = executor.record_runtime_backend_event(route.backend, "thread", no_fallback=True)
        assert event["backend"] == route.backend
        assert executor.validate_certificate(route.certificate) is True
        assert executor.summary()["last_runtime_backend_event"] == event
    finally:
        executor.shutdown(wait=True)


def test_hybrid_executor_submit_records_event(loaded):
    from factor_engine.runtime.hybrid_executor import HybridExecutor
    from factor_engine.runtime.resource_broker import ResourceBroker

    executor = HybridExecutor(
        broker=ResourceBroker(hard_memory_limit=8 * 1024**3, cpu_slots=4)
    )
    try:
        fut = executor.submit("duckdb_sql", lambda: 42)
        assert fut.result(timeout=30) == 42
        event = executor.summary()["last_runtime_backend_event"]
        assert event is not None
        assert event["backend"] == "duckdb_sql"
        assert event["no_fallback"] is True
    finally:
        executor.shutdown(wait=True)


# ---------------------------------------------------------------------------
# (c) PerformanceRunSummary
# ---------------------------------------------------------------------------


def test_run_summary_from_counters_maps_fields():
    from factor_engine.runtime.perf_counters import PerfCounters
    from factor_engine.runtime.performance_run_summary import PerformanceRunSummary

    c = PerfCounters()
    c.add("factor_count", 100)
    c.add("future_count", 240)
    c.add("physical_query_count", 12)
    c.add("parquet_file_open_count", 30)
    c.add("parquet_file_write_count", 25)
    c.add("fsync_count", 40)
    c.add("sqlite_transaction_count", 3)
    c.add("full_factor_rescan_count", 0)
    c.add("watchdog_thread_created_count", 1)

    timing = {
        "compile_ms": 12.5,
        "planning_ms": 3.2,
        "scan_ms": 200.0,
        "compute_ms": 800.0,
        "conversion_ms": 50.0,
        "writer_wait_ms": 4.0,
        "serialize_ms": 60.0,
        "fsync_ms": 15.0,
        "catalog_commit_ms": 5.0,
        "total_ttdc_ms": 1150.0,
        "scan_bytes": 500_000_000,
        "minimum_required_scan_bytes": 400_000_000,
        "conversion_bytes": 100_000_000,
        "output_bytes": 50_000_000,
        "write_bytes": 52_000_000,
        "rewrite_bytes": 0,
        "metadata_scan_bytes": 1_000_000,
    }
    s = PerformanceRunSummary.from_counters(c, timing)
    assert s.factor_count == 100
    assert s.future_count == 240
    assert s.physical_query_count == 12
    assert s.parquet_file_open_count == 30
    assert s.parquet_file_write_count == 25
    assert s.fsync_count == 40
    assert s.sqlite_transaction_count == 3
    assert s.full_factor_rescan_count == 0
    assert s.watchdog_thread_created_count == 1
    assert s.compile_ms == 12.5
    assert s.total_ttdc_ms == 1150.0
    assert s.scan_bytes == 500_000_000
    assert s.minimum_required_scan_bytes == 400_000_000
    assert s.conversion_bytes == 100_000_000
    assert s.output_bytes == 50_000_000
    assert s.write_bytes == 52_000_000
    assert s.rewrite_bytes == 0
    assert s.metadata_scan_bytes == 1_000_000

    amp = s.amplification()
    assert amp["scan_amplification"] == pytest.approx(1.25)
    assert amp["conversion_amplification"] == pytest.approx(2.0)
    assert amp["write_amplification"] == pytest.approx(1.04)
    assert amp["rewrite_amplification"] == 0.0


def test_run_summary_from_counters_defaults_missing():
    from factor_engine.runtime.performance_run_summary import PerformanceRunSummary

    s = PerformanceRunSummary.from_counters(None, {})
    assert s.factor_count == 0
    assert s.total_ttdc_ms == 0.0
    assert s.scan_bytes == 0


# ---------------------------------------------------------------------------
# (d) render_r39_json
# ---------------------------------------------------------------------------


def test_render_r39_json_writes_parseable_payload(tmp_path):
    from factor_engine.runtime.perf_counters import PerfCounters
    from factor_engine.runtime.performance_run_summary import PerformanceRunSummary, render_r39_json

    c = PerfCounters()
    c.add("factor_count", 10)
    c.add("future_count", 22)
    s = PerformanceRunSummary.from_counters(
        c,
        {"compile_ms": 1.0, "planning_ms": 0.5, "scan_ms": 10.0, "compute_ms": 20.0,
         "conversion_ms": 2.0, "writer_wait_ms": 0.1, "serialize_ms": 1.0,
         "fsync_ms": 0.5, "catalog_commit_ms": 0.2, "total_ttdc_ms": 36.0,
         "scan_bytes": 1000, "minimum_required_scan_bytes": 800,
         "conversion_bytes": 200, "output_bytes": 100, "write_bytes": 110,
         "rewrite_bytes": 0, "metadata_scan_bytes": 5},
    )
    out = tmp_path / "r39_perf.json"
    path = render_r39_json(s, out)
    assert path == str(out)
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))

    for key in (
        "factor_count",
        "compile_ms",
        "planning_ms",
        "scan_ms",
        "compute_ms",
        "conversion_ms",
        "writer_wait_ms",
        "serialize_ms",
        "fsync_ms",
        "catalog_commit_ms",
        "total_ttdc_ms",
        "scan_bytes",
        "minimum_required_scan_bytes",
        "conversion_bytes",
        "output_bytes",
        "write_bytes",
        "rewrite_bytes",
        "metadata_scan_bytes",
        "future_count",
        "physical_query_count",
        "parquet_file_open_count",
        "parquet_file_write_count",
        "fsync_count",
        "sqlite_transaction_count",
        "full_factor_rescan_count",
        "watchdog_thread_created_count",
        "amplification",
        "environment",
    ):
        assert key in payload, f"missing required key: {key}"

    env = payload["environment"]
    assert "git_sha" in env
    assert "build_identity" in env
    assert "python_version" in env
    assert "numpy_version" in env
    assert "pandas_version" in env
    assert "polars_version" in env
    assert "duckdb_version" in env
    assert "pyarrow_version" in env
    assert "cpu" in env
    assert "memory" in env
    assert "storage_class" in env
    assert payload["factor_count"] == 10
    assert payload["future_count"] == 22


# ---------------------------------------------------------------------------
# (e) capture_environment
# ---------------------------------------------------------------------------


def test_capture_environment_expected_keys():
    from factor_engine.runtime.performance_run_summary import capture_environment

    env = capture_environment()
    for key in (
        "git_sha",
        "build_identity",
        "python_version",
        "numpy_version",
        "pandas_version",
        "polars_version",
        "duckdb_version",
        "pyarrow_version",
        "cpu",
        "memory",
        "storage_class",
        "platform",
        "source_snapshot",
    ):
        assert key in env, f"missing environment key: {key}"
    assert env["python_version"]  # non-empty
    assert isinstance(env["cpu"], dict)
    assert isinstance(env["memory"], dict)
    assert env["source_snapshot"] is None

    env2 = capture_environment(source_snapshot={"generation": "g1"})
    assert env2["source_snapshot"] == {"generation": "g1"}
