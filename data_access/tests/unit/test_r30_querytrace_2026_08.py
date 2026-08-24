"""R30-P0-010 / R30-P1-028 单测：QueryTrace + Observability + 版本治理。

纯 Python fake（不依赖真实 store / 数据）。覆盖：
    - QueryTracer 阶段计时与嵌套 enter 压栈/恢复；
    - QueryTrace.to_dict JSON 可序列化；
    - run_traced_read 成功路径返回 trace+result、异常路径不丢 trace；
    - MetricsRegistry histogram 聚合与 reset；
    - prometheus_text 含 name_total 序列；
    - data_access_version_manifest 含 build_sha；assert_version_compat。

独立可跑：``python3 -m pytest tests/unit/test_r30_querytrace_2026_08.py``
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from data_access.r30.query_trace import (
    QueryTrace,
    QueryTracer,
    run_traced_read,
)
from data_access.r30.metrics import (
    MetricsRegistry,
    get_metrics,
    otel_dict,
    prometheus_text,
    reset_metrics,
)
from data_access.r30.versioning import (
    API_VERSION,
    CONTRACT_SCHEMA_VERSION,
    assert_version_compat,
    data_access_version_manifest,
)


# ---- fakes ----

class FakeTable:
    def __init__(self, num_rows: int, col_names: list[str]) -> None:
        self.num_rows = num_rows
        self.column_names = list(col_names or [])
        self.num_columns = len(col_names or [])
        self.nbytes = 12_345


class FakeResult:
    def __init__(
        self,
        num_rows: int = 10,
        col_names: list[str] | None = None,
        cache_status: str | None = None,
    ) -> None:
        col_names = list(col_names or ["a", "b"])
        self.num_rows = num_rows
        self.num_columns = len(col_names)
        self.columns = col_names
        self.table = FakeTable(num_rows, col_names)
        self.snapshot = SimpleNamespace(snapshot_id="snap-1")
        self.stats = SimpleNamespace(rows=num_rows, bytes=12_345, elapsed_ms=1.0, paths=())
        self.lineage = SimpleNamespace(dataset="ds")
        self.cache_status = cache_status
        self.result_bytes = 12_345


class FakeStore:
    def __init__(self) -> None:
        self.prepare_calls = 0
        self.execute_calls = 0
        self.fail_prepare = False
        self.fail_execute = False
        self.num_rows = 10
        self.columns = ["a", "b"]

    def prepare_read(self, dataset, *, columns=None, time_range=None, params=None, **kwargs):
        self.prepare_calls += 1
        if self.fail_prepare:
            raise RuntimeError("prepare boom")
        return SimpleNamespace(
            dataset=dataset,
            resolved_source_snapshot=SimpleNamespace(
                snapshot_id="snap-1", content_digest="digest-1"
            ),
            backend_plan={"backend": "duckdb"},
            physical_scope=(),
        )

    def execute_prepared_read(self, prepared, *, verify_after=True):
        self.execute_calls += 1
        if self.fail_execute:
            raise RuntimeError("execute boom")
        return FakeResult(self.num_rows, self.columns)


@pytest.fixture(autouse=True)
def _clean_tracer_and_metrics():
    from data_access.r30.query_trace import reset_tracer

    reset_tracer()
    reset_metrics()
    yield
    reset_tracer()
    reset_metrics()


# ---- QueryTracer / QueryTrace ----

def test_stage_timing_recorded():
    tracer = QueryTracer()
    with tracer.enter(request_id="r1", dataset="ashare_daily") as g:
        with g.stage("auth"):
            time.sleep(0.005)
        with g.stage("duckdb_execute"):
            time.sleep(0.005)
        trace = tracer.finish()
    assert trace is not None
    assert trace.request_id == "r1"
    assert trace.dataset == "ashare_daily"
    assert "auth" in trace.stage_timings
    assert "duckdb_execute" in trace.stage_timings
    assert trace.stage_timings["auth"] >= 3.0
    assert trace.stage_timings["duckdb_execute"] >= 3.0
    assert trace.elapsed_ms is not None
    assert trace.elapsed_ms >= 10.0


def test_stage_timing_accumulates_on_repeat():
    tracer = QueryTracer()
    with tracer.enter(request_id="r2") as g:
        for _ in range(3):
            with g.stage("resolution"):
                time.sleep(0.001)
        trace = tracer.finish()
    assert trace is not None
    assert trace.stage_timings["resolution"] >= 3.0


def test_nested_enter_pushes_and_restores_stack():
    tracer = QueryTracer()
    assert tracer.current() is None
    with tracer.enter(request_id="outer") as outer:
        assert tracer.current() is not None
        assert tracer.current().request_id == "outer"
        with tracer.enter(request_id="inner") as inner:
            assert tracer.current().request_id == "inner"
            # inner 内 stage 记到 inner trace，不串到 outer
            with inner.stage("prepare"):
                time.sleep(0.001)
            assert tracer.current().stage_timings["prepare"] > 0
        assert tracer.current().request_id == "outer"
    assert tracer.current() is None


def test_query_trace_to_dict_json_serializable():
    trace = QueryTrace(
        request_id="r",
        dataset="d",
        columns=["a", "b"],
        started_monotonic=123.0,
        elapsed_ms=4.5,
    )
    trace.add_stage("prepare", 1.5)
    trace.add_stage("duckdb_execute", 2.5)
    d = trace.to_dict()
    # JSON 可序列化
    json.dumps(d)
    assert d["request_id"] == "r"
    assert d["dataset"] == "d"
    assert d["columns"] == ["a", "b"]
    assert d["stage_timings"]["prepare"] == 1.5
    assert d["elapsed_ms"] == 4.5
    assert d["error"] is None


# ---- run_traced_read ----

def test_run_traced_read_success_returns_trace_and_result():
    store = FakeStore()
    trace, result = run_traced_read(
        store, "ashare_daily", columns=["a"], time_range=("2026-01-01", "2026-01-02")
    )
    assert store.prepare_calls == 1
    assert store.execute_calls == 1
    assert trace.dataset == "ashare_daily"
    assert trace.rows == 10
    assert trace.columns == ["a", "b"]
    assert trace.backend == "duckdb"
    assert trace.source_snapshot == "snap-1"
    assert "prepare" in trace.stage_timings
    assert "duckdb_execute" in trace.stage_timings
    assert trace.elapsed_ms is not None
    assert result is not None
    assert result.num_rows == 10


def test_run_traced_read_execute_exception_preserves_trace():
    store = FakeStore()
    store.fail_execute = True
    trace, result = run_traced_read(store, "ds")
    assert result is None
    assert trace.error is not None
    assert "execute boom" in trace.error
    assert trace.elapsed_ms is not None
    assert "prepare" in trace.stage_timings
    assert "duckdb_execute" in trace.stage_timings


def test_run_traced_read_prepare_exception_preserves_trace():
    store = FakeStore()
    store.fail_prepare = True
    trace, result = run_traced_read(store, "ds")
    assert result is None
    assert trace.error is not None
    assert "prepare boom" in trace.error
    assert "prepare" in trace.stage_timings


def test_run_traced_read_resolution_cache_hit():
    from data_access.runtime.read_session_context import (
        reset_resolution_cache,
        set_resolution_cache,
    )

    cache = {("ds", "fp", "time_range", None): (("p",), ())}
    token = set_resolution_cache(cache)
    try:
        store = FakeStore()
        trace, _ = run_traced_read(store, "ds")
        assert trace.resolution_cache_hit is True
    finally:
        reset_resolution_cache(token)


def test_run_traced_read_records_metrics():
    store = FakeStore()
    run_traced_read(store, "ds")
    reg = get_metrics()
    assert reg.count("query_duration_ms", dataset="ds") >= 1
    assert reg.count("rows") == 1


# ---- Metrics ----

def test_metrics_histogram_aggregation_and_reset():
    reg = MetricsRegistry()
    reg.record("query_duration_ms", 1.5, dataset="d1")
    reg.record("query_duration_ms", 2.5, dataset="d1")
    reg.record("query_duration_ms", 10.0, dataset="d2")
    snap = reg.snapshot()
    d1 = snap["query_duration_ms"][(("dataset", "d1"),)]
    assert d1["count"] == 2
    assert d1["sum"] == pytest.approx(4.0)
    assert d1["min"] == pytest.approx(1.5)
    assert d1["max"] == pytest.approx(2.5)
    assert d1["last"] == pytest.approx(2.5)
    assert reg.count("query_duration_ms", dataset="d1") == 2
    assert reg.count("query_duration_ms", dataset="d2") == 1
    reg.reset()
    assert reg.snapshot() == {}
    assert reg.count("query_duration_ms", dataset="d1") == 0


def test_prometheus_text_contains_name_total():
    reg = MetricsRegistry()
    reg.record("query_duration_ms", 5.0, dataset="d1")
    text = prometheus_text(reg)
    assert "query_duration_ms_total" in text
    assert "query_duration_ms_sum" in text
    assert "query_duration_ms_count" in text
    assert "query_duration_ms_bucket" in text
    assert 'dataset="d1"' in text
    assert text.endswith("\n")


def test_otel_dict_returns_data_points():
    reg = MetricsRegistry()
    reg.record("rows", 42, dataset="d1")
    points = otel_dict(reg)
    assert len(points) == 1
    point = points[0]
    assert point["name"] == "rows"
    assert point["attributes"] == {"dataset": "d1"}
    assert point["value"] == 42
    assert point["count"] == 1
    assert "timestamp_ns" in point


def test_metrics_errors_by_type():
    reg = MetricsRegistry()
    store = FakeStore()
    store.fail_execute = True
    run_traced_read(store, "ds")  # 记录到全局 registry
    reg.record("errors_by_type", 1, type="RuntimeError")
    snap = reg.snapshot()
    assert snap["errors_by_type"][(("type", "RuntimeError"),)]["count"] == 1


# ---- versioning ----

def test_version_manifest_has_build_sha():
    manifest = data_access_version_manifest()
    assert manifest["build_sha"] is not None
    assert isinstance(manifest["build_sha"], str)
    assert len(manifest["build_sha"]) > 0
    assert manifest["api"] == API_VERSION
    assert manifest["contract_schema"] == CONTRACT_SCHEMA_VERSION
    assert "package_version" in manifest
    assert "storage_format" in manifest
    assert manifest["identity_schema"] == "2"


def test_assert_version_compat():
    manifest = data_access_version_manifest()
    required = {
        "api": manifest["api"],
        "contract_schema": manifest["contract_schema"],
    }
    assert assert_version_compat(required) == []
    assert assert_version_compat({"api": "9.9.9"}) == ["api"]
    assert assert_version_compat({"api": "9.9.9", "storage_format": "999"}) == [
        "api",
        "storage_format",
    ]
    assert assert_version_compat({"identity_schema": manifest["identity_schema"]}) == []
    assert assert_version_compat({"identity_schema": "1"}) == ["identity_schema"]
