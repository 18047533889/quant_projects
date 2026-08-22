"""R30-P0-001 —— Benchmark Suite gate 测试。

只用 tiny fixture（<30s）：fixtures 能建、一个 workload 能跑并产生 report 指标、
compare/gate 结构正确、benchmark_env 有版本字段、write_evidence 写出三个文件。
"""
from __future__ import annotations

from data_access.benchmarks import benchmark_factor_batch as bf
from data_access.benchmarks import benchmark_local_daily as b01
from data_access.benchmarks.fixtures import tiny_fixture
from data_access.benchmarks.report import (
    BenchmarkReport,
    benchmark_env,
    compare,
)


def test_fixture_builds_tiny(tmp_path):
    """tiny_fixture 能建出 store + 注册数据集 + 落盘 + fresh manifest。"""
    store, paths = tiny_fixture(tmp_path)
    names = store._registry.names()
    assert "ashare_stock_daily" in names
    assert "ashare_stock_income" in names
    assert "ashare_universe" in names
    assert "ashare_stock_minute" in names
    assert list(paths["ashare_stock_daily"].glob("part-*.parquet"))
    mv = store.manifest_version("ashare_stock_daily")
    assert mv.get("has_manifest") is True
    assert mv.get("fresh") is True


def test_fixture_deterministic_rebuild(tmp_path):
    """同一 seed 重建 → 同一数据。"""
    store1, _ = tiny_fixture(tmp_path / "a")
    store2, _ = tiny_fixture(tmp_path / "b")
    t1 = store1.read("ashare_stock_daily", columns=["Close"]).to_arrow()
    t2 = store2.read("ashare_stock_daily", columns=["Close"]).to_arrow()
    assert t1.to_pylist() == t2.to_pylist()
    assert t1.num_rows > 0


def test_b01_workload_produces_report(tmp_path):
    """一个固定 workload（B01）能跑并产生 report 指标。"""
    store, paths = tiny_fixture(tmp_path)
    rep = b01.run_workload(store, paths, scale="tiny", repeat=1)
    d = rep.to_dict()
    assert d["name"] == "B01"
    assert d["workload"].startswith("B01")
    assert rep.metrics["wall_ms"] is not None and rep.metrics["wall_ms"] > 0
    assert rep.metrics["rows_per_sec"] is not None
    assert rep.metrics["physical_object_count"] is not None
    assert rep.metrics["duckdb_execute_ms"] is not None
    assert rep.gate_verdict() in ("PASS", "FAIL", "CHECK")


def test_factor_batch_produces_report(tmp_path):
    """B04 批量因子 workload：factor_count / source_group_count / scan_count。"""
    store, paths = tiny_fixture(tmp_path)
    rep = bf.run_factor_batch(store, paths, factor_count=20, scale="tiny", label="B04")
    assert rep.extra["factor_count"] == 20
    assert rep.extra["source_group_count"] > 0
    assert rep.extra["source_group_count"] <= 20
    assert rep.metrics["physical_scan_count"] > 0
    assert rep.metrics["session_reuse"] is not None


def test_compare_and_gate_structure(tmp_path):
    """compare(before, after) 结构 + verdict 合法。"""
    store, paths = tiny_fixture(tmp_path)
    rep1 = b01.run_workload(store, paths, scale="tiny", repeat=1)
    rep2 = b01.run_workload(store, paths, scale="tiny", repeat=1)
    cmp = compare(rep1, rep2)
    assert set(cmp) >= {"deltas", "gates", "verdict"}
    assert isinstance(cmp["deltas"], dict)
    assert isinstance(cmp["gates"], list)
    assert cmp["verdict"] in ("PASS", "FAIL", "CHECK")
    assert rep1.gate_verdict() in ("PASS", "FAIL", "CHECK")


def test_cos_skips_without_real_cos(tmp_path):
    """无真实 COS → B07/B08 如实 SKIP（不假通过）。"""
    from data_access.benchmarks import benchmark_cos as bc

    store, paths = tiny_fixture(tmp_path)
    reports = bc.run_workload(store, paths, scale="tiny")
    assert len(reports) == 2
    for r in reports:
        assert r.metrics.get("status") == "SKIP"
        assert r.extra.get("skip_reason")
        assert r.gate_verdict() == "SKIP"


def test_gate_verdict_orders_fail_over_check():
    """FAIL 优先于 CHECK。"""
    rep = BenchmarkReport("x", "y", "tiny")
    rep.add_gate("peak_rss_mb", "rss <= +15%", "CHECK", 0.2)
    rep.add_gate("rows_per_sec", "throughput >= 90%", "FAIL", -0.5)
    assert rep.gate_verdict() == "FAIL"


def test_benchmark_env_has_version_fields():
    """benchmark_env 含 commit_sha / platform / 引擎版本。"""
    env = benchmark_env()
    for key in (
        "commit_sha",
        "platform",
        "python",
        "cpu_count",
        "duckdb_version",
        "polars_version",
        "pyarrow_version",
        "pandas_version",
        "numpy_version",
    ):
        assert key in env, f"benchmark_env 缺少 {key}"
    assert env["pyarrow_version"] is not None
    assert env["duckdb_version"] is not None


def test_write_evidence_three_files(tmp_path):
    """write_evidence 写出 BENCHMARK_ENV.json / RESULTS(.parquet) / SUMMARY.md。"""
    store, paths = tiny_fixture(tmp_path)
    rep = b01.run_workload(store, paths, scale="tiny", repeat=1)
    out = rep.write_evidence(tmp_path / "evidence")
    assert (out / "BENCHMARK_ENV.json").exists()
    assert (out / "BENCHMARK_SUMMARY.md").exists()
    assert (out / "BENCHMARK_RESULTS.parquet").exists() or (out / "BENCHMARK_RESULTS.json").exists()
    summary = (out / "BENCHMARK_SUMMARY.md").read_text(encoding="utf-8")
    assert "B01" in summary
    env = __import__("json").loads((out / "BENCHMARK_ENV.json").read_text(encoding="utf-8"))
    assert "duckdb_version" in env
