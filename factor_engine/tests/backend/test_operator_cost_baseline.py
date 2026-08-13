# -*- coding: utf-8
"""operator_cost 读取 benchmark JSON。"""
from __future__ import annotations


def test_benchmark_cost_ignores_seed_defaults():
    import backend.operator_cost as oc

    oc._BENCHMARK_CACHE = None
    # Round-8 #354: the loader now returns (table, status) and treats a
    # seed_defaults / non-measured baseline as stale instead of returning {}.
    bench, status = oc._load_benchmark_costs()
    assert bench is None
    assert status == "stale"
    bc = oc.get_backend_cost("ts_mean", "duckdb_sql")
    table = oc._BACKEND_COST_TABLE["ts_mean"]["duckdb_sql"]
    assert bc.per_million_rows_ms == table.per_million_rows_ms


def test_benchmark_cost_affects_fastpath_coverage():
    from cleaned_operators import load_all
    from backend.fastpath_coverage import build_fastpath_coverage_row
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()
    row = build_fastpath_coverage_row("ts_mean")
    assert row.benchmark_available is False


def test_backend_cost_baseline_json_freshness():
    """committed baseline 应含 ts_mean duckdb_sql 成本（CI freshness gate）。"""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "benchmarks" / "backend_cost_baseline.json"
    assert path.is_file(), f"missing baseline: {path}"
    payload = json.loads(path.read_text(encoding="utf-8"))
    ops = payload.get("operators") or {}
    ts_mean = ops.get("ts_mean") or {}
    assert "duckdb_sql" in ts_mean or any("duckdb_sql" in k for k in ts_mean), ts_mean
    cost = ts_mean.get("duckdb_sql") or next(v for k, v in ts_mean.items() if "duckdb_sql" in k)
    assert float(cost.get("per_million_rows_ms", 0) > 0
