# -*- coding: utf-8
"""operator_cost 读取 benchmark JSON。"""
from __future__ import annotations

import json
import subprocess
import sys


def test_benchmark_cost_distinguishes_missing_and_seed_stale(tmp_path, monkeypatch):
    import factor_engine.backend.operator_cost as oc

    missing = tmp_path / "missing.json"
    monkeypatch.setattr(oc, "_BENCHMARK_JSON", missing)
    monkeypatch.setattr(oc, "_BENCHMARK_CACHE", None)
    bench, status = oc._load_benchmark_costs()
    assert bench is None
    assert status == "missing"

    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps({"generated_by": "seed_defaults"}), encoding="utf-8")
    monkeypatch.setattr(oc, "_BENCHMARK_JSON", seed)
    monkeypatch.setattr(oc, "_BENCHMARK_CACHE", None)
    bench, status = oc._load_benchmark_costs()
    assert bench is None
    assert status == "stale"

    bc = oc.get_backend_cost("ts_mean", "duckdb_sql")
    table = oc._BACKEND_COST_TABLE["ts_mean"]["duckdb_sql"]
    assert bc.per_million_rows_ms == table.per_million_rows_ms


def test_benchmark_cost_affects_fastpath_coverage(tmp_path):
    script = """
import sys
from pathlib import Path

import factor_engine.backend.operator_cost as oc
from factor_engine.cleaned_operators import load_all
from factor_engine.backend.fastpath_coverage import build_fastpath_coverage_row
from factor_engine.backend.sql_pushdown.sql_registry import register_sql_backends

oc._BENCHMARK_JSON = Path(sys.argv[1])
oc._BENCHMARK_CACHE = None
load_all()
register_sql_backends()
assert build_fastpath_coverage_row("ts_mean").benchmark_available is False
print("fastpath-coverage-ok")
"""
    missing = tmp_path / "missing.json"
    result = subprocess.run(
        [sys.executable, "-c", script, str(missing)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "fastpath-coverage-ok" in result.stdout


def test_measured_baseline_requires_complete_provenance(tmp_path, monkeypatch):
    import factor_engine.backend.operator_cost as oc

    path = tmp_path / "backend_cost_baseline.json"
    monkeypatch.setattr(oc, "_BENCHMARK_JSON", path)
    complete = {
        "measured": True,
        **{key: "test-only" for key in oc._BENCHMARK_PROVENANCE_REQUIRED},
    }

    for missing_key in oc._BENCHMARK_PROVENANCE_REQUIRED:
        incomplete = {key: value for key, value in complete.items() if key != missing_key}
        path.write_text(
            json.dumps({"provenance": incomplete, "operators": {}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(oc, "_BENCHMARK_CACHE", None)
        assert oc._load_benchmark_costs() == (None, "stale")

    path.write_text(
        json.dumps({"provenance": complete, "operators": {}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(oc, "_BENCHMARK_CACHE", None)
    assert oc._load_benchmark_costs() == ({}, "ok")
