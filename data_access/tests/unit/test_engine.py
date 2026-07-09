"""DuckDB engine / config 单元测试。"""

from __future__ import annotations

import duckdb
import pytest

from data_access.duckdb_config import DuckDBConfig, apply_pragmas, resolve_duckdb_config
from data_access.engine import DuckDBEngine


def test_resolve_duckdb_config_from_env(monkeypatch):
    monkeypatch.setenv("DUCKDB_THREADS", "2")
    monkeypatch.setenv("DUCKDB_MEMORY_LIMIT", "1GB")
    monkeypatch.setenv("DUCKDB_TEMP_DIRECTORY", "/tmp/duckdb_spill")
    monkeypatch.setenv("DUCKDB_MAX_TEMP_DIRECTORY_SIZE", "10GB")
    cfg = resolve_duckdb_config()
    assert cfg.threads == 2
    assert cfg.memory_limit == "1GB"
    assert cfg.temp_directory == "/tmp/duckdb_spill"
    assert cfg.max_temp_directory_size == "10GB"


def test_apply_pragmas_sets_threads():
    conn = duckdb.connect(":memory:")
    try:
        apply_pragmas(conn, DuckDBConfig(threads=3, memory_limit="512MB"))
        rows = conn.execute("SELECT current_setting('threads')").fetchall()
        assert int(rows[0][0]) == 3
    finally:
        conn.close()


def test_engine_explain_returns_text():
    engine = DuckDBEngine(threads=2)
    try:
        plan = engine.explain("SELECT 1 AS x")
        assert "SCAN" in plan.upper() or "PROJECTION" in plan.upper() or "1" in plan
    finally:
        engine.close()


def test_engine_register_and_drop_temp_views():
    engine = DuckDBEngine(threads=2)
    try:
        engine.register_temp_views([("__da_test_v", "SELECT 42 AS n")])
        val = engine._conn.execute("SELECT n FROM __da_test_v").fetchone()[0]
        assert val == 42
        engine.drop_temp_views(["__da_test_v"])
        rows = engine._conn.execute(
            "SELECT view_name FROM duckdb_views() WHERE view_name = '__da_test_v'"
        ).fetchall()
        assert rows == []
    finally:
        engine.close()


def test_engine_profile_analyze_returns_text():
    engine = DuckDBEngine(threads=2)
    try:
        profile = engine.profile_analyze("SELECT 1 AS x")
        assert profile
    finally:
        engine.close()


def test_execute_isolated_arrow_does_not_touch_shared_catalog():
    engine = DuckDBEngine(threads=2)
    try:
        with pytest.raises(Exception):
            engine.execute_isolated_arrow(
                "SELECT * FROM read_parquet('/no/such/file.parquet')"
            )
        assert engine.execute_arrow("SELECT 1 AS ok").column("ok")[0].as_py() == 1
    finally:
        engine.close()
