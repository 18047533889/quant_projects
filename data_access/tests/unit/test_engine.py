"""DuckDB engine / config 单元测试。"""

from __future__ import annotations

import duckdb
import pytest

from data_access.duckdb_config import DuckDBConfig, _sql_string, apply_pragmas, resolve_duckdb_config
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


def test_sql_string_escapes_single_quotes():
    assert _sql_string("512'MB") == "'512''MB'"
    assert _sql_string("/tmp/duck's_spill") == "'/tmp/duck''s_spill'"


def test_apply_pragmas_escapes_single_quotes_in_temp_directory():
    conn = duckdb.connect(":memory:")
    try:
        apply_pragmas(
            conn,
            DuckDBConfig(
                threads=1,
                temp_directory="/tmp/duck's_spill",
            ),
        )
        rows = conn.execute("SELECT current_setting('temp_directory')").fetchall()
        assert rows[0][0] == "/tmp/duck's_spill"
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


def test_scoped_sql_configures_s3_when_specs_contain_s3(monkeypatch):
    engine = DuckDBEngine(threads=2)
    calls: list[object] = []

    def _fake_configure(conn):
        calls.append(conn)

    monkeypatch.setattr(
        "data_access.s3_duckdb.configure_fresh_duckdb_s3",
        _fake_configure,
    )
    try:
        # s3 路径在 view SQL 中：应触发 configure；随后 SELECT 不真正读 s3
        specs = [("__v", "SELECT 1 AS x FROM read_parquet('s3://bucket/x.parquet')")]
        # DuckDB 会在 CREATE VIEW 时校验——若 httpfs 未真正配置可能失败。
        # 我们只断言 configure 被调用；若执行失败也接受（无真实凭证）。
        try:
            engine.execute_scoped_sql_arrow(specs, "SELECT 1 AS ok")
        except Exception:
            pass
        assert len(calls) == 1
    finally:
        engine.close()


def test_scoped_sql_skips_s3_for_local_specs(monkeypatch):
    engine = DuckDBEngine(threads=2)
    calls: list[object] = []
    monkeypatch.setattr(
        "data_access.s3_duckdb.configure_fresh_duckdb_s3",
        lambda conn: calls.append(conn),
    )
    try:
        table = engine.execute_scoped_sql_arrow(
            [("__v", "SELECT 7 AS n")],
            "SELECT n FROM __v",
        )
        assert table.column("n")[0].as_py() == 7
        assert calls == []
    finally:
        engine.close()
