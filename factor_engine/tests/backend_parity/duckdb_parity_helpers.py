# -*- coding: utf-8
"""DuckDB parity 断言：须真实 SQL 执行，禁止 false pushdown。"""
from __future__ import annotations


def assert_duckdb_real_sql_execution(run_out: dict) -> None:
    """Require the tested operator tree—not merely a leaf—to run in DuckDB."""
    assert int(run_out.get("sql_query_count") or 0) > 0, run_out
    assert run_out.get("used_sql_pushdown") is True, run_out
    assert int(run_out.get("sql_fallback_subtree_count") or 0) == 0, run_out
    assert run_out.get("sql_full_execution_failed") is not True, run_out
    assert run_out.get("fully_sql") is True, run_out
    assert run_out.get("sql_fully_pushed") is True, run_out
