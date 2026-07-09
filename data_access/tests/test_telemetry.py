"""
data_access/tests/test_telemetry —— PR6 telemetry 冒烟测试

覆盖：
- record_query 在正常路径下计数
- 慢查询触发 WARNING
- 配额阈值触发 80% 和 100% 告警，且同一进程不重复触发
- engine.execute_arrow 真实走 telemetry 路径
"""

from __future__ import annotations

import logging
import os

import pytest

from data_access import get_shared_engine, reset_shared_engine
from data_access.telemetry import (
    get_counters_snapshot,
    record_query,
    reset_counters,
)


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """每个用例跑前重置 counters 和环境变量，跑后再重置一次 engine。"""
    reset_counters()
    monkeypatch.delenv("QUANT_SLOW_QUERY_MS", raising=False)
    monkeypatch.delenv("QUANT_OPERATOR_QUERY_QUOTA", raising=False)
    # 用一个稳定的 operator，否则测试 assert 不出 key
    monkeypatch.setenv("QUANT_OPERATOR", "test_operator")
    yield
    reset_counters()
    reset_shared_engine()


def test_record_query_counts_single_query(caplog):
    caplog.set_level(logging.WARNING, logger="data_access.telemetry")
    record_query(elapsed_ms=10.0, sql="SELECT 1")
    snapshot = get_counters_snapshot()
    assert "test_operator" in snapshot
    counters = snapshot["test_operator"]
    assert counters.total_queries == 1
    assert counters.slow_queries == 0
    assert counters.total_elapsed_ms == pytest.approx(10.0)
    # 正常查询不打 WARNING
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == []


def test_slow_query_triggers_warning(monkeypatch, caplog):
    monkeypatch.setenv("QUANT_SLOW_QUERY_MS", "100")
    caplog.set_level(logging.WARNING, logger="data_access.telemetry")
    record_query(elapsed_ms=500.0, sql="SELECT slow_thing")
    record_query(elapsed_ms=50.0, sql="SELECT fast_thing")
    snapshot = get_counters_snapshot()["test_operator"]
    assert snapshot.total_queries == 2
    assert snapshot.slow_queries == 1
    assert any(
        "慢查询" in r.getMessage() and "SELECT slow_thing" in r.getMessage()
        for r in caplog.records
    )


def test_quota_warning_fires_at_80_and_100(monkeypatch, caplog):
    monkeypatch.setenv("QUANT_OPERATOR_QUERY_QUOTA", "10")
    caplog.set_level(logging.WARNING, logger="data_access.telemetry")
    for _ in range(12):
        record_query(elapsed_ms=1.0, sql="SELECT 1")
    # 应该至少触发 80% 和 100% 各一次
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    quota_msgs = [m for m in msgs if "查询配额" in m]
    # 80% 和 100% 各一次，共 2 条（且不会因为后续 query 再重复）
    assert any("80%" in m for m in quota_msgs)
    assert any("100%" in m for m in quota_msgs)
    # 不重复：80%/100% 各自只来一次
    assert sum("80%" in m for m in quota_msgs) == 1
    assert sum("100%" in m for m in quota_msgs) == 1


def test_telemetry_hook_invoked_via_engine(monkeypatch):
    monkeypatch.setenv("QUANT_OPERATOR_QUERY_QUOTA", "100000")  # 防止测试里触告警
    reset_shared_engine()
    engine = get_shared_engine()
    engine.execute_arrow("SELECT 1 AS x")
    engine.execute_arrow("SELECT 2 AS y")
    snapshot = get_counters_snapshot()
    assert "test_operator" in snapshot
    assert snapshot["test_operator"].total_queries == 2
    assert snapshot["test_operator"].total_elapsed_ms > 0


def test_record_query_swallows_errors(monkeypatch):
    """telemetry 内部失败不能把业务查询搞挂。"""
    # 故意注入一个会 boom 的 operator resolver
    import data_access.telemetry as tele

    def boom() -> str:
        raise RuntimeError("operator resolver 挂了")

    monkeypatch.setattr(tele, "resolve_operator", boom)
    # 不抛异常就算通过
    record_query(elapsed_ms=1.0, sql="SELECT 1")


def test_record_polars_scan_counts(monkeypatch):
    from data_access.telemetry import record_polars_scan

    monkeypatch.setenv("QUANT_OPERATOR", "test_operator")
    record_polars_scan(dataset="us_stock_daily", elapsed_ms=12.5, paths_count=3)
    snapshot = get_counters_snapshot()["test_operator"]
    assert snapshot.total_queries == 1
    assert snapshot.total_elapsed_ms == pytest.approx(12.5)


def test_slow_query_explain_when_enabled(monkeypatch, caplog):
    from data_access.engine import DuckDBEngine
    from data_access.telemetry import maybe_log_slow_query_plan

    monkeypatch.setenv("QUANT_EXPLAIN_SLOW_QUERY", "1")
    monkeypatch.setenv("QUANT_EXPLAIN_QUERY_MS", "50")
    monkeypatch.setenv("QUANT_OPERATOR", "test_operator")
    caplog.set_level(logging.WARNING, logger="data_access.telemetry")
    engine = DuckDBEngine(threads=2)
    try:
        maybe_log_slow_query_plan(
            engine,
            elapsed_ms=100.0,
            sql="SELECT 1 AS x",
            params=None,
            op="arrow",
        )
    finally:
        engine.close()
    assert any("慢查询 EXPLAIN" in r.getMessage() for r in caplog.records)
