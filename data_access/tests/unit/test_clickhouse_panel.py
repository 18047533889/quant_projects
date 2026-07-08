# -*- coding: utf-8 -*-
"""ClickHouse panel SQL 组装测试（不连真实 CH）。"""
from __future__ import annotations

from data_access.clickhouse_panel import ClickHouseConfig, _build_select_sql


def test_build_select_sql_with_filters():
    sql, params = _build_select_sql(
        table="panel_daily",
        columns=["close", "volume"],
        timestamp_column="trade_date",
        instrument_column="ticker",
        time_range=("2024-01-01", "2024-06-01"),
        instrument_filter=["AAPL", "MSFT"],
    )
    assert "panel_daily" in sql
    assert "close" in sql and "volume" in sql
    assert "trade_date >=" in sql
    assert "ticker IN" in sql
    assert params["start"] == "2024-01-01"
    assert params["instruments"] == ("AAPL", "MSFT")


def test_clickhouse_config_from_env(monkeypatch):
    monkeypatch.setenv("CLICKHOUSE_HOST", "ch.example.com")
    monkeypatch.setenv("CLICKHOUSE_PORT", "9440")
    cfg = ClickHouseConfig.from_env()
    assert cfg.host == "ch.example.com"
    assert cfg.port == 9440
