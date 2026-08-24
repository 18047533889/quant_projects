# -*- coding: utf-8
"""P0-10 source-ordering contract smoke test (pure, DataAccessSource-independent)."""
from __future__ import annotations

import pytest

pytest.importorskip("polars")


def _unsorted_df():
    import polars as pl

    return pl.DataFrame(
        {
            "inst": ["B", "A", "B", "A"],
            "ts": ["2024-01-02", "2024-01-02", "2024-01-01", "2024-01-01"],
        }
    )


def test_enforce_source_ordering_daily_default():
    from factor_engine.backend.polars_lazy import enforce_source_ordering

    out = enforce_source_ordering(_unsorted_df(), instrument_col="inst", time_col="ts")
    rows = out.select(["inst", "ts"]).to_numpy().tolist()
    assert rows == [
        ["A", "2024-01-01"],
        ["A", "2024-01-02"],
        ["B", "2024-01-01"],
        ["B", "2024-01-02"],
    ]


def test_enforce_source_ordering_minute_session():
    import polars as pl

    from factor_engine.backend.polars_lazy import enforce_source_ordering

    df = pl.DataFrame(
        {
            "inst": ["A", "A", "A", "A"],
            "ts": ["2024-01-01 09:31", "2024-01-01 09:30", "2024-01-01 09:30", "2024-01-01 09:31"],
            "session": ["AM", "AM", "PM", "PM"],
        }
    )
    out = enforce_source_ordering(
        df,
        instrument_col="inst",
        time_col="ts",
        session_col="session",
        frequency="minute",
    )
    rows = out.select(["inst", "ts", "session"]).to_numpy().tolist()
    assert rows == [
        ["A", "2024-01-01 09:30", "AM"],
        ["A", "2024-01-01 09:30", "PM"],
        ["A", "2024-01-01 09:31", "AM"],
        ["A", "2024-01-01 09:31", "PM"],
    ]


def test_enforce_source_ordering_lazyframe():
    from factor_engine.backend.polars_lazy import enforce_source_ordering

    out = enforce_source_ordering(
        _unsorted_df().lazy(), instrument_col="inst", time_col="ts"
    )
    rows = out.collect().select(["inst", "ts"]).to_numpy().tolist()
    assert rows[0] == ["A", "2024-01-01"]
    assert rows[1] == ["A", "2024-01-02"]
    assert rows[2] == ["B", "2024-01-01"]
    assert rows[3] == ["B", "2024-01-02"]


def test_clickhouse_scan_sql_carries_order_by():
    from factor_engine.backend.polars_lazy import build_clickhouse_scan_sql

    sql = build_clickhouse_scan_sql(
        "panel_daily",
        timestamp_column="trade_date",
        instrument_column="ticker",
        physical_columns=["Close"],
        time_range=("2024-01-01", "2024-01-31"),
    )
    assert sql.endswith("ORDER BY `ticker`, `trade_date`")
    assert "WHERE" in sql

    minute = build_clickhouse_scan_sql(
        "minute_bar",
        timestamp_column="trade_time",
        instrument_column="ticker",
        physical_columns=["Close"],
        frequency="minute",
        session_column="session_id",
    )
    assert "`session_id`" in minute
    assert minute.endswith("ORDER BY `ticker`, `trade_time`, `session_id`")
