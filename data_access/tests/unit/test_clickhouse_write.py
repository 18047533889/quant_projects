# -*- coding: utf-8 -*-
"""ClickHouse 写入单元测试（mock，不连真实 CH）。"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_access.clickhouse_panel import ClickHouseConfig
from data_access.clickhouse_write import (
    _assert_select_only,
    ensure_factor_table,
    ensure_panel_table,
    execute_select,
    insert_dataframe,
    insert_factor_dataframe,
    insert_factor_series,
    load_parquet_to_panel_table,
)


def test_assert_select_only_blocks_dml():
    with pytest.raises(Exception):
        _assert_select_only("DROP TABLE x")


def test_assert_select_only_allows_select():
    _assert_select_only("SELECT 1")


def test_ensure_factor_table_ddl():
    cfg = ClickHouseConfig(host="localhost")
    mock_client = MagicMock()
    mock_mod = MagicMock()
    mock_mod.get_client.return_value = mock_client

    with patch.dict(sys.modules, {"clickhouse_connect": mock_mod}):
        ensure_factor_table(config=cfg, table="factor_values")

    mock_client.command.assert_called_once()
    ddl = mock_client.command.call_args[0][0]
    assert "ReplacingMergeTree" in ddl
    assert "factor_values" in ddl


def test_ensure_panel_table_ddl():
    cfg = ClickHouseConfig(host="localhost")
    mock_client = MagicMock()
    mock_mod = MagicMock()
    mock_mod.get_client.return_value = mock_client

    with patch.dict(sys.modules, {"clickhouse_connect": mock_mod}):
        ensure_panel_table(
            config=cfg,
            table="panel_daily",
            value_columns=["close", "volume"],
        )

    mock_client.command.assert_called_once()
    ddl = mock_client.command.call_args[0][0]
    assert "MergeTree" in ddl
    assert "close" in ddl and "volume" in ddl


def test_ensure_panel_table_requires_columns():
    cfg = ClickHouseConfig(host="localhost")
    with pytest.raises(Exception):
        ensure_panel_table(config=cfg, table="panel_daily", value_columns=[])


def test_insert_dataframe_mock():
    cfg = ClickHouseConfig(host="localhost")
    frame = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2024-01-02").date()],
            "instrument": ["A"],
            "close": [1.0],
        }
    )
    mock_client = MagicMock()
    mock_mod = MagicMock()
    mock_mod.get_client.return_value = mock_client

    with patch.dict(sys.modules, {"clickhouse_connect": mock_mod}):
        rows = insert_dataframe(
            config=cfg,
            table="panel_daily",
            frame=frame,
            column_order=["trade_date", "instrument", "close"],
        )

    assert rows == 1
    mock_client.insert_df.assert_called_once()


def test_insert_dataframe_empty():
    cfg = ClickHouseConfig(host="localhost")
    assert insert_dataframe(config=cfg, table="panel_daily", frame=pd.DataFrame()) == 0


def test_insert_factor_series_mock():
    cfg = ClickHouseConfig(host="localhost")
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=2), ["A", "B"]],
        names=["ts", "inst"],
    )
    series = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx, dtype=float)
    mock_client = MagicMock()
    mock_mod = MagicMock()
    mock_mod.get_client.return_value = mock_client

    with patch.dict(sys.modules, {"clickhouse_connect": mock_mod}):
        rows = insert_factor_series(
            config=cfg,
            table="factor_values",
            factor_id="test_factor",
            series=series,
            ensure_table=False,
        )

    assert rows == 4
    mock_client.insert_df.assert_called_once()


def test_insert_factor_dataframe_mock():
    cfg = ClickHouseConfig(host="localhost")
    frame = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2024-01-02").date()],
            "instrument": ["A"],
            "factor_id": ["f1"],
            "value": [1.5],
            "factor_version": ["v1"],
            "data_snapshot_id": ["snap"],
        }
    )
    mock_client = MagicMock()
    mock_mod = MagicMock()
    mock_mod.get_client.return_value = mock_client

    with patch.dict(sys.modules, {"clickhouse_connect": mock_mod}):
        rows = insert_factor_dataframe(
            config=cfg,
            table="factor_values",
            frame=frame,
            ensure_table=False,
        )

    assert rows == 1
    mock_client.insert_df.assert_called_once()
    inserted = mock_client.insert_df.call_args[0][1]
    assert "is_valid" in inserted.columns
    assert "invalid_reason" in inserted.columns


def test_load_parquet_to_panel_table_mock(tmp_path):
    cfg = ClickHouseConfig(host="localhost")
    frame = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-02", periods=3, freq="D"),
            "instrument": ["A", "B", "A"],
            "close": [1.0, 2.0, 3.0],
            "volume": [10.0, 20.0, 30.0],
        }
    )
    parquet_path = tmp_path / "panel.parquet"
    frame.to_parquet(parquet_path, index=False)

    mock_client = MagicMock()
    mock_mod = MagicMock()
    mock_mod.get_client.return_value = mock_client

    with patch.dict(sys.modules, {"clickhouse_connect": mock_mod}):
        rows = load_parquet_to_panel_table(
            config=cfg,
            parquet_path=str(parquet_path),
            table="panel_daily",
            timestamp_column="trade_date",
            instrument_column="instrument",
            value_columns=["close", "volume"],
            ensure_table=True,
        )

    assert rows == 3
    mock_client.command.assert_called_once()
    mock_client.insert_df.assert_called_once()


def test_load_parquet_to_panel_table_time_range(tmp_path):
    cfg = ClickHouseConfig(host="localhost")
    frame = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-02", periods=3, freq="D"),
            "instrument": ["A", "B", "A"],
            "close": [1.0, 2.0, 3.0],
        }
    )
    parquet_path = tmp_path / "panel.parquet"
    frame.to_parquet(parquet_path, index=False)

    mock_client = MagicMock()
    mock_mod = MagicMock()
    mock_mod.get_client.return_value = mock_client

    with patch.dict(sys.modules, {"clickhouse_connect": mock_mod}):
        rows = load_parquet_to_panel_table(
            config=cfg,
            parquet_path=str(parquet_path),
            table="panel_daily",
            timestamp_column="trade_date",
            instrument_column="instrument",
            value_columns=["close"],
            time_range=(pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-03")),
            ensure_table=False,
        )

    assert rows == 1
    inserted = mock_client.insert_df.call_args[0][1]
    assert len(inserted) == 1
    assert inserted.iloc[0]["trade_date"] == pd.Timestamp("2024-01-03").date()


def test_load_parquet_to_panel_table_batching(tmp_path):
    cfg = ClickHouseConfig(host="localhost")
    frame = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-02", periods=5, freq="D"),
            "instrument": ["A"] * 5,
            "close": [float(i) for i in range(5)],
        }
    )
    parquet_path = tmp_path / "panel.parquet"
    frame.to_parquet(parquet_path, index=False)

    mock_client = MagicMock()
    mock_mod = MagicMock()
    mock_mod.get_client.return_value = mock_client

    with patch.dict(sys.modules, {"clickhouse_connect": mock_mod}):
        rows = load_parquet_to_panel_table(
            config=cfg,
            parquet_path=str(parquet_path),
            table="panel_daily",
            timestamp_column="trade_date",
            instrument_column="instrument",
            value_columns=["close"],
            batch_size=2,
            ensure_table=False,
        )

    assert rows == 5
    assert mock_client.insert_df.call_count == 3


def test_execute_select_mock():
    cfg = ClickHouseConfig(host="localhost")
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.arrow.return_value = MagicMock()
    mock_client.query.return_value = mock_result
    mock_mod = MagicMock()
    mock_mod.get_client.return_value = mock_client

    with patch.dict(sys.modules, {"clickhouse_connect": mock_mod}):
        table = execute_select(config=cfg, sql="SELECT 1 AS value")

    mock_client.query.assert_called_once_with("SELECT 1 AS value")
    assert table is mock_result.arrow.return_value


def test_load_parquet_to_panel_table_empty_after_filter(tmp_path):
    cfg = ClickHouseConfig(host="localhost")
    frame = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-02", periods=2, freq="D"),
            "instrument": ["A", "B"],
            "close": [1.0, 2.0],
        }
    )
    parquet_path = tmp_path / "panel.parquet"
    frame.to_parquet(parquet_path, index=False)

    mock_client = MagicMock()
    mock_mod = MagicMock()
    mock_mod.get_client.return_value = mock_client

    with patch.dict(sys.modules, {"clickhouse_connect": mock_mod}):
        rows = load_parquet_to_panel_table(
            config=cfg,
            parquet_path=str(parquet_path),
            table="panel_daily",
            timestamp_column="trade_date",
            instrument_column="instrument",
            value_columns=["close"],
            time_range=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02")),
            ensure_table=False,
        )

    assert rows == 0
    mock_client.insert_df.assert_not_called()


def test_insert_factor_series_metadata_columns():
    cfg = ClickHouseConfig(host="localhost")
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=2), ["A"]],
        names=["ts", "inst"],
    )
    series = pd.Series([1.0, 2.0], index=idx, dtype=float)
    mock_client = MagicMock()
    mock_mod = MagicMock()
    mock_mod.get_client.return_value = mock_client

    with patch.dict(sys.modules, {"clickhouse_connect": mock_mod}):
        rows = insert_factor_series(
            config=cfg,
            table="factor_values",
            factor_id="meta_factor",
            series=series,
            factor_version="v2",
            data_snapshot_id="snap-001",
            is_valid=0,
            ensure_table=False,
        )

    assert rows == 2
    inserted = mock_client.insert_df.call_args[0][1]
    assert inserted["factor_id"].iloc[0] == "meta_factor"
    assert inserted["factor_version"].iloc[0] == "v2"
    assert inserted["data_snapshot_id"].iloc[0] == "snap-001"
    assert int(inserted["is_valid"].iloc[0]) == 0


def test_load_parquet_to_panel_table_missing_file(tmp_path):
    cfg = ClickHouseConfig(host="localhost")
    missing = tmp_path / "missing.parquet"
    with pytest.raises(Exception):
        load_parquet_to_panel_table(
            config=cfg,
            parquet_path=str(missing),
            table="panel_daily",
            ensure_table=False,
        )


def test_execute_select_rejects_mutation():
    cfg = ClickHouseConfig(host="localhost")
    with pytest.raises(Exception):
        execute_select(config=cfg, sql="INSERT INTO panel_daily VALUES (1)")
