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
    insert_factor_series,
)


def test_assert_select_only_blocks_dml():
    with pytest.raises(Exception):
        _assert_select_only("DROP TABLE x")


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
