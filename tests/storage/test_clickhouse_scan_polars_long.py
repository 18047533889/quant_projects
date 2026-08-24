# -*- coding: utf-8
"""ClickHouseSource.scan_polars_long 测试。"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest


@pytest.fixture(autouse=True)
def _path():
    from factor_engine.workspace_paths import quant_projects_root

    root = str(quant_projects_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    yield


def test_clickhouse_scan_polars_long_renames_columns():
    pytest.importorskip("polars")

    from factor_engine.storage.sources.clickhouse_source import ClickHouseSource

    src = ClickHouseSource(
        table="panel_daily",
        timestamp_column="trade_date",
        instrument_column="ticker",
        fields={"close": "Close"},
    )
    table = pa.table(
        {
            "trade_date": pa.array(["2024-01-01", "2024-01-01"]),
            "ticker": pa.array(["A", "B"]),
            "Close": pa.array([10.0, 20.0], type=pa.float64()),
        }
    )
    mock_cfg = MagicMock()

    with patch("data_access.clickhouse.panel.ClickHouseConfig.from_env", return_value=mock_cfg):
        with patch("data_access.clickhouse.panel.execute_query", return_value=table) as mock_eq:
            lf = src.scan_polars_long(["close"])

    mock_eq.assert_called_once()
    sql = mock_eq.call_args[0][1]
    assert "panel_daily" in sql
    assert "Close" in sql
    cols = lf.collect_schema().names()
    assert cols == ["ts", "inst", "close"]
