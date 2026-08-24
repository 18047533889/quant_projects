# -*- coding: utf-8 -*-
"""ClickHouse execute_query 单元测试（mock，不连真实 CH）。"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pyarrow as pa

from data_access.clickhouse.panel import ClickHouseConfig, execute_query


def test_execute_query_returns_arrow():
    cfg = ClickHouseConfig(host="localhost", port=8123)
    table = pa.table({"ts": ["2024-01-01"], "inst": ["A"], "value": [1.0]})
    mock_client = MagicMock()
    mock_client.query.return_value.arrow.return_value = table
    mock_mod = MagicMock()
    mock_mod.get_client.return_value = mock_client

    with patch.dict(sys.modules, {"clickhouse_connect": mock_mod}):
        out = execute_query(config=cfg, sql="SELECT 1")

    assert out.num_rows == 1
    mock_mod.get_client.assert_called_once()
