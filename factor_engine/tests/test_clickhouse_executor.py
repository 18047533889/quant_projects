# -*- coding: utf-8 -*-
"""ClickHouse SQL 执行器单元测试（mock，无真实实例）。"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pytest

from backend.context import ExecutionContext
from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql
from planner.logical_plan import PlanNode
from tests.helpers import InMemorySeriesSource
from workspace_paths import quant_projects_root


@pytest.fixture(autouse=True)
def _path():
    root = str(quant_projects_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    yield


def _col(name: str) -> PlanNode:
    return PlanNode(op="column", attrs={"name": name})


class _CHSource:
    table = "panel_daily"
    timestamp_column = "trade_date"
    instrument_column = "instrument"
    start_date = None
    end_date = None
    instrument_filter = []


def test_execute_clickhouse_delegates_to_execute_query():
    from backend.sql_pushdown.executor import (
        PushdownContext,
        execute_compiled_sql,
    )

    plan = PlanNode(op="ts_mean", inputs=[_col("close")], attrs={"window": 5})
    compiled = compile_plan_to_sql(
        plan,
        table="panel_daily",
        time_column="trade_date",
        instrument_column="instrument",
        dialect=SqlDialect.CLICKHOUSE,
    )
    assert compiled is not None

    table = pa.table(
        {
            "ts": pa.array(["2024-01-01", "2024-01-02"]),
            "inst": pa.array(["A", "A"]),
            "value": pa.array([1.0, 2.0], type=pa.float64()),
        }
    )
    mock_cfg = MagicMock()
    pctx = PushdownContext(
        dialect=SqlDialect.CLICKHOUSE,
        table="panel_daily",
        time_column="trade_date",
        instrument_column="instrument",
        ch_config={},
    )

    with patch("data_access.clickhouse_panel.ClickHouseConfig.from_env", return_value=mock_cfg):
        with patch("data_access.clickhouse_panel.execute_query", return_value=table) as mock_eq:
            series = execute_compiled_sql(compiled, pctx, None)

    mock_eq.assert_called_once()
    assert len(series) == 2


def test_try_execute_sql_pushdown_none_without_ch_source():
    from backend.sql_pushdown.executor import try_execute_sql_pushdown

    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=2), ["A"]],
        names=["timestamp", "instrument"],
    )
    src = InMemorySeriesSource(data={"close": pd.Series([1.0, 2.0], index=idx)})
    ctx = ExecutionContext(data_source=src)
    plan = PlanNode(op="ts_mean", inputs=[_col("close")], attrs={"window": 2})
    assert try_execute_sql_pushdown(plan, ctx) is None
