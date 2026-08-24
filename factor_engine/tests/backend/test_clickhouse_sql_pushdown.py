# -*- coding: utf-8
"""ClickHouse 数据源 + SqlBackend 端到端（mock execute_query，无真实 CH）。"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pytest

from factor_engine.backend.context import ExecutionContext
from factor_engine.backend.factory import build_backend
from factor_engine.backend.sql_pushdown.executor import extract_pushdown_context, try_execute_sql_pushdown
from factor_engine.backend.sql_pushdown.source_resolver import resolve_pushdown_source
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.runtime.config import load_config
from factor_engine.storage.factory import build_data_source
from factor_engine.util.workspace_paths import quant_projects_root


@pytest.fixture(autouse=True)
def _path():
    root = str(quant_projects_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    yield


def _col(name: str) -> PlanNode:
    return PlanNode(op="column", attrs={"name": name})


def test_resolve_clickhouse_source_for_pushdown():
    source = build_data_source(
        {
            "type": "clickhouse",
            "table": "panel_daily",
            "timestamp_column": "trade_date",
            "instrument_column": "ticker",
        }
    )
    resolved = resolve_pushdown_source(source)
    assert resolved is source
    assert resolved.table == "panel_daily"

    ctx = extract_pushdown_context(ExecutionContext(data_source=source))
    assert ctx is not None
    assert ctx.dialect.value == "clickhouse"
    assert ctx.table == "panel_daily"


def test_try_execute_sql_pushdown_clickhouse_ts_mean():
    source = build_data_source(
        {
            "type": "clickhouse",
            "table": "panel_daily",
            "timestamp_column": "trade_date",
            "instrument_column": "ticker",
        }
    )
    plan = PlanNode(op="ts_mean", inputs=[_col("close")], attrs={"window": 3})
    ctx = ExecutionContext(data_source=source)

    table = pa.table(
        {
            "ts": pa.array(["2024-01-01", "2024-01-02"]),
            "inst": pa.array(["A", "A"]),
            "value": pa.array([1.0, 2.0], type=pa.float64()),
        }
    )
    mock_cfg = MagicMock()

    with patch("data_access.clickhouse.panel.ClickHouseConfig.from_env", return_value=mock_cfg):
        with patch("data_access.clickhouse.panel.execute_query", return_value=table) as mock_eq:
            series = try_execute_sql_pushdown(plan, ctx)

    mock_eq.assert_called_once()
    assert series is not None
    assert len(series) == 2


def test_try_execute_sql_pushdown_clickhouse_ffill():
    source = build_data_source(
        {
            "type": "clickhouse",
            "table": "panel_daily",
            "timestamp_column": "trade_date",
            "instrument_column": "ticker",
        }
    )
    plan = PlanNode(op="ffill", inputs=[_col("close")])
    ctx = ExecutionContext(data_source=source)

    table = pa.table(
        {
            "ts": pa.array(["2024-01-01", "2024-01-02"]),
            "inst": pa.array(["A", "A"]),
            "value": pa.array([1.0, 2.0], type=pa.float64()),
        }
    )
    mock_cfg = MagicMock()

    with patch("data_access.clickhouse.panel.ClickHouseConfig.from_env", return_value=mock_cfg):
        with patch("data_access.clickhouse.panel.execute_query", return_value=table) as mock_eq:
            series = try_execute_sql_pushdown(plan, ctx)

    mock_eq.assert_called_once()
    sql = mock_eq.call_args.kwargs.get("sql") or mock_eq.call_args.args[1]
    assert "anyLast" in sql
    assert series is not None
    assert len(series) == 2


def test_try_execute_sql_pushdown_clickhouse_coalesce():
    source = build_data_source(
        {
            "type": "clickhouse",
            "table": "panel_daily",
            "timestamp_column": "trade_date",
            "instrument_column": "ticker",
        }
    )
    plan = PlanNode(
        op="coalesce",
        inputs=[_col("close"), _col("open")],
    )
    ctx = ExecutionContext(data_source=source)
    table = pa.table(
        {
            "ts": pa.array(["2024-01-01"]),
            "inst": pa.array(["A"]),
            "value": pa.array([3.0], type=pa.float64()),
        }
    )
    mock_cfg = MagicMock()

    with patch("data_access.clickhouse.panel.ClickHouseConfig.from_env", return_value=mock_cfg):
        with patch("data_access.clickhouse.panel.execute_query", return_value=table) as mock_eq:
            series = try_execute_sql_pushdown(plan, ctx)

    mock_eq.assert_called_once()
    sql = mock_eq.call_args.kwargs.get("sql") or mock_eq.call_args.args[1]
    assert "coalesce(" in sql
    assert series is not None


def test_try_execute_sql_pushdown_clickhouse_protected_div():
    source = build_data_source(
        {
            "type": "clickhouse",
            "table": "panel_daily",
            "timestamp_column": "trade_date",
            "instrument_column": "ticker",
        }
    )
    plan = PlanNode(
        op="protected_div",
        inputs=[_col("close"), _col("open")],
    )
    ctx = ExecutionContext(data_source=source)
    table = pa.table(
        {
            "ts": pa.array(["2024-01-01"]),
            "inst": pa.array(["A"]),
            "value": pa.array([2.0], type=pa.float64()),
        }
    )
    mock_cfg = MagicMock()

    with patch("data_access.clickhouse.panel.ClickHouseConfig.from_env", return_value=mock_cfg):
        with patch("data_access.clickhouse.panel.execute_query", return_value=table) as mock_eq:
            series = try_execute_sql_pushdown(plan, ctx)

    mock_eq.assert_called_once()
    sql = mock_eq.call_args.kwargs.get("sql") or mock_eq.call_args.args[1]
    assert "CASE WHEN l._v IS NULL OR r._v IS NULL THEN NULL" in sql
    assert series is not None


def test_try_execute_sql_pushdown_clickhouse_nan_to_num():
    source = build_data_source(
        {
            "type": "clickhouse",
            "table": "panel_daily",
            "timestamp_column": "trade_date",
            "instrument_column": "ticker",
        }
    )
    plan = PlanNode(op="nan_to_num", inputs=[_col("close"), PlanNode(op="literal", attrs={"value": 0.0})])
    ctx = ExecutionContext(data_source=source)
    table = pa.table(
        {
            "ts": pa.array(["2024-01-01"]),
            "inst": pa.array(["A"]),
            "value": pa.array([0.0], type=pa.float64()),
        }
    )
    mock_cfg = MagicMock()

    with patch("data_access.clickhouse.panel.ClickHouseConfig.from_env", return_value=mock_cfg):
        with patch("data_access.clickhouse.panel.execute_query", return_value=table) as mock_eq:
            series = try_execute_sql_pushdown(plan, ctx)

    # nan_to_num was moved out of the active Factor DSL.  A dormant emitter
    # branch must not make the removed research tool SQL-capable.
    mock_eq.assert_not_called()
    assert series is None


def test_try_execute_sql_pushdown_clickhouse_where_is_finite():
    source = build_data_source(
        {
            "type": "clickhouse",
            "table": "panel_daily",
            "timestamp_column": "trade_date",
            "instrument_column": "ticker",
        }
    )
    zero = PlanNode(
        op="multiply",
        inputs=[_col("close"), PlanNode(op="literal", attrs={"value": 0.0})],
    )
    plan = PlanNode(
        op="where",
        inputs=[
            PlanNode(op="is_finite", inputs=[_col("close")]),
            _col("close"),
            zero,
        ],
    )
    ctx = ExecutionContext(data_source=source)
    table = pa.table(
        {
            "ts": pa.array(["2024-01-01"]),
            "inst": pa.array(["A"]),
            "value": pa.array([1.0], type=pa.float64()),
        }
    )
    mock_cfg = MagicMock()

    with patch("data_access.clickhouse.panel.ClickHouseConfig.from_env", return_value=mock_cfg):
        with patch("data_access.clickhouse.panel.execute_query", return_value=table) as mock_eq:
            series = try_execute_sql_pushdown(plan, ctx)

    mock_eq.assert_called_once()
    sql = mock_eq.call_args.kwargs.get("sql") or mock_eq.call_args.args[1]
    assert "isFinite" in sql or "isfinite" in sql
    assert "isNotNull(c._v)" in sql or "IS NOT NULL" in sql
    assert series is not None


def test_build_backend_clickhouse_sql_alias():
    backend = build_backend("clickhouse_sql")
    from factor_engine.backend.sql_backend import SqlBackend

    assert isinstance(backend, SqlBackend)


def test_clickhouse_panel_rank_config_loads():
    root = quant_projects_root() / "factor_engine"
    loaded = load_config(root / "examples" / "configs" / "clickhouse_panel_rank.yaml")
    assert loaded.backend.type == "clickhouse_sql"
    assert loaded.data_source.type == "clickhouse"
    assert loaded.data_source.options["table"] == "panel_daily"
