# -*- coding: utf-8 -*-
"""R21-CLICKHOUSE-ADAPTER: regression tests for ClickHouse execution adapter boundary."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pytest

from backend.context import ExecutionContext
from backend.factory import build_backend, build_backend_execution_certificate
from backend.sql_backend import SqlBackend
from backend.sql_pushdown.executor import extract_pushdown_context, try_execute_sql_pushdown
from backend.sql_pushdown.source_resolver import resolve_pushdown_source
from runtime.production_execution_certificate import ProductionExecutionCertificate
from storage.factory import build_data_source


def _ch_source():
    return build_data_source(
        {
            "type": "clickhouse",
            "table": "panel_daily",
            "timestamp_column": "trade_date",
            "instrument_column": "ticker",
        }
    )


def _col(name: str):
    from planner.logical_plan import PlanNode

    return PlanNode(op="column", attrs={"name": name})


def test_clickhouse_pushdown_backend_has_expected_identity_and_base():
    backend = build_backend("clickhouse_sql")

    assert isinstance(backend, SqlBackend)
    assert getattr(backend, "runtime_backend_label", None) == "clickhouse_sql"
    assert getattr(backend, "_requested_backend", None) is None
    assert getattr(backend, "_resolved_dialect", None) is None

    source = _ch_source()
    resolved = resolve_pushdown_source(source)
    assert resolved is source
    ctx = extract_pushdown_context(ExecutionContext(data_source=source))
    assert ctx is not None
    assert ctx.dialect.value == "clickhouse"


def test_clickhouse_certificate_chain_inherits_adapter_identity():
    backend = build_backend("clickhouse_sql")

    cert = build_backend_execution_certificate(
        backend,
        structural_hash="a" * 8,
        bound_ops=["ts_mean"],
        backend_eligibility=["clickhouse_sql"],
        output_shape_hash="b" * 8,
    )
    assert isinstance(cert, ProductionExecutionCertificate)
    assert cert.requested_backend == "clickhouse_sql"
    assert cert.resolved_dialect == "clickhouse"
    assert cert.validate({"backend": "clickhouse_sql", "no_fallback": True}) is True
    assert cert.validate({"backend": "duckdb_sql", "no_fallback": True}) is False


def test_clickhouse_pushdown_sql_runtime_identity_matches_factory():
    source = _ch_source()
    ctx = ExecutionContext(data_source=source)
    plan = _col("close")

    table = pa.table(
        {
            "ts": pa.array(["2024-01-01", "2024-01-02"]),
            "inst": pa.array(["A", "A"]),
            "value": pa.array([1.5, 2.5], type=pa.float64()),
        }
    )
    mock_cfg = MagicMock()
    mock_backend = build_backend("clickhouse_sql")

    with patch("data_access.clickhouse.panel.ClickHouseConfig.from_env", return_value=mock_cfg):
        with patch("data_access.clickhouse.panel.execute_query", return_value=table) as mock_eq:
            result = mock_backend.execute(plan, ctx)

    mock_eq.assert_called_once()
    assert isinstance(result, pd.Series)
    assert len(result) == 2
    stats = getattr(ctx, "runtime_stats", None) or {}
    assert stats.get("sql_dialect") == "clickhouse"
    assert stats.get("backend") == "clickhouse_sql"
