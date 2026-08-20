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
from backend.sql_pushdown.executor import extract_pushdown_context
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
    assert getattr(backend, "_requested_backend", None) == "clickhouse_sql"
    assert getattr(backend, "_resolved_dialect", None) == "clickhouse"

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
    assert cert.validate({"backend": "clickhouse_sql", "no_fallback": True, "dialect": "clickhouse"}) is True
    assert cert.validate({"backend": "clickhouse_sql", "no_fallback": True}) is True

    # Note: `validate` only rejects cross-dialect when BOTH certificate and event
    # declare a dialect. An event without a dialect remains backward-compatible.
    duck_cert = ProductionExecutionCertificate.build(
        structural_hash="a" * 8,
        bound_ops=["ts_mean"],
        backend_eligibility=["clickhouse_sql"],
        output_shape_hash="b" * 8,
        requested_backend="clickhouse_sql",
        resolved_dialect="duckdb",
    )
    assert duck_cert.validate({"backend": "clickhouse_sql", "no_fallback": True, "dialect": "clickhouse"}) is False


def test_clickhouse_pushdown_compiler_identity_matches_factory():
    source = _ch_source()
    ctx = ExecutionContext(data_source=source)
    mock_backend = build_backend("clickhouse_sql")

    stats_before = dict(getattr(ctx, "runtime_stats", None) or {})
    pctx = extract_pushdown_context(ctx)

    assert pctx is not None
    assert pctx.dialect.value == "clickhouse"
    assert pctx.table == "panel_daily"
    assert getattr(mock_backend, "runtime_backend_label", None) == "clickhouse_sql"

    stats_after = dict(getattr(ctx, "runtime_stats", None) or {})
    assert stats_before == stats_after
