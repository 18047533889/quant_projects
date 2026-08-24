# -*- coding: utf-8 -*-
"""R21-P026: ClickHouse route identity regression tests.

Bug: for a ClickHouse data source the router computed
``sql_backend = "clickhouse_sql"`` but still wrote the candidate key as
``"duckdb_sql"`` — so the route, the candidate_costs, the runtime stats and
the execution certificate all reported DuckDB for a ClickHouse execution, and
a DuckDB certificate silently validated ClickHouse runs (and vice versa).

These tests pin:
  1. ``choose_plan_route`` writes the SQL candidate under its true backend
     name (``clickhouse_sql`` for a ClickHouse datasource).
  2. The certificate / runtime event identity chain stays consistent, and a
     DuckDB-issued certificate does NOT validate a ClickHouse runtime event.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from factor_engine.planner.logical_plan import PlanNode
from factor_engine.backend.plan_cost_router import choose_plan_route, record_plan_route
from factor_engine.runtime.production_execution_certificate import (
    ProductionExecutionCertificate,
    normalize_backend,
)
from workspace_paths import quant_projects_root


@pytest.fixture(autouse=True)
def _path():
    root = str(quant_projects_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    yield


def _col(name: str) -> PlanNode:
    return PlanNode(op="column", attrs={"name": name})


def _sql_plan() -> PlanNode:
    return PlanNode(op="ts_mean", inputs=[_col("close")], attrs={"window": 5})


class _CHCapabilities:
    engine_kind = "clickhouse"
    dialect = "clickhouse"
    supports_sql_pushdown = True


class _CHSource:
    """Data source whose capabilities declare ClickHouse."""

    table = "panel_daily"
    timestamp_column = "trade_date"
    instrument_column = "instrument"
    start_date = "2024-01-01"
    end_date = "2024-01-31"
    instrument_filter = ["A", "B"]
    capabilities = _CHCapabilities()


class _DuckCapabilities:
    engine_kind = "duckdb"
    dialect = "duckdb"
    supports_sql_pushdown = True


class _DuckSource:
    table = "panel_daily"
    timestamp_column = "trade_date"
    instrument_column = "instrument"
    start_date = "2024-01-01"
    end_date = "2024-01-31"
    instrument_filter = ["A", "B"]
    capabilities = _DuckCapabilities()


def _ctx(source) -> MagicMock:
    ctx = MagicMock()
    ctx.run_mode = "research"
    ctx.data_source = source
    ctx.runtime_stats = {}
    ctx.perf = None
    ctx.scan_shape = None
    ctx.scan_cost = None
    ctx.downstream_duckdb_fused = False
    return ctx


def test_clickhouse_route_candidate_key_is_clickhouse_sql():
    """The SQL candidate for a ClickHouse datasource must be keyed (and routed)
    as ``clickhouse_sql`` — never ``duckdb_sql``."""
    ctx = _ctx(_CHSource())
    route = choose_plan_route(_sql_plan(), ctx)

    cand = dict(route.candidate_costs)
    # Exactly one SQL candidate may exist, under the datasource's own name.
    sql_cands = [k for k in cand if k in {"duckdb_sql", "clickhouse_sql"}]
    assert len(sql_cands) == 1, f"expected exactly one SQL candidate, got {sql_cands}"
    assert sql_cands[0] == "clickhouse_sql", (
        f"ClickHouse datasource must key its SQL candidate as clickhouse_sql, got {sql_cands[0]}"
    )
    assert "duckdb_sql" not in cand
    if route.backend in {"duckdb_sql", "clickhouse_sql"}:
        assert route.backend == "clickhouse_sql"
        assert cand["clickhouse_sql"] == pytest.approx(route.estimated_cost)

    # Runtime event recording keeps the same identity.
    record_plan_route(ctx, route)
    stats = ctx.runtime_stats["plan_backend_route"]
    assert "duckdb_sql" not in dict(stats["candidate_costs"])
    cert = ctx.runtime_stats.get("production_execution_certificate")
    if cert is not None and route.backend == "clickhouse_sql":
        assert cert["requested_backend"] == "clickhouse_sql"
        assert cert["resolved_dialect"] == "clickhouse"
        assert cert["datasource_identity"] == "clickhouse"

    # A DuckDB datasource must keep its own name (no regression the other way).
    duck_route = choose_plan_route(_sql_plan(), _ctx(_DuckSource()))
    assert "duckdb_sql" in dict(duck_route.candidate_costs)
    assert "clickhouse_sql" not in dict(duck_route.candidate_costs)


def test_duckdb_certificate_does_not_validate_clickhouse_event():
    """A DuckDB-issued certificate must fail a ClickHouse runtime event even
    though ``normalize_backend`` maps both into the ``duckdb_sql`` executor
    family."""
    # normalize_backend family mapping itself is unchanged (executor ABI).
    assert normalize_backend("clickhouse_sql") == "duckdb_sql"
    assert normalize_backend("duckdb_sql") == "duckdb_sql"

    def _cert(dialect: str) -> ProductionExecutionCertificate:
        return ProductionExecutionCertificate.build(
            structural_hash="s" * 8,
            bound_ops=["ts_mean"],
            backend_eligibility=["duckdb_sql"],
            output_shape_hash="o" * 8,
            requested_backend="duckdb_sql" if dialect == "duckdb" else "clickhouse_sql",
            resolved_dialect=dialect,
            datasource_identity=dialect,
        )

    duck_cert = _cert("duckdb")
    ch_cert = _cert("clickhouse")

    duck_event = {"backend": "duckdb_sql", "execution_kind": "thread", "no_fallback": True, "dialect": "duckdb"}
    ch_event = {"backend": "duckdb_sql", "execution_kind": "thread", "no_fallback": True, "dialect": "clickhouse"}

    # Same-dialect events validate.
    assert duck_cert.validate(duck_event) is True
    assert ch_cert.validate(ch_event) is True
    # Cross-dialect validation fails closed.
    assert duck_cert.validate(ch_event) is False, (
        "DuckDB certificate must not validate a ClickHouse execution"
    )
    assert ch_cert.validate(duck_event) is False, (
        "ClickHouse certificate must not validate a DuckDB execution"
    )
    # Events without a dialect remain backward-compatible (no new rejection).
    legacy_event = {"backend": "duckdb_sql", "execution_kind": "thread", "no_fallback": True}
    assert duck_cert.validate(legacy_event) is True

    # Router-built certificates carry the datasource dialect through.
    from factor_engine.backend.plan_cost_router import _build_execution_certificate

    ch_router_cert = _build_execution_certificate(
        _sql_plan(),
        ("ts_mean",),
        {"clickhouse_sql": 10.0},
        chosen_backend="clickhouse_sql",
        rows=1000,
        occurrence_count=1,
        data_kind="clickhouse",
    )
    if ch_router_cert is not None:
        assert ch_router_cert.resolved_dialect == "clickhouse"
        assert ch_router_cert.validate(ch_event) is True
        assert ch_router_cert.validate(duck_event) is False
