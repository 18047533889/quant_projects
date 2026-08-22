# -*- coding: utf-8
"""ClickHouse / DuckDB capability 探测与 tier 降级。"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="legacy SQL P0 whitelist contract; certified DuckDB evidence is authoritative")


@pytest.fixture(scope="module")
def _loaded():
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def test_duckdb_downgrade_only_applies_to_production_safe(monkeypatch):
    from backend.sql_pushdown.duckdb_capabilities import (
        DuckdbCapabilityReport,
        downgrade_sql_canonicals,
    )
    from backend.sql_tiers import SQL_PRODUCTION_SAFE_CANONICALS

    report = DuckdbCapabilityReport(features={"quantile_cont_window": "unsupported"})
    down = downgrade_sql_canonicals(report)
    assert down <= SQL_PRODUCTION_SAFE_CANONICALS


def test_duckdb_downgrade_corr_when_in_production_safe(monkeypatch):
    from backend.sql_pushdown.duckdb_capabilities import (
        DuckdbCapabilityReport,
        downgrade_sql_canonicals,
    )
    from backend.sql_tiers import SQL_PRODUCTION_SAFE_CANONICALS

    report = DuckdbCapabilityReport(features={"corr_window": "unsupported"})
    down = downgrade_sql_canonicals(report)
    assert down <= SQL_PRODUCTION_SAFE_CANONICALS


def test_cs_mad_sql_emitter_ok(_loaded):
    from backend.operator_capability import _sql_emitter_ok

    assert _sql_emitter_ok("cs_mad")
    assert _sql_emitter_ok("cs_mad_zscore")


def test_fastpath_coverage_clickhouse_fields(_loaded):
    from backend.fastpath_coverage import build_fastpath_coverage_row

    row = build_fastpath_coverage_row("ts_mean")
    assert hasattr(row, "clickhouse_sql_production_safe")
    assert hasattr(row, "duckdb_sql_production_safe")


def test_clickhouse_capability_probe_without_client():
    from backend.sql_pushdown.clickhouse_capabilities import probe_clickhouse_capabilities

    report = probe_clickhouse_capabilities(None)
    assert report.features.get("quantile_cont_window") == "unsupported"


def test_p0_operators_allow_in_production_after_fastpath_align():
    from cleaned_operators import load_all
    from cleaned_operators.operator_spec import build_operator_spec
    from backend.production_fastpath_tiers import P0_PRODUCTION_FASTPATH_CANONICALS

    load_all()
    sample = ["vwap", "power", "c_mean", "is_nan", "log_returns", "volatility"]
    for canon in sample:
        assert canon in P0_PRODUCTION_FASTPATH_CANONICALS
        spec = build_operator_spec(canon)
        assert spec is not None, canon
        assert spec.allow_in_production, canon
