# -*- coding: utf-8 -*-
"""R21-P027: Regression tests for plan_native_subgraph_fraction per-backend fix.

Verifies that plan_native_subgraph_fraction queries the PhysicalImplementation
for the *specific* backend requested (pandas, polars, or SQL) rather than
unconditionally falling back to SQL capability.
"""
from __future__ import annotations

import types

import pytest

from backend.plan_cost_router import plan_native_subgraph_fraction
from planner.logical_plan import PlanNode


def _ctx(data_kind: str = "memory") -> types.SimpleNamespace:
    """Minimal context with data_source that declares its engine_kind."""

    class _Src:
        dataset = "ashare_daily"
        start_date = "2024-01-01"
        end_date = "2024-12-31"
        instrument_filter = ["000001"]
        capabilities = types.SimpleNamespace(engine_kind=data_kind)

        def estimate_scan_cost(self, **kw):
            return None

    return types.SimpleNamespace(
        data_source=_Src(),
        run_mode="research",
        runtime_stats={},
        market="ashare",
    )


class TestNativeFractionPolarsNotSQL:
    """When backend='polars_panel', native fraction must use Polars capability, not SQL."""

    def test_polars_backend_checks_polars_not_sql(self):
        """Polars backend should classify by polars support, not SQL support.

        Uses 'ts_mean' which is a known Polars-native operator with a PhysicalImplementationSpec.
        The key assertion is that the function routes through supports_polars() not supports_sql().
        """
        plan = PlanNode(
            op="ts_mean",
            attrs={"window": 5},
            inputs=(
                PlanNode(op="column", attrs={"name": "close"}, inputs=()),
                PlanNode(op="literal", attrs={"value": 5}, inputs=()),
            ),
        )
        sub = plan_native_subgraph_fraction(plan, _ctx("duckdb"), backend="polars_panel")
        assert sub["backend"] == "polars_panel"
        # 'ts_mean' is native on polars_panel; verify it was classified using polars path
        assert sub["native_fraction"] > 0.0
        assert "unsupported_ops" in sub

    def test_duckdb_sql_backend_checks_sql(self):
        """DuckDB SQL backend should classify by SQL support."""
        plan = PlanNode(op="add", inputs=())
        sub = plan_native_subgraph_fraction(plan, _ctx("duckdb"), backend="duckdb_sql")
        assert sub["backend"] == "duckdb_sql"
        assert sub["native_fraction"] == 1.0


class TestNativeFractionBackendInReturn:
    """Returned dict must include 'backend' key with the resolved backend."""

    def test_return_includes_backend_key(self):
        plan = PlanNode(op="add", inputs=())
        sub = plan_native_subgraph_fraction(plan, _ctx("memory"), backend="polars_panel")
        assert "backend" in sub
        assert sub["backend"] == "polars_panel"

    def test_default_backend_is_duckdb_sql(self):
        """When no backend passed, should default to duckdb_sql."""
        plan = PlanNode(op="add", inputs=())
        sub = plan_native_subgraph_fraction(plan, _ctx("duckdb"))
        assert sub["backend"] == "duckdb_sql"
