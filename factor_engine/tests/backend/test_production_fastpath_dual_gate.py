# -*- coding: utf-8
"""Dual-backend production gate（require_mode=all）。"""
from __future__ import annotations

import pytest

from backend.production_fastpath_gate import check_production_fastpath_plan_ops
from backend.sql_pushdown.plan_fixtures import minimal_plan


@pytest.fixture(scope="module")
def _loaded():
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def test_require_mode_all_blocks_single_backend_only(_loaded, monkeypatch):
    plan = minimal_plan("ts_mean")
    monkeypatch.setattr(
        "backend.production_fastpath_gate._duckdb_fastpath_ok",
        lambda _c: False,
    )
    monkeypatch.setattr(
        "backend.production_fastpath_gate._polars_native_fastpath_ok",
        lambda _c: True,
    )
    any_mode = check_production_fastpath_plan_ops(plan, require_mode="any", strict=True, check_full_plan=False)
    all_mode = check_production_fastpath_plan_ops(plan, require_mode="all", strict=True, check_full_plan=False)
    assert any_mode.ok
    assert not all_mode.ok


def test_dual_backend_env_forces_all_mode(_loaded, monkeypatch):
    plan = minimal_plan("ts_mean")
    monkeypatch.setenv("FACTOR_ENGINE_PRODUCTION_REQUIRE_DUAL_BACKEND", "1")
    monkeypatch.setattr(
        "backend.production_fastpath_gate._duckdb_fastpath_ok",
        lambda _c: True,
    )
    monkeypatch.setattr(
        "backend.production_fastpath_gate._polars_native_fastpath_ok",
        lambda _c: False,
    )
    result = check_production_fastpath_plan_ops(plan, require_mode="any", strict=True, check_full_plan=False)
    assert not result.ok


def test_bfill_blocked_causal_strict_gate(_loaded):
    from planner.logical_plan import PlanNode

    plan = PlanNode(op="bfill", inputs=[PlanNode(op="column", attrs={"name": "close"})])
    result = check_production_fastpath_plan_ops(plan, strict=True, check_full_plan=False)
    assert not result.ok
    assert any("bfill" in v for v in result.violations)
