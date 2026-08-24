# -*- coding: utf-8 -*-
"""Review-10 cache scope + production precompiled-plan gate (R10-P0-001..003).

* R10-P0-001 — the execution cache namespace is built AFTER compile and binds
  the secondary SourceRef dependency hash + execution scope, so two factors
  sharing an anchor source but differing in secondary deps / execution
  semantics get different cache scopes.
* R10-P0-002 — ``compute_data_scope`` includes the DataAccess options that
  change the actual read result (``read_mode`` / ``semantic_filters`` /
  ``snapshot_now_only`` / mining gates).
* R10-P0-003 — a pre-compiled plan handed to production ``run`` is re-run
  through the production plan gates (a research-only operator is rejected even
  when the plan was precompiled).
"""
from __future__ import annotations

import pytest

from factor_engine.ir.nodes import IRNode
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.runtime.engine import _assert_production_plan_gates
from factor_engine.runtime.production_policy import ProductionPolicyViolation
from factor_engine.storage.data_scope import (
    DataExecutionScope,
    compute_data_scope,
    compute_execution_cache_scope,
)


class _FakeSource:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# R10-P0-002: data_scope covers read-affecting DataAccess options
# ---------------------------------------------------------------------------

def test_data_scope_differs_by_read_mode():
    a = _FakeSource(dataset="ashare_daily", read_mode="panel", snapshot_now_only=False)
    b = _FakeSource(dataset="ashare_daily", read_mode="event", snapshot_now_only=False)
    assert compute_data_scope(a) != compute_data_scope(b)


def test_data_scope_differs_by_semantic_filters():
    a = _FakeSource(dataset="ashare_daily", read_mode="panel", semantic_filters={}, snapshot_now_only=False)
    b = _FakeSource(
        dataset="ashare_daily", read_mode="panel",
        semantic_filters={"sector": ["BANK"]}, snapshot_now_only=False,
    )
    assert compute_data_scope(a) != compute_data_scope(b)


def test_data_scope_differs_by_snapshot_now_only():
    a = _FakeSource(dataset="ashare_daily", read_mode="panel", snapshot_now_only=False)
    b = _FakeSource(dataset="ashare_daily", read_mode="panel", snapshot_now_only=True)
    assert compute_data_scope(a) != compute_data_scope(b)


def test_data_scope_differs_by_mining_gates():
    a = _FakeSource(dataset="ashare_daily", enforce_mining_gate=False, mining_coverage_threshold=None)
    b = _FakeSource(dataset="ashare_daily", enforce_mining_gate=True, mining_coverage_threshold=0.7)
    assert compute_data_scope(a) != compute_data_scope(b)


# ---------------------------------------------------------------------------
# R10-P0-001: execution cache namespace binds secondary deps + execution scope
# ---------------------------------------------------------------------------

def test_execution_cache_scope_differs_by_secondary_dependencies():
    ds = _FakeSource(dataset="ashare_daily")
    s1 = compute_execution_cache_scope(ds, source_dependencies=("StockIncome.net_profit",))
    s2 = compute_execution_cache_scope(ds, source_dependencies=("StockBalance.total_asset",))
    assert s1 != s2


def test_execution_cache_scope_differs_by_execution_semantics():
    ds = _FakeSource(dataset="ashare_daily")
    s1 = compute_execution_cache_scope(
        ds, execution=DataExecutionScope(frequency="1d"), source_dependencies=("a",)
    )
    s2 = compute_execution_cache_scope(
        ds,
        execution=DataExecutionScope(frequency="1d", decision_time_policy="eod"),
        source_dependencies=("a",),
    )
    assert s1 != s2
    # the namespace is a stable 24-hex prefix (same input -> same key)
    s3 = compute_execution_cache_scope(
        ds, execution=DataExecutionScope(frequency="1d"), source_dependencies=("a",)
    )
    assert s1 == s3
    assert len(s1) == 24


def test_execution_cache_scope_deterministic_structure():
    ns = compute_execution_cache_scope(
        _FakeSource(dataset="d"),
        execution=DataExecutionScope(frequency="1d"),
        source_dependencies=("a", "b"),
        evidence_version="ev1",
    )
    assert isinstance(ns, str) and len(ns) == 24


# ---------------------------------------------------------------------------
# R10-P0-003: precompiled plan re-certified in production
# ---------------------------------------------------------------------------

def _col(name: str) -> PlanNode:
    return PlanNode(op="column", attrs={"name": name}, inputs=[])


class _FakeAnalysis:
    ir = IRNode(op="column", attrs={"name": "x"})


def test_production_gate_rejects_research_only_precompiled_plan():
    plan = PlanNode(
        op="panel_rolling_pca_loading",  # experimental -> research-only
        attrs={"window": 60, "component": 0},
        inputs=[_col("ret")],
    )
    with pytest.raises(ProductionPolicyViolation, match="production"):
        _assert_production_plan_gates(
            plan, _FakeAnalysis(), run_mode="production", context="run:test"
        )


def test_production_gate_accepts_clean_plan():
    # a plan with no operators (just a column) has nothing to reject
    plan = _col("close")
    _assert_production_plan_gates(
        plan, _FakeAnalysis(), run_mode="production", context="run:test"
    )


def test_production_gate_noop_in_research_mode():
    # research mode never enforces the production operator gate
    plan = PlanNode(
        op="panel_rolling_pca_loading",
        attrs={"window": 60, "component": 0},
        inputs=[_col("ret")],
    )
    _assert_production_plan_gates(
        plan, _FakeAnalysis(), run_mode="research", context="run:test"
    )
