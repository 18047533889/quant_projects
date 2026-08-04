# -*- coding: utf-8
"""第一阶段冻结 + 认证基础设施测试。"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def _loaded():
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def test_phase1_scope_loaded(_loaded):
    from backend.phase1_scope import phase1_summary

    s = phase1_summary()
    assert s["primitive_count"] >= 80
    assert s["composite_count"] >= 16
    assert s["primitive_certified_dual"] >= 13
    assert s["composite_full_parity_count"] >= 16
    # Recipe migrations are governed by recipe evidence, not retained as
    # production primitive/composite policy rows; a small number of
    # composites remain certified directly until their recipe migration.
    assert s["composite_production_certified_count"] <= 1


def test_phase1_rank_six_way_certified(_loaded):
    from backend.phase1_scope import get_phase1_entry, phase1_production_certified

    assert phase1_production_certified("rank")
    assert phase1_production_certified("ts_mean")
    assert get_phase1_entry("rank") is not None
    assert get_phase1_entry("ts_mean") is not None


def test_fillna_bfill_forbidden_capability(_loaded):
    from backend.operator_call_capability import CapabilityLevel, check_operator_call_capability
    from planner.logical_plan import PlanNode

    plan = PlanNode(
        op="fillna",
        inputs=[
            PlanNode(op="column", attrs={"name": "close"}, inputs=[]),
            PlanNode(op="literal", attrs={"value": "bfill"}, inputs=[]),
        ],
    )
    r = check_operator_call_capability("fillna", node=plan, production=False)
    assert r.level == CapabilityLevel.FORBIDDEN


def test_fillna_ffill_redirect(_loaded):
    from backend.operator_call_capability import check_operator_call_capability
    from planner.logical_plan import PlanNode

    plan = PlanNode(
        op="fillna",
        inputs=[
            PlanNode(op="column", attrs={"name": "close"}, inputs=[]),
            PlanNode(op="literal", attrs={"value": "ffill"}, inputs=[]),
        ],
    )
    r = check_operator_call_capability("fillna", node=plan, production=False)
    assert r.redirect_canonical == "ffill"


def test_window_spec_rejects_float_window(_loaded):
    from backend.plan_params import PlanParamError
    from backend.window_spec import WindowSpec
    from planner.logical_plan import PlanNode

    node = PlanNode(
        op="ts_mean",
        inputs=[
            PlanNode(op="column", attrs={"name": "close"}, inputs=[]),
            PlanNode(op="literal", attrs={"value": 2.5}, inputs=[]),
        ],
    )
    with pytest.raises(PlanParamError):
        WindowSpec.from_plan_node(node)


def test_explain_factor_rank(_loaded):
    from api.factor_explain import explain_factor

    exp = explain_factor("rank(col('close'))", mode="production")
    assert "rank" in exp.operators
    assert exp.dual_backend_ok


def test_validate_factor_dual_backend(_loaded):
    from api.factor_explain import validate_factor

    ok, exp = validate_factor("rank(col('close'))", target="dual_backend_production")
    assert ok
    assert not exp.violations or exp.dual_backend_ok


def test_certify_operator_ts_mean_dry_run(_loaded):
    from backend.operator_certification import run_certification

    report = run_certification("ts_mean", write_evidence=False, refresh_manifest=False)
    assert any(s.passed for s in report.stages)
