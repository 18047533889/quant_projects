# -*- coding: utf-8
"""Plan 节点参数级 production 策略（fillna method 分支）。"""
from __future__ import annotations

import pytest

from factor_engine.backend.operator_call_capability import CapabilityLevel, check_operator_call_capability
from factor_engine.backend.operator_call_policy import (
    check_plan_operator_calls,
    fillna_call_production_safe,
    is_operator_call_production_safe,
)
from factor_engine.backend.production_fastpath_gate import check_production_fastpath_formula_ops
from factor_engine.cleaned_operators import load_all
from factor_engine.planner.logical_plan import PlanNode


@pytest.fixture(scope="module")
def _loaded():
    load_all()


def _fillna_plan(method) -> PlanNode:
    return PlanNode(
        op="fillna",
        inputs=[
            PlanNode(op="column", attrs={"name": "close"}, inputs=[]),
            PlanNode(op="literal", attrs={"value": method}, inputs=[]),
        ],
        attrs={},
    )


def test_fillna_bfill_rejected(_loaded):
    plan = _fillna_plan("bfill")
    ok, msg = fillna_call_production_safe(plan)
    assert not ok
    assert "bfill" in msg
    assert not is_operator_call_production_safe(plan)


def test_fillna_ffill_redirects_to_ffill(_loaded):
    r = check_operator_call_capability("fillna", node=_fillna_plan("ffill"), production=False)
    assert r.redirect_canonical == "ffill"


def test_fillna_const_numeric_allowed(_loaded):
    plan = _fillna_plan(0.0)
    assert fillna_call_production_safe(plan)[0]
    assert is_operator_call_production_safe(plan)


def test_fillna_production_mode_requires_fillna_const(_loaded):
    from factor_engine.backend.operator_call_capability import CapabilityLevel, check_operator_call_capability
    from factor_engine.planner.logical_plan import PlanNode

    plan = PlanNode(
        op="fillna",
        inputs=[
            PlanNode(op="column", attrs={"name": "close"}, inputs=[]),
            PlanNode(op="literal", attrs={"value": 0.0}, inputs=[]),
        ],
    )
    r = check_operator_call_capability("fillna", node=plan, production=True)
    assert r.level == CapabilityLevel.RESEARCH
    assert r.redirect_canonical == "fillna_const"


def test_formula_gate_rejects_fillna_bfill(_loaded):
    result = check_production_fastpath_formula_ops(
        "fillna(col('close'), 'bfill')",
        use_real_plan=True,
        check_full_plan=False,
        mode="research",
    )
    assert not result.ok
    assert any("bfill" in v or "fillna" in v for v in result.violations)
