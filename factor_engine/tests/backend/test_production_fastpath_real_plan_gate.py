# -*- coding: utf-8
"""真实 PlanNode production fastpath gate 测试。"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def _loaded():
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def test_real_plan_gate_accepts_composite(_loaded):
    from backend.production_fastpath_gate import check_production_fastpath_formula_ops

    result = check_production_fastpath_formula_ops(
        "add(ts_mean(col('close'), 5), rank(col('close')))",
        use_real_plan=True,
    )
    assert result.ok, result.violations


def test_real_plan_gate_rejects_deferred(_loaded):
    from backend.production_fastpath_gate import check_production_fastpath_formula_ops

    result = check_production_fastpath_formula_ops(
        "ts_decay_linear(col('close'), 5)",
        use_real_plan=True,
    )
    assert not result.ok
    assert any("ts_decay_linear" in v or "python_rolling" in v or "deferred" in v for v in result.violations)


def test_minimal_plan_hint_differs_from_real_plan(_loaded):
    from backend.production_fastpath_gate import check_production_fastpath_formula_ops

    formula = "ts_decay_linear(col('close'), 5)"
    real = check_production_fastpath_formula_ops(formula, use_real_plan=True)
    hint = check_production_fastpath_formula_ops(formula, use_real_plan=False)
    assert not real.ok
    assert not hint.ok
