# -*- coding: utf-8
"""Production policy 绕过与 composite reference 语义测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from backend.production_fastpath_gate import (
    check_original_operator_policy,
    check_production_fastpath_formula_ops,
)
from cleaned_operators import load_all
from planner.logical_plan import PlanNode
from planner.optimizer import Optimizer


@pytest.fixture(scope="module")
def _loaded():
    load_all()
    from backend.sql_pushdown.sql_registry import register_sql_backends

    register_sql_backends()


def _col(name: str = "close") -> PlanNode:
    return PlanNode(op="column", attrs={"name": name}, inputs=[])


def test_policy_gate_rejects_current_ratio(_loaded):
    plan = PlanNode(
        op="current_ratio",
        inputs=[_col("assets"), _col("liabilities")],
        attrs={},
    )
    folded = Optimizer()._fold_literals(plan)
    violations = check_original_operator_policy(folded)
    assert any("current_ratio" in v for v in violations)
    assert any("pending" in v or "denied" in v for v in violations)


def test_policy_gate_rejects_micro_spread(_loaded):
    plan = PlanNode(
        op="micro_spread",
        inputs=[_col("high"), _col("low"), _col("close")],
        attrs={},
    )
    violations = check_original_operator_policy(Optimizer()._fold_literals(plan))
    assert any("micro_spread" in v for v in violations)


def test_formula_gate_rejects_current_ratio_even_if_lowered(_loaded):
    result = check_production_fastpath_formula_ops(
        "current_ratio(col('assets'), col('liabilities'))",
        use_real_plan=True,
    )
    assert not result.ok
    assert any("current_ratio" in v for v in result.violations)


def test_formula_gate_rejects_mom_pending_policy(_loaded):
    """MOM 有 lowering 但 production_policy=pending，不得通过 production gate。"""
    result = check_production_fastpath_formula_ops(
        "MOM(col('close'), 5)",
        use_real_plan=True,
        check_full_plan=False,
    )
    assert not result.ok
    assert any("MOM" in v and "pending" in v for v in result.violations)


def test_formula_gate_accepts_ts_mean_primitive(_loaded):
    result = check_production_fastpath_formula_ops(
        "ts_mean(col('close'), 5)",
        use_real_plan=True,
        check_full_plan=True,
    )
    assert result.ok, result.violations


def test_policy_gate_rejects_bfill_permanently_forbidden(_loaded):
    from planner.logical_plan import PlanNode
    from planner.optimizer import Optimizer

    plan = PlanNode(op="bfill", inputs=[PlanNode(op="column", attrs={"name": "close"}, inputs=[])], attrs={"limit": 1})
    violations = check_original_operator_policy(Optimizer()._fold_literals(plan))
    assert any("bfill" in v for v in violations)


def test_formula_gate_rejects_bfill(_loaded):
    result = check_production_fastpath_formula_ops("bfill(col('close'))", use_real_plan=True)
    assert not result.ok


def test_safe_div_null_zero_denominator_is_null(_loaded):
    from cleaned_operators.registry import OperatorRegistry

    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-01"), "A"), (pd.Timestamp("2024-01-02"), "A")],
        names=["timestamp", "instrument"],
    )
    num = pd.DataFrame({"A": [1.0, 2.0]}, index=idx.get_level_values(0).unique())
    den = pd.DataFrame({"A": [0.0, 1.0]}, index=idx.get_level_values(0).unique())
    op = OperatorRegistry.get("safe_div_null")
    out = op.calculate(num, den)
    assert pd.isna(out.iloc[0, 0])
    assert out.iloc[1, 0] == 2.0


def test_obv_first_row_zero_reference(_loaded):
    from tests.backend_parity.composite_reference_helpers import (
        CompositeReferenceCase,
        reference_pandas_calculate,
        series_to_panel,
    )

    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-01"), "A"),
            (pd.Timestamp("2024-01-02"), "A"),
        ],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([10.0, 11.0], index=idx)
    volume = pd.Series([100.0, 110.0], index=idx)
    case = CompositeReferenceCase("OBV", ("close", "volume"))
    panels = {
        "close": series_to_panel(close),
        "price": series_to_panel(close),
        "volume": series_to_panel(volume),
    }
    ref = reference_pandas_calculate(case, panels)
    first = ref.stack(future_stack=True).loc[(pd.Timestamp("2024-01-01"), "A")]
    assert first == 0.0


def test_policy_gate_rejects_composite_allowed_without_evidence(_loaded, monkeypatch):
    """policy=allowed 但 composite_production_safe=False 时 gate 必须拒绝。"""
    from planner.logical_plan import PlanNode
    from planner.optimizer import Optimizer

    plan = PlanNode(
        op="MOM",
        inputs=[PlanNode(op="column", attrs={"name": "close"}, inputs=[])],
        attrs={"window": 5},
    )
    folded = Optimizer()._fold_literals(plan)

    from cleaned_operators import operator_spec as ospec

    real_infer = ospec.infer_production_policy
    real_build = ospec.build_operator_spec

    def _fake_mom_allowed(canon: str) -> str:
        if canon == "MOM":
            return "allowed"
        return real_infer(canon)

    def _fake_mom_spec(canon: str, **kwargs):
        spec = real_build(canon, **kwargs)
        if spec is None or spec.canonical != "MOM":
            return spec
        from dataclasses import replace

        return replace(spec, allow_in_production=True, production_policy="allowed")

    monkeypatch.setattr(ospec, "infer_production_policy", _fake_mom_allowed)
    monkeypatch.setattr(ospec, "build_operator_spec", _fake_mom_spec)
    violations = check_original_operator_policy(folded)
    assert any("MOM" in v and "production evidence incomplete" in v for v in violations)

