# -*- coding: utf-8
"""Composite lowering：高级算子 → 基础 DAG。"""
from __future__ import annotations

from planner.composite_lowering import (
    collect_plan_ops,
    composite_dual_backend_capable,
    lower_composite_operators,
    lowered_primitives,
)
from planner.logical_plan import PlanNode
from planner.optimizer import Optimizer


def _col(name: str = "close") -> PlanNode:
    return PlanNode(op="column", attrs={"name": name}, inputs=[])


def test_mom_lowers_to_ts_delta():
    plan = PlanNode(op="MOM", inputs=[_col()], attrs={"window": 5})
    out = lower_composite_operators(plan)
    assert out.op == "ts_delta"
    assert out.inputs[0].op == "column"


def test_roc_lowers_to_ts_pct_times_100():
    plan = PlanNode(op="ROC", inputs=[_col()], attrs={"window": 3})
    out = lower_composite_operators(plan)
    assert out.op == "multiply"
    assert out.inputs[0].op == "ts_pct"
    assert out.inputs[1].op == "literal"
    assert out.inputs[1].attrs["value"] == 100.0


def test_bollinger_upper_lowers_to_mean_std():
    plan = PlanNode(
        op="BollingerUpper",
        inputs=[_col(), PlanNode(op="literal", attrs={"value": 20}, inputs=[]), PlanNode(op="literal", attrs={"value": 2.0}, inputs=[])],
        attrs={},
    )
    out = lower_composite_operators(plan)
    assert out.op == "add"
    assert out.inputs[0].op == "ts_mean"
    assert out.inputs[1].op == "multiply"
    assert out.inputs[1].inputs[0].op == "literal"
    assert out.inputs[1].inputs[1].op == "ts_std"


def test_bollinger_lower_lowers_to_subtract():
    plan = PlanNode(op="BollingerLower", inputs=[_col()], attrs={"window": 10, "std_dev": 1.5})
    out = lower_composite_operators(plan)
    assert out.op == "subtract"
    assert out.inputs[0].op == "ts_mean"


def test_dpo_lowers_with_shifted_mean():
    plan = PlanNode(
        op="DPO",
        inputs=[_col(), PlanNode(op="literal", attrs={"value": 20}, inputs=[])],
        attrs={},
    )
    out = lower_composite_operators(plan)
    assert out.op == "subtract"
    delayed = out.inputs[1]
    assert delayed.op == "ts_delay"
    assert delayed.inputs[1].op == "literal"
    assert delayed.inputs[1].attrs["value"] == 11


def test_optimizer_applies_composite_before_rewrite():
    col = _col()
    plan = PlanNode(op="MOM", inputs=[col], attrs={"window": 2})
    out = Optimizer().optimize(plan)
    assert out.op == "ts_delta"


def test_lowered_primitives_for_mom():
    prims = lowered_primitives("MOM")
    assert prims is not None
    assert prims == ("ts_delta",)


def test_composite_dual_backend_capable_mom():
    from cleaned_operators import load_all

    load_all()
    assert composite_dual_backend_capable("MOM") is False


def test_collect_plan_ops_after_lowering():
    plan = PlanNode(op="ROC", inputs=[_col()], attrs={"window": 1})
    lowered = lower_composite_operators(plan)
    ops = collect_plan_ops(lowered)
    assert "ts_pct" in ops
    assert "multiply" in ops
    assert "ROC" not in ops


def test_obv_lowers_with_coalesce_on_sign():
    plan = PlanNode(
        op="OBV",
        inputs=[_col("close"), _col("volume")],
        attrs={},
    )
    out = lower_composite_operators(plan)
    assert out.op == "cum_sum"
    inner = out.inputs[0]
    assert inner.op == "multiply"
    assert inner.inputs[0].op == "fillna_const"


def test_obv_lowers_to_cum_sum_chain():
    plan = PlanNode(
        op="OBV",
        inputs=[_col("close"), _col("volume")],
        attrs={},
    )
    out = lower_composite_operators(plan)
    assert out.op == "cum_sum"
    inner = out.inputs[0]
    assert inner.op == "multiply"
    assert inner.inputs[0].op == "fillna_const"
    assert inner.inputs[0].inputs[0].op == "sign"
    assert inner.inputs[0].inputs[0].inputs[0].op == "ts_delta"


def test_stochastic_k_lowers_to_min_max_div():
    plan = PlanNode(
        op="StochasticK",
        inputs=[_col("high"), _col("low"), _col("close")],
        attrs={"window": 14},
    )
    out = lower_composite_operators(plan)
    assert out.op == "multiply"
    assert out.inputs[0].op == "safe_div_null"


def test_stochastic_d_lowers_to_mean_of_k():
    plan = PlanNode(
        op="StochasticD",
        inputs=[_col("high"), _col("low"), _col("close")],
        attrs={"window": 14},
    )
    out = lower_composite_operators(plan)
    assert out.op == "ts_mean"
    assert out.inputs[0].op == "multiply"


def test_bollinger_bands_lowers_to_ts_mean():
    plan = PlanNode(op="BollingerBands", inputs=[_col()], attrs={"window": 20})
    out = lower_composite_operators(plan)
    assert out.op == "ts_mean"


def test_operating_margin_lowers_to_protected_div():
    plan = PlanNode(
        op="operating_margin",
        inputs=[_col("oi"), _col("rev")],
        attrs={},
    )
    out = lower_composite_operators(plan)
    assert out.op == "safe_div_null"


def test_quick_ratio_lowers_to_subtract_and_div():
    plan = PlanNode(
        op="quick_ratio",
        inputs=[_col("ca"), _col("inv"), _col("cl")],
        attrs={},
    )
    out = lower_composite_operators(plan)
    assert out.op == "safe_div_null"
    assert out.inputs[0].op == "subtract"


def test_micro_spread_lowers_to_protected_div():
    plan = PlanNode(
        op="micro_spread",
        inputs=[_col("high"), _col("low"), _col("close")],
        attrs={},
    )
    out = lower_composite_operators(plan)
    assert out.op == "safe_div_null"
    assert out.inputs[0].op == "subtract"


def test_list_composite_lowerings_batch_one_size():
    from planner.composite_lowering import list_composite_lowerings

    names = list_composite_lowerings()
    assert len(names) >= 16
    for name in (
        "MOM",
        "ROC",
        "OBV",
        "StochasticK",
        "StochasticD",
        "BollingerBands",
        "operating_margin",
        "micro_spread",
    ):
        assert name in names


def test_composite_dual_backend_capable_obv():
    from cleaned_operators import load_all

    load_all()
    assert composite_dual_backend_capable("OBV") is False
