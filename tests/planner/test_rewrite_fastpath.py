# -*- coding: utf-8
"""Fastpath rewrite rules。"""
from __future__ import annotations

from planner.logical_plan import PlanNode
from planner.rewrite_fastpath import rewrite_plan_for_fastpath


def test_divide_preserves_semantics_by_default():
    plan = PlanNode(
        op="divide",
        inputs=[
            PlanNode(op="column", attrs={"name": "close"}, inputs=[]),
            PlanNode(op="column", attrs={"name": "volume"}, inputs=[]),
        ],
        attrs={},
    )
    out = rewrite_plan_for_fastpath(plan)
    assert out.op == "divide"


def test_divide_can_opt_into_protected_div():
    plan = PlanNode(
        op="divide",
        inputs=[
            PlanNode(op="column", attrs={"name": "close"}, inputs=[]),
            PlanNode(op="column", attrs={"name": "volume"}, inputs=[]),
        ],
        attrs={},
    )
    out = rewrite_plan_for_fastpath(plan, allow_semantic_rewrites=True)
    assert out.op == "protected_div"


def test_ts_zscore_pattern_rewrites():
    col = PlanNode(op="column", attrs={"name": "close"}, inputs=[])
    mean = PlanNode(op="ts_mean", attrs={"d": 5}, inputs=[col])
    std = PlanNode(op="ts_std", attrs={"d": 5}, inputs=[col])
    num = PlanNode(op="subtract", inputs=[col, mean], attrs={})
    plan = PlanNode(op="divide", inputs=[num, std], attrs={})
    out = rewrite_plan_for_fastpath(plan)
    assert out.op == "ts_zscore"
    assert out.attrs.get("d") == 5


def test_subtract_ts_mean_does_not_rewrite_to_ts_zscore():
    """仅 demean 不等于 ts_zscore（缺除以 std）。"""
    col = PlanNode(op="column", attrs={"name": "close"}, inputs=[])
    mean = PlanNode(op="ts_mean", attrs={"d": 5}, inputs=[col])
    plan = PlanNode(op="subtract", inputs=[col, mean], attrs={})
    out = rewrite_plan_for_fastpath(plan)
    assert out.op == "subtract"


def test_log_divide_delay_rewrites_to_log_returns():
    col = PlanNode(op="column", attrs={"name": "close"}, inputs=[])
    delay = PlanNode(op="ts_delay", attrs={"d": 1}, inputs=[col])
    div = PlanNode(op="divide", inputs=[col, delay], attrs={})
    plan = PlanNode(op="log", inputs=[div], attrs={})
    out = rewrite_plan_for_fastpath(plan)
    assert out.op == "ts_log_return"
    assert out.attrs.get("d") == 1


def test_group_mean_subtract_preserves_semantics_by_default():
    col = PlanNode(op="column", attrs={"name": "close"}, inputs=[])
    grp = PlanNode(op="column", attrs={"name": "sector"}, inputs=[])
    gm = PlanNode(op="group_mean", inputs=[col, grp], attrs={})
    plan = PlanNode(op="subtract", inputs=[col, gm], attrs={})
    out = rewrite_plan_for_fastpath(plan)
    assert out.op == "subtract"


def test_group_mean_subtract_can_opt_into_neutralize():
    col = PlanNode(op="column", attrs={"name": "close"}, inputs=[])
    grp = PlanNode(op="column", attrs={"name": "sector"}, inputs=[])
    gm = PlanNode(op="group_mean", inputs=[col, grp], attrs={})
    plan = PlanNode(op="subtract", inputs=[col, gm], attrs={})
    out = rewrite_plan_for_fastpath(plan, allow_semantic_rewrites=True)
    assert out.op == "group_neutralize"
