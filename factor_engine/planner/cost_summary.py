"""计划代价摘要：供 Optimizer / run_many 调度提示。"""

from __future__ import annotations

from typing import Any

from backend.operator_cost import estimate_plan_cost, get_operator_cost
from planner.logical_plan import PlanNode


def summarize_plan_cost(plan: PlanNode) -> dict[str, Any]:
    """单计划代价摘要。"""
    est = estimate_plan_cost(plan)
    return dict(est) if isinstance(est, dict) else {"estimate": est}


def summarize_plans(plans: dict[str, PlanNode]) -> dict[str, Any]:
    """多因子 plan 代价汇总。"""
    per_factor: dict[str, dict[str, Any]] = {}
    tier_hist: dict[int, int] = {}
    high_memory_ops: list[str] = []

    for name, plan in plans.items():
        summary = summarize_plan_cost(plan)
        per_factor[name] = summary
        for node in _walk(plan):
            cost = get_operator_cost(str(node.op))
            tier_hist[cost.tier] = tier_hist.get(cost.tier, 0) + 1
            if cost.memory == "high":
                high_memory_ops.append(str(node.op))

    return {
        "factors": per_factor,
        "tier_histogram": {str(k): v for k, v in sorted(tier_hist.items())},
        "high_memory_ops": sorted(set(high_memory_ops)),
        "factor_count": len(plans),
    }


def _walk(node: PlanNode):
    for child in node.inputs:
        yield from _walk(child)
    yield node
