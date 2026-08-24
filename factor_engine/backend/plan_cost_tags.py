# -*- coding: utf-8
"""Plan 级算子成本标签扩展（requires_sort/shuffle/join/stateful）。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlanCostTags:
    complexity: str
    requires_sort: bool = False
    requires_shuffle: bool = False
    requires_join: bool = False
    stateful: bool = False
    memory: str = "medium"


PLAN_COST_TAGS: dict[str, PlanCostTags] = {
    "rank": PlanCostTags("O(N log N)", requires_sort=True),
    "rank_pct": PlanCostTags("O(N log N)", requires_sort=True),
    "group_rank": PlanCostTags("O(N log N)", requires_sort=True, requires_shuffle=True),
    "ts_mean": PlanCostTags("O(NW)", requires_sort=True),
    "ts_quantile": PlanCostTags("O(NW log W)", requires_sort=True),
    "ts_corr": PlanCostTags("O(NW)", requires_sort=True),
    "add": PlanCostTags("O(N)"),
    "cum_sum": PlanCostTags("O(N)", requires_sort=True, stateful=True),
    "ts_ema": PlanCostTags("O(N)", requires_sort=True, stateful=True),
}


def plan_cost_tags_for(canon: str) -> PlanCostTags:
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return PLAN_COST_TAGS.get(name, PlanCostTags("O(N)"))
