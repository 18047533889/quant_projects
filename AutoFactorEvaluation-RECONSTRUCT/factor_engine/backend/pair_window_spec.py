# -*- coding: utf-8
"""二元 rolling 算子 WindowSpec（ts_corr/ts_cov/ts_beta/vwap）。"""
from __future__ import annotations

from dataclasses import dataclass

from backend.plan_params import window_spec_from_plan_node
from backend.window_spec import WindowSpec
from planner.logical_plan import PlanNode


@dataclass(frozen=True)
class PairWindowSpec:
    """Pairwise rolling 窗口契约。"""

    size: int
    min_periods: int = 2
    ddof: int = 1
    pairwise_null_policy: str = "exclude_either_null"

    @classmethod
    def from_plan_node(cls, node: PlanNode, *, default_size: int = 20) -> PairWindowSpec:
        base = window_spec_from_plan_node(node, default=default_size)
        mp = max(base.min_periods, 2) if base.min_periods else 2
        return cls(
            size=base.size,
            min_periods=mp,
            ddof=base.ddof,
        )

    @classmethod
    def from_window_spec(cls, spec: WindowSpec, *, min_pair: int = 2) -> PairWindowSpec:
        mp = max(spec.min_periods, min_pair)
        return cls(size=spec.size, min_periods=mp, ddof=spec.ddof)
