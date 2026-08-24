# -*- coding: utf-8
"""二元 rolling 算子 WindowSpec（ts_corr/ts_cov/ts_beta/vwap）。"""
from __future__ import annotations

from dataclasses import dataclass

from factor_engine.backend.plan_params import window_spec_from_plan_node
from factor_engine.backend.window_spec import WindowSpec
from factor_engine.planner.logical_plan import PlanNode


@dataclass(frozen=True)
class PairWindowSpec:
    """Pairwise rolling 窗口契约。"""

    size: int
    min_periods: int = 2
    ddof: int = 1
    pairwise_null_policy: str = "exclude_either_null"

    @classmethod
    def from_plan_node(
        cls,
        node: PlanNode,
        *,
        default_size: int = 20,
        default_min_periods: int = 2,
    ) -> PairWindowSpec:
        base = window_spec_from_plan_node(node, default=default_size)
        attrs = node.attrs or {}
        if attrs.get("min_periods") is not None:
            mp = max(int(base.min_periods), 2)
        else:
            # Caller omitted min_periods — use the op-specific production default
            # (ts_corr/ts_cov → 2; ts_beta → 5 / NEW-024), never silent mp=1.
            mp = max(int(default_min_periods), 2)
        return cls(
            size=base.size,
            min_periods=mp,
            ddof=base.ddof,
        )

    @classmethod
    def from_window_spec(cls, spec: WindowSpec, *, min_pair: int = 2) -> PairWindowSpec:
        mp = max(spec.min_periods, min_pair)
        return cls(size=spec.size, min_periods=mp, ddof=spec.ddof)
