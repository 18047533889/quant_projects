"""算子执行 Tier 路由（bottleneck → numba → polars → pandas）。"""

from __future__ import annotations

from backend.operator_cost import OperatorCost, get_operator_cost


def resolve_execution_tier(
    op: str,
    *,
    panel_rows: int = 0,
    window: int = 0,
) -> int:
    """返回算子推荐执行 tier；大窗口 rolling 倾向 numba/pandas。"""
    cost = get_operator_cost(op)
    if window > 512 and cost.tier <= 1 and op.startswith("ts_"):
        return max(cost.tier, 1)
    if panel_rows > 5_000_000 and cost.memory == "high":
        return max(cost.tier, 2)
    return cost.tier


def select_operator_backend(cost: OperatorCost, *, tier: int | None = None) -> str:
    """tier → backend 名称（``bottleneck`` / ``numba`` / ``polars`` / ``pandas``）。"""
    effective = tier if tier is not None else cost.tier
    if effective <= 0:
        return "bottleneck"
    if effective == 1:
        return "numba"
    if effective == 2 and cost.supports_polars:
        return "polars"
    return "pandas"
