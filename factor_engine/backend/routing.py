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


def numba_enabled_for_op(op: str, *, window: int = 0, panel_rows: int = 0) -> bool:
    """``FACTOR_ENGINE_USE_NUMBA`` 且算子声明 ``supports_numba`` 时启用。"""
    import os

    if os.environ.get("FACTOR_ENGINE_USE_NUMBA", "").lower() not in ("1", "true", "yes"):
        return False
    cost = get_operator_cost(op)
    if not cost.supports_numba:
        return False
    tier = resolve_execution_tier(op, panel_rows=panel_rows, window=window)
    backend = select_operator_backend(cost, tier=tier)
    return backend == "numba" or (backend == "bottleneck" and cost.supports_numba)
