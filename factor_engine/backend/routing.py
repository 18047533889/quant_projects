"""算子执行 Tier 路由（bottleneck → numba → polars → pandas）。"""

from __future__ import annotations

from backend.operator_cost import OperatorCost, get_operator_cost


def resolve_execution_tier(
    op: str,
    *,
    panel_rows: int = 0,
    window: int = 0,
) -> int:
    """返回算子推荐执行 tier；大窗口 rolling 倾向 numba/pandas。

    参数:
        op: 算子名称。
        panel_rows: 面板行数估计，用于高内存算子升级 tier。
        window: 滚动窗口长度，大窗口时倾向 numba。

    返回:
        推荐 tier 整数（0=bottleneck, 1=numba, 2=polars, 3=pandas）。
    """
    cost = get_operator_cost(op)
    if window > 512 and cost.tier <= 1 and op.startswith("ts_"):
        return max(cost.tier, 1)
    if panel_rows > 5_000_000 and cost.memory == "high":
        return max(cost.tier, 2)
    return cost.tier


def select_operator_backend(cost: OperatorCost, *, tier: int | None = None) -> str:
    """tier → backend 名称（``bottleneck`` / ``numba`` / ``polars`` / ``pandas``）。

    参数:
        cost: 算子代价元数据。
        tier: 可选覆盖 tier；为 ``None`` 时使用 ``cost.tier``。

    返回:
        backend 名称字符串。
    """
    effective = tier if tier is not None else cost.tier
    if effective <= 0:
        return "bottleneck"
    if effective == 1:
        return "numba"
    if effective == 2 and cost.supports_polars:
        return "polars"
    return "pandas"


def numba_enabled_for_op(op: str, *, window: int = 0, panel_rows: int = 0) -> bool:
    """``FACTOR_ENGINE_USE_NUMBA`` 且算子声明 ``supports_numba`` 时启用。

    参数:
        op: 算子名称。
        window: 滚动窗口长度。
        panel_rows: 面板行数估计。

    返回:
        当前环境与算子特性下是否应启用 Numba 路径。
    """
    import os

    if os.environ.get("FACTOR_ENGINE_USE_NUMBA", "").lower() not in ("1", "true", "yes"):
        return False
    cost = get_operator_cost(op)
    if not cost.supports_numba:
        return False
    tier = resolve_execution_tier(op, panel_rows=panel_rows, window=window)
    backend = select_operator_backend(cost, tier=tier)
    return backend == "numba" or (backend == "bottleneck" and cost.supports_numba)
