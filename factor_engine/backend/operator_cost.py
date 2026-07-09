"""算子代价模型：复杂度 / 内存 / 执行 Tier 路由。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorCost:
    complexity: str
    memory: str
    supports_incremental: bool
    tier: int  # 0=bottleneck/vectorized, 1=numba, 2=polars, 3=pandas
    supports_polars: bool = True
    supports_numba: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "complexity": self.complexity,
            "memory": self.memory,
            "supports_incremental": self.supports_incremental,
            "tier": self.tier,
            "supports_polars": self.supports_polars,
            "supports_numba": self.supports_numba,
        }


_DEFAULT = OperatorCost("O(N)", "medium", False, 3, False, False)

_COSTS: dict[str, OperatorCost] = {
    "col": OperatorCost("O(1)", "low", True, 0, True, False),
    "literal": OperatorCost("O(1)", "low", True, 0, True, False),
    "add": OperatorCost("O(N)", "low", True, 2, True, False),
    "subtract": OperatorCost("O(N)", "low", True, 2, True, False),
    "multiply": OperatorCost("O(N)", "low", True, 2, True, False),
    "divide": OperatorCost("O(N)", "low", True, 2, True, False),
    "rank": OperatorCost("O(N log N)", "medium", False, 2, True, False),
    "zscore": OperatorCost("O(N)", "medium", False, 2, True, False),
    "ts_mean": OperatorCost("O(N)", "low", True, 0, True, True),
    "ts_sum": OperatorCost("O(N)", "low", True, 0, True, True),
    "ts_std": OperatorCost("O(N)", "medium", True, 0, True, True),
    "ts_std_dev": OperatorCost("O(N)", "medium", True, 0, True, True),
    "ts_min": OperatorCost("O(N)", "low", True, 0, True, True),
    "ts_max": OperatorCost("O(N)", "low", True, 0, True, True),
    "ts_delta": OperatorCost("O(N)", "low", True, 2, True, False),
    "delay": OperatorCost("O(N)", "low", True, 0, True, False),
    "ts_delay": OperatorCost("O(N)", "low", True, 0, True, False),
    "ts_rank": OperatorCost("O(NW log W)", "high", False, 1, False, True),
    "ts_corr": OperatorCost("O(NW)", "high", False, 1, True, True),
    "ts_correlation": OperatorCost("O(NW)", "high", False, 1, True, True),
    "neutralize": OperatorCost("O(NK^2)", "high", False, 3, False, False),
    "group_rank": OperatorCost("O(N log N)", "high", False, 3, False, False),
}


def get_operator_cost(op: str) -> OperatorCost:
    return _COSTS.get(str(op), _DEFAULT)


def estimate_plan_cost(plan: object) -> dict[str, object]:
    """逻辑计划子树代价摘要（节点数 + 最大 tier）。"""
    max_tier = 0
    expensive: list[str] = []
    node_count = 0

    def walk(node: object) -> None:
        nonlocal max_tier, node_count
        node_count += 1
        op = str(getattr(node, "op", "") or "")
        if op:
            cost = get_operator_cost(op)
            max_tier = max(max_tier, cost.tier)
            if cost.tier >= 3 or cost.memory == "high":
                expensive.append(op)
        for child in getattr(node, "inputs", []) or []:
            walk(child)

    walk(plan)
    return {
        "node_count": node_count,
        "max_tier": max_tier,
        "expensive_ops": sorted(set(expensive)),
    }
