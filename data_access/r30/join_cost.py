"""data_access.r30.join_cost —— JoinCostPlanner 启发式 join 计划（R30-P1-011）。

第一版**启发式**（不是 CBO/动态规划）：针对两路 join 给出行之有效的默认决策：

    1. small dimension first —— 按 ``byte_size`` 升序排 join 顺序（小维度驱动）；
    2. filter before join —— selectivity 低（< 1.0，即过滤会砍掉行）的一侧先过滤；
    3. minute aggregate before fundamental join —— left 为 minute、right 为
       季度/事件 等低频基本面时 pre_aggregate=True；
    4. build_side —— 取小的一侧（哈希 join build side）；
    5. materialize —— 两侧 min(byte_size) 低于阈值（256MB）时物化；
    6. backend —— asof join 走 duckdb；内存表达式 join 走 polars。

输出 ``JoinPlanDecision``（join_order/build_side/filter_before_join/pre_aggregate/
materialize/backend）+ ``estimate_join_cost`` 的行数/字节估算。

完全 additive：不修改既有 store.read_joined / read/scan_cost.py；本层是
R30 的 join 计划立面，供上层调度器消费。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "JOIN_MATERIALIZE_BYTES_THRESHOLD",
    "PREAGGREGATE_RIGHT_FREQUENCIES",
    "JoinCostPlanner",
    "JoinInput",
    "JoinPlanDecision",
    "estimate_join_cost",
]

#: 两侧 min(byte_size) 低于该阈值 → 建议物化（内存 join）。
JOIN_MATERIALIZE_BYTES_THRESHOLD: int = 256 * 1024 * 1024

#: minute 聚合后在 join 前可预聚合的右表低频频率集合。
PREAGGREGATE_RIGHT_FREQUENCIES: frozenset[str] = frozenset({
    "quarterly", "quarter", "event", "annual", "yearly", "monthly",
})


@dataclass
class JoinInput:
    """一路 join 输入的统计概要。"""

    relation: str
    row_count: int = 0
    byte_size: int = 0
    selectivity: float = 1.0          # 过滤后保留比例 (0~1)；越低越该先过滤
    unique_keys: int = 1
    time_range: tuple[Any, Any] | None = None
    coverage: float = 1.0
    local: bool = True
    frequency: str = "daily"          # minute/daily/quarterly/event/...

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation": self.relation,
            "row_count": self.row_count,
            "byte_size": self.byte_size,
            "selectivity": self.selectivity,
            "unique_keys": self.unique_keys,
            "time_range": (
                list(self.time_range) if self.time_range is not None else None
            ),
            "coverage": self.coverage,
            "local": self.local,
            "frequency": self.frequency,
        }


@dataclass
class JoinPlanDecision:
    """一次两路 join 的启发式计划决策。"""

    join_order: list[str] = field(default_factory=list)
    build_side: str | None = None
    filter_before_join: list[str] = field(default_factory=list)
    pre_aggregate: bool = False
    materialize: bool = False
    backend: str = "polars"

    def to_dict(self) -> dict[str, Any]:
        return {
            "join_order": list(self.join_order),
            "build_side": self.build_side,
            "filter_before_join": list(self.filter_before_join),
            "pre_aggregate": self.pre_aggregate,
            "materialize": self.materialize,
            "backend": self.backend,
        }


class JoinCostPlanner:
    """两路 join 的启发式成本规划器（第一版，不引入复杂优化器）。"""

    def plan(
        self,
        left: JoinInput,
        right: JoinInput,
        *,
        asof: bool = False,
    ) -> JoinPlanDecision:
        """对 left × right 给出 JoinPlanDecision。"""
        sides = [left, right]

        # 1) small dimension first：按 byte_size 升序排驱动/被驱动顺序
        ordered = sorted(sides, key=lambda s: (s.byte_size, s.row_count))
        join_order = [s.relation for s in ordered]

        # 2) build_side = 小的一侧（哈希 join 把 build 放小表）
        build_side = ordered[0].relation

        # 3) filter before join：selectivity < 1.0 的一侧先过滤（过滤能砍行）
        filter_before_join = [
            s.relation for s in sides if s.selectivity < 1.0
        ]

        # 4) minute aggregate before fundamental join
        left_freq = str(left.frequency or "").strip().lower()
        right_freq = str(right.frequency or "").strip().lower()
        pre_aggregate = (
            left_freq == "minute"
            and right_freq in PREAGGREGATE_RIGHT_FREQUENCIES
        )

        # 5) materialize：两侧 min(byte_size) 低于阈值（或复用需要——第一版只按阈值）
        min_bytes = min(s.byte_size for s in sides)
        materialize = min_bytes < JOIN_MATERIALIZE_BYTES_THRESHOLD

        # 6) backend：asof → duckdb；内存表达式 join → polars
        backend = "duckdb" if asof else "polars"

        return JoinPlanDecision(
            join_order=join_order,
            build_side=build_side,
            filter_before_join=filter_before_join,
            pre_aggregate=pre_aggregate,
            materialize=materialize,
            backend=backend,
        )

    def estimate(
        self,
        left: JoinInput,
        right: JoinInput,
        decision: JoinPlanDecision,
    ) -> dict[str, Any]:
        """按计划估算输出 rows/bytes（第一版近似，不做复杂 selectivity 传播）。"""
        return estimate_join_cost(left, right, decision)


def estimate_join_cost(
    left: JoinInput,
    right: JoinInput,
    decision: JoinPlanDecision,
) -> dict[str, Any]:
    """估算 join 输出行数/字节数。

    行数：过滤后两侧行数相乘 ÷ 两侧 key 基数较小者（哈希 join 的一阶近似）。
    字节：估算平均行宽 = 两侧平均行宽之和，再乘输出行数。
    """
    rows_left = max(0.0, float(left.row_count) * left.selectivity)
    rows_right = max(0.0, float(right.row_count) * right.selectivity)
    key_cardinality = max(1, int(min(left.unique_keys, right.unique_keys)))
    est_rows = int(rows_left * rows_right / key_cardinality)

    avg_row_width = 0.0
    if left.row_count > 0:
        avg_row_width += left.byte_size / max(1, left.row_count)
    if right.row_count > 0:
        avg_row_width += right.byte_size / max(1, right.row_count)
    est_bytes = int(est_rows * avg_row_width)

    return {
        "estimated_rows": est_rows,
        "estimated_bytes": est_bytes,
        "build_side": decision.build_side,
        "pre_aggregate": decision.pre_aggregate,
        "backend": decision.backend,
        "join_order": list(decision.join_order),
        "filter_before_join": list(decision.filter_before_join),
        "materialize": decision.materialize,
        "asof": decision.backend == "duckdb",
    }
