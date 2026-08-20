"""多因子 DAG 计划容器：各因子根节点与 CSE 共享子树定义。"""

import json
from dataclasses import dataclass, field
from typing import Any

from .logical_plan import PlanNode


class DuplicateFactorNameError(ValueError):
    """``run_many``/``compile_many`` 传入重复 ``Factor.name`` 时 fail-fast。

    ``analyses``/``roots`` 索引以 ``factor.name`` 为键，重复名会导致结果互相
    覆盖；批量入口应在编译前先检测并抛出本错误（#322）。
    """


def assert_unique_factor_names(names: list[str]) -> None:
    """#322：重复 ``factor.name`` fail-fast。

    ``run_many``/``compile_many`` 的 ``analyses``/``roots`` 均以 ``factor.name``
    为键，重复名会导致结果互相覆盖；批量入口应在编译前先调用本函数。
    """
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise DuplicateFactorNameError(
                f"重复的 factor name={name!r}；run_many/compile_many 要求 "
                "factor.name 唯一（analyses/roots 均以 name 为键）"
            )
        seen.add(name)


@dataclass(frozen=True)
class FactorExecutionScope:
    """因子执行作用域：CSE / rolling-CSE 只在同一作用域内共享。

    CSE 必须绑定因子执行作用域（freq/universe/market/calendar）——不同执行
    作用域的同结构子树不能共享，否则一次计算的结果会被错误复用到另一作用域
    （#321）。字段均可从 ``Factor`` 对象推断；取不到时使用默认值。

    R20-056..057: ``market`` 默认改为空串（UNKNOWN），**绝不**默认 ``"A"``。
    A 股默认会污染 US/其它市场的因子（把 US 因子当 A 算）；market 未声明时
    保持空串，交给下游 gate / universe 判定决定。

    字段：
        frequency: 频率（默认 ``"1d"``）
        universe_id: 股票池标识（默认 ``"ALL"``）
        market: 市场标识（默认 ``""`` —— UNKNOWN，绝不默认 ``"A"``）
        calendar_id: 日历 ID（默认 ``""``）
        source_scope_hash: 数据源作用域哈希（默认 ``""``）
        decision_time_policy: 决策时间策略（默认 ``""``）
    """

    frequency: str = "1d"
    universe_id: str = "ALL"
    market: str = ""
    calendar_id: str = ""
    source_scope_hash: str = ""
    decision_time_policy: str = ""

    def scope_key(self) -> str:
        """返回规范化 JSON 键（``sort_keys``），用于作用域分组比较。"""
        return json.dumps(
            {
                "frequency": self.frequency,
                "universe_id": self.universe_id,
                "market": self.market,
                "calendar_id": self.calendar_id,
                "source_scope_hash": self.source_scope_hash,
                "decision_time_policy": self.decision_time_policy,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def to_dict(self) -> dict[str, Any]:
        """序列化为可 JSON 化的字典（供审计/日志）。"""
        return {
            "frequency": self.frequency,
            "universe_id": self.universe_id,
            "market": self.market,
            "calendar_id": self.calendar_id,
            "source_scope_hash": self.source_scope_hash,
            "decision_time_policy": self.decision_time_policy,
        }


@dataclass
class FactorPlan:
    """单因子逻辑计划条目。

    字段：
        factor_name: 因子名称（与 ``Factor.name`` 一致）
        root: 该因子的逻辑计划根 ``PlanNode``
        execution_scope: 该因子的执行作用域（R9-P0-011）；缺省
            ``FactorExecutionScope()``（``universe_id="ALL"``）以保持向后兼容。
    """

    factor_name: str
    root: PlanNode
    execution_scope: FactorExecutionScope = field(default_factory=FactorExecutionScope)


@dataclass
class DAGPlan:
    """多因子批量计划：根列表 + 结构 CSE 共享节点表。

    字段：
        roots: 各因子的 ``FactorPlan`` 列表
        shared_nodes: 结构键 → 共享子树定义（``plan_ref`` 通过 ``attrs["sid"]`` 引用）
    """

    roots: list[FactorPlan]
    shared_nodes: dict[str, PlanNode] = field(default_factory=dict)
