"""多因子 DAG 计划容器：各因子根节点与 CSE 共享子树定义。"""

from dataclasses import dataclass, field

from .logical_plan import PlanNode


@dataclass
class FactorPlan:
    """单因子逻辑计划条目。

    字段：
        factor_name: 因子名称（与 ``Factor.name`` 一致）
        root: 该因子的逻辑计划根 ``PlanNode``
    """

    factor_name: str
    root: PlanNode


@dataclass
class DAGPlan:
    """多因子批量计划：根列表 + 结构 CSE 共享节点表。

    字段：
        roots: 各因子的 ``FactorPlan`` 列表
        shared_nodes: 结构键 → 共享子树定义（``plan_ref`` 通过 ``attrs["sid"]`` 引用）
    """

    roots: list[FactorPlan]
    shared_nodes: dict[str, PlanNode] = field(default_factory=dict)
