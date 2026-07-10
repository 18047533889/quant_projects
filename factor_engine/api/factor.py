"""因子容器：名称 + ``Expr`` 树 + 业务元数据（频率、股票池、描述）。

``Factor`` 本身不参与计算；经 ``FactorEngine.compile`` 变为 ``PlanNode`` 后由 backend 执行。
``freq`` / ``universe`` 写入 YAML 与物化元数据，不改变 IR 类型推导。
"""

from dataclasses import dataclass

from expr.base import Expr


@dataclass(frozen=True)
class Factor:
    """可编译、可执行的一条因子；核心负载是 ``expr: Expr``。

    经 ``FactorEngine.compile`` 编译为 ``PlanNode`` 后由 backend 执行；
    ``freq`` / ``universe`` 仅写入 YAML 与物化元数据，不改变 IR 类型推导。

    Attributes
    ----------
    name : str
        因子唯一标识（YAML key / 物化列名）。
    expr : Expr
        由 ``col``、``CleanedCall`` 等节点组成的表达式树。
    freq : str
        业务语义频率（默认 ``"1d"``），供配置与文档。
    universe : str | None
        股票池或标签名；执行时以数据源 universe 为准。
    description : str | None
        人类可读说明（与 ``source_expr`` 不同）。
    source_expr : str | None
        原始 DSL 字符串，用于 lineage / 审计。
    """

    name: str
    expr: Expr
    freq: str = "1d"  # 业务语义频率，写入 YAML/物化元数据，不参与 IR 类型推导
    universe: str | None = None  # 股票池/标签，供配置与文档；执行时以数据源为准
    description: str | None = None
    source_expr: str | None = None  # 原始 DSL 字符串（lineage 用，勿与 description 混用）
