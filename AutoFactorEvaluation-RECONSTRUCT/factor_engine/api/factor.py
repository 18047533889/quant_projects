"""因子容器：名称 + ``Expr`` 树 + 业务元数据（频率、股票池、描述）。

``Factor`` 本身不参与计算；经 ``FactorEngine.compile`` 变为 ``PlanNode`` 后由 backend 执行。

``calc_mode``:
  - ``"expr"``: 通过算子与显式表达式驱动计算（默认）。
  - ``"code"``: 通过 Python 函数字符串直接计算。
"""

from dataclasses import dataclass

from expr.base import Expr


@dataclass(frozen=True)
class Factor:
    """可编译、可执行的一条因子。"""

    name: str
    expr: Expr
    calc_mode: str = "expr"  # "expr" | "code"
    freq: str = "1d"
    universe: str | None = None
    description: str | None = None
