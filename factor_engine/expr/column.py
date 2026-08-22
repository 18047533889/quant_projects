"""列引用：表示从数据源按名称取一列（如收盘价 ``close``）。"""

from dataclasses import dataclass

from .base import Expr


@dataclass(frozen=True)
class ColumnRef(Expr):
    """引用名为 ``name`` 的数据列，执行时由 ``DataSource.load_column`` 加载。"""

    name: str
