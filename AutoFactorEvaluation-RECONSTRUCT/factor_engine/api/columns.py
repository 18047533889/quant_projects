"""列引用：``col('close')`` → ``ColumnRef``，表示从数据源读取该字段的 MultiIndex Series。"""

from expr.column import ColumnRef


def col(name: str) -> ColumnRef:
    """按列名字符串构造 :class:`~expr.column.ColumnRef`。"""
    return ColumnRef(name=name)
