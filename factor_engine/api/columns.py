"""列引用：``col('close')`` → ``ColumnRef``，表示从数据源读取该字段的 MultiIndex Series。"""

from expr.column import ColumnRef


def col(name: str) -> ColumnRef:
    """按列名字符串构造 :class:`~expr.column.ColumnRef`。

    Parameters
    ----------
    name : str
        数据源逻辑字段名（如 ``"close"``、``"volume"``）。

    Returns
    -------
    ColumnRef
        表示从 panel 读取该列的 MultiIndex Series 引用节点。
    """
    return ColumnRef(name=name)
