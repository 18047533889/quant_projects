"""列引用：``col('close')`` → ``ColumnRef``，字段语义可由 ``field()`` 校验。"""

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


def field(name: str, *, table: str | None = None, strict: bool = True) -> ColumnRef:
    """Construct a semantically resolved field reference.

    With no ``table``, the canonical field name is emitted as a normal
    :class:`ColumnRef`, preserving the old DSL/runtime representation.  With a
    table, the reference uses the existing encoded ``SourceRef`` transport so
    no new expression node or IR opcode is introduced.
    """

    from fields import FIELD_REGISTRY

    spec = FIELD_REGISTRY.get(name, table=table, strict=strict)
    if spec is None:
        return col(name)
    if table is None:
        return col(spec.name)
    from api.source_ref import source_col

    return source_col(spec.table, spec.source_name)


__all__ = ["col", "field"]
