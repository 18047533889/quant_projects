"""列引用：``col('close')`` → ``ColumnRef``，字段语义可由 ``field()`` 校验。"""

from expr.column import ColumnRef
from expr.field import FieldRef


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


def field(name: str, *, table: str | None = None, strict: bool = True) -> FieldRef | ColumnRef:
    """Construct a catalog-bound field reference.

    Unlike :func:`col`, a resolved field keeps its canonical table and physical
    source identity in the expression tree.  The runtime transport name stays a
    normal logical column for anchor fields and an encoded ``SourceRef`` for
    secondary tables, preserving compatibility with existing data sources.

    Resolution order (review §4.3): an explicit ``table=`` qualifier wins;
    otherwise a bare name resolves uniquely, or falls back to the
    ``StockDailyBar`` anchor when multiple tables share the physical name (e.g.
    ``close``/``volume`` across DailyBar / IndexDailyBar / EtfDailyBar /
    StockMinuteBar).  A bare name that is still ambiguous with **no** anchor
    candidate (e.g. ``pub_date`` shared by all financial tables) raises in
    ``strict`` mode exactly as the catalog contract requires.
    """

    from fields import FIELD_REGISTRY

    spec = FIELD_REGISTRY.get(name, table=table, strict=False)
    if spec is None and table is None:
        # Anchor preference: bare ambiguous OHLCV-family names keep the daily
        # panel binding so legacy recipes stay catalog-bound.
        spec = FIELD_REGISTRY.get(name, table="StockDailyBar", strict=False)
    if spec is None:
        if strict:
            qualifier = f" in table {table!r}" if table else ""
            raise KeyError(f"unknown field {name!r}{qualifier}")
        return col(name)
    if table is None and spec.table == "StockDailyBar":
        transport = spec.name
    else:
        from api.source_ref import source_col

        transport = source_col(spec.table, spec.source_name).name
    return FieldRef(
        name=transport,
        field_id=str(spec.field_id),
        canonical_name=spec.name,
        table=spec.table,
        source_name=spec.source_name,
        catalog_hash=FIELD_REGISTRY.catalog_hash(),
    )


__all__ = ["col", "field"]
