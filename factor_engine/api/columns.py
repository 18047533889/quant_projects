"""列引用：``col('close')`` → ``ColumnRef``，字段语义可由 ``field()`` 校验。"""

from typing import Any

from expr.column import ColumnRef
from expr.field import FieldRef

#: Contract attributes embedded on a resolved :class:`~expr.field.FieldRef`
#: (round-7 WS-E #281).  ``FieldRef`` is a frozen dataclass; the enrichment is
#: attached via ``object.__setattr__`` so the reference carries the catalog
#: contract into the IR leaf without changing dataclass equality/hash.
_FIELD_CONTRACT_ATTRS = (
    "mining_allowed",
    "strict_pit_allowed",
    "current_snapshot_only",
    "required_filters",
    "applicability",
    "allowed_operator_families",
    "null_policy",
    "semantic_kind",
    "role",
    "coverage",
)


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


def _embed_field_contract(ref: FieldRef, spec: Any) -> FieldRef:
    """Attach the catalog field contract onto a ``FieldRef`` (round-7 WS-E #281).

    The analyzer / planner already read ``field_id`` / ``table`` /
    ``catalog_hash``; the additional contract attributes let a mining /
    backfill preflight enforce ``mining_allowed``, ``current_snapshot_only`` and
    the four-layer PIT eligibility without a second registry lookup.
    """
    values: dict[str, Any] = {attr: getattr(spec, attr, None) for attr in _FIELD_CONTRACT_ATTRS}
    # ``current_snapshot_only`` is a table-level contract; a FieldSpec does not
    # carry it, so resolve it from the owning table when the field does not.
    if not values.get("current_snapshot_only"):
        table_name = getattr(ref, "table", None) or getattr(spec, "table", None)
        if table_name:
            try:
                from fields import FIELD_REGISTRY

                table_spec = FIELD_REGISTRY.resolve_table(str(table_name))
                if table_spec is not None:
                    values["current_snapshot_only"] = bool(
                        getattr(table_spec, "current_snapshot_only", False)
                    )
            except Exception:
                pass
    for attr, value in values.items():
        object.__setattr__(ref, attr, value)
    return ref


def check_mining_gate(spec: Any, *, for_mining: bool = False, field_name: str | None = None) -> None:
    """Production hard gate: reject using a ``mining_allowed=False`` field as input.

    Round-7 WS-E #279.  ``for_mining=True`` is the explicit signal from the
    AlphaMiner / production preflight that the field is being consumed as a
    mined factor input; a field whose catalog contract forbids mining then
    raises instead of relying on DataAccess readability.
    """
    if not for_mining:
        return
    if not bool(getattr(spec, "mining_allowed", True)):
        label = field_name or getattr(spec, "field_id", None) or getattr(spec, "name", "?")
        raise ValueError(
            f"field {label!r} has mining_allowed=False and cannot be used as a "
            "mined factor input"
        )


def field(
    name: str,
    *,
    table: str | None = None,
    strict: bool = True,
    for_mining: bool = False,
) -> FieldRef | ColumnRef:
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

    ``for_mining`` (round-7 WS-E #279) marks the reference as a mined factor
    input; a catalog field with ``mining_allowed=False`` then raises.
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
    check_mining_gate(spec, for_mining=for_mining, field_name=name)
    if table is None and spec.table == "StockDailyBar":
        transport = spec.name
    else:
        from api.source_ref import source_col

        transport = source_col(spec.table, spec.source_name).name
    return _embed_field_contract(
        FieldRef(
            name=transport,
            field_id=str(spec.field_id),
            canonical_name=spec.name,
            table=spec.table,
            source_name=spec.source_name,
            catalog_hash=FIELD_REGISTRY.catalog_hash(),
        ),
        spec,
    )


__all__ = ["col", "field"]
