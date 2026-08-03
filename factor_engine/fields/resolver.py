"""Resolution helpers for ColumnRef and encoded SourceRef leaves."""
from __future__ import annotations

from typing import Any

from .registry import FieldRegistry
from .spec import FieldSpec


def resolve_field(
    value: Any,
    *,
    table: str | None = None,
    registry: FieldRegistry | None = None,
    strict: bool = False,
) -> FieldSpec | None:
    """Resolve a string, :class:`ColumnRef`, or encoded ``SourceRef``.

    Legacy plain columns remain valid: unknown values return ``None`` unless
    ``strict=True``.  Encoded source refs are matched by physical field name
    within their explicitly named table.
    """

    from . import FIELD_REGISTRY

    active = registry or FIELD_REGISTRY
    name = getattr(value, "name", value)
    if not isinstance(name, str):
        if strict:
            raise TypeError(f"field reference must be a string or ColumnRef, got {type(value).__name__}")
        return None
    try:
        from api.source_ref import decode_source_ref

        source = decode_source_ref(name)
    except (ImportError, ValueError, TypeError):
        source = None
    if source is not None:
        return active.get(source.field, table=source.table, strict=strict)
    return active.get(name, table=table, strict=strict)


def require_field(value: Any, *, table: str | None = None, registry: FieldRegistry | None = None) -> FieldSpec:
    result = resolve_field(value, table=table, registry=registry, strict=True)
    assert result is not None
    return result


__all__ = ["require_field", "resolve_field"]
