"""Deterministic canonical representation for expression lineage and gates."""
from __future__ import annotations

import json
from typing import Any

from .cleaned_call import CleanedCall
from .column import ColumnRef
from .field import FieldRef
from .literal import Literal


def _value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _value(item) for key, item in sorted(value.items())}
    return repr(value)


def expression_payload(node: Any) -> dict[str, Any]:
    if isinstance(node, FieldRef):
        return {
            "kind": "field",
            "field_id": node.field_id,
            "canonical_name": node.canonical_name,
            "table": node.table,
            "source_name": node.source_name,
            "catalog_hash": node.catalog_hash,
        }
    if isinstance(node, ColumnRef):
        return {"kind": "column", "name": node.name}
    if isinstance(node, Literal):
        return {"kind": "literal", "value": _value(node.value)}
    if isinstance(node, CleanedCall):
        return {
            "kind": "call",
            "op": node.op,
            "args": [expression_payload(child) for child in node.args],
            "kwargs": {str(key): _value(value) for key, value in sorted(node.kwargs)},
        }
    raise TypeError(f"unsupported expression node: {type(node).__name__}")


def canonical_expression(node: Any) -> str:
    return json.dumps(
        expression_payload(node), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
