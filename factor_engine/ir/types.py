"""IR static types and field-to-IR type inference."""

from __future__ import annotations

from enum import Enum
from typing import Any


class ValueType(str, Enum):
    """编译期/运行期对因子结果形态的粗分类（目前 schema 层使用较少）。"""

    SCALAR = "scalar"   # 单值
    SERIES = "series"   # 一维序列（通常指单标的时序）
    PANEL = "panel"     # 二维面板 date × instrument


_DTYPE_ALIASES = {
    "bool": "bool",
    "boolean": "bool",
    "date": "date",
    "date32": "date",
    "datetime": "datetime64[ns]",
    "timestamp": "datetime64[ns]",
    "float": "float64",
    "double": "float64",
    "float64": "float64",
    "int": "int64",
    "integer": "int64",
    "int64": "int64",
    "uint64": "uint64",
    "str": "string",
    "string": "string",
    "object": "string",
}


def normalize_dtype(dtype: str | None) -> str:
    """Normalize source/catalog dtype spellings without requiring NumPy."""

    raw = str(dtype or "float64").strip().lower()
    if raw.startswith("timestamp") or raw.startswith("datetime"):
        return "datetime64[ns]"
    return _DTYPE_ALIASES.get(raw, raw)


def infer_field_type(value: Any, *, table: str | None = None):
    """Infer a registered ``FieldSpec`` for a name/ColumnRef/SourceRef.

    ``None`` means the legacy column has no semantic registration; callers
    should retain their historical float-panel fallback in that case.
    """

    from fields import resolve_field

    return resolve_field(value, table=table, strict=False)


__all__ = ["ValueType", "infer_field_type", "normalize_dtype"]
