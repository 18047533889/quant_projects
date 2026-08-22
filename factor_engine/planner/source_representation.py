# -*- coding: utf-8 -*-
"""R39-P0-PERF-011：SourceRepresentation —— read wave 输出表示不再是固定 pandas。

原则（R39 §4.11）：
    source-native → largest native compute region → terminal writer

而不是：
    source → pandas → native → pandas → writer

``consumer_backend_mask`` 用 bitmask 表达下游后端集合；``preferred_representation``
按 mask 选最「原生」的表示；混合 / 未知时回退 PANDAS_COLUMNS（保持现有行为）。
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable


class SourceRepresentation(Enum):
    """ReadWave 输出的数据表示（R39-P0-PERF-011）。"""

    DUCKDB_RELATION = "duckdb_relation"
    ARROW_BATCH_BLOCK = "arrow_batch_block"
    POLARS_LAZY = "polars_lazy"
    NUMPY_COLUMN_BLOCK = "numpy_column_block"
    PANDAS_COLUMNS = "pandas_columns"  # reference fallback

    @property
    def bit(self) -> int:
        return _REPR_BIT[self]


#: backend 关键字 → 最贴近的表示（便宜可判定时用；否则回退 pandas）。
#: 注意顺序：引擎默认 backend 是 ``pandas_numpy``（PhysicalFactorTask 默认），
#: 必须先于 numpy/numba 匹配 pandas，保持现有默认路径。
_BACKEND_KEYWORDS: tuple[tuple[str, SourceRepresentation], ...] = (
    ("duckdb", SourceRepresentation.DUCKDB_RELATION),
    ("arrow", SourceRepresentation.ARROW_BATCH_BLOCK),
    ("polars", SourceRepresentation.POLARS_LAZY),
    ("pandas", SourceRepresentation.PANDAS_COLUMNS),
    ("numpy", SourceRepresentation.NUMPY_COLUMN_BLOCK),
    ("numba", SourceRepresentation.NUMPY_COLUMN_BLOCK),
)

_REPR_BIT: dict[SourceRepresentation, int] = {
    SourceRepresentation.DUCKDB_RELATION: 1 << 0,
    SourceRepresentation.ARROW_BATCH_BLOCK: 1 << 1,
    SourceRepresentation.POLARS_LAZY: 1 << 2,
    SourceRepresentation.NUMPY_COLUMN_BLOCK: 1 << 3,
    SourceRepresentation.PANDAS_COLUMNS: 1 << 4,
}

#: 原生（非 pandas）表示，按「越原生越优先」排序（DuckDB 原生 → arrow → polars
#: → numpy block）。
_NATIVE_ORDER: tuple[SourceRepresentation, ...] = (
    SourceRepresentation.DUCKDB_RELATION,
    SourceRepresentation.ARROW_BATCH_BLOCK,
    SourceRepresentation.POLARS_LAZY,
    SourceRepresentation.NUMPY_COLUMN_BLOCK,
)


def representation_for_backend(backend: str) -> SourceRepresentation:
    """backend 名 → 最贴近的原生表示（便宜可判定；未知回退 pandas）。"""
    b = str(backend or "").lower()
    for kw, rep in _BACKEND_KEYWORDS:
        if kw in b:
            return rep
    return SourceRepresentation.PANDAS_COLUMNS


def consumer_backend_mask(backends: Iterable[str]) -> int:
    """下游后端集合 → bitmask（0 = 无信息 → 默认 pandas 路径）。"""
    mask = 0
    for b in backends:
        mask |= _REPR_BIT[representation_for_backend(b)]
    return mask


def preferred_representation_for_mask(mask: int) -> SourceRepresentation:
    """从 bitmask 选 preferred representation（R39-P0-PERF-011）。

    - mask == 0 或含 pandas → PANDAS_COLUMNS（默认 / 混合回退）。
    - 只有一个原生 backend → 该原生表示。
    - 多个不同原生 backend 混合 → PANDAS_COLUMNS（避免为一种下游做转换）。
    """
    mask = int(mask or 0)
    if mask == 0 or (mask & _REPR_BIT[SourceRepresentation.PANDAS_COLUMNS]):
        return SourceRepresentation.PANDAS_COLUMNS
    for rep in _NATIVE_ORDER:
        if mask == _REPR_BIT[rep]:
            return rep
    return SourceRepresentation.PANDAS_COLUMNS
