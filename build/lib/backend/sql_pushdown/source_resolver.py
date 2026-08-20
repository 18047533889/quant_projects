# -*- coding: utf-8
"""解析可 SQL 下推的数据源（支持 LongTable / Composite 包装）。

沿 ``_inner``、``anchor_source`` / ``sources`` 链解包，直至找到具备
``dataset``（DuckDB）或 ``table``（ClickHouse）字段的数据源对象。
"""
from __future__ import annotations

from typing import Any


def resolve_pushdown_source(data_source: Any) -> Any | None:
    """解包数据源，返回具备 ``dataset`` 或 ``table`` 的底层对象。

    支持 Composite / LongTable 等包装；循环引用或无法解析时返回 ``None``。
    """
    seen: set[int] = set()
    ds = data_source
    while ds is not None and id(ds) not in seen:
        seen.add(id(ds))
        if getattr(ds, "dataset", None) or getattr(ds, "table", None):
            return ds
        inner = getattr(ds, "_inner", None)
        if inner is not None and inner is not ds:
            ds = inner
            continue
        anchor = getattr(ds, "anchor_source", None)
        sources = getattr(ds, "sources", None)
        if anchor and sources and anchor in sources:
            ds = sources[anchor]
            continue
        return None
    return None
