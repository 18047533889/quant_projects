# -*- coding: utf-8
"""解析可 SQL 下推的数据源（支持 LongTable / Composite 包装）。"""
from __future__ import annotations

from typing import Any


def resolve_pushdown_source(data_source: Any) -> Any | None:
    """unwrap 至具备 ``dataset`` 或 ``table`` 的数据源（保留 LongTable 包装）。"""
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
