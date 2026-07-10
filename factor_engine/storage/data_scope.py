# -*- coding: utf-8 -*-
"""数据源窗口指纹：plan 子树缓存键的作用域（避免跨窗口/跨源污染）。"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _jsonable(value: Any) -> Any:
    """将值递归转为 JSON 可序列化结构。
    
    参数:
        value: 缓存值
    
    返回:
        Any
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    return repr(value)


def compute_data_scope(data_source: Any) -> str:
    """从数据源提取稳定作用域指纹（plan 缓存键）。
    
    参数:
        data_source: 数据源实例
    
    返回:
        str
    """
    payload: dict[str, Any] = {}
    for attr in (
        "dataset",
        "start_date",
        "end_date",
        "root",
        "timestamp_column",
        "instrument_column",
    ):
        val = getattr(data_source, attr, None)
        if val is not None and str(val).strip():
            payload[attr] = _jsonable(val)

    fields = getattr(data_source, "fields", None)
    if isinstance(fields, dict) and fields:
        payload["fields"] = _jsonable(fields)

    instrument_filter = getattr(data_source, "instrument_filter", None)
    if instrument_filter:
        payload["instrument_filter"] = _jsonable(sorted(instrument_filter))

    if not payload:
        return f"ephemeral:{id(data_source)}"

    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
