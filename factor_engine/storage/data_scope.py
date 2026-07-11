# -*- coding: utf-8 -*-
"""数据源窗口指纹：plan/column 缓存键的稳定作用域。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


def _jsonable(value: Any) -> Any:
    """将常见配置值转为跨进程稳定的 JSON 结构。

    禁止优先使用任意对象 ``repr``：很多对象的 repr 含内存地址，会让同一配置
    在不同 worker 上得到不同 cache key。确实无法结构化时退回类型名 + 字符串值。
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value.expanduser())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True, default=str))
    if isinstance(value, Mapping):
        return {
            str(k): _jsonable(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    return {
        "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": str(value),
    }


def compute_data_scope(data_source: Any) -> str:
    """从数据源提取稳定作用域指纹，防止跨参数/快照/读模式缓存污染。"""
    payload: dict[str, Any] = {}
    for attr in (
        "dataset",
        "start_date",
        "end_date",
        "root",
        "timestamp_column",
        "instrument_column",
        "timestamp_unit",
        "normalize_timestamp",
        "read_auto",
        "lazy_scan",
        "data_snapshot_id",
    ):
        val = getattr(data_source, attr, None)
        if val is not None and (not isinstance(val, str) or val.strip()):
            payload[attr] = _jsonable(val)

    params = getattr(data_source, "params", None)
    if isinstance(params, Mapping) and params:
        payload["params"] = _jsonable(params)

    fields = getattr(data_source, "fields", None)
    if isinstance(fields, Mapping) and fields:
        payload["fields"] = _jsonable(fields)

    instrument_filter = getattr(data_source, "instrument_filter", None)
    if instrument_filter:
        payload["instrument_filter"] = _jsonable(sorted(instrument_filter))

    if not payload:
        # 没有任何可描述字段的数据源无法跨实例安全复用缓存，只在当前对象生命周期内稳定。
        return f"ephemeral:{id(data_source)}"

    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
