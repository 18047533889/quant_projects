# -*- coding: utf-8 -*-
"""R30-P0-004: FieldRequestCoalescer —— 同源字段请求合并。

多个消费者（因子 / 分析 / 中间层）会对同一 dataset + 同一 source snapshot 发出
几乎相同的字段请求。本模块只对**完全兼容**（:class:`RequestCompatibilityKey` 全
等）的请求做 ``fields = union(fields)`` 合并，产出合并后的 :class:`FieldRequest`
列表和统计（``to_dict()``）。

**绝不合并**不同 timeframe / price_basis / pit_policy 的请求——这些维度全部
进入 :class:`RequestCompatibilityKey`，键不同即不合并。

顶层 ``coalesce(requests)`` 提供便捷调用。纯 Python，不读数据。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "FieldRequest",
    "FieldRequestCoalescer",
    "RequestCompatibilityKey",
    "coalesce",
]


def _fmt_bound(value: Any) -> Any:
    if value is None:
        return None
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        try:
            return iso()
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def _norm_tr(time_range: Any) -> Any:
    """time_range 归一化为可哈希元组（None 保留）。"""
    if time_range is None:
        return None
    if isinstance(time_range, str):
        return (time_range,)
    as_tuple = getattr(time_range, "as_tuple", None)
    if callable(as_tuple):
        try:
            time_range = as_tuple()
        except Exception:  # noqa: BLE001
            return (str(time_range),)
    if isinstance(time_range, (tuple, list)) and len(time_range) == 2:
        return (_fmt_bound(time_range[0]), _fmt_bound(time_range[1]))
    return (str(time_range),)


def _norm_universe(values: Any) -> tuple[str, ...] | None:
    """instrument universe 归一化为排序去重元组（空 → None）。"""
    if values is None:
        return None
    if isinstance(values, str):
        items = [values]
    else:
        items = list(values or ())
    cleaned = tuple(sorted({str(v) for v in items if v not in (None, "")}))
    return cleaned or None


@dataclass(frozen=True)
class RequestCompatibilityKey:
    """合并判定的全部语义维度（frozen + hashable）。

    任一维度不同 ⇒ 请求**不可**合并（绝不合并不同 timeframe / price_basis /
    pit_policy）。
    """

    dataset: str = ""
    source_snapshot: str = ""
    market: str = ""
    time_range: Any = None
    instrument_universe: Any = None
    pit_policy: str = "none"
    timeframe: str = "default"
    security_digest: str = ""
    frequency: str = "daily"
    price_basis: str = "raw"
    aggregation_semantics: str = "none"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset", str(self.dataset))
        object.__setattr__(self, "source_snapshot", str(self.source_snapshot))
        object.__setattr__(self, "market", str(self.market))
        object.__setattr__(self, "time_range", _norm_tr(self.time_range))
        object.__setattr__(self, "instrument_universe", _norm_universe(self.instrument_universe))
        object.__setattr__(self, "pit_policy", str(self.pit_policy))
        object.__setattr__(self, "timeframe", str(self.timeframe))
        object.__setattr__(self, "security_digest", str(self.security_digest))
        object.__setattr__(self, "frequency", str(self.frequency))
        object.__setattr__(self, "price_basis", str(self.price_basis))
        object.__setattr__(self, "aggregation_semantics", str(self.aggregation_semantics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "source_snapshot": self.source_snapshot,
            "market": self.market,
            "time_range": list(self.time_range) if isinstance(self.time_range, tuple) else self.time_range,
            "instrument_universe": list(self.instrument_universe) if isinstance(self.instrument_universe, tuple) else None,
            "pit_policy": self.pit_policy,
            "timeframe": self.timeframe,
            "security_digest": self.security_digest,
            "frequency": self.frequency,
            "price_basis": self.price_basis,
            "aggregation_semantics": self.aggregation_semantics,
        }


@dataclass(frozen=True)
class FieldRequest:
    """一个（去重后的）字段请求：key = 合并判定，fields = 请求字段，payload 附带。"""

    key: RequestCompatibilityKey
    fields: tuple[str, ...]
    payload: Any | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "fields", tuple(sorted(str(f) for f in (self.fields or ())))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key.to_dict(),
            "fields": list(self.fields),
            "n_fields": len(self.fields),
        }


class FieldRequestCoalescer:
    """同源字段请求合并器。

    用法::

        merged = FieldRequestCoalescer().coalesce(requests)
        stats = coalescer.to_dict()  # 最近一次调用的统计

    ``coalesce`` 只合并 key 全等的请求；不同 timeframe / price_basis /
    pit_policy 永不合并（键不同）。
    """

    def __init__(self) -> None:
        self.last_stats: dict[str, int] = {}

    def coalesce(self, requests: Sequence[FieldRequest]) -> list[FieldRequest]:
        buckets: dict[RequestCompatibilityKey, set[str]] = {}
        payloads: dict[RequestCompatibilityKey, list[Any]] = {}
        for req in requests or ():
            buckets.setdefault(req.key, set()).update(req.fields or ())
            payloads.setdefault(req.key, []).append(req.payload)
        merged = [
            FieldRequest(
                key=key,
                fields=tuple(sorted(fields)),
                payload=payloads[key],
            )
            for key, fields in buckets.items()
        ]
        self.last_stats = self._compute_stats(requests, merged)
        return merged

    def to_dict(self) -> dict[str, int]:
        """统计 ``{input_requests, merged_requests, saved_scans, field_union_total}``。"""
        return dict(self.last_stats)

    @staticmethod
    def _compute_stats(
        requests: Sequence[FieldRequest], merged: Sequence[FieldRequest]
    ) -> dict[str, int]:
        return {
            "input_requests": len(requests),
            "merged_requests": len(merged),
            "saved_scans": len(requests) - len(merged),
            "field_union_total": sum(len(m.fields) for m in merged),
        }


def coalesce(requests: Sequence[FieldRequest]) -> list[FieldRequest]:
    """顶层便捷：一次调用完成合并。"""
    return FieldRequestCoalescer().coalesce(requests)
