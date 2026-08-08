"""
data_access.read.query_cache —— 查询结果缓存（LRU + TTL，默认关闭）

为什么需要
    因子挖掘里同一 (dataset, params, time_range, instruments) 会被反复读取
    （多因子共享同一 anchor 数据）。``store.read_cached(...)`` 提供进程内
    LRU 缓存：命中直接返回 Arrow Table，省重复 IO。

设计
    - 默认关闭（``store.enable_result_cache(True)`` 或 env
      ``DATA_ACCESS_QUERY_CACHE=1`` 才生效）；
    - key = dataset + 规范化 params + manifest 版本 token + time_range +
      instrument 集合；
    - 写路径 bump manifest epoch 后，``is_snapshot_stale`` 让缓存自动失效；
    - LRU 容量默认 128 条，TTL 默认 300s（可配）。

非职责
    不做跨进程缓存；不做磁盘缓存（那交给 serving layer）。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Mapping, Sequence


class QueryResultCache:
    """进程内 LRU+TTL 查询结果缓存。线程安全。"""

    def __init__(self, capacity: int = 128, ttl_seconds: float = 300.0) -> None:
        self._capacity = max(1, capacity)
        self._ttl = ttl_seconds
        self._store: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            ts, value = item
            if time.monotonic() - ts > self._ttl:
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = (time.monotonic(), value)
            self._store.move_to_end(key)
            while len(self._store) > self._capacity:
                self._store.popitem(last=False)

    def invalidate_prefix(self, prefix: str) -> None:
        with self._lock:
            keys = [k for k in self._store if k.startswith(prefix)]
            for k in keys:
                self._store.pop(k, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


def _stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def query_cache_key(
    *,
    dataset: str,
    params: Mapping[str, Any],
    time_range: tuple[Any, Any] | None,
    instruments: Sequence[str] | None,
    columns: Sequence[str] | None,
    manifest_token: Mapping[str, Any] | None,
) -> str:
    """缓存 key：dataset + 规范化参数 + manifest 版本 + 查询窗口 + 列。"""
    payload = {
        "dataset": dataset,
        "params": dict(params or {}),
        "time_range": tuple(map(str, time_range)) if time_range else None,
        "instruments": sorted(instruments) if instruments else None,
        "columns": sorted(columns) if columns else None,
        "manifest": {
            k: manifest_token.get(k)
            for k in (
                "dataset_version",
                "partition_version",
                "manifest_epoch",
            )
            if manifest_token and manifest_token.get(k)
        },
    }
    return hashlib.sha256(_stable(payload).encode("utf-8")).hexdigest()[:24]


_cache_enabled = None


def result_cache_enabled() -> bool:
    """查询结果缓存是否启用（默认关闭，env DATA_ACCESS_QUERY_CACHE=1 或显式开启）。"""
    global _cache_enabled
    if _cache_enabled is not None:
        return _cache_enabled
    return os.environ.get("DATA_ACCESS_QUERY_CACHE", "").lower() in {
        "1",
        "true",
        "yes",
    }


def set_result_cache_enabled(enabled: bool) -> None:
    """显式开关查询结果缓存（进程级）。"""
    global _cache_enabled
    _cache_enabled = bool(enabled)


# 进程级单例
_QUERY_CACHE = QueryResultCache()


def get_query_cache() -> QueryResultCache:
    return _QUERY_CACHE


__all__ = [
    "QueryResultCache",
    "query_cache_key",
    "get_query_cache",
    "result_cache_enabled",
    "set_result_cache_enabled",
]
