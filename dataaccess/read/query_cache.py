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
      instrument 集合 + columns + **canonical Filter AST hash** + limit + mode +
      allow_sparse + allow_effective_time + normalize_units + 语义 catalog 指纹
      + Contract IR 指纹（#3：缺 filters/limit 会命中错误结果）；
    - 写路径 bump source_epoch 后，``is_snapshot_stale`` 让缓存自动失效；
    - **联合容量**：``max_bytes``（按结果 nbytes 估算）+ ``max_entries`` +
      ``ttl_seconds``（#4：不再按条数一刀切——128 个 2GB 表和 128 个 1MB 表
      占的内存完全不同）。

非职责
    不做跨进程缓存；不做磁盘缓存（那交给 serving layer）。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Mapping, Sequence


def _estimate_bytes(value: Any) -> int:
    """估算缓存对象的字节数：Arrow Table 用 nbytes，其余退回 sys.getsizeof。"""
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int) and nbytes > 0:
        return int(nbytes)
    try:
        return sys.getsizeof(value)
    except TypeError:
        return 0


class QueryResultCache:
    """进程内 LRU+TTL 查询结果缓存。线程安全。

    约束（#4）：``max_bytes``（内存上界，None=不限）与 ``max_entries`` 联合
    生效——evict 直到同时满足两者；``ttl_seconds`` 过期条目读取时惰性剔除。
    """

    def __init__(
        self,
        capacity: int = 128,
        ttl_seconds: float = 300.0,
        *,
        max_bytes: int | None = None,
        max_entries: int | None = None,
    ) -> None:
        self._max_entries = max(1, int(max_entries or capacity))
        self._max_bytes = max_bytes  # None = 不限字节
        self._ttl = ttl_seconds
        # key -> (ts, value, nbytes)
        self._store: "OrderedDict[str, tuple[float, Any, int]]" = OrderedDict()
        self._lock = threading.Lock()

    def _total_bytes(self) -> int:
        return sum(item[2] for item in self._store.values())

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            ts, value, _ = item
            if time.monotonic() - ts > self._ttl:
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        size = _estimate_bytes(value)
        with self._lock:
            self._store[key] = (time.monotonic(), value, size)
            self._store.move_to_end(key)
            self._evict_locked()

    def set_with_size(self, key: str, value: Any, nbytes: int | None = None) -> None:
        """显式指定缓存条目字节数（避免对已物化对象重复估算）。"""
        size = int(nbytes) if nbytes is not None else _estimate_bytes(value)
        with self._lock:
            self._store[key] = (time.monotonic(), value, size)
            self._store.move_to_end(key)
            self._evict_locked()

    def _evict_locked(self) -> None:
        """逐出 LRU 尾（最旧）直到条目数 <= max_entries 且总字节 <= max_bytes。"""
        while len(self._store) > self._max_entries:
            self._store.popitem(last=False)
        if self._max_bytes is not None:
            while self._store and self._total_bytes() > self._max_bytes:
                self._store.popitem(last=False)

    def invalidate_prefix(self, prefix: str) -> None:
        with self._lock:
            keys = [k for k in self._store if k.startswith(prefix)]
            for k in keys:
                self._store.pop(k, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    @property
    def max_bytes(self) -> int | None:
        return self._max_bytes

    @property
    def max_entries(self) -> int:
        return self._max_entries

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes()


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
    filters: Any = None,
    limit: int | None = None,
    mode: str | None = None,
    allow_sparse: bool | None = None,
    allow_effective_time: bool | None = None,
    normalize_units: bool | None = None,
    catalog_fingerprint: str | None = None,
    contract_ir_fingerprint: str | None = None,
    calendar_version: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """缓存 key：dataset + 参数 + manifest 版本 + 查询窗口 + 列 + 完整语义。

    filters 用 canonical Filter AST hash（#3）：两个语义等价的过滤（书写顺序
    不同）得到同一 key；缺 filters/limit 会造成错误命中，因此必须进 key。
    """
    from data_access.read.predicate_ast import canonical_filter_hash

    payload: dict[str, Any] = {
        "dataset": dataset,
        "params": dict(params or {}),
        "time_range": tuple(map(str, time_range)) if time_range else None,
        "instruments": sorted(instruments) if instruments else None,
        "columns": sorted(columns) if columns else None,
        "filters": canonical_filter_hash(filters),
        "limit": limit,
        "mode": mode,
        "allow_sparse": allow_sparse,
        "allow_effective_time": allow_effective_time,
        "normalize_units": normalize_units,
        "catalog_fingerprint": catalog_fingerprint,
        "contract_ir_fingerprint": contract_ir_fingerprint,
        "calendar_version": calendar_version,
        "manifest": {
            k: manifest_token.get(k)
            for k in (
                "dataset_version",
                "partition_version",
                "source_epoch",
                "manifest_epoch",
            )
            if manifest_token and manifest_token.get(k)
        },
    }
    if extra:
        payload["extra"] = _stable(extra)
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


def configure_query_cache(
    *,
    max_bytes: int | None = None,
    max_entries: int | None = None,
    ttl_seconds: float | None = None,
) -> None:
    """运行时调整进程级查询缓存的联合约束（#4）。None 表示保持现值。"""
    global _QUERY_CACHE
    cache = _QUERY_CACHE
    with cache._lock:
        if max_bytes is not None:
            cache._max_bytes = max(1, int(max_bytes))
        if max_entries is not None:
            cache._max_entries = max(1, int(max_entries))
        if ttl_seconds is not None:
            cache._ttl = float(ttl_seconds)
        cache._evict_locked()


# 进程级单例
_QUERY_CACHE = QueryResultCache(
    max_entries=128,
    max_bytes=int(os.environ.get("DATA_ACCESS_QUERY_CACHE_BYTES", "0") or "0")
    or None,
)


def get_query_cache() -> QueryResultCache:
    return _QUERY_CACHE


def reset_query_cache() -> None:
    """清空进程级缓存（测试用）。"""
    global _QUERY_CACHE
    _QUERY_CACHE.clear()


__all__ = [
    "QueryResultCache",
    "query_cache_key",
    "get_query_cache",
    "result_cache_enabled",
    "set_result_cache_enabled",
    "configure_query_cache",
    "reset_query_cache",
]
