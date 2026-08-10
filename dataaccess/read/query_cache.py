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
        # key -> (ts, value, nbytes, meta)
        # ``meta``：命中时恢复 provenance（original snapshot_id / security
        # digest / source generation），cache hit 不再丢失审计与 lineage。
        self._store: "OrderedDict[str, tuple[float, Any, int, dict[str, Any]]]" = OrderedDict()
        self._lock = threading.Lock()

    def _total_bytes(self) -> int:
        return sum(item[2] for item in self._store.values())

    def get(self, key: str) -> Any | None:
        """返回缓存值（历史契约）；命中返回裸值。provenance 用 ``get_entry``。"""
        value, _meta = self.get_entry(key)
        return value

    def get_entry(self, key: str) -> tuple[Any, dict[str, Any] | None]:
        """R27（Cache/Write/Unsafe-Surface Closure）：命中返回 ``(value, meta)``。

        ``meta`` 携带写缓存时记录的 provenance（original_snapshot_id /
        security_digest / principal_id / source_generation / dataset），供
        cache hit 的审计与 lineage 恢复（R27-A/C：命中不能丢审计）。
        """
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None, None
            ts, value, _size, meta = item
            if time.monotonic() - ts > self._ttl:
                self._store.pop(key, None)
                return None, None
            self._store.move_to_end(key)
            return value, meta

    def set(self, key: str, value: Any, meta: dict[str, Any] | None = None) -> None:
        size = _estimate_bytes(value)
        with self._lock:
            self._store[key] = (time.monotonic(), value, size, dict(meta or {}))
            self._store.move_to_end(key)
            self._evict_locked()

    def set_with_size(
        self,
        key: str,
        value: Any,
        nbytes: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """显式指定缓存条目字节数（避免对已物化对象重复估算）。

        ``meta`` 保存 provenance，cache hit 恢复审计/lineage（R27-C）。
        """
        size = int(nbytes) if nbytes is not None else _estimate_bytes(value)
        with self._lock:
            self._store[key] = (time.monotonic(), value, size, dict(meta or {}))
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


class CachedReadResult:
    """R27-A/C —— ``read_cached_result()`` 的返回值：裸 Arrow Table + provenance。

    cache hit 不再丢审计/lineage：即使命中也携带 original_snapshot_id、
    security digest、principal、source generation 与 cache_provenance，
    供上层（FactorEngine 等）恢复 lineage 并记录 audit。
    """

    __slots__ = (
        "table",
        "cache_hit",
        "dataset",
        "snapshot_id",
        "source_generation",
        "security_digest",
        "principal_id",
        "cache_key",
        "cache_provenance",
        "rows",
        "nbytes",
    )

    def __init__(
        self,
        *,
        table: Any,
        cache_hit: bool,
        dataset: str,
        snapshot_id: str | None = None,
        source_generation: str | None = None,
        security_digest: str | None = None,
        principal_id: str | None = None,
        cache_key: str | None = None,
        cache_provenance: Mapping[str, Any] | None = None,
    ) -> None:
        self.table = table
        self.cache_hit = cache_hit
        self.dataset = dataset
        self.snapshot_id = snapshot_id
        self.source_generation = source_generation
        self.security_digest = security_digest
        self.principal_id = principal_id
        self.cache_key = cache_key
        self.cache_provenance = dict(cache_provenance or {})
        self.rows = getattr(table, "num_rows", None)
        self.nbytes = getattr(table, "nbytes", None)

    def to_arrow(self) -> Any:
        return self.table

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_hit": self.cache_hit,
            "dataset": self.dataset,
            "rows": self.rows,
            "nbytes": self.nbytes,
            "snapshot_id": self.snapshot_id,
            "source_generation": self.source_generation,
            "security_digest": self.security_digest,
            "principal_id": self.principal_id,
            "cache_key": self.cache_key,
            "cache_provenance": self.cache_provenance,
        }


def _security_scope_digest(*, principal: Any, access_policy: Any) -> str:
    """R27-A：缓存 key 的 security scope = principal + policy + run_mode。

    principal 缺省/默认（local, DEFAULT）返回空串——表示「未隔离身份」，
    由调用方决定是否允许共享缓存（restricted/premium 数据一律不缓存）。
    """
    payload: dict[str, Any] = {
        "principal_id": (
            getattr(principal, "principal_id", None) if principal is not None else None
        ),
        "policy_digest": (
            access_policy.digest()
            if access_policy is not None
            and callable(getattr(access_policy, "digest", None))
            else None
        ),
        "clearance": (
            getattr(principal, "clearance", None) if principal is not None else None
        ),
        "entitlements": (
            sorted(getattr(principal, "entitlements", ()) or ())
            if principal is not None
            else None
        ),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def classification_cache_blocked(classification: str | None) -> bool:
    """R27-A：restricted/premium 数据默认不进跨 principal shared result cache。

    即使 key 已含 security digest（per-principal 隔离），restricted/premium 的
    结果仍不允许进入进程级共享缓存——避免「高权限读取 → 同一进程其它身份/后续
    任务仍能拿到该表的内存引用」。classification 未知（unclassified）按可缓存
    处理（读本身仍受逻辑授权保护）。
    """
    return str(classification or "").strip().lower() in {"restricted", "premium"}


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
        # #P0-C1 严格区分 instrument_filter=[]（空股票池）与 None（全市场）：
        # None → null；[] → 空数组；["A","B"] → 排序后数组。旧写法
        # `sorted(...) if instruments else None` 把 [] 折叠成 None，全市场结果
        # 和空池结果会串 cache，命中错误结果。
        "instruments": (
            None
            if instruments is None
            else tuple(sorted(str(i) for i in instruments))
        ),
        # #P0-35 保留列顺序：read 输出列顺序与请求一致，[A,B] 和 [B,A] 是不同结果，
        # 不能用 sorted 合并成同一 key（会命中错误列序的缓存）。
        "columns": tuple(columns) if columns else None,
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
