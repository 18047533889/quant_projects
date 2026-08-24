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

import json
import logging
import os
import sys
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Mapping, Sequence

from data_access.core.identity_encoder import (
    hash_correctness_identity,
    hash_security_identity,
)

logger = logging.getLogger("data_access.query_cache")

# R39 #71：fail-closed 默认字节上限（env 未设置 / FE coordinator 不存在时）。
# 256 MiB 是保守 safe default——有上限永远好过 None≈无限。
_DEFAULT_CACHE_BYTES = 256 * 1024 * 1024


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

    def shrink_to(self, target_bytes: int) -> None:
        """R39 #72：动态收缩 hook——逐出 LRU 最旧条目直到总字节 <= target。

        ResourceAutopilot 在内存压力下把 cache budget 收缩到 target 时调用。
        """
        target = max(0, int(target_bytes))
        with self._lock:
            while self._store and self._total_bytes() > target:
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
    return hash_security_identity(payload).digest


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
    from data_access.read.read_contract import canonicalize_params

    payload: dict[str, Any] = {
        "dataset": dataset,
        "params": canonicalize_params(params),
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
        payload["extra"] = dict(extra)
    return hash_correctness_identity(payload).digest


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


def _fe_cache_budget_bytes() -> int | None:
    """从 FE ResourceAutopilotService / HostResourceCoordinator 读取 cache budget。

    FE 是 flat package（``runtime.*``）；DA/FE 同进程时懒加载。都不可用 →
    返回 None（调用方回退保守默认）。绝不抛异常。
    """
    # 优先读 autopilot 最新 decision（无副作用；stale → 不看）。
    try:
        from factor_engine.runtime.resource_autopilot_service import get_resource_autopilot

        autopilot = get_resource_autopilot()
        if autopilot is not None:
            snap = autopilot.last_decision()
            if snap is not None and not getattr(snap, "is_stale", True):
                budget = getattr(snap.decision, "cache_budget_bytes", None)
                if budget:
                    return max(1, int(budget))
    except Exception:
        pass
    try:
        from factor_engine.runtime.host_resource_coordinator import get_host_coordinator

        decision = get_host_coordinator().decision()
        budget = getattr(decision, "cache_budget_bytes", None)
        if budget:
            return max(1, int(budget))
    except Exception:
        pass
    return None


def _ensure_query_cache_byte_cap() -> None:
    """R39 #71：字节上限 fail-closed 默认。

    进程级缓存初始化时若 env 未给 ``DATA_ACCESS_QUERY_CACHE_BYTES``，max_bytes
    为 None（≈无限）是内存安全漏洞。首次访问时：优先取 FE coordinator 的
    cache_budget_bytes，否则保守默认 256 MiB。
    """
    cache = _QUERY_CACHE
    if cache.max_bytes is not None:
        return
    cap = _fe_cache_budget_bytes() or _DEFAULT_CACHE_BYTES
    with cache._lock:
        if cache._max_bytes is None:
            cache._max_bytes = max(1, int(cap))
            cache._evict_locked()


def _sync_cache_inventory() -> None:
    """R39 #74：把 DA QueryResultCache 注册进统一 cache inventory（幂等）。"""
    try:
        from data_access.runtime.cache_inventory import register_cache_owner

        register_cache_owner(
            "da_query_result_cache",
            current_bytes=_QUERY_CACHE.total_bytes,
            shrink_to=_QUERY_CACHE.shrink_to,
            max_bytes=lambda: _QUERY_CACHE.max_bytes,
        )
    except Exception:  # pragma: no cover
        pass


def configure_query_cache(
    *,
    max_bytes: int | None = None,
    max_entries: int | None = None,
    ttl_seconds: float | None = None,
) -> None:
    """运行时调整进程级查询缓存的联合约束（#4）。None 表示保持现值。"""
    global _QUERY_CACHE
    cache = _QUERY_CACHE
    if max_bytes is not None:
        # 显式配置永远优先于 fail-closed 默认。
        cache._max_bytes = max(1, int(max_bytes))
    else:
        _ensure_query_cache_byte_cap()
    with cache._lock:
        if max_entries is not None:
            cache._max_entries = max(1, int(max_entries))
        if ttl_seconds is not None:
            cache._ttl = float(ttl_seconds)
        cache._evict_locked()
    _sync_cache_inventory()


def apply_resource_cache_budget(max_bytes: int) -> None:
    """R39 #72：FE ResourceAutopilotService 的 cache 消费者入口。

    autopilot 每个 control tick 把 ``cache_budget_bytes`` 应用到 DA 缓存：
    设新上限 + 超限立即 shrink（LRU）。这是动态 grow/shrink 的真实接线。
    """
    cache = get_query_cache()
    budget = max(1, int(max_bytes))
    with cache._lock:
        cache._max_bytes = budget
        cache._evict_locked()
        while cache._store and cache._total_bytes() > budget:
            cache._store.popitem(last=False)
    _sync_cache_inventory()


def _resolve_env_cache_bytes() -> int | None:
    """env DATA_ACCESS_QUERY_CACHE_BYTES 显式上限；未设置/0 → None（惰性默认）。"""
    raw = os.environ.get("DATA_ACCESS_QUERY_CACHE_BYTES", "").strip()
    if raw and raw != "0":
        try:
            return max(1, int(raw))
        except ValueError:
            logger.warning(
                "DATA_ACCESS_QUERY_CACHE_BYTES=%r 非整数，忽略（使用惰性默认）", raw
            )
    return None


# 进程级单例（R39 #71：字节上限惰性解析 fail-closed 默认，见 _ensure_query_cache_byte_cap）。
_QUERY_CACHE = QueryResultCache(
    max_entries=128,
    max_bytes=_resolve_env_cache_bytes(),
)


def get_query_cache() -> QueryResultCache:
    _ensure_query_cache_byte_cap()
    _sync_cache_inventory()
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
    "apply_resource_cache_budget",
    "reset_query_cache",
]
