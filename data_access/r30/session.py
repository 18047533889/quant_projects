"""data_access.r30.session —— R30-P0-005：R30ReadSession 源块复用 + 成本模型。

R30 在 R29 DataReadSession（resolution 缓存 + 上下文绑定）之上再进一步：
**不仅复用 path resolution，还复用 PreparedRead + 物理 source block**，由成本模型
决定 ``materialize vs relation reuse vs rescan``，并服从共享 resource envelope。

本模块完全 **additive**（不改 read/read_session.py / store.py / runtime/*）：
    - ``SourceBlockEntry`` / ``SourceBlockCache``    LRU 源块缓存（带 spillable
      偏好：非 spillable 条目不先 evict）
    - ``source_block_key``                           源块身份（含 snapshot /
      security / price_basis，用 stable_digest 折叠）
    - ``CostModel``                                  物化/复用/重扫决策
    - ``R30ReadSession``                             job 级读会话：resolution /
      prepared / source-block 三层缓存 + 上下文绑定 + calendar 冻结

**读场景只读契约**：job 内写数据会使缓存陈旧（路径/文件集/快照在写后变化）。
写场景请用 Store 直连或新开会话。
"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Mapping

from data_access.core.identity_encoder import CanonicalIdentityEncoder
from data_access.r30._shared import security_scope, stable_digest
from data_access.security.execution_context import (
    _execution_ctx_var,
    resolve_execution_context,
)

__all__ = [
    "CostModel",
    "R30ReadSession",
    "SourceBlockCache",
    "SourceBlockEntry",
    "source_block_key",
]


def _freeze(value: Any) -> Any:
    """把可变序列/dict 冻结成可哈希嵌套结构（进 prepared_cache key）。"""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_freeze(v) for v in value)
    return value


_IDENTITY_ENCODER = CanonicalIdentityEncoder(strict=True)


def _identity_digest(value: Any) -> str:
    """Encode correctness identity through the strict canonical authority."""
    return stable_digest(_IDENTITY_ENCODER.encode(value))


def _snapshot_identity(snapshot: Any) -> Any:
    """Extract an explicit snapshot identity; unsupported values fail closed later."""
    if snapshot is None or isinstance(snapshot, str):
        return snapshot
    for attr in ("snapshot_id", "content_digest", "identity"):
        value = getattr(snapshot, attr, None)
        if value is not None:
            return value
    return snapshot


def _materialize_to_arrow(result: Any):
    """把 execute_prepared_read 的结果物化成 Arrow Table。

    兼容两种形态：``ReadResult``（有 ``.table``）与 ``ReadHandle``（有
    ``.to_arrow()``）。两者都没有 → fail-closed（TypeError）。
    """
    table = getattr(result, "table", None)
    if table is not None:
        return table
    to_arrow = getattr(result, "to_arrow", None)
    if callable(to_arrow):
        return to_arrow()
    raise TypeError(
        f"无法从 {type(result).__name__} 物化 Arrow Table（需要 .table 或 .to_arrow()）"
    )


def _estimate_size(table: Any) -> int:
    """源块字节数估计：优先真实 nbytes，缺失用 CostModel 启发式。"""
    nbytes = getattr(table, "nbytes", None)
    if nbytes:
        try:
            return max(0, int(nbytes))
        except (TypeError, ValueError):
            pass
    rows = getattr(table, "num_rows", 0) or 0
    cols = getattr(table, "num_columns", 0) or 0
    return CostModel().estimate_memory(int(rows), int(cols))


def _validate_expected_fields(table: Any, expected_fields: Any) -> None:
    """fail-closed：期望字段必须出现在源块里（缺字段直接抛，不静默返回子集）。"""
    if not expected_fields:
        return
    names: set[str] = set()
    for attr in ("column_names", "columns"):
        v = getattr(table, attr, None)
        if isinstance(v, (list, tuple)):
            names = {str(x) for x in v}
            break
    if names:
        missing = [str(f) for f in expected_fields if str(f) not in names]
        if missing:
            raise ValueError(f"source block 缺少期望字段: {missing}")


@dataclass
class SourceBlockEntry:
    """一个源块条目的元数据（值本身存在 SourceBlockCache._data）。"""

    key: str
    size: int  # bytes
    reuse_count: int = 0
    refcount: int = 0
    last_access: float = field(default_factory=time.monotonic)
    spillable: bool = True
    security_scope: str | None = None
    snapshot_identity: str | None = None

    def touch(self) -> None:
        """一次命中：复用计数 +1，刷新最近访问时间。"""
        self.reuse_count += 1
        self.last_access = time.monotonic()


class SourceBlockCache:
    """LRU 源块缓存。

    - ``get`` 命中 → ``entry.touch()``（reuse_count+1）+ ``refcount+1``；
    - 超 ``max_bytes`` / ``max_entries`` 按 LRU evict；**spillable=False 的条目
      不先 evict**（等所有 spillable 条目 evict 后才动它们）。
    """

    def __init__(self, max_bytes: int | None, max_entries: int = 1024) -> None:
        if max_bytes is not None and int(max_bytes) <= 0:
            raise ValueError("max_bytes 必须为正或 None（不限）")
        self._max_bytes: int | None = (
            None if max_bytes is None else int(max_bytes)
        )
        self._max_entries = max(1, int(max_entries))
        #: OrderedDict：插入序即 LRU 序（前端最旧，后端最新）。
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._entries: dict[str, SourceBlockEntry] = {}
        self._total_size = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    # ---- 访问 ----

    def get(self, key: str) -> Any | None:
        """命中 → touch + refcount+1 + 移到 MRU；未命中 → None。"""
        if key in self._data:
            entry = self._entries[key]
            entry.touch()
            entry.refcount += 1
            self._data.move_to_end(key)
            self._hits += 1
            return self._data[key]
        self._misses += 1
        return None

    def put(
        self,
        key: str,
        value: Any,
        size: int,
        security_scope: str | None = None,
        snapshot_identity: str | None = None,
        spillable: bool = True,
    ) -> None:
        """写入/更新一个源块；超限按 LRU（spillable 优先）evict。"""
        size = max(0, int(size))
        if key in self._data:
            old = self._entries[key]
            self._total_size += size - old.size
            old.size = size
            old.spillable = bool(spillable)
            old.security_scope = security_scope
            old.snapshot_identity = snapshot_identity
            old.refcount = 0
            self._data[key] = value
            self._data.move_to_end(key)
        else:
            entry = SourceBlockEntry(
                key=key,
                size=size,
                spillable=bool(spillable),
                security_scope=security_scope,
                snapshot_identity=snapshot_identity,
            )
            self._entries[key] = entry
            self._data[key] = value
            self._total_size += size
            self._data.move_to_end(key)
        self._evict()

    def evict(self, until_bytes: int) -> int:
        """LRU evict（spillable 优先）直到总字节数 <= until_bytes；返回剩余字节。"""
        target = max(0, int(until_bytes))
        while self._total_size > target and self._data:
            if not self._evict_one():
                break
        return self._total_size

    # ---- 内部 ----

    def _evict(self) -> None:
        if self._max_entries is not None:
            while len(self._data) > self._max_entries:
                if not self._evict_one():
                    break
        if self._max_bytes is not None:
            while self._total_size > self._max_bytes and self._data:
                if not self._evict_one():
                    break

    def _evict_one(self) -> bool:
        """evict 最旧的一条目：优先 spillable（非 spillable 最后才动）。"""
        for key in list(self._data.keys()):
            if self._entries[key].spillable:
                self._remove(key)
                return True
        for key in list(self._data.keys()):
            self._remove(key)
            return True
        return False

    def _remove(self, key: str) -> None:
        entry = self._entries.pop(key)
        self._data.pop(key, None)
        self._total_size -= entry.size
        self._evictions += 1

    # ---- 状态 ----

    def stats(self) -> dict[str, Any]:
        return {
            "size": self._total_size,
            "max_bytes": self._max_bytes,
            "entries": len(self._data),
            "max_entries": self._max_entries,
            "reuse_count": sum(e.reuse_count for e in self._entries.values()),
            "hit": self._hits,
            "miss": self._misses,
            "evictions": self._evictions,
        }

    def clear(self) -> None:
        self._data.clear()
        self._entries.clear()
        self._total_size = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def __len__(self) -> int:
        return len(self._data)


def source_block_key(
    dataset: str,
    snapshot: Any,
    time_range: Any,
    universe: Any,
    canonical_fields: Any,
    filters: Any,
    pit_contract: Any,
    security_digest: Any,
    price_basis: Any,
) -> str:
    """源块身份：dataset + snapshot + time_range + universe + fields + filters +
    pit_contract + security_digest + price_basis 全部折叠进 stable_digest。

    不同 principal/policy 的 security_digest 不同 → 不同 key（绝不跨 scope 共享）。
    snapshot 身份由 ``_snapshot_identity`` 提取（对象优先 snapshot_id）。
    """
    identity = {
        "dataset": dataset,
        "snapshot": _snapshot_identity(snapshot),
        "time_range": time_range,
        "universe": universe,
        "canonical_fields": canonical_fields,
        "filters": filters,
        "pit_contract": pit_contract,
        "security_digest": security_digest,
        "price_basis": price_basis,
    }
    return _identity_digest(identity)


class CostModel:
    """源块复用成本模型（R30-P0-005）。

    决策三态：
        - ``materialize``  物化到 Arrow 并缓存（多次复用收益 > 物化成本）；
        - ``relation``     保留 relation/不物化（数据太大放不进内存，且重扫比
                           远端更贵 → 偏向复用 relation 而非落盘缓存）；
        - ``rescan``       每次重扫（复用收益不足 / 内存装不下）。
    """

    def should_materialize(
        self,
        reuse_count: int,
        estimated_size: int,
        rescan_cost: int,
        remote_cost: int,
        available_memory: int,
        materialize_cost: int,
    ) -> str:
        """返回决策字符串。"""
        if estimated_size > available_memory:
            # 内存装不下 → 不能物化。重扫 <= 远端成本时宁可重扫，否则偏向
            # relation 复用（保留 relation，不落盘物化）。
            return "rescan" if rescan_cost <= remote_cost else "relation"
        if int(reuse_count) * int(rescan_cost) > int(materialize_cost):
            return "materialize"
        return "rescan"

    def estimate_memory(self, rows: int, columns: int) -> int:
        """启发式内存估计：rows × columns × 8 bytes（float64 单元格）。"""
        rows = max(0, int(rows or 0))
        columns = max(0, int(columns or 0))
        return rows * max(1, columns) * 8


class R30ReadSession:
    """job 级读会话：三层缓存（resolution / prepared / source-block）+ 成本模型。

    - ``resolution_cache``  注入 store（经 read_session_context ContextVar），
      prepare 命中跳过 glob/stat/镜像检查；
    - ``prepared_cache``    (dataset, params_fp, time_range, instrument_filter,
      filters, columns, security_scope) → PreparedRead；命中不重复 prepare；
    - ``source_block_cache`` LRU，key 含 snapshot + security + price_basis；
      由 ``CostModel`` 决定是否物化缓存（不 materialize 就不 put）。

    ContextVar 隔离：A/B 两个 session overlap 时各自持有自己的 token 与 dict，
    互不串（resolution cache 各自 dict，经 ``set_resolution_cache`` 绑定当前
    request context）。

    **读场景只读契约**：job 内写数据会使本会话缓存陈旧——写场景请用 Store 直连
    或新开会话。
    """

    _DEFAULT_MAX_SOURCE_BLOCK_BYTES = 256 * 1024 * 1024  # 256 MiB

    def __init__(
        self,
        store: Any,
        *,
        principal: Any = None,
        authorizer: Any = None,
        credential_provider: Any = None,
        run_mode: Any = None,
        request_id: str | None = None,
        max_source_block_bytes: int | None = _DEFAULT_MAX_SOURCE_BLOCK_BYTES,
        cost_model: Any = None,
    ) -> None:
        self._store = store
        base = resolve_execution_context(
            principal=principal,
            authorizer=authorizer,
            credential_provider=credential_provider,
            run_mode=run_mode,
        )
        # 同 DataReadSession：带上 request_id，security_digest 沿用 resolve 结果。
        self._ctx = type(base)(
            principal=base.principal,
            authorizer=base.authorizer,
            access_policy=base.access_policy,
            credential_provider=base.credential_provider,
            request_id=request_id,
            run_mode=base.run_mode,
            security_digest=base.security_digest,
        )
        self._scope_digest: str | None = (
            self._ctx.security_digest or security_scope(store)
        )
        self.resolution_cache: dict[Any, Any] = {}
        self.prepared_cache: dict[tuple[Any, ...], Any] = {}
        self.source_block_cache = SourceBlockCache(max_bytes=max_source_block_bytes)
        self.cost_model = cost_model if cost_model is not None else CostModel()
        self._max_source_block_bytes: int | None = max_source_block_bytes
        self._prepared_hits = 0
        self._prepared_misses = 0
        self._source_block_requests: dict[str, int] = {}
        self._token: Any = None
        self._cache_token: Any = None
        self._mode_token: Any = None
        self._closed = False

    # ---- 生命周期 ----

    def __enter__(self) -> "R30ReadSession":
        if self._closed:
            raise RuntimeError("R30ReadSession 已关闭，无法再次进入")
        # 1) 绑定请求级执行上下文（整个 job）。
        self._token = _execution_ctx_var.set(self._ctx)
        # 1b) job 级 RuntimeModeIdentity（单一权威；防御性 import）。
        try:
            from data_access.runtime.mode_identity import set_runtime_mode_identity

            rm = getattr(self._ctx, "run_mode", None)
            if rm is not None:
                self._mode_token = set_runtime_mode_identity(rm, source="R30ReadSession")
        except Exception:
            self._mode_token = None
        # 2) resolution 缓存放 ContextVar（request-scoped，并发 session 互不覆盖）。
        from data_access.runtime.read_session_context import set_resolution_cache

        self._cache_token = set_resolution_cache(self.resolution_cache)
        # 3) job 级冻结 calendar 世界（PIT 语义跨因子稳定）。
        try:
            lock_fn = getattr(self._store, "lock_calendars", None)
            if callable(lock_fn):
                lock_fn()
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._token is not None:
            _execution_ctx_var.reset(self._token)
            self._token = None
        if getattr(self, "_cache_token", None) is not None:
            try:
                from data_access.runtime.read_session_context import (
                    reset_resolution_cache,
                )

                reset_resolution_cache(self._cache_token)
            except Exception:
                pass
            self._cache_token = None
        if getattr(self, "_mode_token", None) is not None:
            try:
                from data_access.runtime.mode_identity import (
                    reset_runtime_mode_identity,
                )

                reset_runtime_mode_identity(self._mode_token)
            except Exception:
                pass
            self._mode_token = None
        self._closed = True

    # ---- 读路径 ----

    def prepare_once(
        self,
        dataset: str,
        *,
        columns: Any = None,
        time_range: Any = None,
        instrument_filter: Any = None,
        filters: Any = None,
        params: Mapping[str, Any] | None = None,
        security_digest: str | None = None,
    ):
        """Prepare 一次并缓存：同一 (dataset, params_fp, time_range,
        instrument_filter, filters, columns, security_scope) 只 prepare 一次。

        key 含 security_scope —— 不同 principal/policy 绝不共享 PreparedRead。
        """
        scope = security_digest or self._scope_digest
        params_fp = _identity_digest(params)
        key = (
            dataset,
            params_fp,
            _freeze(time_range),
            _freeze(instrument_filter),
            _freeze(filters),
            _freeze(columns),
            scope,
        )
        cached = self.prepared_cache.get(key)
        if cached is not None:
            self._prepared_hits += 1
            return cached
        self._prepared_misses += 1
        prepared = self._store.prepare_read(
            dataset,
            columns=columns,
            time_range=time_range,
            instrument_filter=instrument_filter,
            filters=filters,
            params=params,
        )
        self.prepared_cache[key] = prepared
        return prepared

    def read_source_block(
        self,
        dataset: str,
        *,
        columns: Any = None,
        time_range: Any = None,
        instrument_filter: Any = None,
        filters: Any = None,
        params: Mapping[str, Any] | None = None,
        universe: Any = None,
        canonical_fields: Any = None,
        pit_contract: Any = None,
        price_basis: Any = None,
        security_digest: str | None = None,
        expected_fields: Any = None,
        snapshot_identity: str | None = None,
    ):
        """读一个物理 source block（Arrow Table）。

        先查 ``source_block_cache``（key 含 snapshot/security/price_basis）；命中
        直接返回。未命中 → ``execute_prepared_read`` → Arrow → 按 CostModel 决定
        是否 put（不 materialize 就返回不缓存）。
        """
        prepared = self.prepare_once(
            dataset,
            columns=columns,
            time_range=time_range,
            instrument_filter=instrument_filter,
            filters=filters,
            params=params,
            security_digest=security_digest,
        )
        if snapshot_identity is None:
            snapshot_identity = _snapshot_identity(
                getattr(prepared, "resolved_source_snapshot", None)
            )
        scope = security_digest or self._scope_digest
        key = source_block_key(
            dataset,
            snapshot_identity,
            time_range,
            universe,
            canonical_fields,
            filters,
            pit_contract,
            scope,
            price_basis,
        )
        cached = self.source_block_cache.get(key)
        if cached is not None:
            _validate_expected_fields(cached, expected_fields)
            return cached
        self._source_block_requests[key] = self._source_block_requests.get(key, 0) + 1
        reuse_count = self._source_block_requests[key]
        result = self._store.execute_prepared_read(prepared)
        table = _materialize_to_arrow(result)
        _validate_expected_fields(table, expected_fields)
        size = _estimate_size(table)
        decision = self.cost_model.should_materialize(
            reuse_count=reuse_count,
            estimated_size=size,
            rescan_cost=size,
            remote_cost=size * 3,
            available_memory=self._max_source_block_bytes,
            materialize_cost=size * 2,
        )
        if decision == "materialize":
            self.source_block_cache.put(
                key,
                table,
                size,
                security_scope=scope,
                snapshot_identity=snapshot_identity,
            )
        return table

    def read(self, dataset: str, **kwargs: Any):
        """便捷入口 = prepare_once + read_source_block。"""
        return self.read_source_block(dataset, **kwargs)

    # ---- 状态 ----

    def stats(self) -> dict[str, Any]:
        """resolution / prepared / source-block 三层命中率 + 缓存大小。"""
        sc = self.source_block_cache.stats()
        return {
            "resolution_cache": {"entries": len(self.resolution_cache)},
            "prepared_cache": {
                "entries": len(self.prepared_cache),
                "hits": self._prepared_hits,
                "misses": self._prepared_misses,
            },
            "source_block_cache": sc,
        }

    def clear_resolution_cache(self) -> None:
        """job 内显式失效全部缓存（如外部数据变化后）。"""
        self.resolution_cache.clear()
        self.prepared_cache.clear()
        self.source_block_cache.clear()
