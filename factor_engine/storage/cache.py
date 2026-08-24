# -*- coding: utf-8 -*-
"""计划子树执行缓存：内存 + 可选磁盘持久化。"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

#: 磁盘缓存格式版本（R20-153..172：schema_version 校验，旧格式 fail-closed）。
PLAN_CACHE_SCHEMA_VERSION = 2
#: 编译器语义版本：数值语义 / lowering / optimizer 任一变化都会 invalidate
#: persistent cache 命名空间（R20-159 / R20-468）。
_OPTIMIZER_COMPILER_SEMANTIC_VERSION = "1"
_LOWERING_SEMANTIC_VERSION = "1"
#: R32-P1-045: bounded per-key 写锁 registry（多 writer 同 key 交错生成
#: payload/meta 的防护）。历史实现 ``dict[str, Lock]`` 对每个唯一 key 保留
#: Lock，per-key 表会无限增长。现在用有界 LRU（最多 ``_SAVE_LOCK_MAX`` 个锁，
#: 超限时淘汰最久未用的 key）。
#: R40 #120: 淘汰只允许命中 refcount==0 的 key —— 正被 ``with`` 持有的锁绝不
#: 淘汰。refcount 在 ``__enter__`` 递增 / ``__exit__`` 递减（``_RefCountedSaveLock``）。
_SAVE_LOCKS: dict[str, threading.Lock] = {}
_SAVE_LOCKS_GUARD = threading.Lock()
_SAVE_LOCKS_ORDER: list[str] = []
_SAVE_LOCK_REFS: dict[str, int] = {}
_SAVE_LOCK_MAX = 4096


class _RefCountedSaveLock:
    """R40 #120: context-manager 包装的 per-key 写锁。

    ``__enter__`` 在 ``_SAVE_LOCKS_GUARD`` 内把该 key 的 refcount +1（并重新取
    当前 registry 里的锁 —— 若在 ``_save_lock_for`` 与 ``__enter__`` 之间被
    淘汰，则取/建新锁，绝不在孤儿锁上序列化），``__exit__`` -1。持有期间
    refcount > 0，LRU 淘汰永不命中该 key。
    """

    __slots__ = ("_key", "_lock", "_acquired")

    def __init__(self, key: str) -> None:
        self._key = key
        self._lock: threading.Lock | None = None
        self._acquired = False

    def __enter__(self) -> "_RefCountedSaveLock":
        with _SAVE_LOCKS_GUARD:
            lock = _SAVE_LOCKS.get(self._key)
            if lock is None:
                lock = threading.Lock()
                _SAVE_LOCKS[self._key] = lock
                _SAVE_LOCKS_ORDER.append(self._key)
            else:
                try:
                    _SAVE_LOCKS_ORDER.remove(self._key)
                except ValueError:
                    pass
                _SAVE_LOCKS_ORDER.append(self._key)
            _SAVE_LOCK_REFS[self._key] = _SAVE_LOCK_REFS.get(self._key, 0) + 1
            self._lock = lock
        # P0-FIX: Add timeout to prevent indefinite deadlock
        if not self._lock.acquire(timeout=30.0):
            import logging
            logging.getLogger(__name__).error(
                f"Failed to acquire cache save lock for key={self._key!r} within 30s. "
                f"Potential deadlock detected."
            )
            with _SAVE_LOCKS_GUARD:
                refs = _SAVE_LOCK_REFS.get(self._key, 0) - 1
                if refs > 0:
                    _SAVE_LOCK_REFS[self._key] = refs
                else:
                    _SAVE_LOCK_REFS.pop(self._key, None)
            raise RuntimeError(
                f"Failed to acquire cache save lock for key={self._key!r} within 30s - "
                f"potential deadlock or long-running critical section"
            )
        self._acquired = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            if self._lock is not None:
                self._lock.release()
        finally:
            with _SAVE_LOCKS_GUARD:
                refs = _SAVE_LOCK_REFS.get(self._key, 0) - 1
                if refs > 0:
                    _SAVE_LOCK_REFS[self._key] = refs
                else:
                    _SAVE_LOCK_REFS.pop(self._key, None)
            self._acquired = False

    @property
    def lock(self) -> threading.Lock:
        assert self._lock is not None
        return self._lock


def _save_lock_for(key: str) -> _RefCountedSaveLock:
    """R32-P1-045 + R40 #120: bounded lock registry —— 只淘汰 refcount==0 的 key。

    返回 context-manager 包装器（``with _save_lock_for(key):``）。超限时只淘汰
    最久未用且**当前无 writer 持有**（refcount==0）的 key；全部 key 都在持有时
    暂不淘汰（有界性退化为"同时持有锁的数量"）。
    """
    global _SAVE_LOCKS, _SAVE_LOCKS_ORDER
    with _SAVE_LOCKS_GUARD:
        lock = _SAVE_LOCKS.get(key)
        if lock is not None:
            try:
                _SAVE_LOCKS_ORDER.remove(key)
            except ValueError:
                pass
            _SAVE_LOCKS_ORDER.append(key)
        else:
            lock = threading.Lock()
            _SAVE_LOCKS[key] = lock
            _SAVE_LOCKS_ORDER.append(key)
        while len(_SAVE_LOCKS) > _SAVE_LOCK_MAX:
            oldest = next(
                (c for c in _SAVE_LOCKS_ORDER if _SAVE_LOCK_REFS.get(c, 0) == 0),
                None,
            )
            if oldest is None:
                break  # 所有锁都在持有中：绝不淘汰正被持有的锁
            _SAVE_LOCKS_ORDER.remove(oldest)
            _SAVE_LOCKS.pop(oldest, None)
            _SAVE_LOCK_REFS.pop(oldest, None)
        return _RefCountedSaveLock(key)


def _governor():
    """惰性导入全局 MemoryGovernor，避免 storage → runtime → storage 循环导入。"""
    from factor_engine.runtime.resource_governor import global_memory_governor

    return global_memory_governor()


class _SharedCache:
    """Shared mutable backing store + byte counter for ``with_scope`` instances.

    R13 P1-62: the old ``with_scope`` copied ``_cache`` and ``_bytes`` but never
    reserved the copied bytes in the MemoryGovernor, so a child scope calling
    ``set``/``evict``/``clear`` released accounting it never reserved and the
    global accounting drifted below true resident bytes.  Sharing ONE mutable
    backing store (dict + byte counter) makes every operation on any scope touch
    the same bytes, so governor accounting always equals resident bytes.

    R20-119..124: ``lock``（``threading.RLock``）保护 dict 与 byte counter —— 多个
    scope 并发 set/get/evict/clear 必须同一个原子 critical section。
    """

    __slots__ = ("cache", "nbytes", "refs", "lock")

    def __init__(self) -> None:
        self.cache: dict[str, object] = {}
        self.nbytes: int = 0
        self.refs: int = 1
        self.lock = threading.RLock()


class CacheManager:
    """计划子树执行结果的内存键值缓存。
    
    参数:
        data_scope: 数据作用域指纹（可选）
    
    
    - **列数据**可由数据源自行管理；本类主要用于 ``PandasBackend`` **子树求值结果**复用
          （键为计划子树的结构化字符串 + 可选 ``data_scope``）。
    """

    def __init__(
        self,
        *,
        data_scope: str | None = None,
        budget_bytes: int | None = None,
        layer_name: str = "l2_subplan",
    ) -> None:
        """初始化实例。

        参数:
            data_scope: 数据作用域指纹（可选）
            budget_bytes: 子计划结果字节预算（Phase 5 R6；``None`` 按进程预算比例）
            layer_name: MemoryGovernor 记账层名（审计 #333；``PersistentPlanCache``
                继承本类也用 ``"l2_subplan"``）

        返回:
            无
        """
        self.data_scope = data_scope
        self.layer_name = layer_name
        # R6-151: insertion-ordered dict so eviction is true LRU — a ``get()``
        # moves the entry to the most-recent end, and eviction drops the oldest
        # (least-recently-used) entry.  The previous plain dict evicted by
        # creation order, so a hot entry created early was evicted first.
        self._shared = _SharedCache()
        self._budget_bytes = budget_bytes

    @property
    def _cache(self) -> dict[str, object]:
        """LRU dict of the shared backing store (R13 P1-62)."""
        return self._shared.cache

    @_cache.setter
    def _cache(self, value: dict[str, object]) -> None:
        self._shared.cache = value

    @property
    def _bytes(self) -> int:
        """Resident byte counter of the shared backing store (R13 P1-62)."""
        return self._shared.nbytes

    @_bytes.setter
    def _bytes(self, value: int) -> None:
        self._shared.nbytes = int(value)

    @property
    def budget_bytes(self) -> int:
        if self._budget_bytes is not None and self._budget_bytes > 0:
            return self._budget_bytes
        try:
            from factor_engine.runtime.resource_governor import global_memory_governor

            return int(global_memory_governor().process_budget_bytes * 0.05)
        except Exception:
            return 512 * 1024 * 1024

    def _scoped_key(self, key: str) -> str:
        """生成防碰撞的 scoped key。

        参数:
            key: 缓存键

        返回:
            str

        修复：使用 JSON 编码的 [scope, key] 对避免分隔符碰撞。旧实现
        ``f"{scope}:{key}"`` 对 ``(scope="a", key="b:c")`` 与
        ``(scope="a:b", key="c")`` 产生相同字符串 ``"a:b:c"``，导致不同逻辑
        缓存条目共享同一磁盘文件。新实现生成可区分的 scoped key：
            - ``(scope="a", key="b:c")`` → ``'["a","b:c"]'``
            - ``(scope="a:b", key="c")`` → ``'["a:b","c"]'``
        """
        if self.data_scope:
            return json.dumps([self.data_scope, key], separators=(",", ":"), sort_keys=False)
        return key

    def get(self, key: str):
        """get。

        参数:
            key: 缓存键

        返回:
            无
        """
        scoped = self._scoped_key(key)
        with self._shared.lock:
            value = self._cache.get(scoped)
            if value is not None:
                # R6-151: record a hit as "most recently used" so LRU eviction
                # (drop the least-recently-used entry) reflects access, not just
                # insertion.  Plain-dict insertion order made eviction FIFO.
                self._cache.pop(scoped, None)
                self._cache[scoped] = value
            return value

    def set(self, key: str, value) -> None:
        """set；受字节预算约束（LRU 逐出）。

        参数:
            key: 缓存键
            value: 缓存值

        返回:
            无
        """
        from factor_engine.runtime.resource_governor import estimate_object_bytes

        gov = _governor()
        scoped = self._scoped_key(key)
        size = estimate_object_bytes(value)
        if size > self.budget_bytes:
            return
        # R20-119..124：dict mutation / byte counter / governor accounting 必须
        # 同一个原子 critical section。锁顺序恒为 governor → shared（与
        # ``MemoryGovernor._evict_for`` 一致，避免锁反转死锁）。
        with gov.lock:
            with self._shared.lock:
                if scoped in self._cache:
                    old_size = estimate_object_bytes(self._cache[scoped])
                    self._bytes -= old_size
                    gov.release_accounting(self.layer_name, old_size)
                    # R20-146..152：overwrite 先 pop 再 insert，让新值成为 MRU
                    # （plain dict 对已有 key 重新赋值不改变插入位置）。
                    self._cache.pop(scoped, None)
                self._bytes += size
                gov.reserve_accounting(self.layer_name, size)
                self._cache[scoped] = value
                freed = self._evict_to(self.budget_bytes)
                if freed > 0:
                    gov.release_accounting(self.layer_name, freed)

    def _evict_to(self, target: int) -> int:
        from factor_engine.runtime.resource_governor import estimate_object_bytes

        freed = 0
        while self._cache and self._bytes > target:
            scoped, value = next(iter(self._cache.items()))
            size = estimate_object_bytes(value)
            self._bytes = max(0, self._bytes - size)
            del self._cache[scoped]
            freed += size
        return freed

    def evict_if_over_budget(self, target: int = 0) -> int:
        """MemoryGovernor evict hook：逐出到 ``target`` 字节。

        审计 #332：``target > 0`` 用 ``target``，否则逐出到自身 ``budget_bytes``
        （governor 触发路径的记账由 ``MemoryGovernor._evict_for`` 统一扣减，
        这里只逐出并返回释放字节数，避免重复记账）。
        """
        gov = _governor()
        with gov.lock:
            with self._shared.lock:
                return self._evict_to(target if target > 0 else self.budget_bytes)

    def clear_memory(self) -> None:
        """clear_memory。

        参数:
            无

        返回:
            无
        """
        gov = _governor()
        with gov.lock:
            with self._shared.lock:
                self._cache.clear()
                self._bytes = 0
            gov.release_all(self.layer_name)

    def with_scope(self, data_scope: str, *, clear_memory: bool = False) -> CacheManager:
        """返回同类型实例并切换作用域（用于增量窗口隔离）。

        参数:
            data_scope: 数据作用域指纹
            clear_memory: 见函数签名（可选）

        返回:
            CacheManager

        R13 P1-62：``clear_memory=False`` 时**共享**同一 backing store（dict +
        字节计数），不再复制 ``_bytes`` 而不 reserve。所有 scope 的
        set/evict/clear 都作用于同一字典与同一字节计数，MemoryGovernor 记账
        始终等于真实驻留字节。``clear_memory=True`` 时创建全新的空 store。
        """
        out = type(self)(
            data_scope=data_scope,
            budget_bytes=self._budget_bytes,
            layer_name=self.layer_name,
        )
        if not clear_memory:
            out._shared = self._shared
            out._shared.refs += 1
        return out


UNKNOWN_CACHE_NAMESPACE = "unknown_ops"


def _operator_namespace() -> str:
    """获取算子目录哈希命名空间（用于磁盘缓存路径隔离）。

    参数:
        无

    返回:
        str

    R20-159：命名空间必须覆盖 numeric semantics / lowering semantics / optimizer
    compiler semantic version —— 任一语义变更都 invalidate 整个磁盘缓存目录，
    绝不从旧语义的缓存里拿子树。

    R32-P1-046: 解析失败返回 ``UNKNOWN_CACHE_NAMESPACE``（"unknown_ops"）。
    production 下由调用方（``_namespace_root``）fail-closed 抛错 —— 未解析
    namespace 的 production 缓存读写是语义污染的来源，绝不静默当 cache miss。
    """
    try:
        from factor_engine.cleaned_operators.operator_policy import compute_operator_catalog_hash
        from factor_engine.fields import compute_field_catalog_hash

        return (
            f"{compute_operator_catalog_hash()[:16]}-"
            f"{compute_field_catalog_hash()[:16]}-"
            f"{_compiler_semantic_hash()}"
        )
    except Exception:
        return UNKNOWN_CACHE_NAMESPACE


def _compiler_semantic_hash() -> str:
    """Numeric / lowering / optimizer compiler semantic 摘要（R20-159 / R20-468）。

    - ``numeric_semantics_hash()``：真实数值语义指纹（ddof/tie/div-zero/NaN 等）
    - ``_LOWERING_SEMANTIC_VERSION`` / ``_OPTIMIZER_COMPILER_SEMANTIC_VERSION``：
      模块级版本常量，任何 lowering/optimizer 语义改动 bump 即 invalidate 缓存

    测试可通过 monkeypatch ``storage.cache._LOWERING_SEMANTIC_VERSION`` /
    ``_OPTIMIZER_COMPILER_SEMANTIC_VERSION`` 验证缓存失效。
    """
    import hashlib

    parts: dict[str, str] = {
        "lowering_semantic_version": _LOWERING_SEMANTIC_VERSION,
        "optimizer_compiler_semantic_version": _OPTIMIZER_COMPILER_SEMANTIC_VERSION,
    }
    try:
        from factor_engine.backend.numeric_semantics import numeric_semantics_hash

        parts["numeric_semantics"] = numeric_semantics_hash()
    except Exception:
        parts["numeric_semantics"] = "unknown"
    raw = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _series_to_frame(series: pd.Series) -> pd.DataFrame:
    """将 MultiIndex Series 转为可序列化的 DataFrame。
    
    参数:
        series: MultiIndex Series
    
    返回:
        pd.DataFrame
    """
    frame = series.reset_index()
    if series.name is not None:
        frame = frame.rename(columns={series.name: "__value__"})
    else:
        frame = frame.rename(columns={frame.columns[-1]: "__value__"})
    return frame


def _jsonable_index_names(index: Any) -> list[str | None]:
    """提取 index 名称列表以便 JSON 元数据存储。
    
    参数:
        index: 见函数签名
    
    返回:
        list[str | None]
    """
    names = list(getattr(index, "names", None) or [])
    if not names and getattr(index, "name", None) is not None:
        names = [index.name]
    return names


def _payload_checksum(path: Path) -> str:
    """Parquet 文件字节的 sha256（流式，避免整读大文件）。"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _index_schema_part(index: Any) -> str:
    """Index 的稳定 schema 摘要（names/dtypes/timezone/categorical）。

    R20-155：``schema_hash`` 必须覆盖 index levels/names/dtypes/timezone/
    categorical 与 column order，否则改 index dtype 或 tz 不会 invalidate 磁盘
    缓存。
    """
    parts: list[str] = []
    if isinstance(index, pd.MultiIndex):
        for name, level in zip(index.names, index.levels):
            parts.append(
                f"{name!r}:{level.dtype}:tz={getattr(level, 'tz', None)}"
                f":cat={isinstance(level, pd.CategoricalIndex)}"
            )
    else:
        parts.append(
            f"{getattr(index, 'name', None)!r}:{index.dtype}"
            f":tz={getattr(index, 'tz', None)}"
            f":cat={isinstance(index, pd.CategoricalIndex)}"
        )
    return "|".join(parts)


def _schema_hash(value: Any) -> str:
    """frame/series 的完整 schema 摘要（dtypes + index + column order）。

    R20-153..155：读 parquet 后重算并比较 —— dtypes/columns/index levels/names/
    timezone/categorical 任一变化都得到不同 hash，从而 invalidate 磁盘缓存。
    """
    h = hashlib.sha256()
    if isinstance(value, pd.DataFrame):
        # column order 敏感：显式遍历 columns
        for col in value.columns:
            h.update(f"col:{col}:{value[col].dtype}\x00".encode("utf-8"))
    else:
        h.update(f"series:{getattr(value, 'dtype', 'unknown')}\x00".encode("utf-8"))
        h.update(f"name:{getattr(value, 'name', None)!r}\x00".encode("utf-8"))
    h.update(_index_schema_part(value.index).encode("utf-8"))
    return h.hexdigest()


def _index_schema_meta(index: Any) -> dict[str, Any]:
    """把 index schema 存为 JSON 元数据（供空轴重建，R20-157 / R20-465）。"""
    if isinstance(index, pd.MultiIndex):
        return {
            "kind": "multi",
            "names": [str(n) if n is not None else None for n in index.names],
            "dtypes": [str(l.dtype) for l in index.levels],
            "tz": [str(getattr(l, "tz", None)) if getattr(l, "tz", None) is not None else None for l in index.levels],
            "categorical": [isinstance(l, pd.CategoricalIndex) for l in index.levels],
        }
    return {
        "kind": "single",
        "names": [str(index.name) if index.name is not None else None],
        "dtypes": [str(index.dtype)],
        "tz": [str(getattr(index, "tz", None)) if getattr(index, "tz", None) is not None else None],
        "categorical": [isinstance(index, pd.CategoricalIndex)],
    }


def _rebuild_index_from_meta(meta: dict[str, Any], fallback_columns: list[str]) -> Any:
    """从 ``index_schema`` meta 重建空 Index / MultiIndex（R20-157 / R20-465）。

    普通空 Series 的 round-trip 会退化成 plain ``pd.Series(dtype=float)``，丢失
    MultiIndex 轴语义 —— 这里根据 meta 重建空 index schema。
    """
    schema = meta.get("index_schema") or {}
    kind = schema.get("kind")
    names = list(schema.get("names") or [])
    dtypes = list(schema.get("dtypes") or [])
    tz = list(schema.get("tz") or [])
    if not names:
        names = list(fallback_columns)
        dtypes = dtypes or []
    arrays: list[pd.Index] = []
    for i, name in enumerate(names):
        dt = dtypes[i] if i < len(dtypes) else None
        tz_i = tz[i] if i < len(tz) else None
        if dt == "datetime64[ns]":
            arrays.append(pd.DatetimeIndex([], name=name, tz=tz_i))
        elif tz_i and dt and dt.startswith("datetime64"):
            arrays.append(pd.DatetimeIndex([], name=name, tz=tz_i))
        elif dt and dt != "object":
            arrays.append(pd.Index([], dtype=dt, name=name))
        else:
            arrays.append(pd.Index([], dtype="object", name=name))
    if kind == "multi" or len(arrays) > 1:
        return pd.MultiIndex.from_arrays(arrays, names=names)
    return arrays[0] if arrays else pd.RangeIndex(0)


def _save_value(path: Path, value: Any, *, cache_key: str | None = None) -> None:
    """将 Series/DataFrame 原子写入 Parquet 并附带元数据 JSON。

    审计 #330 + R20-153..172：
        - 先写 ``<key>.tmp.<uuid>.parquet`` 与 ``<key>.tmp.<uuid>.meta.json``；
        - meta 新增 ``payload_checksum``（parquet 字节 sha256）、``schema_hash``
          （**完整** schema：dtypes/columns/index levels/names/timezone/
          categorical）、``generation_id``（uuid）、``schema_version``（缓存格式
          版本）、``index_schema``（空轴重建用）；
        - 两份都写成功后才用 ``os.replace`` 原子改名到正式路径，避免读到
          写了一半的缓存文件；
        - 多 writer 同 key 由 per-key 锁串行化，避免 payload/meta 交错。

    参数:
        path: 文件或目录路径
        value: 缓存值
        cache_key: 可选的 scoped cache key（用于 cache_key_digest 验证）

    返回:
        无
    """
    meta_path = path.with_suffix(".meta.json")
    if isinstance(value, pd.Series):
        frame = _series_to_frame(value)
        meta = {
            "kind": "series",
            "index_columns": [c for c in frame.columns if c != "__value__"],
        }
        is_frame = False
    elif isinstance(value, pd.DataFrame):
        meta = {"kind": "dataframe", "index_name": _jsonable_index_names(value.index)}
        is_frame = True
    else:
        return
    meta["schema_version"] = PLAN_CACHE_SCHEMA_VERSION
    meta["index_schema"] = _index_schema_meta(value.index)
    if cache_key is not None:
        # Verify that a payload is read only under the exact scoped-key digest
        # that created its path; payload/schema checks alone cannot detect a
        # digest collision or an accidentally moved cache file.
        meta["cache_key_digest"] = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()

    # R20-158：per-key 写锁（同 key 多 writer 交错 payload/meta 的防护）。
    # R32-P1-045：bounded lock registry（per-key 表不再无限增长）。
    key = str(path)
    lock = _save_lock_for(key)
    with lock:
        tmp_dir = path.parent
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_base = f"{path.name}.tmp.{uuid.uuid4().hex}"
        tmp_parquet = tmp_dir / f"{tmp_base}.parquet"
        tmp_meta = tmp_dir / f"{tmp_base}.meta.json"
        try:
            if is_frame:
                value.to_parquet(tmp_parquet, index=True)
            else:
                frame.to_parquet(tmp_parquet, index=False)
            meta["payload_checksum"] = _payload_checksum(tmp_parquet)
            meta["schema_hash"] = _schema_hash(value)
            meta["generation_id"] = uuid.uuid4().hex
            tmp_meta.write_text(json.dumps(meta), encoding="utf-8")
            os.replace(tmp_parquet, path)
            os.replace(tmp_meta, meta_path)
        except Exception:
            # 清理残留 tmp 文件后重抛（若 parquet 已替换而 meta 未替换，下次 load
            # 会因 checksum 不匹配而 fail-closed 返回 None，安全）。
            for p in (tmp_parquet, tmp_meta):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
            raise


def _load_value(path: Path, *, cache_key: str | None = None) -> Any | None:
    """从 Parquet + 元数据 JSON 恢复缓存值（校验 checksum + schema，fail-closed）。

    审计 #330：checksum 缺失或不匹配都返回 ``None``，不信任未经验证的磁盘缓存。
    R20-153..172：校验 ``schema_version``（旧格式 fail-closed）、读 parquet 后
    重算 ``schema_hash`` 与 meta 比对（mismatch → cache miss），并按
    ``index_schema`` 重建空轴（R20-465）。

    参数:
        path: 文件或目录路径
        cache_key: 可选的 scoped cache key（用于 cache_key_digest 验证）

    返回:
        Any | None
    """
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        checksum = meta.get("payload_checksum")
        if not checksum:
            # 旧格式无校验和：fail-closed，不信任
            return None
        if int(meta.get("schema_version", 0)) != PLAN_CACHE_SCHEMA_VERSION:
            return None
        if cache_key is not None:
            expected_digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
            if meta.get("cache_key_digest") != expected_digest:
                return None
        if _payload_checksum(path) != checksum:
            return None
        frame = pd.read_parquet(path)
        if meta.get("kind") == "dataframe":
            loaded = frame
        else:
            loaded = _frame_to_series(frame, meta=meta)
        # R20-153：schema mismatch → quarantine/cache miss（不信任未验证缓存）。
        expected = meta.get("schema_hash")
        if expected and _schema_hash(loaded) != expected:
            _logger_cache_mismatch(path, expected)
            return None
        return loaded
    except FileNotFoundError:
        # Expected cache miss - no logging needed
        return None
    except Exception as e:
        # P0-FIX: Log unexpected cache load failures for observability
        import logging
        logging.getLogger(__name__).warning(
            f"Cache load failed unexpectedly for {path}: {type(e).__name__}: {e}"
        )
        return None


def _logger_cache_mismatch(path: Path, expected: str) -> None:
    """R20-153：schema mismatch 的 telemetry（log + 不抛）。"""
    try:
        import logging

        logging.getLogger("factor_engine.storage.cache").warning(
            "persistent cache schema mismatch for %s (expected %s); cache miss",
            path,
            expected,
        )
    except Exception:
        pass


def _frame_to_series(frame: pd.DataFrame, meta: dict[str, Any] | None = None) -> pd.Series:
    """将缓存 DataFrame 还原为 MultiIndex Series。

    参数:
        frame: 长表 DataFrame
        meta: 可选的持久化 meta（R20-157：空轴重建用）

    返回:
        pd.Series
    """
    value_col = "__value__"
    if value_col not in frame.columns:
        value_col = frame.columns[-1]
    idx_cols = [c for c in frame.columns if c != value_col]
    if frame.empty:
        # R20-157 / R20-465：空 Series round-trip 必须重建原 index schema，
        # 不能退化成普通空 Series（MultiIndex 轴语义丢失）。
        if not idx_cols:
            return pd.Series(dtype=float)
        if meta is not None and meta.get("index_schema"):
            idx = _rebuild_index_from_meta(meta, fallback_columns=idx_cols)
            return pd.Series(dtype=float, index=idx)
        arrays = [pd.Index([], dtype=frame[c].dtype) for c in idx_cols]
        if len(arrays) == 1:
            return pd.Series(dtype=float, index=arrays[0])
        return pd.Series(
            dtype=float,
            index=pd.MultiIndex.from_arrays(arrays, names=list(idx_cols)),
        )
    if not idx_cols:
        return frame[value_col]
    out = frame.set_index(idx_cols)[value_col]
    out.name = None
    # Fix: set_index with single-element list creates index.name from column name;
    # reset to None to match original Series that had unnamed index.
    if len(idx_cols) == 1:
        out.index.name = None
    return out


class PersistentPlanCache(CacheManager):
    """带磁盘 Parquet 持久化的计划子树缓存（L1 内存 + L2 磁盘）。
    
    参数:
        root: 根目录路径
        data_scope: 数据作用域指纹（可选）
    """

    def __init__(
        self,
        root: str | Path,
        *,
        data_scope: str | None = None,
        budget_bytes: int | None = None,
        layer_name: str = "l2_subplan",
    ) -> None:
        """初始化实例。

        参数:
            root: 根目录路径
            data_scope: 数据作用域指纹（可选）
            budget_bytes: 内存层字节预算（可选）
            layer_name: MemoryGovernor 记账层名（默认 ``"l2_subplan"``；磁盘部分
                仍走 disk hit accounting，不单独计层）

        返回:
            无
        """
        super().__init__(
            data_scope=data_scope, budget_bytes=budget_bytes, layer_name=layer_name
        )
        self.root = Path(root)

    def _namespace_root(self) -> Path:
        """_namespace_root。

        R32-P1-046: production 下 namespace 解析失败（"unknown_ops"）必须抛错 —
        — production 禁写/禁读未解析 namespace（否则两个不同 operator/field 集合
        的缓存共享同一稳定目录）。research 下 "unknown_ops" 视为 cache miss。

        参数:
            无

        返回:
            Path
        """
        ns = _operator_namespace()
        if ns == UNKNOWN_CACHE_NAMESPACE:
            from factor_engine.runtime.production_policy import is_production_mode

            if is_production_mode():
                raise RuntimeError(
                    "persistent cache namespace is unresolved ('unknown_ops'): "
                    "operator/field catalog hash resolution failed. Production "
                    "forbids reading or writing an unresolved cache namespace "
                    "(R32-P1-046)."
                )
            # research：未知 namespace 不命中任何缓存 —— 返回一个恒空的哨兵目录
            # 由调用方当作 cache miss（磁盘路径不存在）。
            return self.root / ".unknown_ops_miss"
        return self.root / ns

    def _disk_path(self, scoped_key: str) -> Path:
        """_disk_path。
        
        参数:
            scoped_key: 见函数签名
        
        返回:
            Path
        """
        digest = hashlib.sha256(scoped_key.encode("utf-8")).hexdigest()
        return self._namespace_root() / f"{digest}.parquet"

    def get(self, key: str):
        """get。

        参数:
            key: 缓存键

        返回:
            无
        """
        scoped = self._scoped_key(key)
        with self._shared.lock:
            hit = self._cache.get(scoped)
            if hit is not None:
                # R6-151: record the hit as most-recently-used (LRU).
                self._cache.pop(scoped, None)
                self._cache[scoped] = hit
                return hit

        # 磁盘读放在锁外（I/O 不进 critical section）。
        path = self._disk_path(scoped)
        if not path.is_file():
            return None
        value = _load_value(path, cache_key=scoped)
        if value is None:
            return None
        # R6-152: a disk (L2) hit must still count toward the memory budget and
        # be subject to eviction, or repeated L3 hits bypass the entire in-memory
        # limit (resource P0).  Same accounting as ``set``: size the value, drop
        # it if it cannot fit alone, else account + evict to budget.
        from factor_engine.runtime.resource_governor import estimate_object_bytes

        gov = _governor()
        size = estimate_object_bytes(value)
        if size > self.budget_bytes:
            return None
        with gov.lock:
            with self._shared.lock:
                # 锁内重查：并发另一个线程可能已把同 key 放入内存。
                existing = self._cache.get(scoped)
                if existing is not None:
                    self._cache.pop(scoped, None)
                    self._cache[scoped] = existing
                    return existing
                self._bytes += size
                gov.reserve_accounting(self.layer_name, size)
                self._cache[scoped] = value
                freed = self._evict_to(self.budget_bytes)
                if freed > 0:
                    gov.release_accounting(self.layer_name, freed)
        return value

    def set(self, key: str, value) -> None:
        """set；受字节预算约束。

        参数:
            key: 缓存键
            value: 缓存值

        返回:
            无
        """
        if not isinstance(value, (pd.Series, pd.DataFrame)):
            return
        from factor_engine.runtime.resource_governor import estimate_object_bytes

        gov = _governor()
        scoped = self._scoped_key(key)
        size = estimate_object_bytes(value)
        if size > self.budget_bytes:
            return
        # R20-119..124：内存记账原子化；磁盘写放在锁外（per-key 锁在
        # ``_save_value`` 内）。
        with gov.lock:
            with self._shared.lock:
                if scoped in self._cache:
                    old_size = estimate_object_bytes(self._cache[scoped])
                    self._bytes -= old_size
                    gov.release_accounting(self.layer_name, old_size)
                    # R20-146..152：overwrite → MRU。
                    self._cache.pop(scoped, None)
                self._bytes += size
                gov.reserve_accounting(self.layer_name, size)
                self._cache[scoped] = value
                freed = self._evict_to(self.budget_bytes)
                if freed > 0:
                    gov.release_accounting(self.layer_name, freed)
        path = self._disk_path(scoped)
        path.parent.mkdir(parents=True, exist_ok=True)
        _save_value(path, value, cache_key=scoped)

    def with_scope(self, data_scope: str, *, clear_memory: bool = False) -> PersistentPlanCache:
        """with_scope。

        参数:
            data_scope: 数据作用域指纹
            clear_memory: 见函数签名（可选）

        返回:
            PersistentPlanCache

        R13 P1-62：同 :meth:`CacheManager.with_scope`，``clear_memory=False`` 时
        共享 backing store，内存记账对称。
        """
        out = PersistentPlanCache(
            self.root,
            data_scope=data_scope,
            budget_bytes=self._budget_bytes,
            layer_name=self.layer_name,
        )
        if not clear_memory:
            out._shared = self._shared
            out._shared.refs += 1
        return out
