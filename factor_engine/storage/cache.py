# -*- coding: utf-8 -*-
"""计划子树执行缓存：内存 + 可选磁盘持久化。"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd


def _governor():
    """惰性导入全局 MemoryGovernor，避免 storage → runtime → storage 循环导入。"""
    from runtime.resource_governor import global_memory_governor

    return global_memory_governor()


class _SharedCache:
    """Shared mutable backing store + byte counter for ``with_scope`` instances.

    R13 P1-62: the old ``with_scope`` copied ``_cache`` and ``_bytes`` but never
    reserved the copied bytes in the MemoryGovernor, so a child scope calling
    ``set``/``evict``/``clear`` released accounting it never reserved and the
    global accounting drifted below true resident bytes.  Sharing ONE mutable
    backing store (dict + byte counter) makes every operation on any scope touch
    the same bytes, so governor accounting always equals resident bytes.
    """

    __slots__ = ("cache", "nbytes", "refs")

    def __init__(self) -> None:
        self.cache: dict[str, object] = {}
        self.nbytes: int = 0
        self.refs: int = 1


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
            from runtime.resource_governor import global_memory_governor

            return int(global_memory_governor().process_budget_bytes * 0.05)
        except Exception:
            return 512 * 1024 * 1024

    def _scoped_key(self, key: str) -> str:
        """_scoped_key。

        参数:
            key: 缓存键

        返回:
            str
        """
        if self.data_scope:
            return f"{self.data_scope}:{key}"
        return key

    def get(self, key: str):
        """get。

        参数:
            key: 缓存键

        返回:
            无
        """
        scoped = self._scoped_key(key)
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
        from runtime.resource_governor import estimate_object_bytes

        gov = _governor()
        scoped = self._scoped_key(key)
        size = estimate_object_bytes(value)
        if size > self.budget_bytes:
            return
        if scoped in self._cache:
            old_size = estimate_object_bytes(self._cache[scoped])
            self._bytes -= old_size
            gov.release_accounting(self.layer_name, old_size)
        self._bytes += size
        gov.reserve_accounting(self.layer_name, size)
        self._cache[scoped] = value
        freed = self._evict_to(self.budget_bytes)
        if freed > 0:
            gov.release_accounting(self.layer_name, freed)

    def _evict_to(self, target: int) -> int:
        from runtime.resource_governor import estimate_object_bytes

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
        return self._evict_to(target if target > 0 else self.budget_bytes)

    def clear_memory(self) -> None:
        """clear_memory。

        参数:
            无

        返回:
            无
        """
        self._cache.clear()
        self._bytes = 0
        _governor().release_all(self.layer_name)

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


def _operator_namespace() -> str:
    """获取算子目录哈希命名空间（用于磁盘缓存路径隔离）。
    
    参数:
        无
    
    返回:
        str
    """
    try:
        from cleaned_operators.operator_policy import compute_operator_catalog_hash
        from fields import compute_field_catalog_hash

        return (
            f"{compute_operator_catalog_hash()[:16]}-"
            f"{compute_field_catalog_hash()[:16]}"
        )
    except Exception:
        return "unknown_ops"


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


def _schema_hash(value: Any) -> str:
    """frame dtypes/columns 摘要（审计 #330 meta ``schema_hash``）。"""
    if isinstance(value, pd.DataFrame):
        cols = "|".join(f"{c}:{value[c].dtype}" for c in value.columns)
    else:
        cols = f"series:{getattr(value, 'dtype', 'unknown')}"
    return hashlib.sha256(cols.encode("utf-8")).hexdigest()


def _save_value(path: Path, value: Any) -> None:
    """将 Series/DataFrame 原子写入 Parquet 并附带元数据 JSON。

    审计 #330：
        - 先写 ``<key>.tmp.<uuid>.parquet`` 与 ``<key>.tmp.<uuid>.meta.json``；
        - meta 新增 ``payload_checksum``（parquet 字节 sha256）、``schema_hash``
          （dtypes/columns 摘要）、``generation_id``（uuid）；
        - 两份都写成功后才用 ``os.replace`` 原子改名到正式路径，避免读到
          写了一半的缓存文件。

    参数:
        path: 文件或目录路径
        value: 缓存值

    返回:
        无
    """
    meta_path = path.with_suffix(".meta.json")
    if isinstance(value, pd.Series):
        frame = _series_to_frame(value)
        meta = {"kind": "series", "index_columns": [c for c in frame.columns if c != "__value__"]}
        is_frame = False
    elif isinstance(value, pd.DataFrame):
        meta = {"kind": "dataframe", "index_name": _jsonable_index_names(value.index)}
        is_frame = True
    else:
        return

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


def _load_value(path: Path) -> Any | None:
    """从 Parquet + 元数据 JSON 恢复缓存值（校验 ``payload_checksum``，fail-closed）。

    审计 #330：checksum 缺失或不匹配都返回 ``None``，不信任未经验证的磁盘缓存。

    参数:
        path: 文件或目录路径

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
        if _payload_checksum(path) != checksum:
            return None
        frame = pd.read_parquet(path)
    except Exception:
        return None
    if meta.get("kind") == "dataframe":
        return frame
    return _frame_to_series(frame)


def _frame_to_series(frame: pd.DataFrame) -> pd.Series:
    """将缓存 DataFrame 还原为 MultiIndex Series。
    
    参数:
        frame: 长表 DataFrame
    
    返回:
        pd.Series
    """
    if frame.empty:
        return pd.Series(dtype=float)
    value_col = "__value__"
    if value_col not in frame.columns:
        value_col = frame.columns[-1]
    idx_cols = [c for c in frame.columns if c != value_col]
    if not idx_cols:
        return frame[value_col]
    out = frame.set_index(idx_cols)[value_col]
    out.name = None
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
        
        参数:
            无
        
        返回:
            Path
        """
        return self.root / _operator_namespace()

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
        hit = self._cache.get(scoped)
        if hit is not None:
            # R6-151: record the hit as most-recently-used (LRU).
            self._cache.pop(scoped, None)
            self._cache[scoped] = hit
            return hit

        path = self._disk_path(scoped)
        if not path.is_file():
            return None
        value = _load_value(path)
        if value is None:
            return None
        # R6-152: a disk (L2) hit must still count toward the memory budget and
        # be subject to eviction, or repeated L3 hits bypass the entire in-memory
        # limit (resource P0).  Same accounting as ``set``: size the value, drop
        # it if it cannot fit alone, else account + evict to budget.
        from runtime.resource_governor import estimate_object_bytes

        gov = _governor()
        size = estimate_object_bytes(value)
        if size > self.budget_bytes:
            return None
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
        from runtime.resource_governor import estimate_object_bytes

        gov = _governor()
        scoped = self._scoped_key(key)
        size = estimate_object_bytes(value)
        if size > self.budget_bytes:
            return
        if scoped in self._cache:
            old_size = estimate_object_bytes(self._cache[scoped])
            self._bytes -= old_size
            gov.release_accounting(self.layer_name, old_size)
        self._bytes += size
        gov.reserve_accounting(self.layer_name, size)
        self._cache[scoped] = value
        freed = self._evict_to(self.budget_bytes)
        if freed > 0:
            gov.release_accounting(self.layer_name, freed)
        path = self._disk_path(scoped)
        path.parent.mkdir(parents=True, exist_ok=True)
        _save_value(path, value)

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
