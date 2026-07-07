# -*- coding: utf-8 -*-
"""计划子树执行缓存：内存 + 可选磁盘持久化。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


class CacheManager:
    """简单键值缓存。

    - **列数据**可由数据源自行管理；本类主要用于 ``PandasBackend`` **子树求值结果**复用
      （键为计划子树的结构化字符串 + 可选 ``data_scope``）。
    """

    def __init__(self, *, data_scope: str | None = None) -> None:
        self.data_scope = data_scope
        self._cache: dict[str, object] = {}

    def _scoped_key(self, key: str) -> str:
        if self.data_scope:
            return f"{self.data_scope}:{key}"
        return key

    def get(self, key: str):
        return self._cache.get(self._scoped_key(key))

    def set(self, key: str, value) -> None:
        self._cache[self._scoped_key(key)] = value

    def clear_memory(self) -> None:
        self._cache.clear()

    def with_scope(self, data_scope: str, *, clear_memory: bool = False) -> CacheManager:
        """返回同类型实例并切换作用域（用于增量窗口隔离）。"""
        out = type(self)(data_scope=data_scope)
        if not clear_memory and type(out) is CacheManager:
            out._cache = dict(self._cache)
        return out


def _operator_namespace() -> str:
    try:
        from cleaned_operators.operator_policy import compute_operator_catalog_hash

        return compute_operator_catalog_hash()[:16]
    except Exception:
        return "unknown_ops"


def _series_to_frame(series: pd.Series) -> pd.DataFrame:
    frame = series.reset_index()
    if series.name is not None:
        frame = frame.rename(columns={series.name: "__value__"})
    else:
        frame = frame.rename(columns={frame.columns[-1]: "__value__"})
    return frame


def _jsonable_index_names(index: Any) -> list[str | None]:
    names = list(getattr(index, "names", None) or [])
    if not names and getattr(index, "name", None) is not None:
        names = [index.name]
    return names


def _save_value(path: Path, value: Any) -> None:
    meta_path = path.with_suffix(".meta.json")
    if isinstance(value, pd.Series):
        frame = _series_to_frame(value)
        meta = {"kind": "series", "index_columns": [c for c in frame.columns if c != "__value__"]}
        frame.to_parquet(path, index=False)
    elif isinstance(value, pd.DataFrame):
        meta = {"kind": "dataframe", "index_name": _jsonable_index_names(value.index)}
        value.to_parquet(path, index=True)
    else:
        return
    meta_path.write_text(json.dumps(meta), encoding="utf-8")


def _load_value(path: Path) -> Any | None:
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        frame = pd.read_parquet(path)
    except Exception:
        return None
    if meta.get("kind") == "dataframe":
        return frame
    return _frame_to_series(frame)


def _frame_to_series(frame: pd.DataFrame) -> pd.Series:
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
    """磁盘持久化 plan 子树缓存：L1 内存 + L2 parquet。"""

    def __init__(
        self,
        root: str | Path,
        *,
        data_scope: str | None = None,
    ) -> None:
        super().__init__(data_scope=data_scope)
        self.root = Path(root)

    def _namespace_root(self) -> Path:
        return self.root / _operator_namespace()

    def _disk_path(self, scoped_key: str) -> Path:
        digest = hashlib.sha256(scoped_key.encode("utf-8")).hexdigest()
        return self._namespace_root() / f"{digest}.parquet"

    def get(self, key: str):
        scoped = self._scoped_key(key)
        hit = self._cache.get(scoped)
        if hit is not None:
            return hit

        path = self._disk_path(scoped)
        if not path.is_file():
            return None
        value = _load_value(path)
        if value is None:
            return None
        self._cache[scoped] = value
        return value

    def set(self, key: str, value) -> None:
        if not isinstance(value, (pd.Series, pd.DataFrame)):
            return
        scoped = self._scoped_key(key)
        self._cache[scoped] = value
        path = self._disk_path(scoped)
        path.parent.mkdir(parents=True, exist_ok=True)
        _save_value(path, value)

    def with_scope(self, data_scope: str, *, clear_memory: bool = False) -> PersistentPlanCache:
        out = PersistentPlanCache(self.root, data_scope=data_scope)
        if not clear_memory:
            out._cache = dict(self._cache)
        return out
