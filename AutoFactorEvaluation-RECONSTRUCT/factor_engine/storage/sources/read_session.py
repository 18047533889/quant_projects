"""统一列读会话：batch prefetch + input_dq 与列缓存复用。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from cache.column_cache import column_cache_scope


class DataSourceReadSession:
    """数据源列读会话（batch prefetch + DQ）。
    
    参数:
        data_source: 数据源实例
    """

    def __init__(self, data_source: Any) -> None:
        """初始化实例。
        
        参数:
            data_source: 数据源实例
        
        返回:
            无
        """
        self._source = data_source
        self._last_scope_key: str | None = None

    @property
    def source(self) -> Any:
        """source。
        
        参数:
            无
        
        返回:
            Any
        """
        return self._source

    def prefetch(self, columns: Iterable[str]) -> None:
        """prefetch。
        
        参数:
            columns: 列名列表
        
        返回:
            无
        """
        names = sorted(set(columns))
        if not names:
            return
        prefetch = getattr(self._source, "prefetch_columns", None)
        if callable(prefetch):
            prefetch(names)
            return
        load_columns = getattr(self._source, "load_columns", None)
        if callable(load_columns):
            load_columns(names)
            return
        for name in names:
            self._source.load_column(name)

    def load_columns_once(
        self,
        columns: Iterable[str],
        *,
        data_snapshot_id: str | None = None,
    ) -> dict[str, Any]:
        """单次 batch 读列；已缓存列不再触发 IO。
        
        参数:
            columns: 列名列表
            data_snapshot_id: 见函数签名（可选）
        
        返回:
            dict[str, Any]
        """
        names = sorted(set(columns))
        snap = data_snapshot_id or getattr(self._source, "data_snapshot_id", None)
        scope = column_cache_scope(
            self._source,
            names,
            data_snapshot_id=snap,
        )
        self._last_scope_key = scope.key
        self.prefetch(names)
        load_column = getattr(self._source, "load_column", None)
        if not callable(load_column):
            cache = getattr(self._source, "_column_cache", {})
            return {n: cache[n] for n in names}
        return {n: load_column(n) for n in names}

    def prepare_batch(
        self,
        columns: Iterable[str],
        *,
        input_dq_check: bool = False,
        input_dq_strict: bool = True,
        input_dq_thresholds=None,
        data_snapshot_id: str | None = None,
    ):
        """input_dq + prefetch 合并为一次 ``load_columns`` 批次。
        
        参数:
            columns: 列名列表
            input_dq_check: 见函数签名（可选）
            input_dq_strict: 见函数签名（可选）
            input_dq_thresholds: 见函数签名（可选）
            data_snapshot_id: 见函数签名（可选）
        
        返回:
            无
        """
        names = sorted(set(columns))
        if not names:
            return None
        snap = data_snapshot_id or getattr(self._source, "data_snapshot_id", None)
        scope = column_cache_scope(
            self._source,
            names,
            data_snapshot_id=snap,
        )
        self._last_scope_key = scope.key
        input_report = None
        if input_dq_check:
            from runtime.input_dq import (
                adjust_input_dq_thresholds_from_stats,
                load_dataset_stats_for_source,
            )

            stats = load_dataset_stats_for_source(self._source)
            effective_thresholds = adjust_input_dq_thresholds_from_stats(
                input_dq_thresholds,
                stats,
                names,
            )
            input_report = self.assert_input_dq(
                names,
                raise_on_fail=input_dq_strict,
                thresholds=effective_thresholds,
            )
        self.prefetch(names)
        return input_report

    def assert_input_dq(
        self,
        columns: Iterable[str],
        *,
        raise_on_fail: bool = True,
        thresholds=None,
    ):
        """assert_input_dq。
        
        参数:
            columns: 列名列表
            raise_on_fail: 见函数签名（可选）
            thresholds: 见函数签名（可选）
        
        返回:
            无
        """
        from runtime.input_dq import assert_input_dq

        return assert_input_dq(
            self._source,
            columns,
            raise_on_fail=raise_on_fail,
            thresholds=thresholds,
        )

    def cached_columns(self) -> frozenset[str]:
        """cached_columns。
        
        参数:
            无
        
        返回:
            frozenset[str]
        """
        cache = getattr(self._source, "_column_cache", None)
        if isinstance(cache, dict):
            return frozenset(cache)
        return frozenset()

    def cache_stats(self) -> dict[str, int]:
        """cache_stats。
        
        参数:
            无
        
        返回:
            dict[str, int]
        """
        stats_fn = getattr(self._source, "column_cache_stats", None)
        if callable(stats_fn):
            return dict(stats_fn())
        cache = getattr(self._source, "_column_cache", None)
        panels = getattr(self._source, "_panel_cache", None)
        return {
            "cached_columns": len(cache) if isinstance(cache, dict) else 0,
            "cached_panels": len(panels) if isinstance(panels, dict) else 0,
        }

    @property
    def last_scope_key(self) -> str | None:
        """last_scope_key。
        
        参数:
            无
        
        返回:
            str | None
        """
        return self._last_scope_key
