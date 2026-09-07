"""Unified column read session: batch prefetch, SourceRef resolution and input DQ."""
from __future__ import annotations
from collections.abc import Iterable
from typing import Any
from factor_engine.cache.column_cache import column_cache_scope

class DataSourceReadSession:
    def __init__(self, data_source: Any) -> None:
        from .lqtp_logical_source_v2 import LQTPLogicalDataSource
        if isinstance(data_source, LQTPLogicalDataSource):
            self._source = data_source
        else:
            wrapper = LQTPLogicalDataSource(data_source)
            try:
                setattr(data_source, "_factor_engine_lqtp_wrapper", wrapper)
            except Exception:
                pass
            self._source = wrapper
        self._last_scope_key: str | None = None

    @property
    def source(self) -> Any: return self._source
    def prefetch(self, columns: Iterable[str]) -> dict[str, Any]:
        names=sorted(set(columns))
        if not names: return {}
        load_columns=getattr(self._source,"load_columns",None)
        if callable(load_columns):
            loaded=load_columns(names)
            return loaded if isinstance(loaded,dict) else {}
        prefetch=getattr(self._source,"prefetch_columns",None)
        if callable(prefetch):
            loaded=prefetch(names)
            return loaded if isinstance(loaded,dict) else {}
        return {name:self._source.load_column(name) for name in names}
    def load_columns_once(self, columns: Iterable[str], *, data_snapshot_id: str | None=None) -> dict[str,Any]:
        names=sorted(set(columns)); snap=data_snapshot_id or getattr(self._source,"data_snapshot_id",None)
        scope=column_cache_scope(self._source,names,data_snapshot_id=snap); self._last_scope_key=scope.key
        loaded=self.prefetch(names)
        if all(name in loaded for name in names): return loaded
        load=getattr(self._source,"load_column",None)
        if not callable(load):
            cache=getattr(self._source,"_column_cache",{}); return {n:cache[n] for n in names}
        return {n:load(n) for n in names}
    def prepare_batch(self, columns: Iterable[str], *, input_dq_check: bool=False,
                      input_dq_strict: bool=True, input_dq_thresholds=None,
                      data_snapshot_id: str | None=None):
        names=sorted(set(columns))
        if not names: return None
        snap=data_snapshot_id or getattr(self._source,"data_snapshot_id",None)
        scope=column_cache_scope(self._source,names,data_snapshot_id=snap); self._last_scope_key=scope.key
        report=None
        # Populate the batch cache once before DQ.  DQ may otherwise call
        # load_column per field, defeating the single-read batch contract when
        # worker governance clamps a parallel run to one worker.
        loaded=self.prefetch(names)
        if input_dq_check:
            from factor_engine.runtime.input_dq import adjust_input_dq_thresholds_from_stats, load_dataset_stats_for_source
            stats=load_dataset_stats_for_source(self._source)
            thresholds=adjust_input_dq_thresholds_from_stats(input_dq_thresholds,stats,names)
            report=self.assert_input_dq(names,raise_on_fail=input_dq_strict,thresholds=thresholds,prefetched=loaded)
        return report

    def prepare_waves(self, waves: Iterable[Any], *, input_dq_check: bool=False,
                      input_dq_strict: bool=True, input_dq_thresholds=None,
                      data_snapshot_id: str | None=None) -> list[Any]:
        """R27-170/248：按 ReadWave 逐波读取，**不**全 batch 一次性 union prefetch。

        每个 wave 只投影该 wave 真实需要的列（R27-067），scan 后 source buffer 喂
        多个 task（R27-066）。返回每波 prefetch 的 input_dq report（None 表示跳过）。
        """
        reports=[]
        for wave in waves:
            names=sorted(set(getattr(wave,"columns",[]) or []))
            if not names:
                continue
            snap=data_snapshot_id or getattr(self._source,"data_snapshot_id",None)
            scope=column_cache_scope(self._source,names,data_snapshot_id=snap); self._last_scope_key=scope.key
            report=None
            loaded=self.prefetch(names)
            if input_dq_check:
                from factor_engine.runtime.input_dq import adjust_input_dq_thresholds_from_stats, load_dataset_stats_for_source
                stats=load_dataset_stats_for_source(self._source)
                thresholds=adjust_input_dq_thresholds_from_stats(input_dq_thresholds,stats,names)
                report=self.assert_input_dq(names,raise_on_fail=input_dq_strict,thresholds=thresholds,prefetched=loaded)
            reports.append(report)
        return reports
    def assert_input_dq(self, columns: Iterable[str], *, raise_on_fail: bool=True, thresholds=None, prefetched=None):
        from factor_engine.runtime.input_dq import assert_input_dq
        return assert_input_dq(self._source,columns,raise_on_fail=raise_on_fail,thresholds=thresholds,prefetched=prefetched)
    def cached_columns(self) -> frozenset[str]:
        cache=getattr(self._source,"_column_cache",None)
        if isinstance(cache,dict): return frozenset(cache)
        cache=getattr(self._source,"_cache",None)
        return frozenset(cache) if isinstance(cache,dict) else frozenset()
    def cache_stats(self) -> dict[str,int]:
        fn=getattr(self._source,"column_cache_stats",None)
        if callable(fn): return dict(fn())
        cache=getattr(self._source,"_cache",None)
        if not isinstance(cache,dict): cache=getattr(self._source,"_column_cache",None)
        panels=getattr(self._source,"_panel_cache",None)
        return {"cached_columns":len(cache) if isinstance(cache,dict) else 0,"cached_panels":len(panels) if isinstance(panels,dict) else 0}
    @property
    def last_scope_key(self) -> str | None: return self._last_scope_key
