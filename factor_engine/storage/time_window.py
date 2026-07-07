# -*- coding: utf-8 -*-
"""数据时间窗口：增量生产加载裁剪与配置传播。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from .datasource import DataSource
from .trading_calendar import TradingCalendar, trading_day_offset


def business_day_offset(
    base: str | pd.Timestamp,
    n: int,
    *,
    calendar: TradingCalendar | None = None,
) -> pd.Timestamp:
    """相对交易日偏移（n>0 向前，n<0 向后）。"""
    return trading_day_offset(base, n, calendar=calendar)


def resolve_incremental_window(
    *,
    watermark_end: str | None,
    lookback_bars: int,
    since: str | None = None,
    end_date: str | None = None,
    recompute_tail_bars: int | None = None,
    calendar: TradingCalendar | None = None,
) -> dict[str, pd.Timestamp | None]:
    """解析增量窗口：加载区间 vs 落盘输出区间。

    - **load_start**：含 lookback 预热，保证 rolling 在输出边界正确
    - **output_start**：写入因子湖的起始时间（含 tail 重算）
    - **load_end / output_end**：可选上界
    """
    lb = max(0, int(lookback_bars))
    tail = max(0, int(recompute_tail_bars)) if recompute_tail_bars is not None else lb

    wm_end = pd.Timestamp(since or watermark_end).normalize() if (since or watermark_end) else None
    load_end = pd.Timestamp(end_date).normalize() if end_date else None

    if wm_end is None:
        return {
            "load_start": None,
            "load_end": load_end,
            "output_start": None,
            "output_end": load_end,
        }

    output_start = (
        business_day_offset(wm_end, -tail, calendar=calendar) if tail > 0 else wm_end
    )
    load_start = (
        business_day_offset(output_start, -lb, calendar=calendar) if lb > 0 else output_start
    )

    return {
        "load_start": load_start,
        "load_end": load_end,
        "output_start": output_start,
        "output_end": load_end,
    }


def _to_date_str(value: str | pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _merge_date_bound(
    existing: str | None,
    new: str | None,
    *,
    kind: str,
) -> str | None:
    """合并日期边界：start 取较晚，end 取较早（更窄窗口）。"""
    if new is None:
        return existing
    if existing is None:
        return new
    if kind == "start":
        return max(existing, new)
    return min(existing, new)


def _normalize_bound_for_index(
    bound: pd.Timestamp | None,
    index_tz,
) -> pd.Timestamp | None:
    """对齐 index 时区后再比较，避免 naive/aware 混比。"""
    if bound is None:
        return None
    ts = pd.Timestamp(bound).normalize()
    if index_tz is not None:
        if ts.tz is None:
            ts = ts.tz_localize(index_tz)
        else:
            ts = ts.tz_convert(index_tz)
    elif ts.tz is not None:
        ts = ts.tz_localize(None)
    return ts


def slice_series_time_window(
    series: pd.Series,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    exclusive_start: bool = False,
) -> pd.Series:
    """按 MultiIndex 第 0 级（timestamp）裁剪 Series。"""
    if not isinstance(series.index, pd.MultiIndex) or series.index.nlevels < 1:
        return series
    ts = series.index.get_level_values(0)
    index_tz = getattr(ts, "tz", None)
    start_cmp = _normalize_bound_for_index(start, index_tz)
    end_cmp = _normalize_bound_for_index(end, index_tz)

    if start_cmp is None and end_cmp is None:
        return series

    mask = np.ones(len(ts), dtype=bool)
    if start_cmp is not None:
        if exclusive_start:
            mask &= ts > start_cmp
        else:
            mask &= ts >= start_cmp
    if end_cmp is not None:
        mask &= ts <= end_cmp
    return series[mask]


def apply_time_window_to_config(
    config: dict[str, Any],
    *,
    start_date: str | None,
    end_date: str | None,
) -> dict[str, Any]:
    """递归为 data_source 配置注入 start_date/end_date（composite 子源同步）。"""
    cfg = deepcopy(config)
    if start_date is not None:
        cfg["start_date"] = start_date
    if end_date is not None:
        cfg["end_date"] = end_date

    if str(cfg.get("type", "")).lower() == "composite":
        sources = cfg.get("sources") or {}
        for name, sub in sources.items():
            if isinstance(sub, dict):
                sources[name] = apply_time_window_to_config(
                    sub, start_date=start_date, end_date=end_date
                )
        cfg["sources"] = sources
    return cfg


class WindowedDataSource(DataSource):
    """包装已有 DataSource，按 timestamp 裁剪列（用于增量加载）。"""

    def __init__(
        self,
        inner: DataSource,
        *,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> None:
        self._inner = inner
        self._start = pd.Timestamp(start_date).normalize() if start_date else None
        self._end = pd.Timestamp(end_date).normalize() if end_date else None
        self._column_cache: dict[str, Any] = {}

    def load_column(self, name: str):
        if name in self._column_cache:
            return self._column_cache[name]
        series = self._inner.load_column(name)
        out = slice_series_time_window(series, start=self._start, end=self._end)
        self._column_cache[name] = out
        return out

    def load_columns(self, names: list[str]) -> dict[str, Any]:
        missing = [n for n in names if n not in self._column_cache]
        load_columns = getattr(self._inner, "load_columns", None)
        if missing and callable(load_columns):
            fetched = load_columns(missing)
            for n, s in fetched.items():
                self._column_cache[n] = slice_series_time_window(
                    s, start=self._start, end=self._end
                )
        return {n: self.load_column(n) for n in names}


def narrow_data_source_for_window(
    source: DataSource,
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> DataSource:
    """尽可能在数据源层裁剪（DuckDB/parquet 谓词下推），否则内存切片。"""
    start_s = _to_date_str(start_date)
    end_s = _to_date_str(end_date)
    if start_s is None and end_s is None:
        return source

    from .composite_source import CompositeDataSource
    from .data_access_source import DataAccessSource
    from .kline_parquet_source import KlineParquetSource
    from .parquet_source import ParquetSource

    if isinstance(source, KlineParquetSource):
        return replace(
            source,
            start_date=_merge_date_bound(source.start_date, start_s, kind="start"),
            end_date=_merge_date_bound(source.end_date, end_s, kind="end"),
        )
    if isinstance(source, DataAccessSource):
        return DataAccessSource(
            dataset=source.dataset,
            fields=source.fields,
            start_date=_merge_date_bound(source.start_date, start_s, kind="start"),
            end_date=_merge_date_bound(source.end_date, end_s, kind="end"),
            instrument_filter=source.instrument_filter,
            normalize_timestamp=source.normalize_timestamp,
            timestamp_unit=source.timestamp_unit,
        )
    if isinstance(source, ParquetSource):
        cls = type(source)
        return cls(
            root=source.root,
            timestamp_column=source.timestamp_column,
            instrument_column=source.instrument_column,
            fields=source.fields,
            max_files=source.max_files,
            timestamp_unit=source.timestamp_unit,
            start_date=_merge_date_bound(source.start_date, start_s, kind="start"),
            end_date=_merge_date_bound(source.end_date, end_s, kind="end"),
            recursive=source.recursive,
        )
    if isinstance(source, CompositeDataSource):
        return CompositeDataSource(
            anchor_source=source.anchor_source,
            anchor_column=source.anchor_column,
            sources={
                name: narrow_data_source_for_window(
                    sub,
                    start_date=start_date,
                    end_date=end_date,
                )
                for name, sub in source.sources.items()
            },
            joins=source.joins,
            aliases=source.aliases,
            allow_unqualified_anchor_columns=source.allow_unqualified_anchor_columns,
        )
    return WindowedDataSource(source, start_date=start_date, end_date=end_date)
