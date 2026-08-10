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
    """相对交易日偏移（封装 trading_day_offset）。
    
    参数:
        base: 见函数签名
        n: 见函数签名
        calendar: 交易日历实例（可选）
    
    返回:
        pd.Timestamp
    """
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
    """解析日频增量加载与输出时间窗口。
    
    参数:
        watermark_end: 水位线结束日期（可选）
        lookback_bars: 预热 bar 数（可选）
        since: 增量起始覆盖日期（可选）
        end_date: 结束日期（可选）
        recompute_tail_bars: 尾部重算 bar 数（可选）
        calendar: 交易日历实例（可选）
    
    返回:
        dict[str, pd.Timestamp | None]
    
    
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


def resolve_incremental_window_for_bar_freq(
    *,
    watermark_end: str | None,
    lookback_bars: int,
    since: str | None = None,
    end_date: str | None = None,
    recompute_tail_bars: int | None = None,
    calendar: TradingCalendar | None = None,
    bar_freq: str | None = None,
    use_tick_precise: bool = True,
    market: str | None = None,
) -> dict[str, pd.Timestamp | None | str]:
    """按 bar 频率解析增量时间窗口。

    参数:
        watermark_end: 水位线结束日期（可选）
        lookback_bars: 预热 bar 数（可选）
        since: 增量起始覆盖日期（可选）
        end_date: 结束日期（可选）
        recompute_tail_bars: 尾部重算 bar 数（可选）
        calendar: 交易日历实例（可选）
        bar_freq: bar 频率字符串（可选）
        use_tick_precise: 是否使用 tick 精确扩窗（可选）
        market: 市场标识（ashare/us/CN…，决定 session 结构，可选）

    返回:
        dict[str, pd.Timestamp | None | str]


    日内源默认 ``intraday_session_clock``（R32-P0-004）：分钟 lookback/tail 在
    真实 session slot grid 上偏移（认识 09:30 开盘 / 11:30–13:00 午休 / 半日市 /
    early close / DST），不再是「交易日零点 + bar_duration × N」；
        ``use_tick_precise=False`` 时回退 ``intraday_calendar_approx``（+1 日缓冲）。
    """
    from cleaned_operators.operator_policy import (
        bar_freq_to_timedelta,
        bars_per_day,
        bars_to_calendar_trading_days,
    )

    bpd = bars_per_day(bar_freq)
    if bpd <= 1:
        out = resolve_incremental_window(
            watermark_end=watermark_end,
            lookback_bars=lookback_bars,
            since=since,
            end_date=end_date,
            recompute_tail_bars=recompute_tail_bars,
            calendar=calendar,
        )
        out["window_mode"] = "daily"
        return out

    lb = max(0, int(lookback_bars))
    tail = max(0, int(recompute_tail_bars)) if recompute_tail_bars is not None else lb
    cal_buffer = 0 if use_tick_precise else 1
    lb_days = bars_to_calendar_trading_days(lb, bar_freq) + cal_buffer
    tail_days = bars_to_calendar_trading_days(tail, bar_freq) + cal_buffer
    out = resolve_incremental_window(
        watermark_end=watermark_end,
        lookback_bars=lb_days,
        since=since,
        end_date=end_date,
        recompute_tail_bars=tail_days,
        calendar=calendar,
    )

    if use_tick_precise:
        from runtime.session_calendar import SessionCalendar

        # R32-P0-004: 所有分钟 lookback/tail 必须在真实 session slot grid 上偏移
        # （认识 09:30 开盘 / 11:30–13:00 午休 / 集合竞价 / 半日市 / early close）。
        # 不再用「交易日零点 + bar_duration × bars_per_day」的线性算术。
        session_market = str(market or "US").strip().upper()
        session_cal = SessionCalendar(
            market=session_market,
            bar_freq=bar_freq,
            timestamp_convention="bar_end",
        )
        wm_raw = since or watermark_end
        wm = pd.Timestamp(wm_raw).normalize() if wm_raw else None
        if wm is not None:
            # 以当日最后一个合法 bar slot 为 end_anchor（bar_end 惯例）。
            day_slots = session_cal.bar_slots(wm)
            end_anchor = day_slots[-1] if day_slots else wm
            out["load_start"] = session_cal.offset_bars(end_anchor, -lb)
            if tail > 0:
                output_precise = session_cal.offset_bars(end_anchor, -tail)
                cal_out = out.get("output_start")
                if cal_out is None:
                    out["output_start"] = output_precise
                else:
                    out["output_start"] = min(pd.Timestamp(cal_out), output_precise)
        out["window_mode"] = "intraday_session_clock"
    else:
        out["window_mode"] = "intraday_calendar_approx"

    out["source_bar_freq"] = str(bar_freq)
    return out


def _to_date_str(value: str | pd.Timestamp | None) -> str | None:
    """将时间戳规范为 YYYY-MM-DD 字符串。
    
    参数:
        value: 缓存值
    
    返回:
        str | None
    """
    if value is None:
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _bound_for_io(
    value: str | pd.Timestamp | None,
    bar_freq: str | None = None,
) -> str | None:
    """生成 IO 下推用的时间边界字符串。
    
    参数:
        value: 缓存值
        bar_freq: bar 频率字符串（可选）
    
    返回:
        str | None
    """
    if value is None:
        return None
    ts = pd.Timestamp(value)
    from cleaned_operators.operator_policy import bars_per_day

    if bars_per_day(bar_freq) > 1 or ts.hour or ts.minute or ts.second:
        return ts.isoformat()
    return ts.strftime("%Y-%m-%d")


def _merge_timestamp_bound_for_freq(
    existing: str | None,
    new: str | None,
    *,
    kind: str,
    bar_freq: str | None = None,
) -> str | None:
    """合并时间边界（日频/日内格式自适应）。
    
    参数:
        existing: 见函数签名
        new: 见函数签名
        kind: 见函数签名（可选）
        bar_freq: bar 频率字符串（可选）
    
    返回:
        str | None
    """
    if new is None:
        return existing
    if existing is None:
        return new
    ex = pd.Timestamp(existing)
    nv = pd.Timestamp(new)
    if kind == "start":
        chosen = max(ex, nv)
    else:
        chosen = min(ex, nv)
    from cleaned_operators.operator_policy import bars_per_day

    if bars_per_day(bar_freq) <= 1 and not (
        chosen.hour or chosen.minute or chosen.second or chosen.microsecond
    ):
        return chosen.strftime("%Y-%m-%d")
    return chosen.isoformat()


def _merge_date_bound(
    existing: str | None,
    new: str | None,
    *,
    kind: str,
) -> str | None:
    """合并日期边界（取更窄窗口）。
    
    参数:
        existing: 见函数签名
        new: 见函数签名
        kind: 见函数签名（可选）
    
    返回:
        str | None
    """
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
    """对齐 index 时区后规范化边界时间戳。
    
    参数:
        bound: 见函数签名
        index_tz: 见函数签名
    
    返回:
        pd.Timestamp | None
    """
    if bound is None:
        return None
    ts = pd.Timestamp(bound)
    has_time = bool(ts.hour or ts.minute or ts.second or ts.microsecond)
    if not has_time:
        ts = ts.normalize()
    if index_tz is not None:
        if ts.tz is None:
            ts = ts.tz_localize(index_tz)
        else:
            ts = ts.tz_convert(index_tz)
    elif ts.tz is not None:
        # R32-P0-005: 禁止 ``tz_localize(None)`` 直接 strip tz —— 那只是丢弃
        # tz 标签保留本地墙钟，等于把「跨时区转换」当成「去掉 tz」。naive index
        # 的显式 timestamp convention 是 UTC：先 tz_convert 到 UTC 保住 instante，
        # 再 drop tz 标签。
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def slice_series_time_window(
    series: pd.Series,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    exclusive_start: bool = False,
) -> pd.Series:
    """按 MultiIndex 时间层裁剪 Series。
    
    参数:
        series: MultiIndex Series
        start: 起始时间（含）（可选）
        end: 结束时间（含）（可选）
        exclusive_start: 起始边界是否开区间（可选）
    
    返回:
        pd.Series
    """
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
    """递归为 data_source 配置注入日期范围。
    
    参数:
        config: 数据源或运行时配置
        start_date: 起始日期（可选）
        end_date: 结束日期（可选）
    
    返回:
        dict[str, Any]
    """
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
    """包装数据源并按时间边界内存裁剪列。
    
    参数:
        inner: 内层数据源
        start_date: 起始日期（可选）
        end_date: 结束日期（可选）
        intraday: 是否按日内精度处理边界（可选）
    """

    def __init__(
        self,
        inner: DataSource,
        *,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        intraday: bool = False,
    ) -> None:
        """初始化实例。
        
        参数:
            inner: 内层数据源
            start_date: 起始日期（可选）
            end_date: 结束日期（可选）
            intraday: 是否按日内精度处理边界（可选）
        
        返回:
            无
        """
        self._inner = inner
        self._intraday = intraday
        self._start = self._coerce_bound(start_date)
        self._end = self._coerce_bound(end_date)
        self._column_cache: dict[str, Any] = {}

    def _coerce_bound(self, value: str | pd.Timestamp | None) -> pd.Timestamp | None:
        """_coerce_bound。
        
        参数:
            value: 缓存值
        
        返回:
            pd.Timestamp | None
        """
        if value is None:
            return None
        ts = pd.Timestamp(value)
        if self._intraday or ts.hour or ts.minute or ts.second or ts.microsecond:
            return ts
        return ts.normalize()

    def load_column(self, name: str):
        """load_column。
        
        参数:
            name: 逻辑列名
        
        返回:
            无
        """
        if name in self._column_cache:
            return self._column_cache[name]
        series = self._inner.load_column(name)
        out = slice_series_time_window(series, start=self._start, end=self._end)
        self._column_cache[name] = out
        return out

    def load_columns(self, names: list[str]) -> dict[str, Any]:
        """load_columns。
        
        参数:
            names: 逻辑列名列表
        
        返回:
            dict[str, Any]
        """
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
    bar_freq: str | None = None,
) -> DataSource:
    """尽可能在数据源层下推时间窗口裁剪。
    
    参数:
        source: 见函数签名
        start_date: 起始日期（可选）
        end_date: 结束日期（可选）
        bar_freq: bar 频率字符串（可选）
    
    返回:
        DataSource
    """
    from cleaned_operators.operator_policy import bars_per_day

    resolved_bar_freq = bar_freq or getattr(source, "bar_freq", None)
    if resolved_bar_freq is None:
        inner = getattr(source, "inner", None) or getattr(source, "_inner", None)
        if inner is not None:
            resolved_bar_freq = getattr(inner, "bar_freq", None)

    intraday = bars_per_day(resolved_bar_freq) > 1
    start_s = _bound_for_io(start_date, resolved_bar_freq)
    end_s = _bound_for_io(end_date, resolved_bar_freq)
    if start_s is None and end_s is None:
        return source

    from .composite_source import CompositeDataSource
    from .data_access_source import DataAccessSource
    from .kline_parquet_source import KlineParquetSource
    from .parquet_source import ParquetSource

    if isinstance(source, KlineParquetSource):
        return replace(
            source,
            start_date=_merge_timestamp_bound_for_freq(
                source.start_date, start_s, kind="start", bar_freq=resolved_bar_freq
            ),
            end_date=_merge_timestamp_bound_for_freq(
                source.end_date, end_s, kind="end", bar_freq=resolved_bar_freq
            ),
        )
    if isinstance(source, DataAccessSource):
        return DataAccessSource(
            dataset=source.dataset,
            fields=source.fields,
            start_date=_merge_timestamp_bound_for_freq(
                source.start_date, start_s, kind="start", bar_freq=resolved_bar_freq
            ),
            end_date=_merge_timestamp_bound_for_freq(
                source.end_date, end_s, kind="end", bar_freq=resolved_bar_freq
            ),
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
            start_date=_merge_timestamp_bound_for_freq(
                source.start_date, start_s, kind="start", bar_freq=resolved_bar_freq
            ),
            end_date=_merge_timestamp_bound_for_freq(
                source.end_date, end_s, kind="end", bar_freq=resolved_bar_freq
            ),
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
                    bar_freq=resolved_bar_freq,
                )
                for name, sub in source.sources.items()
            },
            joins=source.joins,
            aliases=source.aliases,
            allow_unqualified_anchor_columns=source.allow_unqualified_anchor_columns,
        )
    return WindowedDataSource(
        source,
        start_date=start_date,
        end_date=end_date,
        intraday=intraday,
    )
