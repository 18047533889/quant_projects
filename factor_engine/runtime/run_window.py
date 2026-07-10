# -*- coding: utf-8
"""全量运行的 warm-up 扩窗与输出 trim。

增量路径（``run_incremental``）已有 lookback 扩窗；本模块补齐 **全量 run/materialize**：
compile 得到 lookback → 向前扩展 load_start → 计算 → trim 回用户请求区间。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from storage.time_window import business_day_offset
from storage.trading_calendar import TradingCalendar


@dataclass(frozen=True)
class RunWindow:
    """一次运行的请求窗口 vs 实际加载窗口。"""

    requested_start: str | None
    requested_end: str | None
    actual_load_start: str | None
    actual_load_end: str | None
    warmup_bars: int
    trim_output: bool

    def to_dict(self) -> dict[str, Any]:
        """序列化为可 JSON 化的字典。"""
        return {
            "requested_start": self.requested_start,
            "requested_end": self.requested_end,
            "actual_load_start": self.actual_load_start,
            "actual_load_end": self.actual_load_end,
            "warmup_bars": self.warmup_bars,
            "trim_output": self.trim_output,
        }


def _to_date_str(value: str | pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return pd.Timestamp(text).strftime("%Y-%m-%d")


def extract_source_date_bounds(data_source: Any) -> tuple[str | None, str | None]:
    """从 DataSource 实例读取用户配置的 start/end（若有）。"""
    time_range_fn = getattr(data_source, "time_range", None)
    if callable(time_range_fn):
        tr = time_range_fn()
        if tr and len(tr) >= 2:
            return _to_date_str(tr[0]), _to_date_str(tr[1])
    start = getattr(data_source, "start_date", None)
    end = getattr(data_source, "end_date", None)
    return _to_date_str(start), _to_date_str(end)


def build_full_run_window(
    *,
    requested_start: str | None,
    requested_end: str | None,
    lookback_bars: int,
    trim_output: bool = True,
    calendar: TradingCalendar | None = None,
) -> RunWindow:
    """根据 lookback 计算实际加载起点；无 start 或无 lookback 时不扩窗。"""
    lb = max(0, int(lookback_bars))
    req_start = _to_date_str(requested_start)
    req_end = _to_date_str(requested_end)

    if lb <= 0 or req_start is None:
        return RunWindow(
            requested_start=req_start,
            requested_end=req_end,
            actual_load_start=req_start,
            actual_load_end=req_end,
            warmup_bars=0,
            trim_output=trim_output,
        )

    load_start = business_day_offset(req_start, -lb, calendar=calendar)
    return RunWindow(
        requested_start=req_start,
        requested_end=req_end,
        actual_load_start=_to_date_str(load_start),
        actual_load_end=req_end,
        warmup_bars=lb,
        trim_output=trim_output,
    )
