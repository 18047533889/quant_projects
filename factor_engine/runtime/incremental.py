# -*- coding: utf-8 -*-
"""增量因子生产：结合 watermark + lookback 缓冲。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from cleaned_operators.operator_policy import effective_lookback
from storage.time_window import (
    resolve_incremental_window_for_bar_freq,
    slice_series_time_window,
)


@dataclass(frozen=True)
class IncrementalPlan:
    """增量执行计划。"""

    factor_id: str
    lookback_bars: int
    load_start: pd.Timestamp | None
    load_end: pd.Timestamp | None
    output_start: pd.Timestamp | None
    output_end: pd.Timestamp | None
    watermark_end: str | None
    is_full_run: bool
    factor_freq: str | None = None
    source_bar_freq: str | None = None
    window_mode: str = "daily"

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "lookback_bars": self.lookback_bars,
            "load_start": None if self.load_start is None else self.load_start.isoformat(),
            "load_end": None if self.load_end is None else self.load_end.isoformat(),
            "output_start": None if self.output_start is None else self.output_start.isoformat(),
            "output_end": None if self.output_end is None else self.output_end.isoformat(),
            "watermark_end": self.watermark_end,
            "is_full_run": self.is_full_run,
            "factor_freq": self.factor_freq,
            "source_bar_freq": self.source_bar_freq,
            "window_mode": self.window_mode,
        }


def build_incremental_plan(
    *,
    factor_id: str,
    analysis_lookback: int,
    watermark: dict | None,
    since: str | None = None,
    end_date: str | None = None,
    lookback_extra: int = 5,
    recompute_tail_bars: int | None = None,
    market: str | None = None,
    calendar=None,
    factor_freq: str | None = None,
    source_bar_freq: str | None = None,
) -> IncrementalPlan:
    """构建增量计划；日内源按 ``bars_to_calendar_trading_days`` 换算窗口。"""
    from storage.trading_calendar import get_trading_calendar

    wm_end: str | None = None
    if since:
        wm_end = str(since)
    elif watermark is not None:
        raw = watermark.get("end_date")
        if raw:
            wm_end = str(raw)

    load_lookback = effective_lookback(
        analysis_lookback,
        factor_freq=factor_freq,
        source_bar_freq=source_bar_freq,
        extra=lookback_extra,
    )
    if recompute_tail_bars is None:
        # 输出 tail 只需覆盖因子 IR lookback + 1 bar lag，不必重算整个 load 缓冲
        tail_bars = max(1, int(analysis_lookback) + 1)
    else:
        tail_bars = max(0, int(recompute_tail_bars))

    # resolve_incremental_window 使用交易日偏移；日内 bar 用 bar_freq 近似 + 安全日缓冲
    cal = calendar if calendar is not None else get_trading_calendar(market)
    window = resolve_incremental_window_for_bar_freq(
        watermark_end=wm_end,
        lookback_bars=load_lookback,
        since=None,
        end_date=end_date,
        recompute_tail_bars=tail_bars,
        calendar=cal,
        bar_freq=source_bar_freq,
    )
    window_mode = str(window.get("window_mode") or "daily")

    is_full = wm_end is None
    return IncrementalPlan(
        factor_id=factor_id,
        lookback_bars=load_lookback,
        load_start=window["load_start"],
        load_end=window["load_end"],
        output_start=window["output_start"],
        output_end=window["output_end"],
        watermark_end=wm_end,
        is_full_run=is_full,
        factor_freq=factor_freq,
        source_bar_freq=source_bar_freq,
        window_mode=window_mode,
    )


def slice_factor_result_for_incremental(
    result: pd.Series,
    plan: IncrementalPlan,
) -> pd.Series:
    if plan.is_full_run or plan.output_start is None:
        return result
    return slice_series_time_window(
        result,
        start=plan.output_start,
        end=plan.output_end,
        exclusive_start=False,
    )
