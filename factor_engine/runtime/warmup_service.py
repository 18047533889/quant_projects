# -*- coding: utf-8
"""全量 run 的 auto_warmup 扩窗与引擎切换。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cleaned_operators.operator_policy import (
    bars_per_day,
    effective_lookback,
    infer_source_bar_freq,
)
from logging_utils import get_logger
from runtime.run_window import RunWindow, build_full_run_window, extract_source_date_bounds
from runtime.session_calendar import SessionBarCalendar
from storage.cache import PersistentPlanCache

logger = get_logger("runtime.warmup_service")


@dataclass(frozen=True)
class WarmupContext:
    engine: Any
    run_window: RunWindow | None
    source_bar_freq: str
    history_buffer: int
    bars_per_day: int


def _build_intraday_run_window(
    *,
    requested_start: str | None,
    requested_end: str | None,
    lookback_bars: int,
    bar_freq: str,
    trim_output: bool,
) -> RunWindow:
    from runtime.run_window import _to_date_str

    req_start = _to_date_str(requested_start)
    req_end = _to_date_str(requested_end)
    lb = max(0, int(lookback_bars))
    if lb <= 0 or req_start is None:
        return RunWindow(
            requested_start=req_start,
            requested_end=req_end,
            actual_load_start=req_start,
            actual_load_end=req_end,
            warmup_bars=0,
            trim_output=trim_output,
        )
    cal = SessionBarCalendar(bar_freq)
    load_start = cal.warmup_load_start(req_start, lookback_bars=lb)
    return RunWindow(
        requested_start=req_start,
        requested_end=req_end,
        actual_load_start=load_start.isoformat(),
        actual_load_end=req_end,
        warmup_bars=lb,
        trim_output=trim_output,
    )


def prepare_run_warmup(
    engine: Any,
    factor: Any,
    analysis: Any,
    *,
    auto_warmup: bool,
    trim_warmup: bool,
    market: str | None,
) -> WarmupContext:
    """计算 warmup 计划；必要时返回 narrowed data_source 的 engine 副本。"""
    source_bar_freq = infer_source_bar_freq(
        engine.data_source,
        fallback=getattr(factor, "freq", None),
    )
    history_buffer = effective_lookback(
        getattr(analysis, "lookback", 0),
        factor_freq=getattr(factor, "freq", None),
        source_bar_freq=source_bar_freq,
    )
    s_bpd = bars_per_day(source_bar_freq)
    warmup_calendar_bars = history_buffer
    if s_bpd > 1:
        warmup_calendar_bars = max(1, (history_buffer + s_bpd - 1) // s_bpd)

    run_window = None
    engine_to_use = engine

    if auto_warmup and history_buffer > 0:
        from storage.time_window import narrow_data_source_for_window
        from storage.trading_calendar import get_trading_calendar, infer_market

        req_start, req_end = extract_source_date_bounds(engine.data_source)
        dataset = getattr(engine.data_source, "dataset", None)
        resolved_market = market or infer_market(
            universe=getattr(factor, "universe", None),
            dataset=str(dataset) if dataset else None,
        )
        if s_bpd > 1:
            run_window = _build_intraday_run_window(
                requested_start=req_start,
                requested_end=req_end,
                lookback_bars=history_buffer,
                bar_freq=source_bar_freq,
                trim_output=trim_warmup,
            )
        else:
            cal = get_trading_calendar(resolved_market)
            run_window = build_full_run_window(
                requested_start=req_start,
                requested_end=req_end,
                lookback_bars=warmup_calendar_bars,
                trim_output=trim_warmup,
                calendar=cal,
            )
        if (
            run_window.actual_load_start is not None
            and run_window.actual_load_start != req_start
        ):
            narrowed = narrow_data_source_for_window(
                engine.data_source,
                start_date=run_window.actual_load_start,
                end_date=run_window.actual_load_end,
                bar_freq=source_bar_freq if s_bpd > 1 else None,
            )
            use_fresh = not isinstance(engine.cache, PersistentPlanCache)
            engine_to_use = engine.with_data_source(narrowed, fresh_cache=use_fresh)
            logger.info(
                "因子 '%s' auto_warmup: load_start %s → %s（warmup_bars=%d, bar_freq=%s）",
                factor.name,
                req_start,
                run_window.actual_load_start,
                run_window.warmup_bars,
                source_bar_freq,
            )
    elif history_buffer > 0:
        logger.info(
            "因子 '%s' 建议历史缓冲 >= %d bars（lookback=%s）；可设 auto_warmup=True",
            factor.name,
            history_buffer,
            getattr(analysis, "lookback", 0),
        )

    return WarmupContext(
        engine=engine_to_use,
        run_window=run_window,
        source_bar_freq=source_bar_freq,
        history_buffer=history_buffer,
        bars_per_day=s_bpd,
    )
