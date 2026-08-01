# -*- coding: utf-8 -*-
"""Auto-warmup and full-history replay planning for full factor runs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from cleaned_operators.operator_policy import effective_lookback, infer_source_bar_freq
from logging_utils import get_logger
from runtime.run_window import (
    RunWindow,
    build_full_history_run_window,
    build_full_run_window,
    extract_source_date_bounds,
)
from runtime.session_calendar import SessionBarCalendar
from storage.cache import PersistentPlanCache

logger = get_logger("runtime.warmup_service")
FULL_HISTORY_LOOKBACK_SENTINEL = 1_000_000_000


@dataclass(frozen=True)
class WarmupContext:
    engine: Any
    run_window: RunWindow | None
    source_bar_freq: str
    history_buffer: int
    bars_per_day: int
    full_history_required: bool = False
    full_history_start: str | None = None
    full_history_satisfied: bool = True


def _build_intraday_run_window(
    *,
    requested_start: str | None,
    requested_end: str | None,
    lookback_bars: int,
    bar_freq: str,
    trim_output: bool,
    market: str | None = None,
) -> RunWindow:
    from runtime.run_window import _to_date_str

    requested_start = _to_date_str(requested_start)
    requested_end = _to_date_str(requested_end)
    lookback = max(0, int(lookback_bars))
    if lookback <= 0 or requested_start is None:
        return RunWindow(
            requested_start=requested_start,
            requested_end=requested_end,
            actual_load_start=requested_start,
            actual_load_end=requested_end,
            warmup_bars=0,
            trim_output=trim_output,
        )
    from cleaned_operators.operator_policy import normalize_bars_market

    calendar = SessionBarCalendar(
        bar_freq, market=normalize_bars_market(market)
    )
    load_start = calendar.warmup_load_start(
        requested_start, lookback_bars=lookback
    )
    return RunWindow(
        requested_start=requested_start,
        requested_end=requested_end,
        actual_load_start=load_start.isoformat(),
        actual_load_end=requested_end,
        warmup_bars=lookback,
        trim_output=trim_output,
    )


def _normalise_date(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return pd.Timestamp(text).strftime("%Y-%m-%d")


def _direct_full_history_start(data_source: Any) -> str | None:
    direct = getattr(data_source, "full_history_start", None)
    if direct is not None:
        return _normalise_date(direct)
    params = dict(getattr(data_source, "params", {}) or {})
    for key in ("full_history_start", "history_origin", "listing_history_start"):
        if params.get(key) is not None:
            return _normalise_date(params[key])
    return None


def resolve_full_history_start(data_source: Any) -> str | None:
    """Resolve one explicit history origin from a logical/composite source."""
    direct = _direct_full_history_start(data_source)
    if direct is not None:
        return direct

    seen: set[int] = set()
    candidates: set[str] = set()

    def visit(source: Any) -> None:
        if source is None or id(source) in seen:
            return
        seen.add(id(source))
        value = _direct_full_history_start(source)
        if value is not None:
            candidates.add(value)
        for attribute in ("inner", "_inner", "anchor", "anchor_source"):
            visit(getattr(source, attribute, None))
        children = getattr(source, "sources", None)
        if isinstance(children, dict):
            for child in children.values():
                visit(child)

    visit(data_source)
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(
            "ambiguous full_history_start across data sources: "
            + ", ".join(sorted(candidates))
        )
    return next(iter(candidates))


def _preserve_source_contracts(original: Any, narrowed: Any) -> Any:
    """Copy immutable read contracts, never caches or stale snapshot state."""
    original_params = dict(getattr(original, "params", {}) or {})
    if hasattr(narrowed, "params") and original_params:
        narrowed.params = original_params
    if hasattr(original, "read_auto") and hasattr(narrowed, "read_auto"):
        narrowed.read_auto = bool(original.read_auto)
    if hasattr(original, "lazy_scan") and hasattr(narrowed, "_lazy_scan"):
        narrowed._lazy_scan = bool(original.lazy_scan)
    for attribute in (
        "full_history_start",
        "bar_freq",
        "session_open",
        "session_close",
        "session_minutes",
        "timestamp_convention",
    ):
        if hasattr(original, attribute):
            try:
                setattr(narrowed, attribute, getattr(original, attribute))
            except Exception:
                pass
    return narrowed


def _switch_engine_to_window(
    engine: Any,
    run_window: RunWindow,
    *,
    source_bar_freq: str,
    bars_per_day: int,
) -> Any:
    if (
        run_window.actual_load_start is None
        or run_window.actual_load_start == run_window.requested_start
    ):
        return engine
    from storage.time_window import narrow_data_source_for_window

    narrowed = narrow_data_source_for_window(
        engine.data_source,
        start_date=run_window.actual_load_start,
        end_date=run_window.actual_load_end,
        bar_freq=source_bar_freq if bars_per_day > 1 else None,
    )
    narrowed = _preserve_source_contracts(engine.data_source, narrowed)
    use_fresh_cache = not isinstance(engine.cache, PersistentPlanCache)
    return engine.with_data_source(narrowed, fresh_cache=use_fresh_cache)


def prepare_run_warmup(
    engine: Any,
    factor: Any,
    analysis: Any,
    *,
    auto_warmup: bool,
    trim_warmup: bool,
    market: str | None,
) -> WarmupContext:
    source_bar_freq = infer_source_bar_freq(
        engine.data_source,
        fallback=getattr(factor, "freq", None),
    )
    from cleaned_operators.operator_policy import bars_per_day, normalize_bars_market
    from runtime.production_policy import ProductionPolicyViolation, is_production_mode
    from storage.trading_calendar import infer_market

    requires_full_history = bool(
        getattr(analysis, "requires_full_history", False)
        or int(getattr(analysis, "lookback", 0)) >= FULL_HISTORY_LOOKBACK_SENTINEL
    )
    raw_lookback = int(getattr(analysis, "lookback", 0))
    finite_lookback = 0 if requires_full_history else raw_lookback
    history_buffer = effective_lookback(
        finite_lookback,
        factor_freq=getattr(factor, "freq", None),
        source_bar_freq=source_bar_freq,
    )
    resolved_market = market or infer_market(
        universe=getattr(factor, "universe", None),
        dataset=str(getattr(engine.data_source, "dataset", None) or ""),
    )
    bars_market = normalize_bars_market(resolved_market)
    source_bars_per_day = bars_per_day(source_bar_freq, market=bars_market)
    warmup_calendar_bars = history_buffer
    if source_bars_per_day > 1:
        warmup_calendar_bars = max(
            1, (history_buffer + source_bars_per_day - 1) // source_bars_per_day
        )

    requested_start, requested_end = extract_source_date_bounds(engine.data_source)
    run_window: RunWindow | None = None
    engine_to_use = engine
    full_history_start = None
    full_history_satisfied = True

    if requires_full_history:
        full_history_start = resolve_full_history_start(engine.data_source)
        if full_history_start is None:
            full_history_satisfied = False
            message = (
                f"factor {factor.name!r} requires full-history replay but the "
                "data source does not declare full_history_start"
            )
            if is_production_mode(engine.run_mode):
                raise ProductionPolicyViolation(message)
            logger.warning(message)
        elif not auto_warmup:
            full_history_satisfied = (
                requested_start is None
                or pd.Timestamp(requested_start) <= pd.Timestamp(full_history_start)
            )
            message = (
                f"factor {factor.name!r} requires full-history replay from "
                f"{full_history_start}; enable auto_warmup"
            )
            if is_production_mode(engine.run_mode):
                raise ProductionPolicyViolation(message)
            logger.warning(message)
        else:
            try:
                run_window = build_full_history_run_window(
                    requested_start=requested_start,
                    requested_end=requested_end,
                    full_history_start=full_history_start,
                    trim_output=trim_warmup,
                )
            except ValueError as error:
                if is_production_mode(engine.run_mode):
                    raise ProductionPolicyViolation(str(error)) from error
                raise
            engine_to_use = _switch_engine_to_window(
                engine,
                run_window,
                source_bar_freq=source_bar_freq,
                bars_per_day=source_bars_per_day,
            )
            logger.info(
                "factor '%s' full-history replay: load_start %s → %s",
                factor.name,
                requested_start,
                full_history_start,
            )
    elif auto_warmup and history_buffer > 0:
        from storage.trading_calendar import get_trading_calendar

        if source_bars_per_day > 1:
            run_window = _build_intraday_run_window(
                requested_start=requested_start,
                requested_end=requested_end,
                lookback_bars=history_buffer,
                bar_freq=source_bar_freq,
                trim_output=trim_warmup,
                market=resolved_market,
            )
        else:
            calendar = get_trading_calendar(resolved_market)
            run_window = build_full_run_window(
                requested_start=requested_start,
                requested_end=requested_end,
                lookback_bars=warmup_calendar_bars,
                trim_output=trim_warmup,
                calendar=calendar,
            )
        engine_to_use = _switch_engine_to_window(
            engine,
            run_window,
            source_bar_freq=source_bar_freq,
            bars_per_day=source_bars_per_day,
        )
        if run_window.actual_load_start != requested_start:
            logger.info(
                "factor '%s' auto_warmup: load_start %s → %s "
                "(warmup_bars=%d, bar_freq=%s)",
                factor.name,
                requested_start,
                run_window.actual_load_start,
                run_window.warmup_bars,
                source_bar_freq,
            )
    elif history_buffer > 0:
        logger.info(
            "factor '%s' recommends >= %d history bars; set auto_warmup=True",
            factor.name,
            history_buffer,
        )

    return WarmupContext(
        engine=engine_to_use,
        run_window=run_window,
        source_bar_freq=source_bar_freq,
        history_buffer=history_buffer,
        bars_per_day=source_bars_per_day,
        full_history_required=requires_full_history,
        full_history_start=full_history_start,
        full_history_satisfied=full_history_satisfied,
    )
