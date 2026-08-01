# -*- coding: utf-8 -*-
"""Incremental factor production with watermark and causal history contracts."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import pandas as pd

from cleaned_operators.operator_policy import effective_lookback
from storage.time_window import (
    resolve_incremental_window_for_bar_freq,
    slice_series_time_window,
)

FULL_HISTORY_LOOKBACK_SENTINEL = 1_000_000_000
_PENDING_INCREMENTAL_HISTORY: ContextVar[dict[str, str] | None] = ContextVar(
    "factor_engine_pending_incremental_history", default=None
)
_INCREMENTAL_HISTORY_CERTIFICATE: ContextVar[bool] = ContextVar(
    "factor_engine_incremental_history_certificate", default=False
)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).isoformat()


def _stage_incremental_history_contract(plan: "IncrementalPlan") -> None:
    """Stage expected window bounds; do not certify until narrowing occurs."""
    _INCREMENTAL_HISTORY_CERTIFICATE.set(False)
    if (
        plan.is_full_run
        or plan.full_history_required
        or plan.load_start is None
        or plan.output_start is None
    ):
        _PENDING_INCREMENTAL_HISTORY.set(None)
        return
    _PENDING_INCREMENTAL_HISTORY.set(
        {
            "factor_id": plan.factor_id,
            "load_start": _iso(plan.load_start) or "",
            "load_end": _iso(plan.load_end) or "",
            "output_start": _iso(plan.output_start) or "",
            "output_end": _iso(plan.output_end) or "",
        }
    )


def certify_narrowed_incremental_window(
    *,
    start_date: Any,
    end_date: Any,
) -> bool:
    """Certify only when the actual narrowed source matches the staged plan."""
    pending = _PENDING_INCREMENTAL_HISTORY.get()
    if not pending:
        return False
    actual_start = _iso(start_date) or ""
    actual_end = _iso(end_date) or ""
    if actual_start != pending["load_start"] or actual_end != pending["load_end"]:
        _PENDING_INCREMENTAL_HISTORY.set(None)
        _INCREMENTAL_HISTORY_CERTIFICATE.set(False)
        return False
    _PENDING_INCREMENTAL_HISTORY.set(None)
    _INCREMENTAL_HISTORY_CERTIFICATE.set(True)
    return True


def clear_incremental_history_contract() -> None:
    _PENDING_INCREMENTAL_HISTORY.set(None)
    _INCREMENTAL_HISTORY_CERTIFICATE.set(False)


def consume_incremental_history_certificate() -> bool:
    """Consume and clear the current task's one-shot certificate."""
    value = bool(_INCREMENTAL_HISTORY_CERTIFICATE.get())
    _INCREMENTAL_HISTORY_CERTIFICATE.set(False)
    return value


@dataclass(frozen=True)
class IncrementalPlan:
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
    full_history_required: bool = False

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
            "full_history_required": self.full_history_required,
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
    """Build an incremental plan, forcing full replay for recursive factors."""
    from storage.trading_calendar import get_trading_calendar

    raw_lookback = int(analysis_lookback)
    full_history_required = raw_lookback >= FULL_HISTORY_LOOKBACK_SENTINEL
    finite_lookback = 0 if full_history_required else max(0, raw_lookback)

    watermark_end: str | None = None
    if since:
        watermark_end = str(since)
    elif watermark is not None:
        raw = watermark.get("end_date")
        if raw:
            watermark_end = str(raw)

    window_watermark = None if full_history_required else watermark_end
    load_lookback = effective_lookback(
        finite_lookback,
        factor_freq=factor_freq,
        source_bar_freq=source_bar_freq,
        extra=lookback_extra,
    )
    tail_bars = (
        max(1, finite_lookback + 1)
        if recompute_tail_bars is None
        else max(0, int(recompute_tail_bars))
    )

    calendar = calendar if calendar is not None else get_trading_calendar(market)
    window = resolve_incremental_window_for_bar_freq(
        watermark_end=window_watermark,
        lookback_bars=load_lookback,
        since=None,
        end_date=end_date,
        recompute_tail_bars=tail_bars,
        calendar=calendar,
        bar_freq=source_bar_freq,
    )
    window_mode = str(window.get("window_mode") or "daily")
    is_full = full_history_required or watermark_end is None

    plan = IncrementalPlan(
        factor_id=factor_id,
        lookback_bars=load_lookback,
        load_start=window["load_start"],
        load_end=window["load_end"],
        output_start=window["output_start"],
        output_end=window["output_end"],
        watermark_end=watermark_end,
        is_full_run=is_full,
        factor_freq=factor_freq,
        source_bar_freq=source_bar_freq,
        window_mode="full_history" if full_history_required else window_mode,
        full_history_required=full_history_required,
    )
    _stage_incremental_history_contract(plan)
    return plan


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
