# -*- coding: utf-8 -*-
"""Full-run warmup windows and output trimming."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from factor_engine.storage.time_window import business_day_offset
from factor_engine.storage.trading_calendar import TradingCalendar


@dataclass(frozen=True)
class RunWindow:
    """Requested output window versus actual data-load window."""

    requested_start: str | None
    requested_end: str | None
    actual_load_start: str | None
    actual_load_end: str | None
    warmup_bars: int
    trim_output: bool
    full_history_required: bool = False
    full_history_start: str | None = None
    full_history_satisfied: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_start": self.requested_start,
            "requested_end": self.requested_end,
            "actual_load_start": self.actual_load_start,
            "actual_load_end": self.actual_load_end,
            "warmup_bars": self.warmup_bars,
            "trim_output": self.trim_output,
            "full_history_required": self.full_history_required,
            "full_history_start": self.full_history_start,
            "full_history_satisfied": self.full_history_satisfied,
        }


def _to_date_str(value: str | pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return pd.Timestamp(text).strftime("%Y-%m-%d")


def extract_source_date_bounds(data_source: Any) -> tuple[str | None, str | None]:
    time_range_fn = getattr(data_source, "time_range", None)
    if callable(time_range_fn):
        time_range = time_range_fn()
        if time_range and len(time_range) >= 2:
            return _to_date_str(time_range[0]), _to_date_str(time_range[1])
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
    lookback = max(0, int(lookback_bars))
    requested_start = _to_date_str(requested_start)
    requested_end = _to_date_str(requested_end)

    if lookback <= 0 or requested_start is None:
        return RunWindow(
            requested_start=requested_start,
            requested_end=requested_end,
            actual_load_start=requested_start,
            actual_load_end=requested_end,
            warmup_bars=0,
            trim_output=trim_output,
        )

    load_start = business_day_offset(
        requested_start, -lookback, calendar=calendar
    )
    return RunWindow(
        requested_start=requested_start,
        requested_end=requested_end,
        actual_load_start=_to_date_str(load_start),
        actual_load_end=requested_end,
        warmup_bars=lookback,
        trim_output=trim_output,
    )


def build_full_history_run_window(
    *,
    requested_start: str | None,
    requested_end: str | None,
    full_history_start: str,
    trim_output: bool = True,
) -> RunWindow:
    """Build a full-replay window from an explicit immutable history origin."""
    requested_start = _to_date_str(requested_start)
    requested_end = _to_date_str(requested_end)
    origin = _to_date_str(full_history_start)
    if origin is None:
        raise ValueError("full_history_start is required")
    if requested_start is not None and pd.Timestamp(origin) > pd.Timestamp(requested_start):
        raise ValueError(
            "full_history_start must be on or before requested_start"
        )
    return RunWindow(
        requested_start=requested_start,
        requested_end=requested_end,
        actual_load_start=origin,
        actual_load_end=requested_end,
        warmup_bars=0,
        trim_output=trim_output,
        full_history_required=True,
        full_history_start=origin,
        full_history_satisfied=True,
    )
