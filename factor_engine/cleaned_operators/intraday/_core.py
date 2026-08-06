# -*- coding: utf-8 -*-
"""Shared kernels for next-stage intraday operators.

Self-contained helpers (adapted from the reviewed ``microstructure`` contract):
every operator accepts minute-frequency panels (row index is a minute
timestamp, columns are instruments) and returns a daily-frequency panel
(row index is the calendar date, columns are instruments).

Contract
--------
* One scalar per (TradeDate, Symbol).  Never a row per minute.
* All kernels are causal: only the day's own minute data plus that day's daily
  limit / weight panels, never future bars or future days.
* Empty / all-NaN windows yield NaN, never Inf or a fabricated zero.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata

_EPS = 1e-12
_SESSION_TZ = "Asia/Shanghai"
_MORNING = (570, 690)   # 09:30 .. 11:30 minute-of-day
_AFTERNOON = (780, 900)  # 13:00 .. 15:00 minute-of-day
_SEGMENT_RANGES = {"morning": _MORNING, "afternoon": _AFTERNOON}


def metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    unit: str,
    domain: str = "intraday",
    cost: int = 6,
    extra_tags: list[str] | None = None,
) -> OperatorMetadata:
    """Standard metadata for intraday -> daily aggregation operators."""
    tags = [
        "intraday", "daily_agg", "minute", "pit_safe", "causal",
        "typed_v2", "source_blocked",
        f"signature:{','.join(params)}->series", f"unit:{unit}",
        f"cost:{cost}", f"domain:{domain}",
    ]
    if extra_tags:
        tags.extend(extra_tags)
    return OperatorMetadata(
        name=name,
        category="intraday_microstructure",
        description=description,
        param_names=params,
        return_type="series",
        tags=tags,
    )


def as_panel(x: Any) -> pd.DataFrame:
    if isinstance(x, pd.Series):
        return x.to_frame(getattr(x, "name", None) or "value")
    return x


def minute_of_day(times: np.ndarray) -> np.ndarray:
    """Timestamp array -> minute-of-day (integer)."""
    seconds = times.astype("datetime64[s]").astype("int64") % 86400
    return seconds // 60


def log_returns(vals: np.ndarray) -> np.ndarray:
    out = np.full(len(vals), np.nan)
    if len(vals) > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            out[1:] = np.log(vals[1:] / vals[:-1])
    return out


def session_local(frame: pd.DataFrame, tz: str | None = None) -> pd.DataFrame:
    """Convert a tz-aware index to session wall-clock (naive) for minute math.

    A-share COS minute data is stored in UTC; session segments are defined in
    Asia/Shanghai wall-clock.  Naive indexes are assumed to already be local.
    """
    frame = as_panel(frame)
    if isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is not None:
        tz = tz or _SESSION_TZ
        frame = frame.tz_convert(tz)
        frame.index = frame.index.tz_localize(None)
    return frame


def seg_mask(times: np.ndarray, segment: str) -> np.ndarray:
    lo, hi = _SEGMENT_RANGES[str(segment)]
    minutes = minute_of_day(times)
    return (minutes >= lo) & (minutes <= hi)


def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.isfinite(den) & (np.abs(den) > _EPS), num / den, np.nan)
    return out


def daily_agg(frame: pd.DataFrame, fn: Callable[[np.ndarray, np.ndarray], float]) -> pd.DataFrame:
    """Apply per-(instrument, calendar-day) aggregation fn(vals, times)."""
    frame = as_panel(frame)
    out: dict[str, pd.Series] = {}
    for inst in frame.columns:
        col = frame[inst]
        per_day: dict[pd.Timestamp, float] = {}
        for day, group in col.groupby(col.index.normalize()):
            vals = np.asarray(group, dtype=float)
            times = np.asarray(group.index, dtype="datetime64[ns]")
            if not np.any(np.isfinite(vals)):
                per_day[day] = np.nan
                continue
            try:
                per_day[day] = float(fn(vals, times))
            except (ValueError, ZeroDivisionError, OverflowError):
                per_day[day] = np.nan
        out[inst] = pd.Series(per_day, dtype=float)
    if not out:
        return pd.DataFrame(dtype=float)
    return pd.DataFrame(out).sort_index()


def daily_agg_two(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    fn: Callable[[np.ndarray, np.ndarray], float],
) -> pd.DataFrame:
    """Apply fn(a_vals, b_vals) per (instrument, day)."""
    frame_a, frame_b = as_panel(frame_a), as_panel(frame_b)
    out: dict[str, pd.Series] = {}
    for inst in frame_a.columns:
        a, b = frame_a[inst], frame_b[inst]
        joined = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna(subset=["a"])
        joined["day"] = joined.index.normalize()
        per_day: dict[pd.Timestamp, float] = {}
        for day, group in joined.groupby("day"):
            vals_a = np.asarray(group["a"], dtype=float)
            vals_b = np.asarray(group["b"], dtype=float)
            if not np.any(np.isfinite(vals_a)):
                per_day[day] = np.nan
                continue
            try:
                per_day[day] = float(fn(vals_a, vals_b))
            except (ValueError, ZeroDivisionError, OverflowError):
                per_day[day] = np.nan
        out[inst] = pd.Series(per_day, dtype=float)
    if not out:
        return pd.DataFrame(dtype=float)
    return pd.DataFrame(out).sort_index()


def daily_agg_three(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    frame_c: pd.DataFrame,
    fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
) -> pd.DataFrame:
    """Apply fn(a_vals, b_vals, c_vals) per (instrument, day)."""
    frame_a, frame_b, frame_c = as_panel(frame_a), as_panel(frame_b), as_panel(frame_c)
    out: dict[str, pd.Series] = {}
    for inst in frame_a.columns:
        joined = pd.concat(
            [frame_a[inst], frame_b[inst], frame_c[inst]], axis=1, keys=["a", "b", "c"]
        ).dropna(subset=["a"])
        joined["day"] = joined.index.normalize()
        per_day: dict[pd.Timestamp, float] = {}
        for day, group in joined.groupby("day"):
            vals_a = np.asarray(group["a"], dtype=float)
            vals_b = np.asarray(group["b"], dtype=float)
            vals_c = np.asarray(group["c"], dtype=float)
            if not np.any(np.isfinite(vals_a)):
                per_day[day] = np.nan
                continue
            try:
                per_day[day] = float(fn(vals_a, vals_b, vals_c))
            except (ValueError, ZeroDivisionError, OverflowError):
                per_day[day] = np.nan
        out[inst] = pd.Series(per_day, dtype=float)
    if not out:
        return pd.DataFrame(dtype=float)
    return pd.DataFrame(out).sort_index()


def broadcast_daily_panel(close_frame: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Broadcast a daily panel (index=date) onto a minute panel by date."""
    daily = as_panel(daily)
    out: dict[str, pd.Series] = {}
    for inst in close_frame.columns:
        if inst not in daily.columns:
            continue
        close_col = close_frame[inst]
        daily_col = daily[inst]
        days = close_col.index.normalize()
        mapped = daily_col.reindex(pd.DatetimeIndex(days.unique()))
        out[inst] = pd.Series(
            mapped.reindex(pd.DatetimeIndex(days)).to_numpy(),
            index=close_col.index,
        )
    return pd.DataFrame(out)


def register_surface(canonicals: list[str]) -> None:
    """Append new canonicals to the reviewed extended surface.

    Union is order-independent so concurrent modules may each add their own
    names without clobbering one another.
    """
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | set(canonicals)
    )


def register_research_surface(canonicals: list[str]) -> None:
    """Append new canonicals to the research surface (experimental operators)."""
    import cleaned_operators.operator_surface as _surface

    _surface.RESEARCH_ONLY_CANONICALS = frozenset(
        set(_surface.RESEARCH_ONLY_CANONICALS) | set(canonicals)
    )


# Silent-numpy context for statistic kernels that may overflow intermediate
# products on extreme minute moves.
_np_err = {"divide": "ignore", "invalid": "ignore", "over": "ignore"}


def np_errstate():
    return np.errstate(**_np_err)


def mu_p(p: float) -> float:
    """E|Z|^p for a standard normal Z."""
    return float(2.0 ** (p / 2.0) * math.gamma((p + 1.0) / 2.0) / math.gamma(0.5))


def tripower_scale() -> float:
    """mu_{4/3}^{-3} constant for tripower quarticity."""
    return float(mu_p(4.0 / 3.0) ** -3)
