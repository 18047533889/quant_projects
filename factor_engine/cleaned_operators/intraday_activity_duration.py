# -*- coding: utf-8 -*-
"""Intraday activity-duration curvature (2026-08-08 Gemini round).

``intraday_activity_duration_curvature`` — one scalar per (date, symbol): split
the day's non-negative minute ``activity`` (volume / amount) into ``buckets``
equal-activity buckets, record the *trading-session* bar index at which each
bucket completes ``D_1..D_B``, and return the standardised second difference of
the duration curve ``mean(Δ²D) / (MAD(D)+eps)``.

* Positive curvature → activity *decelerating* into the close (later buckets
  take longer session-clock).
* Negative curvature → activity *accelerating* into the close.

Clock semantics (P0-005, R11 #183): the session minute axis is NEVER compressed
— any NaN minute activity fail-closes the day, real zero-activity bars are
preserved, and positions are ORIGINAL session-bar indices.  This is
*trading-session-clock* (A continuous-auction minutes, ~240 bars/day), NOT
wall-clock elapsed time; US has no full-minute OHLCV provider yet, so the
operator is A-share supported and US provider_required.  Minute-source input,
strict-PIT within the day, deterministic, NaN fail-closed.

Round-11 contract (findings #183-#186)
--------------------------------------
* Clock semantics unified on the trading-session clock (module and class docs
  agree with the implementation) — R11 #183.
* The day is reindexed onto the OFFICIAL SessionGrid supplied by an explicit
  exchange ``calendar`` (``runtime.session_calendar.SessionCalendar``); a missing
  whole-minute row is otherwise invisible to a NaN gate — R11 #184.  The grid is
  NEVER inferred from a modal of the observed data (P0-10): without a calendar
  the grid is UNKNOWN and every day fails closed (NaN).
* A partial session (fewer observed minutes than the official grid, or a
  truncated close) fails closed and never emits — R11 #185.
* The output is dimensionless: ``mean(Δ²D) / MAD(D)`` (bar-index differences
  cancel), not a generic level — R11 #186.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like

_EPS = 1e-12
_SESSION_TZ = "Asia/Shanghai"
_WARNED_NO_CALENDAR = False


def _metadata(name: str, description: str, params: list[str]) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="intraday_microstructure",
        description=description,
        param_names=params,
        return_type="series",
        # R11 P0-13: these operators consume minute bars and emit one scalar per
        # (TradeDate, Symbol) — the grain contract is declared HERE so the
        # policy layer classifies them non-shape-preserving from their own
        # contract, not from any hand-maintained canonical list.
        input_grain="minute",
        output_grain="daily",
        available_at="session_close",
        same_session_usable=False,
        tags=[
            "intraday", "minute", "daily", "pit_safe", "causal", "deterministic",
            f"signature:{','.join(params)}->series", "domain:intraday",
            # R11 #186: mean(Δ²D)/MAD(D) is dimensionless (bar-index ratio).
            "unit:ratio", "cost:6",
        ],
    )


def _minute_of_day(times: np.ndarray) -> np.ndarray:
    seconds = times.astype("datetime64[s]").astype("int64") % 86400
    return seconds // 60


def _session_local_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """A-share session wall-clock (naive) for a tz-aware minute panel."""
    idx = frame.index
    if isinstance(idx, pd.DatetimeIndex) and getattr(idx, "tz", None) is not None:
        out = frame.tz_convert(_SESSION_TZ)
        out.index = out.index.tz_localize(None)
        return out
    return frame


def _hhmm_minute(value: str) -> int:
    """Wall-clock ``"HH:MM"`` -> minute-of-day (``"15:00" -> 900``)."""
    hour, minute = (int(part) for part in str(value).split(":"))
    return hour * 60 + minute


def _official_session_grid(calendar: Any) -> tuple[int, ...]:
    """Official full-session minute-of-day grid from an EXPLICIT calendar.

    P0-10: the official SessionGrid is NEVER inferred from the observed minute
    data.  The modal-grid self-certification is removed — a dataset that
    systematically drops the final bar of every session (239 bars instead of
    240) must not be able to re-define which grid is "official".  ``calendar``
    is a ``runtime.session_calendar.SessionCalendar`` (duck-typed: ``segments``
    of ``("HH:MM","HH:MM")`` and ``timestamp_convention`` in
    {``bar_start``, ``bar_end``}); the returned tuple is the full ordered set of
    minute-of-day slots of a complete session (e.g. A-share 1-min bar_start:
    ``570..689`` + ``780..899``).
    """
    segments = list(getattr(calendar, "segments", None) or ())
    if not segments:
        raise ValueError("calendar must define non-empty session segments")
    convention = str(getattr(calendar, "timestamp_convention", "bar_end")).lower()
    if convention not in {"bar_start", "bar_end"}:
        raise ValueError("calendar timestamp_convention must be bar_start or bar_end")
    # R26-056: the official grid is generated at the calendar's DECLARED
    # ``bar_freq`` resolution — a 5-min source must not be judged against a
    # per-minute grid (it would read every 4 of 5 bars as "missing").  The
    # declared width is never inferred from observed deltas.
    from runtime.session_panel import declared_width_minutes

    width = declared_width_minutes(calendar)
    out: list[int] = []
    for start_text, stop_text in segments:
        start, stop = _hhmm_minute(start_text), _hhmm_minute(stop_text)
        if stop <= start:
            raise ValueError("calendar segments must be increasing intervals")
        if convention == "bar_start":
            out.extend(range(start, stop, width))
        else:
            out.extend(range(start + width, stop + 1, width))
    return tuple(out)


def _day_curvature(values: np.ndarray, buckets: int) -> float:
    """Duration-curve second-difference curvature for one day of minute activity.

    Trading-session-clock semantics (P0-005, R11 #183): completion positions are
    ORIGINAL session-bar indices — the minute axis is NEVER compressed by
    dropping NaN bars.  A NaN minute activity fail-closes the whole day (an
    unknown minute is not a zero-volume minute; deleting it would silently
    shorten the session and move the duration curve).  Negative activity is an
    invalid state.  Real ``activity == 0`` bars are preserved: they are "no
    trade" but still occupy a session bar and correctly stretch the duration
    curve.  The result ``mean(Δ²D)/MAD(D)`` is dimensionless (R11 #186).
    """
    v = values.astype(float)
    if not np.all(np.isfinite(v)):
        return np.nan
    if np.any(v < 0.0):
        return np.nan
    b = int(buckets)
    if b < 3:
        raise ValueError("intraday_activity_duration_curvature requires buckets >= 3")
    total = float(v.sum())
    if total <= _EPS or v.size < b:
        return np.nan
    cum = np.cumsum(v)
    # completion bar index (original session position) for each fraction k/B
    D = np.full(b, np.nan)
    for k in range(1, b + 1):
        target = np.where(b * total != 0, float(k) / b * total, np.nan)
        pos = np.flatnonzero(cum >= target)
        if pos.size == 0:
            return np.nan
        D[k - 1] = float(pos[0])
    if np.any(np.isnan(D)):
        return np.nan
    d2 = D[2:] - 2.0 * D[1:-1] + D[:-2]
    mad = float(np.median(np.abs(D - np.median(D))))
    if mad <= _EPS:
        return np.nan
    return float(np.mean(d2) / mad)


@register_operator(
    name="intraday_activity_duration_curvature",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_activity_duration_curvature",
    source="intraday_activity_duration",
    status="implemented",
)
class IntradayActivityDurationCurvature(SeriesOperator):
    """日内 activity 等量分桶后的 trading-session 时长曲线标准化二阶曲率。"""

    metadata = _metadata(
        "intraday_activity_duration_curvature",
        "分钟 activity 等量分桶时长的二阶曲率（mean Δ²D / MAD(D)，无量纲）。",
        ["activity", "buckets", "calendar"],
    )

    def _calculate_series(
        self,
        activity: pd.DataFrame,
        buckets: int = 10,
        calendar: Any = None,
        **_: Any,
    ) -> pd.DataFrame:
        global _WARNED_NO_CALENDAR
        act = _session_local_frame(activity.copy())
        b = int(buckets)
        if b < 3:
            raise ValueError("intraday_activity_duration_curvature requires buckets >= 3")
        # P0-10: the official SessionGrid comes from the explicit exchange
        # calendar, NEVER from a modal of the observed data.  Without a calendar
        # the grid is UNKNOWN -> every day fails closed (NaN).
        if calendar is None:
            if not _WARNED_NO_CALENDAR:
                _WARNED_NO_CALENDAR = True
                warnings.warn(
                    "intraday_activity_duration_curvature requires an explicit "
                    "`calendar` (runtime.session_calendar.SessionCalendar).  Without "
                    "one the official SessionGrid is UNKNOWN and every day fails "
                    "closed (NaN); the grid is never inferred from observed data "
                    "(P0-10).",
                    RuntimeWarning,
                    stacklevel=2,
                )
            official = ()
        else:
            official = _official_session_grid(calendar)
        out: dict[str, pd.Series] = {}
        for inst in act.columns:
            col = act[inst]
            arr = col.to_numpy(dtype=float)
            idx = col.index
            per_day: dict[pd.Timestamp, float] = {}
            for day, group_idx in pd.Series(np.arange(len(idx)), index=idx).groupby(idx.normalize()):
                positions = np.asarray(group_idx, dtype=int)
                if not official:
                    # No official grid to validate completeness against -> fail closed.
                    per_day[day] = np.nan
                    continue
                # R11 #184/#185: reindex the day onto the OFFICIAL session grid so a
                # missing whole-minute row (invisible to a NaN gate on the raw array)
                # becomes NaN and a partial / truncated session fails closed.
                full_idx = pd.DatetimeIndex(
                    [pd.Timestamp(day) + pd.Timedelta(minutes=m) for m in official]
                )
                day_series = col.iloc[positions]
                v = day_series.reindex(full_idx).to_numpy(dtype=float)
                per_day[day] = _day_curvature(v, b)
            out[inst] = pd.Series(per_day, dtype=float)
        if not out:
            return pd.DataFrame(dtype=float)
        return pd.DataFrame(out).sort_index()


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({"intraday_activity_duration_curvature"})
    from cleaned_operators.rolling_pack import register_polars_udf

    register_polars_udf("intraday_activity_duration_curvature")


_register_surface()
