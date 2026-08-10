# -*- coding: utf-8 -*-
"""Minute → daily intraday aggregation operators (microstructure family).

Each operator accepts minute-frequency OHLCV panels (row index is a minute
timestamp, columns are instruments) and emits one daily value per instrument
(session-close available), i.e. every kernel is a shape-changing minute → daily
cross-sectional factor after close.  Operators never emit a row per minute.

* All kernels are causal: they only use the day's own minute data plus that
  day's daily limit/close inputs, so a day's value is known at session close.

* These operators require a certified minute dataset.  Until one is available
  the surface stays SOURCE_BLOCKED; the kernels below are nevertheless the
  single place the numerical semantics are pinned.

R26-013..039 (physical clock): every operator canonicalizes its minute panel
through :class:`runtime.session_panel.SessionPanel` BEFORE any statistic is
computed:

* bar frequency comes from the DECLARED ``bar_freq`` (default ``"1min"``) +
  ``market`` — NEVER from a modal of observed minute deltas (R26-014).  A
  dataset that systematically drops every other bar therefore fails the
  official-grid coverage gate instead of re-certifying itself at 2-min
  resolution.
* physically absent timestamps become explicit missing slots via reindex onto
  the official session grid (R26-017..019).
* log-returns only exist between adjacent official slots; a gap is NaN, never a
  single fused bar (R26-020).
* coverage / share denominators count unique valid official slots, never
  observed row counts; a duplicate official slot is a DQ failure (R26-021).
* segment / lunch-gap endpoints are EXACT official minutes by default
  (EndpointPolicy.EXACT, R26-025..027); ``endpoint_policy="recent_valid"`` is
  an explicit opt-in.
* ``intra_high_time`` / ``intra_low_time`` use the official slot ordinal, never
  the compressed observed-row index (R26-030).
* limit masks stay tri-state (unknown limit / missing bar -> NaN, never a
  fabricated False) and the up/down touch side needs only its own OHLC side
  (R26-031..034).
* incomplete full-day denominators make share operators NaN (R26-035).
* the session trade-date is computed in the session timezone, so UTC-stored
  minute data is split on the correct session day (R26-036..038).
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, ParamSpec, SeriesOperator, register_operator
from runtime.session_calendar import SessionCalendar
from runtime.session_panel import (
    SessionPanel,
    build_session_panel,
    default_ashare_calendar,
    session_trade_dates,
)

_EPS = 1e-12
_MORNING = (571, 690)   # 09:31 .. 11:30 (A-share 240-bar session, matches DataAccess)
_AFTERNOON = (781, 900)  # 13:01 .. 15:00
_SEGMENT_RANGES = {"morning": _MORNING, "afternoon": _AFTERNOON}
_SEGMENT_END_MOD = {"morning": 690, "afternoon": 900}   # exact official close minute
_SEGMENT_START_MOD = {"morning": 571, "afternoon": 781}  # exact official first-bar minute

_DEFAULT_SESSION_TZ = "Asia/Shanghai"
# R26-038: documented storage convention for the A-share minute pipeline (UTC).
_DEFAULT_SOURCE_TZ = {"ashare": "UTC", "cn": "UTC", "a": "UTC"}
_COVERAGE_FLOOR = 0.9  # official-slot coverage required for RV / share / limits


def _metadata(name: str, description: str, params: list[str], *, unit: str) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="intraday_microstructure",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "intraday", "daily_agg", "minute", "pit_safe", "causal",
            "typed_v2", "source_blocked",
            f"signature:{','.join(params)}->series", f"unit:{unit}", "cost:1",
        ],
        # R11 P0-04/05: minute -> daily shape-changing contract + EOD-only
        # availability, machine-readable (not docstring text).
        input_grain="minute",
        output_grain="daily",
        available_at="session_close",
        same_session_usable=False,
    )


def _as_panel(x: Any) -> pd.DataFrame:
    if isinstance(x, pd.Series):
        return x.to_frame(getattr(x, "name", None) or "value")
    return x


def _minute_of_day(times: np.ndarray) -> np.ndarray:
    """Timestamp array -> minute-of-day (integer)."""
    seconds = times.astype("datetime64[s]").astype("int64") % 86400
    return seconds // 60


def _declared_calendar(market: str | None, bar_freq: str | None) -> SessionCalendar:
    """Declared A-share SessionCalendar from ``market`` + ``bar_freq``.

    R26-013/014: bar frequency is a DECLARED contract property (never a modal
    of observed deltas).  ``bar_freq`` defaults to the documented 1-minute
    session; anything else is an explicit declaration.
    """
    freq = str(bar_freq or "1min")
    market_norm = (market or "ashare").lower().replace("_", "").replace(" ", "")
    if market_norm in {"ashare", "cn", "a", "china"}:
        return default_ashare_calendar(bar_freq=freq)
    return SessionCalendar(market=market_norm.upper(), bar_freq=freq, timestamp_convention="bar_end")


def _session_local_frame(
    frame: pd.DataFrame,
    *,
    session_tz: str | None,
    source_timezone: str | None,
    market: str | None,
) -> pd.DataFrame:
    """Convert a tz-aware minute panel to session wall-clock (naive).

    R26-036..038: A-share minute data is stored in UTC; the session segments are
    defined in Asia/Shanghai wall-clock.  A tz-aware index with no declared /
    documented ``source_timezone`` fails closed (raise) — never silently
    assumed session-local.
    """
    frame = _as_panel(frame)
    idx = frame.index
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        src_tz = source_timezone or _DEFAULT_SOURCE_TZ.get((market or "ashare").lower())
        if src_tz is None:
            raise ValueError(
                "tz-aware minute data requires a declared `source_timezone`; "
                "unknown storage timezone fails closed (R26-038)"
            )
        frame = frame.tz_convert(session_tz or _DEFAULT_SESSION_TZ)
        frame.index = frame.index.tz_localize(None)
    return frame


def _daily_agg(
    frame: pd.DataFrame,
    fn: Callable[[SessionPanel], float],
    *,
    market: str | None = None,
    bar_freq: str | None = None,
    session_tz: str | None = None,
    source_timezone: str | None = None,
) -> pd.DataFrame:
    """Apply fn(panel) per (instrument, session-local trade date).

    The panel is grid-aligned (SessionPanel); absent bars are explicit missing
    slots.  The trade-date is the session-local calendar date (R26-036).
    """
    frame = _session_local_frame(
        _as_panel(frame), session_tz=session_tz, source_timezone=source_timezone, market=market
    )
    cal = _declared_calendar(market, bar_freq)
    tz = session_tz or _DEFAULT_SESSION_TZ
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
                panel = build_session_panel(
                    times, vals, cal,
                    market=str(market or "ashare"),
                    session_timezone=tz,
                    source_timezone=None,  # already session-local
                    trade_date=pd.Timestamp(day),
                )
                per_day[day] = float(fn(panel))
            except (ValueError, ZeroDivisionError, OverflowError):
                per_day[day] = np.nan
        out[inst] = pd.Series(per_day, dtype=float)
    if not out:
        return pd.DataFrame(dtype=float)
    return pd.DataFrame(out).sort_index()


def _pair_agg(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    fn: Callable[[SessionPanel, SessionPanel], float],
    *,
    market: str | None = None,
    bar_freq: str | None = None,
    session_tz: str | None = None,
    source_timezone: str | None = None,
) -> pd.DataFrame:
    """Tuple-complete pair aggregation (R26-022..024).

    Both panels are reindexed onto the same official grid; a slot is usable only
    when BOTH carry a valid bar.  A bar with exactly one finite member is
    data-invalid and never acts as a single-series bar.
    """
    a = _session_local_frame(
        _as_panel(frame_a), session_tz=session_tz, source_timezone=source_timezone, market=market
    )
    b = _session_local_frame(
        _as_panel(frame_b), session_tz=session_tz, source_timezone=source_timezone, market=market
    )
    cal = _declared_calendar(market, bar_freq)
    tz = session_tz or _DEFAULT_SESSION_TZ
    out: dict[str, pd.Series] = {}
    for inst in a.columns:
        per_day: dict[pd.Timestamp, float] = {}
        for day, ga in a[inst].groupby(a[inst].index.normalize()):
            gb = b[inst]
            gb = gb[gb.index.normalize() == day]
            va = np.asarray(ga, dtype=float)
            vb = np.asarray(gb, dtype=float)
            ta = np.asarray(ga.index, dtype="datetime64[ns]")
            tb = np.asarray(gb.index, dtype="datetime64[ns]")
            if not np.any(np.isfinite(va)) or not np.any(np.isfinite(vb)):
                per_day[day] = np.nan
                continue
            try:
                pa = build_session_panel(ta, va, cal, market=str(market or "ashare"),
                                         session_timezone=tz, source_timezone=None, trade_date=pd.Timestamp(day))
                pb = build_session_panel(tb, vb, cal, market=str(market or "ashare"),
                                         session_timezone=tz, source_timezone=None, trade_date=pd.Timestamp(day))
                per_day[day] = float(fn(pa, pb))
            except (ValueError, ZeroDivisionError, OverflowError):
                per_day[day] = np.nan
        out[inst] = pd.Series(per_day, dtype=float)
    if not out:
        return pd.DataFrame(dtype=float)
    return pd.DataFrame(out).sort_index()


def _triple_agg(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    frame_c: pd.DataFrame,
    fn: Callable[[SessionPanel, SessionPanel, SessionPanel], float],
    *,
    market: str | None = None,
    bar_freq: str | None = None,
    session_tz: str | None = None,
    source_timezone: str | None = None,
) -> pd.DataFrame:
    """Tuple-complete three-input aggregation (R26-022..024).

    A required slot is usable only when ALL required inputs are valid — the old
    helper dropped on ``a`` alone and let ``b``/``c`` NaNs silently change the
    cohort / denominator (R26-154).
    """
    a = _session_local_frame(
        _as_panel(frame_a), session_tz=session_tz, source_timezone=source_timezone, market=market
    )
    b = _session_local_frame(
        _as_panel(frame_b), session_tz=session_tz, source_timezone=source_timezone, market=market
    )
    c = _session_local_frame(
        _as_panel(frame_c), session_tz=session_tz, source_timezone=source_timezone, market=market
    )
    cal = _declared_calendar(market, bar_freq)
    tz = session_tz or _DEFAULT_SESSION_TZ
    out: dict[str, pd.Series] = {}
    for inst in a.columns:
        per_day: dict[pd.Timestamp, float] = {}
        for day, ga in a[inst].groupby(a[inst].index.normalize()):
            gb = b[inst]; gb = gb[gb.index.normalize() == day]
            gc = c[inst]; gc = gc[gc.index.normalize() == day]
            va = np.asarray(ga, dtype=float)
            vb = np.asarray(gb, dtype=float)
            vc = np.asarray(gc, dtype=float)
            ta = np.asarray(ga.index, dtype="datetime64[ns]")
            tb = np.asarray(gb.index, dtype="datetime64[ns]")
            tc = np.asarray(gc.index, dtype="datetime64[ns]")
            if not (np.any(np.isfinite(va)) and np.any(np.isfinite(vb)) and np.any(np.isfinite(vc))):
                per_day[day] = np.nan
                continue
            try:
                pa = build_session_panel(ta, va, cal, market=str(market or "ashare"),
                                         session_timezone=tz, source_timezone=None, trade_date=pd.Timestamp(day))
                pb = build_session_panel(tb, vb, cal, market=str(market or "ashare"),
                                         session_timezone=tz, source_timezone=None, trade_date=pd.Timestamp(day))
                pc = build_session_panel(tc, vc, cal, market=str(market or "ashare"),
                                         session_timezone=tz, source_timezone=None, trade_date=pd.Timestamp(day))
                per_day[day] = float(fn(pa, pb, pc))
            except (ValueError, ZeroDivisionError, OverflowError):
                per_day[day] = np.nan
        out[inst] = pd.Series(per_day, dtype=float)
    if not out:
        return pd.DataFrame(dtype=float)
    return pd.DataFrame(out).sort_index()


def _seg_mask(minute_of_day: np.ndarray, segment: str) -> np.ndarray:
    """Segment mask.  Accepts either minute-of-day (int) or timestamp arrays."""
    arr = np.asarray(minute_of_day)
    if arr.dtype.kind in "mM":
        arr = _minute_of_day(arr)
    lo, hi = _SEGMENT_RANGES[str(segment)]
    return (arr >= lo) & (arr <= hi)


def _coverage_ok(panel: SessionPanel, floor: float = _COVERAGE_FLOOR) -> bool:
    """Coverage gate (R26-021): unique valid official slots / expected slots."""
    return panel.coverage() >= floor


# ---------------------------------------------------------------------------
# § Segment aggregation
# ---------------------------------------------------------------------------

def _seg_return(panel: SessionPanel, segment: str, endpoint_policy: str = "exact") -> float:
    """Segment return (close at end / close at start - 1).

    R26-025/026: EndpointPolicy.EXACT by default — the official segment start
    and end minutes are required.  ``endpoint_policy="recent_valid"`` is an
    explicit opt-in that uses the last/first finite close inside the segment.
    """
    if endpoint_policy == "exact":
        lo = _SEGMENT_START_MOD[str(segment)]
        hi = _SEGMENT_END_MOD[str(segment)]
        start = panel.value_at_minute(lo)
        end = panel.value_at_minute(hi)
        if not np.isfinite(start) or not np.isfinite(end) or start <= _EPS:
            return np.nan
        return float(end / start - 1.0)
    # explicit recent-valid policy: last/first finite within the segment
    mods = panel.grid.expected_minutes
    seg = _seg_mask(mods, segment)
    vals = np.where(seg, panel.values, np.nan)
    valid = panel.is_valid_bar & np.isfinite(vals)
    idx = np.where(valid)[0]
    if len(idx) < 2 or vals[idx[0]] <= _EPS:
        return np.nan
    return float(vals[idx[-1]] / vals[idx[0]] - 1.0)


@register_operator(
    name="intra_segment_return",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_segment_return",
    source="intraday_agg",
    backend="pandas_numpy",
    status="experimental",
)
class IntraSegmentReturn(SeriesOperator):
    metadata = _metadata(
        "intra_segment_return",
        "指定时段（morning/afternoon）收盘/开盘收益 - 1（默认官方端点精确）。",
        ["close", "segment", "session_tz", "endpoint_policy"],
        unit="return",
    )
    metadata.param_specs = {
        "endpoint_policy": ParamSpec(dtype=str, choices=("exact", "recent_valid"), searchable=False),
    }

    def _calculate_series(self, close, segment="morning", session_tz=None,
                          endpoint_policy="exact", **_):
        if segment not in _SEGMENT_RANGES:
            raise ValueError(f"segment must be in {{morning, afternoon}}, got {segment!r}")
        return _daily_agg(
            close,
            lambda panel: _seg_return(panel, segment, endpoint_policy),
            session_tz=session_tz,
        )


def _seg_volume_share(panel: SessionPanel, segment: str) -> float:
    """Segment volume / full-day volume.

    R26-035: an incomplete full-day denominator must not produce a normal
    share.  Below 90% official-slot coverage the denominator is unknown -> NaN.
    """
    if not _coverage_ok(panel):
        return np.nan
    mods = panel.grid.expected_minutes
    seg = _seg_mask(mods, segment)
    valid = panel.is_valid_bar & np.isfinite(panel.values)
    total = float(panel.values[valid].sum())
    seg_total = float(panel.values[valid & seg].sum())
    if total <= _EPS:
        return np.nan
    return seg_total / total


def _make_seg_share(unit: str):
    def _calculate_series(self, value, segment="morning", session_tz=None, **_):
        if segment not in _SEGMENT_RANGES:
            raise ValueError(f"segment must be in {{morning, afternoon}}, got {segment!r}")
        return _daily_agg(
            value,
            lambda panel: _seg_volume_share(panel, segment),
            session_tz=session_tz,
        )
    return _calculate_series


@register_operator(
    name="intra_segment_volume_share",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_segment_volume_share",
    source="intraday_agg",
    backend="pandas_numpy",
    status="experimental",
)
class IntraSegmentVolumeShare(SeriesOperator):
    metadata = _metadata(
        "intra_segment_volume_share",
        "指定时段成交量占全天比例（完整日 denominator 才有效）。",
        ["volume", "segment", "session_tz"], unit="ratio",
    )
    metadata.param_specs = {}

    _calculate_series = _make_seg_share("volume")


@register_operator(
    name="intra_segment_amount_share",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_segment_amount_share",
    source="intraday_agg",
    backend="pandas_numpy",
    status="experimental",
)
class IntraSegmentAmountShare(SeriesOperator):
    metadata = _metadata(
        "intra_segment_amount_share",
        "指定时段成交额占全天比例（完整日 denominator 才有效）。",
        ["amount", "segment", "session_tz"], unit="ratio",
    )
    metadata.param_specs = {}

    _calculate_series = _make_seg_share("amount")


def _seg_vwap_deviation(pc: SessionPanel, pa: SessionPanel, pv: SessionPanel, segment: str) -> float:
    """Segment-end close vs cumulative segment VWAP.

    Exact endpoint for the close (R26-025); the VWAP is the tuple-complete
    amount/volume over the segment.  An incomplete segment fails closed.
    """
    if not _coverage_ok(pc):
        return np.nan
    end = pc.value_at_minute(_SEGMENT_END_MOD[str(segment)])
    if not np.isfinite(end):
        return np.nan
    mods = pc.grid.expected_minutes
    seg = _seg_mask(mods, segment)
    valid = pc.is_valid_bar & pa.is_valid_bar & pv.is_valid_bar & seg
    a = np.nansum(np.where(valid, pa.values, np.nan))
    v = np.nansum(np.where(valid, pv.values, np.nan))
    if v <= _EPS or not np.isfinite(end):
        return np.nan
    return float(end / (a / v) - 1.0)


@register_operator(
    name="intra_segment_vwap_deviation",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_segment_vwap_deviation",
    source="intraday_agg",
    backend="pandas_numpy",
    status="experimental",
)
class IntraSegmentVwapDeviation(SeriesOperator):
    metadata = _metadata(
        "intra_segment_vwap_deviation",
        "指定时段末价相对该时段累计 VWAP 的偏差。",
        ["close", "amount", "volume", "segment", "session_tz"],
        unit="ratio",
    )
    metadata.param_specs = {}


    def _calculate_series(self, close, amount, volume, segment="morning", session_tz=None, **_):
        if segment not in _SEGMENT_RANGES:
            raise ValueError(f"segment must be in {{morning, afternoon}}, got {segment!r}")
        return _triple_agg(
            close, amount, volume,
            lambda pa, pb, pc: _seg_vwap_deviation(pa, pb, pc, segment),
            session_tz=session_tz,
        )


def _seg_realized_vol(panel: SessionPanel, segment: str) -> float:
    if not _coverage_ok(panel):
        return np.nan
    mods = panel.grid.expected_minutes
    seg = _seg_mask(mods, segment)
    r = np.where(seg, panel.log_returns(), np.nan)
    with np.errstate(invalid="ignore"):
        return float(math.sqrt(float(np.nansum(r * r))))


@register_operator(
    name="intra_segment_realized_vol",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_segment_realized_vol",
    source="intraday_agg",
    backend="pandas_numpy",
    status="experimental",
)
class IntraSegmentRealizedVol(SeriesOperator):
    metadata = _metadata(
        "intra_segment_realized_vol",
        "指定时段已实现波动率 sqrt(sum(r_t^2))。",
        ["close", "segment", "session_tz"], unit="volatility",
    )
    metadata.param_specs = {}


    def _calculate_series(self, close, segment="morning", session_tz=None, **_):
        if segment not in _SEGMENT_RANGES:
            raise ValueError(f"segment must be in {{morning, afternoon}}, got {segment!r}")
        return _daily_agg(
            close, lambda panel: _seg_realized_vol(panel, segment),
            session_tz=session_tz,
        )


# ---------------------------------------------------------------------------
# § Realized variance / semivariance / bipower / jump
# ---------------------------------------------------------------------------

def _rv(panel: SessionPanel) -> float:
    if not _coverage_ok(panel):
        return np.nan
    r = panel.log_returns()
    with np.errstate(invalid="ignore"):
        return float(np.nansum(r * r))


@register_operator(
    name="intra_realized_variance",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_realized_variance",
    source="intraday_agg",
    backend="pandas_numpy",
    status="experimental",
)
class IntraRealizedVariance(SeriesOperator):
    metadata = _metadata(
        "intra_realized_variance",
        "日内已实现方差 sum(r_t^2)（官方网格逐槽）。",
        ["close"], unit="variance",
    )
    metadata.param_specs = {}


    def _calculate_series(self, close, **_):
        return _daily_agg(close, _rv)


def _semivariance(panel: SessionPanel, side: str) -> float:
    if not _coverage_ok(panel):
        return np.nan
    r = panel.log_returns()
    if side == "down":
        r = np.where(r < 0, r, 0.0)
    elif side == "up":
        r = np.where(r > 0, r, 0.0)
    else:
        raise ValueError("intra_realized_semivariance requires side in {'up', 'down'}")
    with np.errstate(invalid="ignore"):
        return float(np.nansum(r * r))


@register_operator(
    name="intra_realized_semivariance",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_realized_semivariance",
    source="intraday_agg",
    backend="pandas_numpy",
    status="experimental",
)
class IntraRealizedSemivariance(SeriesOperator):
    metadata = _metadata(
        "intra_realized_semivariance",
        "日内上/下半方差 sum(r_t^2 * 1(sign))。",
        ["close", "side"], unit="variance",
    )
    metadata.param_specs = {}


    def _calculate_series(self, close, side="down", **_):
        if side not in ("up", "down"):
            raise ValueError("intra_realized_semivariance requires side in {'up', 'down'}")
        return _daily_agg(close, lambda panel: _semivariance(panel, side))


def _bipower(panel: SessionPanel) -> float:
    if not _coverage_ok(panel):
        return np.nan
    r = panel.log_returns()
    # Bipower variation uses *real adjacent* grid returns: a missing slot is an
    # explicit NaN, so no pair spans a gap (R26-020).
    with np.errstate(invalid="ignore"):
        return float((math.pi / 2.0) * np.nansum(np.abs(r[1:]) * np.abs(r[:-1])))


@register_operator(
    name="intra_bipower_variation",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_bipower_variation",
    source="intraday_agg",
    backend="pandas_numpy",
    status="experimental",
)
class IntraBipowerVariation(SeriesOperator):
    metadata = _metadata(
        "intra_bipower_variation",
        "日内双幂变差 (pi/2)*sum(|r_t||r_{t-1}|)。",
        ["close"], unit="variance",
    )
    metadata.param_specs = {}


    def _calculate_series(self, close, **_):
        return _daily_agg(close, _bipower)


def _jump_ratio(panel: SessionPanel) -> float:
    rv = _rv(panel)
    bv = _bipower(panel)
    if not np.isfinite(rv) or not np.isfinite(bv) or rv <= _EPS:
        return np.nan
    return float(max(rv - bv, 0.0) / rv)


@register_operator(
    name="intra_jump_ratio",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_jump_ratio",
    source="intraday_agg",
    backend="pandas_numpy",
    status="experimental",
)
class IntraJumpRatio(SeriesOperator):
    metadata = _metadata(
        "intra_jump_ratio",
        "日内跳跃占比 max(RV-BV,0)/RV。",
        ["close"], unit="ratio",
    )
    metadata.param_specs = {}


    def _calculate_series(self, close, **_):
        return _daily_agg(close, _jump_ratio)


# ---------------------------------------------------------------------------
# § Intraday price path
# ---------------------------------------------------------------------------

def _path_efficiency(panel: SessionPanel) -> float:
    """|net displacement| / arc length over the longest trailing contiguous
    complete run (R26-028/029).

    A missing slot BREAKS the path: interior gaps are never bridged by a
    straight segment between gap endpoints (which would shorten the path and
    inflate efficiency).
    """
    run = panel.contiguous_complete_run()
    if run.sum() < 2:
        return np.nan
    pts = panel.values[run]
    length = float(np.sum(np.abs(np.diff(pts))))
    if length <= _EPS:
        return 0.0
    return float(abs(pts[-1] - pts[0]) / length)


@register_operator(
    name="intra_path_efficiency",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_path_efficiency",
    source="intraday_agg",
    backend="pandas_numpy",
    status="experimental",
)
class IntraPathEfficiency(SeriesOperator):
    metadata = _metadata(
        "intra_path_efficiency",
        "日内路径效率 |净位移|/路径长度（缺口断开路径）。",
        ["close"], unit="ratio",
    )
    metadata.param_specs = {}


    def _calculate_series(self, close, **_):
        return _daily_agg(close, _path_efficiency)


def _position_of(panel: SessionPanel, *, low: bool) -> float:
    """Position of the extreme value as official-slot ordinal / session slot count.

    R26-030: the slot_id (official minute ordinal) is used, never the compressed
    observed-row index — missing/duplicate bars must not move "when the high
    occurred".
    """
    valid = panel.is_valid_bar & np.isfinite(panel.values)
    if not np.any(valid):
        return np.nan
    vals = np.where(valid, panel.values, np.nan)
    target = int(np.nanargmin(vals)) if low else int(np.nanargmax(vals))
    return float(panel.slot_id[target]) / float(panel.n_slots)


@register_operator(
    name="intra_high_time",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_high_time",
    source="intraday_agg",
    backend="pandas_numpy",
    status="experimental",
)
class IntraHighTime(SeriesOperator):
    metadata = _metadata(
        "intra_high_time",
        "全天最高价首次出现位置（官方 slot ordinal / 会话 slot 数）。",
        ["high"], unit="position",
    )
    metadata.param_specs = {}


    def _calculate_series(self, high, **_):
        return _daily_agg(high, lambda panel: _position_of(panel, low=False))


@register_operator(
    name="intra_low_time",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_low_time",
    source="intraday_agg",
    backend="pandas_numpy",
    status="experimental",
)
class IntraLowTime(SeriesOperator):
    metadata = _metadata(
        "intra_low_time",
        "全天最低价首次出现位置（官方 slot ordinal / 会话 slot 数）。",
        ["low"], unit="position",
    )
    metadata.param_specs = {}


    def _calculate_series(self, low, **_):
        return _daily_agg(low, lambda panel: _position_of(panel, low=True))


def _vwap_above_ratio(pc: SessionPanel, pa: SessionPanel, pv: SessionPanel) -> float:
    valid = pc.is_valid_bar & pa.is_valid_bar & pv.is_valid_bar & (pv.values > 0)
    a = np.where(valid, pa.values, 0.0)
    v = np.where(valid, pv.values, 0.0)
    tv = float(v.sum())
    if tv <= _EPS:
        return np.nan
    day_vwap = float(a.sum()) / tv
    ok = valid & np.isfinite(pc.values)
    if ok.sum() == 0:
        return np.nan
    return float(np.mean(pc.values[ok] > day_vwap))


@register_operator(
    name="intra_vwap_above_ratio",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_vwap_above_ratio",
    source="intraday_agg",
    backend="pandas_numpy",
    status="experimental",
)
class IntraVwapAboveRatio(SeriesOperator):
    metadata = _metadata(
        "intra_vwap_above_ratio",
        "收盘价高于当日 VWAP 的分钟占比。",
        ["close", "amount", "volume"], unit="ratio",
    )
    metadata.param_specs = {}


    def _calculate_series(self, close, amount, volume, **_):
        return _triple_agg(close, amount, volume, _vwap_above_ratio)


def _vwap_cross_count(pc: SessionPanel, pa: SessionPanel, pv: SessionPanel) -> float:
    # Tuple-complete cumulative VWAP over the aligned grid (R26-022..024).
    valid = pc.is_valid_bar & pa.is_valid_bar & pv.is_valid_bar
    vol = np.where(valid, pv.values, np.nan)
    amt = np.where(valid, pa.values, np.nan)
    cum_v = np.nancumsum(vol)
    cum_a = np.nancumsum(amt)
    with np.errstate(divide="ignore", invalid="ignore"):
        cum_vwap = np.where(cum_v > _EPS, cum_a / cum_v, np.nan)
    ok = valid & np.isfinite(pc.values) & np.isfinite(cum_vwap)
    if ok.sum() < 2:
        return np.nan
    c = np.where(ok, pc.values, np.nan)
    sign = np.sign(c - cum_vwap)
    sign = sign[ok]
    if len(sign) < 2:
        return 0.0
    return float(np.sum(sign[1:] != sign[:-1]))


@register_operator(
    name="intra_vwap_cross_count",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_vwap_cross_count",
    source="intraday_agg",
    backend="pandas_numpy",
    status="experimental",
)
class IntraVwapCrossCount(SeriesOperator):
    metadata = _metadata(
        "intra_vwap_cross_count",
        "收盘价相对累计 VWAP 的方向变化次数。",
        ["close", "amount", "volume"], unit="count",
    )
    metadata.param_specs = {}


    def _calculate_series(self, close, amount, volume, **_):
        return _triple_agg(close, amount, volume, _vwap_cross_count)


# ---------------------------------------------------------------------------
# § Intraday volume / amount distribution
# ---------------------------------------------------------------------------

def _concentration(panel: SessionPanel) -> float:
    v = np.where(panel.is_valid_bar, panel.values, np.nan)
    if np.any(v < 0.0):
        return np.nan
    finite = v[np.isfinite(v)]
    total = float(finite.sum())
    if total <= _EPS:
        return np.nan
    w = finite / total
    return float(np.sum(w * w))


@register_operator(
    name="intra_concentration",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_concentration",
    source="intraday_agg",
    backend="pandas_numpy",
    status="experimental",
)
class IntraConcentration(SeriesOperator):
    metadata = _metadata(
        "intra_concentration",
        "日内成交集中度 sum((value/sum)^2)。",
        ["value"], unit="hhi",
    )
    metadata.param_specs = {}


    def _calculate_series(self, value, **_):
        return _daily_agg(value, _concentration)


def _entropy(panel: SessionPanel, normalize: bool = True) -> float:
    v = np.where(panel.is_valid_bar, panel.values, np.nan)
    if np.any(v < 0.0):
        return np.nan
    finite = v[np.isfinite(v)]
    total = float(finite.sum())
    n = int(np.count_nonzero(finite))
    if total <= _EPS or n < 2:
        return np.nan
    w = finite / total
    w = w[w > 0]
    entropy = -float(np.sum(w * np.log(w)))
    if normalize:
        return entropy / math.log(n)
    return entropy


@register_operator(
    name="intra_entropy",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_entropy",
    source="intraday_agg",
    backend="pandas_numpy",
    status="experimental",
)
class IntraEntropy(SeriesOperator):
    metadata = _metadata(
        "intra_entropy",
        "日内成交分布熵（归一化）。",
        ["value", "normalize"], unit="entropy",
    )
    metadata.param_specs = {
        "normalize": ParamSpec(dtype=bool, searchable=False),
    }

    def _calculate_series(self, value, normalize=True, **_):
        return _daily_agg(value, lambda panel: _entropy(panel, bool(normalize)))


def _signed_imbalance_proxy(pc: SessionPanel, pv: SessionPanel) -> float:
    r = pc.log_returns()
    valid = pc.is_valid_bar & pv.is_valid_bar
    value = np.where(valid, pv.values, 0.0)
    total = float(value.sum())
    if total <= _EPS:
        return np.nan
    signed = np.where(np.isfinite(r) & valid, np.sign(r), 0.0) * value
    return float(signed.sum() / total)


@register_operator(
    name="intra_signed_imbalance_proxy",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_signed_imbalance_proxy",
    source="intraday_agg",
    backend="pandas_numpy",
    status="experimental",
)
class IntraSignedImbalanceProxy(SeriesOperator):
    metadata = _metadata(
        "intra_signed_imbalance_proxy",
        "基于分钟价格方向的成交不平衡代理 sum(sign(r)*value)/sum(value)。",
        ["close", "value"], unit="ratio",
    )
    metadata.param_specs = {}


    def _calculate_series(self, close, value, **_):
        return _pair_agg(close, value, _signed_imbalance_proxy)


def _return_activity_corr(pc: SessionPanel, pa: SessionPanel, absolute_return: bool) -> float:
    r = pc.log_returns()
    if absolute_return:
        r = np.abs(r)
    valid = pc.is_valid_bar & pa.is_valid_bar & np.isfinite(pa.values)
    r, a = r[valid], pa.values[valid]
    if len(r) < 2 or np.std(r) <= _EPS or np.std(a) <= _EPS:
        return np.nan
    return float(np.corrcoef(r, a)[0, 1])


@register_operator(
    name="intra_return_activity_corr",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_return_activity_corr",
    source="intraday_agg",
    backend="pandas_numpy",
    status="experimental",
)
class IntraReturnActivityCorr(SeriesOperator):
    metadata = _metadata(
        "intra_return_activity_corr",
        "分钟收益与成交活动相关性（可选绝对值）。",
        ["close", "activity", "absolute_return"], unit="corr",
    )
    metadata.param_specs = {
        "absolute_return": ParamSpec(dtype=bool, searchable=False),
    }

    def _calculate_series(self, close, activity, absolute_return=False, **_):
        return _pair_agg(
            close, activity,
            lambda pa, pb: _return_activity_corr(pa, pb, bool(absolute_return)),
        )


# ---------------------------------------------------------------------------
# § Intraday liquidity
# ---------------------------------------------------------------------------

def _intra_amihud(pc: SessionPanel, pa: SessionPanel, scale: float) -> float:
    r = np.abs(pc.log_returns())
    valid = pc.is_valid_bar & pa.is_valid_bar
    amount = np.where(valid, pa.values, np.nan)
    denom = np.maximum(amount, _EPS)
    ratio = np.where(np.isfinite(r) & valid, r / denom, np.nan)
    ratio = ratio[np.isfinite(ratio)]
    if len(ratio) == 0:
        return np.nan
    return float(np.mean(ratio) * float(scale))


@register_operator(
    name="intra_amihud",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_amihud",
    source="intraday_agg",
    backend="pandas_numpy",
    status="experimental",
)
class IntraAmihud(SeriesOperator):
    metadata = _metadata(
        "intra_amihud",
        "日内 Amihud 非流动性 mean(|r|/max(amount,eps))*scale。",
        ["close", "amount", "scale"], unit="illiquidity",
    )
    # R6-189: ``scale`` is a pure output-unit rescaling — never a search param.
    metadata.param_specs = {
        "scale": ParamSpec(dtype=float, searchable=False),
    }

    def _calculate_series(self, close, amount, scale=1e8, **_):
        return _pair_agg(
            close, amount, lambda pa, pb: _intra_amihud(pa, pb, float(scale)),
        )


def _kyle_lambda_proxy(pc: SessionPanel, pa: SessionPanel) -> float:
    r = pc.log_returns()
    valid = pc.is_valid_bar & pa.is_valid_bar
    amount = np.where(valid, pa.values, 0.0)
    total = float(amount.sum())
    if total <= _EPS:
        return np.nan
    signed_share = np.sign(r) * amount / total
    mask = valid & np.isfinite(r) & np.isfinite(signed_share)
    r, s = r[mask], signed_share[mask]
    if len(r) < 3 or np.std(s) <= _EPS:
        return np.nan
    beta = float(np.cov(r, s)[0, 1] / np.var(s))
    return beta


@register_operator(
    name="intra_kyle_lambda_proxy",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_kyle_lambda_proxy",
    source="intraday_agg",
    backend="pandas_numpy",
    status="experimental",
)
class IntraKyleLambdaProxy(SeriesOperator):
    metadata = _metadata(
        "intra_kyle_lambda_proxy",
        "日内 Kyle Lambda 代理（收益对方向性成交额占比回归）。",
        ["close", "amount"], unit="lambda",
    )
    metadata.param_specs = {}


    def _calculate_series(self, close, amount, **_):
        return _pair_agg(close, amount, _kyle_lambda_proxy)


def _extreme_bar_return(panel: SessionPanel, side: str) -> float:
    r = panel.log_returns()
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return np.nan
    return float(np.max(r)) if side == "max" else float(np.min(r))


@register_operator(
    name="intra_extreme_bar_return",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_extreme_bar_return",
    source="intraday_agg",
    backend="pandas_numpy",
    status="experimental",
)
class IntraExtremeBarReturn(SeriesOperator):
    metadata = _metadata(
        "intra_extreme_bar_return",
        "日内单分钟最大/最小收益。",
        ["close", "side"], unit="return",
    )
    metadata.param_specs = {}


    def _calculate_series(self, close, side="max", **_):
        if side not in ("max", "min"):
            raise ValueError("side must be 'max' or 'min'")
        return _daily_agg(close, lambda panel: _extreme_bar_return(panel, side))


def _lunch_gap_return(
    pc: SessionPanel,
    po: SessionPanel,
    morning_cutoff: str,
    afternoon_start: str,
    endpoint_policy: str = "exact",
) -> float:
    """Lunch gap: afternoon first-bar Open / morning last-bar Close - 1.

    R26-027: EXACT official endpoints by default — 11:30 close and 13:01 open
    are required.  A missing 11:30 must NOT be replaced by 11:29, nor 13:01 by
    13:02, unless ``endpoint_policy="recent_valid"`` is declared.
    """
    if endpoint_policy == "exact":
        mc = pc.value_at_minute(_SEGMENT_END_MOD["morning"])  # 11:30
        ao = po.value_at_minute(_SEGMENT_START_MOD["afternoon"])  # 13:01
        if not np.isfinite(mc) or not np.isfinite(ao) or mc <= _EPS:
            return np.nan
        return float(ao / mc - 1.0)
    # explicit recent-valid policy
    mcm = _minute_of_day(pc.grid.expected)  # grid minute-of-day
    am = _minute_of_day(po.grid.expected)
    morning_mask = mcm <= _minute_hm(morning_cutoff)
    afternoon_mask = am >= _minute_hm(afternoon_start)
    mc_vals = np.where(morning_mask & pc.is_valid_bar, pc.values, np.nan)
    ao_vals = np.where(afternoon_mask & po.is_valid_bar, po.values, np.nan)
    mc_vals = mc_vals[np.isfinite(mc_vals)]
    ao_vals = ao_vals[np.isfinite(ao_vals)]
    if len(mc_vals) == 0 or len(ao_vals) == 0 or mc_vals[-1] <= _EPS:
        return np.nan
    return float(ao_vals[0] / mc_vals[-1] - 1.0)


def _minute_hm(text: str) -> int:
    hh, mm = str(text).split(":")
    return int(hh) * 60 + int(mm)


@register_operator(
    name="intra_lunch_gap_return",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_lunch_gap_return",
    source="intraday_agg",
    backend="pandas_numpy",
    status="experimental",
)
class IntraLunchGapReturn(SeriesOperator):
    metadata = _metadata(
        "intra_lunch_gap_return",
        "午间跳空：下午首根 Open/上午末根 Close - 1（默认官方端点精确）。",
        ["close", "open", "morning_cutoff", "afternoon_start", "session_tz", "endpoint_policy"],
        unit="return",
    )
    metadata.param_specs = {  # config knobs, not search parameters
        "morning_cutoff": ParamSpec(dtype=str, searchable=False),
        "afternoon_start": ParamSpec(dtype=str, searchable=False),
        "endpoint_policy": ParamSpec(dtype=str, choices=("exact", "recent_valid"), searchable=False),
    }

    def _calculate_series(self, close, open_px, morning_cutoff="11:30", afternoon_start="13:01",
                          session_tz=None, endpoint_policy="exact", **_):
        return _pair_agg(
            close, open_px,
            lambda pc, po: _lunch_gap_return(pc, po, morning_cutoff, afternoon_start, endpoint_policy),
            session_tz=session_tz,
        )


# ---------------------------------------------------------------------------
# § Intraday limit-up / limit-down behavior
# ---------------------------------------------------------------------------

def _broadcast_daily_limits(panel: SessionPanel, limit_series: pd.Series | None) -> float | np.ndarray:
    """Broadcast a daily limit value onto every official slot of the panel.

    Returns a per-slot limit array (repeated across slots) or NaN when the
    daily limit is unavailable for the panel's date.
    """
    if limit_series is None:
        return np.full(panel.n_slots, np.nan)
    date = pd.Timestamp(panel.trade_date)
    try:
        value = float(limit_series.loc[date])
    except (KeyError, TypeError):
        return np.full(panel.n_slots, np.nan)
    return np.full(panel.n_slots, value)


def _limit_touch_mask(
    pc: SessionPanel, limit_v: np.ndarray, side: str
) -> tuple[bool, np.ndarray, np.ndarray]:
    """Limit-touch tri-state over the official grid.

    Returns ``(day_known, touch, valid)``:
    * ``day_known`` — the daily limit level itself is finite (the limit is a
      daily broadcast, so all-or-nothing).  An unknown limit makes the whole
      day's limit state unknown (R26-031/032).
    * ``touch`` — per-slot True/False where the limit is known AND the required
      bar field is valid.
    * ``valid`` — per-slot bars usable for limit statistics (present + finite
      limit).  An ABSENT bar is excluded from the denominator — it must never
      read as a confirmed non-touch, but it must also not nuke the day.
    """
    limit_known = np.isfinite(limit_v)
    bar_valid = pc.is_valid_bar
    day_known = bool(np.all(limit_known))
    usable = limit_known & bar_valid
    if side == "up":
        touch = usable & (pc.values >= limit_v - _EPS)
    elif side == "down":
        touch = usable & (pc.values <= limit_v + _EPS)
    else:
        raise ValueError(f"unknown limit side: {side!r}")
    return day_known, touch, usable


def _limit_first_hit_time(pc: SessionPanel, limit_v: np.ndarray, side: str) -> float:
    day_known, touch, usable = _limit_touch_mask(pc, limit_v, side)
    if not day_known or pc.n_slots == 0:
        return np.nan
    idx = np.flatnonzero(touch)
    if len(idx) == 0:
        return np.nan
    # R26-034: official slot ordinal (physical clock), never the compressed
    # observed-row index.
    return float(pc.slot_id[idx[0]]) / float(pc.n_slots)


@register_operator(
    name="intra_limit_first_hit_time",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_limit_first_hit_time",
    source="intraday_agg",
    backend="pandas_numpy",
    status="experimental",
)
class IntraLimitFirstHitTime(SeriesOperator):
    metadata = _metadata(
        "intra_limit_first_hit_time",
        "首次触及涨/跌停的分钟位置（官方 slot ordinal / 会话 slot 数）。",
        ["close", "high", "low", "high_limit", "low_limit", "side"],
        unit="position",
    )
    metadata.tags.append("allow_panel_broadcast")
    metadata.param_specs = {}


    def _calculate_series(self, close, high=None, low=None, high_limit=None, low_limit=None,
                          side="up", **_):
        # R26-033: a first-hit TOUCH needs only its own OHLC side.  ``side=up``
        # uses the minute HIGH vs upper limit; ``side=down`` uses the minute LOW
        # vs lower limit.  The OPPOSITE side's panel must never gate the check.
        if side not in ("up", "down"):
            raise ValueError("side must be 'up' or 'down'")
        touch = high if (side == "up" and high is not None) else (low if (side == "down" and low is not None) else close)
        limit_panel = high_limit if side == "up" else low_limit
        lim_series = _as_panel(limit_panel).iloc[:, 0] if limit_panel is not None else None

        def _fn(pc: SessionPanel) -> float:
            limit_v = _broadcast_daily_limits(pc, lim_series)
            return _limit_first_hit_time(pc, np.asarray(limit_v, dtype=float), side)

        return _daily_agg(touch, _fn)


def _limit_duration(pc: SessionPanel, limit_v: np.ndarray, side: str) -> float:
    day_known, touch, usable = _limit_touch_mask(pc, limit_v, side)
    if not day_known:
        return np.nan
    n = int(usable.sum())
    if n <= 0:
        return np.nan
    return float(touch.sum()) / float(n)


@register_operator(
    name="intra_limit_duration",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_limit_duration",
    source="intraday_agg",
    backend="pandas_numpy",
    status="experimental",
)
class IntraLimitDuration(SeriesOperator):
    metadata = _metadata(
        "intra_limit_duration",
        "收盘价处于涨/跌停价附近的分钟占比。",
        ["close", "high_limit", "low_limit", "side"], unit="ratio",
    )
    metadata.tags.append("allow_panel_broadcast")
    metadata.param_specs = {}


    def _calculate_series(self, close, high_limit=None, low_limit=None, side="up", **_):
        limit_panel = high_limit if side == "up" else low_limit
        lim_series = _as_panel(limit_panel).iloc[:, 0] if limit_panel is not None else None

        def _fn(pc: SessionPanel) -> float:
            limit_v = _broadcast_daily_limits(pc, lim_series)
            return _limit_duration(pc, np.asarray(limit_v, dtype=float), side)

        return _daily_agg(close, _fn)


def _limit_reopen_count(pc: SessionPanel, limit_v: np.ndarray, side: str, transition: str) -> float:
    day_known, touch, usable = _limit_touch_mask(pc, limit_v, side)
    if not day_known:
        return np.nan
    m = np.asarray(touch, dtype=bool)
    count = 0.0
    for i in range(1, pc.n_slots):
        if not (usable[i] and usable[i - 1]):
            continue  # grid gap: state not comparable across a missing minute
        if transition == "open" and m[i - 1] and not m[i]:
            count += 1.0
        elif transition == "reseal" and (not m[i - 1]) and m[i]:
            count += 1.0
    return count


@register_operator(
    name="intra_limit_reopen_count",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_limit_reopen_count",
    source="intraday_agg",
    backend="pandas_numpy",
    status="experimental",
)
class IntraLimitReopenCount(SeriesOperator):
    metadata = _metadata(
        "intra_limit_reopen_count",
        "封板打开（open）/ 重新封板（reseal）次数。",
        ["close", "high_limit", "low_limit", "side", "transition"], unit="count",
    )
    metadata.tags.append("allow_panel_broadcast")
    metadata.param_specs = {}


    def _calculate_series(self, close, high_limit=None, low_limit=None, side="up",
                          transition="open", **_):
        limit_panel = high_limit if side == "up" else low_limit
        lim_series = _as_panel(limit_panel).iloc[:, 0] if limit_panel is not None else None

        def _fn(pc: SessionPanel) -> float:
            limit_v = _broadcast_daily_limits(pc, lim_series)
            return _limit_reopen_count(pc, np.asarray(limit_v, dtype=float), side, transition)

        return _daily_agg(close, _fn)


# ---------------------------------------------------------------------------
# R26 legacy-compat shims for ``microstructure/flow_impact.py`` and
# ``advanced_intraday.py``.
#
# flow_impact's BVC / impact operators and advanced_intraday's subsampled /
# signature operators consume the pre-R26 plumbing (``_daily_agg_two`` /
# ``_daily_agg_legacy`` / raw ``_log_returns``).  They are NOT
# grid-canonicalized; the R26 physical-clock kernels live behind the new
# ``_pair_agg`` / ``SessionPanel.log_returns`` path above.  The shims keep
# those modules' behaviour byte-identical; their raw-array log-returns are
# audited separately (R26-126..129).
# ---------------------------------------------------------------------------

def _daily_agg_legacy(
    frame: pd.DataFrame,
    fn: Callable[[np.ndarray, np.ndarray], float],
) -> pd.DataFrame:
    """LEGACY raw-array daily aggregation for advanced_intraday (R26-deprecated).

    New minute kernels MUST use :func:`_daily_agg` (grid-canonicalized).
    """
    frame = _as_panel(frame)
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


def _log_returns(vals: np.ndarray) -> np.ndarray:
    """LEGACY raw-array log returns for flow_impact (R26-deprecated).

    New minute kernels MUST use :meth:`SessionPanel.log_returns` (grid-gated,
    no gap fusion).  Kept only for the flow_impact consumer.
    """
    out = np.full(len(vals), np.nan)
    if len(vals) > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            out[1:] = np.log(vals[1:] / vals[:-1])
            out[1:][(vals[1:] <= 0.0) | (vals[:-1] <= 0.0)] = np.nan
    return out


def _daily_agg_two(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    fn: Callable[[np.ndarray, np.ndarray], float],
) -> pd.DataFrame:
    """LEGACY pair aggregation for flow_impact (R26-deprecated).

    Raw-observed-row pair-drop; new code uses :func:`_pair_agg`.
    """
    a = _as_panel(frame_a)
    b = _as_panel(frame_b)
    out: dict[str, pd.Series] = {}
    for inst in a.columns:
        joined = pd.concat([a[inst], b[inst]], axis=1, keys=["a", "b"]).dropna(subset=["a", "b"])
        joined["day"] = joined.index.normalize()
        per_day: dict[pd.Timestamp, float] = {}
        for day, group in joined.groupby("day"):
            vals_a = np.asarray(group["a"], dtype=float)
            vals_b = np.asarray(group["b"], dtype=float)
            if not np.any(np.isfinite(vals_a)) or not np.any(np.isfinite(vals_b)):
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


__all__ = [
    "intra_segment_return", "intra_segment_volume_share", "intra_segment_amount_share",
    "intra_segment_vwap_deviation", "intra_segment_realized_vol",
    "intra_realized_variance", "intra_realized_semivariance", "intra_bipower_variation",
    "intra_jump_ratio", "intra_path_efficiency", "intra_high_time", "intra_low_time",
    "intra_vwap_above_ratio", "intra_vwap_cross_count", "intra_concentration",
    "intra_entropy", "intra_signed_imbalance_proxy", "intra_return_activity_corr",
    "intra_amihud", "intra_kyle_lambda_proxy", "intra_extreme_bar_return",
    "intra_lunch_gap_return", "intra_limit_first_hit_time", "intra_limit_duration",
    "intra_limit_reopen_count",
]

# These are extended intraday-microstructure operators; classify them so the
# static operator surface covers the final registry exactly.
import cleaned_operators.operator_surface as _surface  # noqa: E402

_surface.extend_extended_only(set(__all__))


# Register the intraday aggregation surface.  Operators stay on the reviewed
# extended surface (DSL-usable) but remain SOURCE_BLOCKED for production until
# a certified minute dataset is available.
from cleaned_operators import operator_surface as _surface  # noqa: E402

_surface.extend_extended_only(set(__all__))
