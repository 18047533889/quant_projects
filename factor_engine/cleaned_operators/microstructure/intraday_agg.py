# -*- coding: utf-8 -*-
"""Minute frequency to daily aggregation operators.

Each operator accepts minute-frequency OHLCV panels (row index is a minute
timestamp, columns are instruments) and returns a daily-frequency panel
(row index is the calendar date, columns are instruments).

Contract
--------
* The output is one scalar per (TradeDate, Symbol): it can be used as a daily
  cross-sectional factor after close.  Operators never emit a row per minute.
* All kernels are causal: they only use the day's own minute data plus that
  day's daily limit prices, never future bars or future days.
* Empty / all-NaN windows yield NaN, never Inf or a fabricated zero.
* These operators require a certified minute dataset.  Until one is available
  they remain ``SOURCE_BLOCKED_CANONICALS`` and are excluded from production
  targets (see ``production_hardening``).

Segment convention (A-share default): morning 09:30--11:30, afternoon
13:00--15:00, expressed in minute-of-day as [570, 690] and [780, 900].
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, ParamSpec, SeriesOperator, register_operator

_EPS = 1e-12
_MORNING = (570, 690)   # 09:30 .. 11:30
_AFTERNOON = (780, 900) # 13:00 .. 15:00
_SEGMENT_RANGES = {"morning": _MORNING, "afternoon": _AFTERNOON}


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
    )


def _as_panel(x: Any) -> pd.DataFrame:
    if isinstance(x, pd.Series):
        return x.to_frame(getattr(x, "name", None) or "value")
    return x


def _minute_of_day(times: np.ndarray) -> np.ndarray:
    """Timestamp array -> minute-of-day (integer)."""
    seconds = times.astype("datetime64[s]").astype("int64") % 86400
    return seconds // 60


def _log_returns(vals: np.ndarray) -> np.ndarray:
    out = np.full(len(vals), np.nan)
    if len(vals) > 1:
        # R6-192: explicit positive-price contract.  log(negative/zero) leaking
        # -inf/NaN into the return silently corrupts downstream aggregates; a
        # non-positive price is invalid data and yields NaN (fail-closed), not
        # a fabricated log-return.
        with np.errstate(divide="ignore", invalid="ignore"):
            out[1:] = np.log(vals[1:] / vals[:-1])
            out[1:][(vals[1:] <= 0.0) | (vals[:-1] <= 0.0)] = np.nan
    return out


def _daily_agg(frame: pd.DataFrame, fn: Callable[[np.ndarray, np.ndarray], float]) -> pd.DataFrame:
    """Apply per-(instrument, calendar-day) aggregation fn(vals, times)."""
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


def _daily_agg_two(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    fn: Callable[[np.ndarray, np.ndarray], float],
) -> pd.DataFrame:
    """Apply fn(a_vals, b_vals) per (instrument, day)."""
    frame_a, frame_b = _as_panel(frame_a), _as_panel(frame_b)
    out: dict[str, pd.Series] = {}
    for inst in frame_a.columns:
        a, b = frame_a[inst], frame_b[inst]
        joined = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna(subset=["a"])
        joined["day"] = joined.index.normalize()
        per_day: dict[pd.Timestamp, float] = {}
        for day, group in joined.groupby("day"):
            vals_a = np.asarray(group["a"], dtype=float)
            vals_b = np.asarray(np.where(np.isfinite(group["b"]), group["b"], np.nan), dtype=float)
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


def _daily_agg_three(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    frame_c: pd.DataFrame,
    fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
) -> pd.DataFrame:
    """Apply fn(a_vals, b_vals, c_vals) per (instrument, day)."""
    frame_a, frame_b, frame_c = _as_panel(frame_a), _as_panel(frame_b), _as_panel(frame_c)
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


_SESSION_TZ = "Asia/Shanghai"


def _session_local(frame: pd.DataFrame, tz: str | None = None) -> pd.DataFrame:
    """Convert a tz-aware index to session wall-clock (naive) for minute-of-day math.

    A-share COS minute data is stored in UTC; the A-share session segments
    (09:30--11:30 / 13:00--15:00) are defined in Asia/Shanghai wall-clock time.
    Naive indexes are assumed to already be in session wall-clock time.
    """
    frame = _as_panel(frame)
    if isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is not None:
        tz = tz or _SESSION_TZ
        frame = frame.tz_convert(tz)
        frame.index = frame.index.tz_localize(None)
    return frame


def _seg_mask(times: np.ndarray, segment: str) -> np.ndarray:
    lo, hi = _SEGMENT_RANGES[str(segment)]
    minutes = _minute_of_day(times)
    return (minutes >= lo) & (minutes <= hi)


# ---------------------------------------------------------------------------
# § Segment aggregation
# ---------------------------------------------------------------------------

def _seg_return(vals, times, segment):
    mask = _seg_mask(times, segment)
    sel = vals[mask]
    finite = sel[np.isfinite(sel)]
    if len(finite) < 2:
        return np.nan
    return float(finite[-1] / finite[0] - 1.0)


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
    metadata = _metadata("intra_segment_return", "指定时段（morning/afternoon）收盘/开盘收益 - 1。", ["close", "segment", "session_tz"], unit="return")
    metadata.param_specs = {"session_tz": ParamSpec(dtype=str, searchable=False)}  # R6-190

    def _calculate_series(self, close, segment="morning", session_tz=None, **_):
        return _daily_agg(_session_local(close, session_tz), lambda v, t: _seg_return(v, t, segment))


def _seg_volume_share(vals, times, segment, total):
    mask = _seg_mask(times, segment)
    seg = float(np.nansum(vals[mask]))
    return seg / total if total > _EPS else np.nan


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
    metadata = _metadata("intra_segment_volume_share", "指定时段成交量占全天比例。", ["volume", "segment", "session_tz"], unit="ratio")
    metadata.param_specs = {"session_tz": ParamSpec(dtype=str, searchable=False)}  # R6-190

    def _calculate_series(self, volume, segment="morning", session_tz=None, **_):
        def fn(v, t):
            total = float(np.nansum(v))
            return _seg_volume_share(v, t, segment, total)
        return _daily_agg(_session_local(volume, session_tz), fn)


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
    metadata = _metadata("intra_segment_amount_share", "指定时段成交额占全天比例。", ["amount", "segment", "session_tz"], unit="ratio")
    metadata.param_specs = {"session_tz": ParamSpec(dtype=str, searchable=False)}  # R6-190

    def _calculate_series(self, amount, segment="morning", session_tz=None, **_):
        def fn(v, t):
            total = float(np.nansum(v))
            return _seg_volume_share(v, t, segment, total)
        return _daily_agg(_session_local(amount, session_tz), fn)


def _seg_vwap_deviation(close_v, amt_v, vol_v, times, segment):
    mask = _seg_mask(times, segment)
    c = close_v[mask]
    a = np.nansum(amt_v[mask])
    v = np.nansum(vol_v[mask])
    if len(c) < 1 or v <= _EPS or not np.isfinite(c[-1]):
        return np.nan
    vwap = a / v
    return float(c[-1] / vwap - 1.0)


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
    metadata = _metadata("intra_segment_vwap_deviation", "指定时段末价相对该时段累计 VWAP 的偏差。", ["close", "amount", "volume", "segment", "session_tz"], unit="ratio")
    metadata.param_specs = {"session_tz": ParamSpec(dtype=str, searchable=False)}  # R6-190

    def _calculate_series(self, close, amount, volume, segment="morning", session_tz=None, **_):
        frame = _session_local(close, session_tz)
        amt = _session_local(amount, session_tz)
        vol = _session_local(volume, session_tz)
        out = {}
        for inst in frame.columns:
            joined = pd.concat([frame[inst], amt[inst], vol[inst]], axis=1, keys=["c", "a", "v"]).dropna(subset=["c"])
            joined["day"] = joined.index.normalize()
            per_day = {}
            for day, group in joined.groupby("day"):
                per_day[day] = _seg_vwap_deviation(
                    np.asarray(group["c"], dtype=float),
                    np.asarray(group["a"], dtype=float),
                    np.asarray(group["v"], dtype=float),
                    np.asarray(group.index, dtype="datetime64[ns]"),
                    segment,
                )
            out[inst] = pd.Series(per_day, dtype=float)
        return pd.DataFrame(out).sort_index()


def _coverage_ok(r: np.ndarray, floor: float = 0.9) -> bool:
    """Coverage gate: realised-variation estimates must not treat data gaps as
    zero-return minutes (P1-56).  A day whose observed grid is < ``floor``
    finite returns fails closed rather than reporting a partial RV."""
    if r.size == 0:
        return False
    return float(np.sum(np.isfinite(r))) / r.size >= floor


def _seg_realized_vol(close_v, times, segment):
    mask = _seg_mask(times, segment)
    r = _log_returns(close_v[mask])
    if not _coverage_ok(r):
        return np.nan
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
    metadata = _metadata("intra_segment_realized_vol", "指定时段已实现波动率 sqrt(sum(r_t^2))。", ["close", "segment", "session_tz"], unit="volatility")
    metadata.param_specs = {"session_tz": ParamSpec(dtype=str, searchable=False)}  # R6-190

    def _calculate_series(self, close, segment="morning", session_tz=None, **_):
        return _daily_agg(_session_local(close, session_tz), lambda v, t: _seg_realized_vol(v, t, segment))


# ---------------------------------------------------------------------------
# § Realized variance / semivariance / bipower / jump
# ---------------------------------------------------------------------------

def _rv(close_v):
    r = _log_returns(close_v)
    if not _coverage_ok(r):
        return np.nan
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
    metadata = _metadata("intra_realized_variance", "日内已实现方差 sum(r_t^2)。", ["close"], unit="variance")

    def _calculate_series(self, close, **_):
        return _daily_agg(close, lambda v, t: _rv(v))


def _semivariance(close_v, side):
    r = _log_returns(close_v)
    if not _coverage_ok(r):
        return np.nan
    if side == "down":
        r = np.where(r < 0, r, 0.0)
    elif side == "up":
        r = np.where(r > 0, r, 0.0)
    else:
        # An invalid ``side`` must fail loudly, never silently fall back to the
        # full RV (P1-55).
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
    metadata = _metadata("intra_realized_semivariance", "日内上/下半方差 sum(r_t^2 * 1(sign))。", ["close", "side"], unit="variance")

    def _calculate_series(self, close, side="down", **_):
        if side not in ("up", "down"):
            # Invalid side must fail loudly (P1-55), not fall back to full RV.
            raise ValueError("intra_realized_semivariance requires side in {'up', 'down'}")
        return _daily_agg(close, lambda v, t: _semivariance(v, side))


def _bipower(close_v):
    r = _log_returns(close_v)
    if not _coverage_ok(r):
        return np.nan
    # Bipower variation must use *real adjacent* minute returns.  Compressing
    # the finite returns and differencing the compressed series pairs 09:40 with
    # 09:42 across a gap — a 3rd-round P1-53 numerical bug.  Multiplying the raw
    # positional array means any product touching a NaN return is dropped and
    # only truly adjacent finite minutes are summed.
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
    metadata = _metadata("intra_bipower_variation", "日内双幂变差 (pi/2)*sum(|r_t||r_{t-1}|)。", ["close"], unit="variance")

    def _calculate_series(self, close, **_):
        return _daily_agg(close, lambda v, t: _bipower(v))


def _jump_ratio(close_v):
    rv = _rv(close_v)
    bv = _bipower(close_v)
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
    metadata = _metadata("intra_jump_ratio", "日内跳跃占比 max(RV-BV,0)/RV。", ["close"], unit="ratio")

    def _calculate_series(self, close, **_):
        return _daily_agg(close, lambda v, t: _jump_ratio(v))


# ---------------------------------------------------------------------------
# § Intraday price path
# ---------------------------------------------------------------------------

def _path_efficiency(close_v):
    mask = np.isfinite(close_v)
    if mask.sum() < 2:
        return np.nan
    idx = np.where(mask)[0]
    pts = close_v[idx]
    # Value-contiguous arc length through consecutive finite closes (bridging
    # interior gaps with the straight segment between their endpoints).  By the
    # triangle inequality this is always >= |net displacement|, so efficiency
    # stays in [0,1] — P1-54 (compressing the finite closes skipped the gap and
    # inflated efficiency).
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
    metadata = _metadata("intra_path_efficiency", "日内路径效率 |净位移|/路径长度。", ["close"], unit="ratio")

    def _calculate_series(self, close, **_):
        return _daily_agg(close, lambda v, t: _path_efficiency(v))


def _position_of(high_v, times, *, low: bool):
    finite_mask = np.isfinite(high_v)
    if not np.any(finite_mask):
        return np.nan
    idx = np.where(finite_mask)[0]
    vals = high_v[finite_mask]
    target = np.argmin(vals) if low else np.argmax(vals)
    # Normalise by the *observed session grid* (all rows, finite or not).  The
    # previous denominator counted only finite bars, so a high at raw slot 200
    # on a 150-finite-bar day produced 200/150 > 1 — a direct numerical error
    # (P0-23).  ``raw_slot / total_grid`` is always in [0, 1).
    total = len(high_v)
    if total <= 0:
        return np.nan
    return float(idx[target]) / float(total)


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
    metadata = _metadata("intra_high_time", "全天最高价首次出现位置 / 有效分钟数。", ["high"], unit="position")

    def _calculate_series(self, high, **_):
        return _daily_agg(high, lambda v, t: _position_of(v, t, low=False))


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
    metadata = _metadata("intra_low_time", "全天最低价首次出现位置 / 有效分钟数。", ["low"], unit="position")

    def _calculate_series(self, low, **_):
        return _daily_agg(low, lambda v, t: _position_of(v, t, low=True))


def _vwap_above_ratio(close_v, amount_v, volume_v):
    valid_vwap = np.isfinite(amount_v) & np.isfinite(volume_v) & (volume_v > 0)
    total_v = float(np.sum(np.where(valid_vwap, volume_v, 0.0)))
    total_a = float(np.sum(np.where(valid_vwap, amount_v, 0.0)))
    if total_v <= _EPS:
        return np.nan
    day_vwap = total_a / total_v
    valid = np.isfinite(close_v) & valid_vwap
    if valid.sum() == 0:
        return np.nan
    return float(np.mean(close_v[valid] > day_vwap))


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
    metadata = _metadata("intra_vwap_above_ratio", "收盘价高于当日 VWAP 的分钟占比。", ["close", "amount", "volume"], unit="ratio")

    def _calculate_series(self, close, amount, volume, **_):
        return _daily_agg_three(close, amount, volume, _vwap_above_ratio)


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
    metadata = _metadata("intra_vwap_cross_count", "收盘价相对累计 VWAP 的方向变化次数。", ["close", "amount", "volume"], unit="count")

    def _calculate_series(self, close, amount, volume, **_):
        frame = _as_panel(close)
        amt = _as_panel(amount)
        vol = _as_panel(volume)
        out = {}
        for inst in frame.columns:
            joined = pd.concat([frame[inst], amt[inst], vol[inst]], axis=1, keys=["c", "a", "v"]).dropna(subset=["c"])
            joined["day"] = joined.index.normalize()
            per_day = {}
            for day, group in joined.groupby("day"):
                per_day[day] = _vwap_cross_count(
                    np.asarray(group["c"], dtype=float),
                    np.asarray(group["a"], dtype=float),
                    np.asarray(group["v"], dtype=float),
                )
            out[inst] = pd.Series(per_day, dtype=float)
        return pd.DataFrame(out).sort_index()


def _vwap_cross_count(close_v, amount_v, volume_v):
    # Cumulative VWAP runs on the full aligned grid (all traded bars), while the
    # cross test only fires on rows where BOTH close and the cumulative VWAP
    # exist.  The previous code compressed ``close`` to its finite rows but kept
    # ``amount``/``volume`` on the full axis — the two arrays could desync when a
    # close was missing (P0-24).
    vol = np.where(np.isfinite(volume_v), volume_v, 0.0)
    amt = np.where(np.isfinite(amount_v), amount_v, 0.0)
    cum_v = np.cumsum(vol)
    cum_a = np.cumsum(amt)
    with np.errstate(divide="ignore", invalid="ignore"):
        cum_vwap = np.where(cum_v > _EPS, cum_a / cum_v, np.nan)
    valid = np.isfinite(close_v) & np.isfinite(cum_vwap)
    if valid.sum() < 2:
        return np.nan
    c = np.where(valid, close_v, np.nan)
    sign = np.sign(c - cum_vwap)
    sign = sign[valid]
    if len(sign) < 2:
        return 0.0
    return float(np.sum(sign[1:] != sign[:-1]))


# ---------------------------------------------------------------------------
# § Intraday volume / amount distribution
# ---------------------------------------------------------------------------

def _concentration(vals):
    # R6-191: intra_concentration is a distribution-of-activity primitive —
    # its input contract is NonnegativeActivity.  A negative value is INVALID,
    # not "the amount that happened in the opposite direction"; silently
    # abs()-ing it manufactures an undeclared semantics.  Fail closed on any
    # negative entry instead of folding it in as positive mass.
    v = np.asarray(vals, dtype=float)
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
    metadata = _metadata("intra_concentration", "日内成交集中度 sum((value/sum)^2)。", ["value"], unit="hhi")

    def _calculate_series(self, value, **_):
        return _daily_agg(value, lambda v, t: _concentration(v))


def _entropy(vals, normalize=True):
    # R6-191: same NonnegativeActivity contract as _concentration — a negative
    # value is data-invalid, never abs()-folded in as positive mass.
    v = np.asarray(vals, dtype=float)
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
    metadata = _metadata("intra_entropy", "日内成交分布熵（归一化）。", ["value", "normalize"], unit="entropy")
    metadata.param_specs = {"normalize": ParamSpec(dtype=bool, searchable=False)}  # R6-190

    def _calculate_series(self, value, normalize=True, **_):
        return _daily_agg(value, lambda v, t: _entropy(v, bool(normalize)))


def _signed_imbalance_proxy(close_v, value_v):
    r = np.sign(_log_returns(close_v))
    value = np.where(np.isfinite(value_v), value_v, 0.0)
    total = float(value.sum())
    if total <= _EPS:
        return np.nan
    return float(np.sum(np.where(np.isfinite(r), r, 0.0) * value) / total)


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
    metadata = _metadata("intra_signed_imbalance_proxy", "基于分钟价格方向的成交不平衡代理 sum(sign(r)*value)/sum(value)。", ["close", "value"], unit="ratio")

    def _calculate_series(self, close, value, **_):
        return _daily_agg_two(close, value, lambda a, b: _signed_imbalance_proxy(a, b))


def _return_activity_corr(close_v, activity_v, absolute_return):
    r = _log_returns(close_v)
    if absolute_return:
        r = np.abs(r)
    mask = np.isfinite(r) & np.isfinite(activity_v)
    r, a = r[mask], activity_v[mask]
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
    metadata = _metadata("intra_return_activity_corr", "分钟收益与成交活动相关性（可选绝对值）。", ["close", "activity", "absolute_return"], unit="corr")

    def _calculate_series(self, close, activity, absolute_return=False, **_):
        return _daily_agg_two(close, activity, lambda a, b: _return_activity_corr(a, b, bool(absolute_return)))


# ---------------------------------------------------------------------------
# § Intraday liquidity
# ---------------------------------------------------------------------------

def _intra_amihud(close_v, amount_v, scale):
    r = np.abs(_log_returns(close_v))
    amount = np.where(np.isfinite(amount_v), amount_v, 0.0)
    denom = np.maximum(amount, _EPS)
    ratio = np.where(np.isfinite(r), r / denom, np.nan)
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
    metadata = _metadata("intra_amihud", "日内 Amihud 非流动性 mean(|r|/max(amount,eps))*scale。", ["close", "amount", "scale"], unit="illiquidity")

    # R6-189: ``scale`` is a pure output-unit rescaling (1e6 vs 1e8 vs 1e10 give
    # identical ordering / alpha) — it must never be a search parameter.
    metadata.param_specs = {"scale": ParamSpec(dtype=float, searchable=False)}

    def _calculate_series(self, close, amount, scale=1e8, **_):
        return _daily_agg_two(close, amount, lambda a, b: _intra_amihud(a, b, float(scale)))


def _kyle_lambda_proxy(close_v, amount_v):
    r = _log_returns(close_v)
    amount = np.where(np.isfinite(amount_v), amount_v, 0.0)
    total = float(amount.sum())
    if total <= _EPS:
        return np.nan
    signed_share = np.sign(r) * amount / total
    mask = np.isfinite(r) & np.isfinite(signed_share)
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
    metadata = _metadata("intra_kyle_lambda_proxy", "日内 Kyle Lambda 代理（收益对方向性成交额占比回归）。", ["close", "amount"], unit="lambda")

    def _calculate_series(self, close, amount, **_):
        return _daily_agg_two(close, amount, lambda a, b: _kyle_lambda_proxy(a, b))


def _extreme_bar_return(close_v, side):
    r = _log_returns(close_v)
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
    metadata = _metadata("intra_extreme_bar_return", "日内单分钟最大/最小收益。", ["close", "side"], unit="return")

    def _calculate_series(self, close, side="max", **_):
        # R6-184: the kernel silently treated ANY side != "max" as "min", so
        # ``side="abc"`` silently produced the min return.  Validate the enum.
        if side not in ("max", "min"):
            raise ValueError("side must be 'max' or 'min'")
        return _daily_agg(close, lambda v, t: _extreme_bar_return(v, side))


def _lunch_gap_return(close_v, open_v, times, morning_cutoff, afternoon_start):
    minutes = _minute_of_day(times)
    morning_mask = minutes <= _minute_hm(morning_cutoff)
    afternoon_mask = minutes >= _minute_hm(afternoon_start)
    morning_close = close_v[morning_mask]
    afternoon_open = open_v[afternoon_mask]
    mc = morning_close[np.isfinite(morning_close)]
    ao = afternoon_open[np.isfinite(afternoon_open)]
    if len(mc) == 0 or len(ao) == 0 or mc[-1] <= _EPS:
        return np.nan
    return float(ao[0] / mc[-1] - 1.0)


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
    metadata = _metadata("intra_lunch_gap_return", "午间跳空：下午首根 Open/上午末根 Close - 1。",
               ["close", "open", "morning_cutoff", "afternoon_start", "session_tz"], unit="return")
    metadata.param_specs = {  # R6-190: config knobs, not search parameters
        "morning_cutoff": ParamSpec(dtype=str, searchable=False),
        "afternoon_start": ParamSpec(dtype=str, searchable=False),
        "session_tz": ParamSpec(dtype=str, searchable=False),
    }

    def _calculate_series(self, close, open_px, morning_cutoff="11:30", afternoon_start="13:00", session_tz=None, **_):
        frame = _session_local(close, session_tz)
        opn = _session_local(open_px, session_tz)
        out = {}
        for inst in frame.columns:
            # R6-185: the OLD code ``dropna(subset=["c","o"])`` required BOTH
            # close and open to be finite on the SAME minute before the join —
            # the morning last-close and the afternoon first-open both had to
            # be complete, dropping a valid morning close if its open was
            # missing and vice versa.  The morning needs only Close, the
            # afternoon only Open, so concatenate without a joint dropna and let
            # ``_lunch_gap_return`` select each side independently.
            joined = pd.concat([frame[inst], opn[inst]], axis=1, keys=["c", "o"])
            joined["day"] = joined.index.normalize()
            per_day = {}
            for day, group in joined.groupby("day"):
                per_day[day] = _lunch_gap_return(
                    np.asarray(group["c"], dtype=float),
                    np.asarray(group["o"], dtype=float),
                    np.asarray(group.index, dtype="datetime64[ns]"),
                    morning_cutoff, afternoon_start,
                )
            out[inst] = pd.Series(per_day, dtype=float)
        return pd.DataFrame(out).sort_index()


# ---------------------------------------------------------------------------
# § Intraday limit-up / limit-down behavior
# ---------------------------------------------------------------------------

def _broadcast_daily_limits(close_frame, limit_frame):
    """Broadcast a daily limit panel (index=date) onto a minute panel by date."""
    lim = _as_panel(limit_frame)
    out = {}
    for inst in close_frame.columns:
        if inst not in lim.columns:
            continue
        close_col = close_frame[inst]
        lim_col = lim[inst]
        days = close_col.index.normalize()
        out[inst] = pd.Series(
            lim_col.reindex(pd.DatetimeIndex(days.unique())).reindex(pd.DatetimeIndex(days)).to_numpy(),
            index=close_col.index,
        )
    return pd.DataFrame(out)


def _limit_mask(close_v, limit_v, side):
    """Limit-touch mask over a per-minute series.

    R6-187: a NaN limit price means the day's limit level is UNKNOWN — it must
    not be read as "no limit state" (which would silently count every minute as
    not-at-limit and fabricate duration/reopen statistics).  The mask is
    tri-state: True = at limit, False = valid bar below/above the known limit,
    None = limit unknown (the caller must emit NaN, not False).
    """
    limit_known = np.isfinite(limit_v)
    bar_valid = np.isfinite(close_v)
    if side == "up":
        base = limit_known & bar_valid & (close_v >= limit_v - _EPS)
    elif side == "down":
        base = limit_known & bar_valid & (close_v <= limit_v + _EPS)
    else:
        raise ValueError(f"unknown limit side: {side!r}")
    out = np.full(len(close_v), np.nan, dtype=object)
    out[base] = True
    out[limit_known & ~base] = False
    return out


def _limit_first_hit_time(close_v, times, limit_v, side):
    mask = _limit_mask(close_v, limit_v, side)
    # R6-187: limit level unknown -> the whole day's limit state is unknown,
    # emit NaN rather than a fabricated "never hit".
    if np.any(pd.isna(mask)):
        return np.nan
    idx = np.flatnonzero(mask)
    n = len(close_v)
    if len(idx) == 0 or n == 0:
        return np.nan
    return float(idx[0]) / float(n)


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
    metadata = _metadata("intra_limit_first_hit_time", "首次触及涨/跌停的分钟位置 / 有效分钟数。", ["close", "high", "low", "high_limit", "low_limit", "side"], unit="position")
    metadata.tags.append("allow_panel_broadcast")

    def _calculate_series(self, close, high=None, low=None, high_limit=None, low_limit=None, side="up", **_):
        # R6-186: "touch" must use the minute HIGH for the up-limit and the
        # minute LOW for the down-limit — a minute whose high == limit_up but
        # close < limit_up really touched the board but a close-based mask
        # missed it.  ``sealed`` states (sustained at limit) can use close, but
        # the FIRST-HIT is a touch event.  ``high``/``low`` are optional
        # panels; when not provided the operator falls back to close (legacy
        # behaviour) so the positional contract stays backward-compatible.
        if side not in ("up", "down"):
            raise ValueError("side must be 'up' or 'down'")
        touch = (high if side == "up" else low) if (high is not None and low is not None) else close
        frame = _as_panel(touch)
        lim = _broadcast_daily_limits(frame, high_limit if side == "up" else low_limit)
        out = {}
        for inst in frame.columns:
            joined = pd.concat([frame[inst], lim[inst]], axis=1, keys=["c", "l"]).dropna(subset=["c"])
            joined["day"] = joined.index.normalize()
            per_day = {}
            for day, group in joined.groupby("day"):
                per_day[day] = _limit_first_hit_time(
                    np.asarray(group["c"], dtype=float),
                    np.asarray(group.index, dtype="datetime64[ns]"),
                    np.asarray(group["l"], dtype=float), side,
                )
            out[inst] = pd.Series(per_day, dtype=float)
        return pd.DataFrame(out).sort_index()


def _limit_duration(close_v, limit_v, side):
    mask = _limit_mask(close_v, limit_v, side)
    # R6-187: unknown limit level -> NaN, not "0 minutes at limit".
    if np.any(pd.isna(mask)):
        return np.nan
    n = len(close_v)
    return float(np.count_nonzero(mask)) / float(n) if n > 0 else np.nan


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
    metadata = _metadata("intra_limit_duration", "收盘价处于涨/跌停价附近的分钟占比。", ["close", "high_limit", "low_limit", "side"], unit="ratio")
    metadata.tags.append("allow_panel_broadcast")

    def _calculate_series(self, close, high_limit=None, low_limit=None, side="up", **_):
        frame = _as_panel(close)
        lim = _broadcast_daily_limits(frame, high_limit if side == "up" else low_limit)
        out = {}
        for inst in frame.columns:
            joined = pd.concat([frame[inst], lim[inst]], axis=1, keys=["c", "l"]).dropna(subset=["c"])
            joined["day"] = joined.index.normalize()
            per_day = {}
            for day, group in joined.groupby("day"):
                per_day[day] = _limit_duration(
                    np.asarray(group["c"], dtype=float),
                    np.asarray(group["l"], dtype=float), side,
                )
            out[inst] = pd.Series(per_day, dtype=float)
        return pd.DataFrame(out).sort_index()


def _limit_reopen_count(close_v, limit_v, side, transition):
    mask = _limit_mask(close_v, limit_v, side)
    # R6-187: unknown limit level -> the day's limit state is unknown; a
    # reopen/reseal count built on a guessed mask is fabricated.
    if np.any(pd.isna(mask)):
        return np.nan
    # R6-188: run the transition state machine on the OFFICIAL minute grid.
    # A missing minute (NaN close) is a boundary: the state before and after a
    # gap cannot be compared as adjacent minutes.  The caller must NOT
    # ``dropna`` the grid (which would turn "1 -> gap -> 0" into an adjacent
    # 1->0 transition).  We treat a NaN close as "episode interrupted": no
    # transition is counted across it.
    m = np.asarray(mask, dtype=bool)
    valid = np.isfinite(close_v) & np.isfinite(limit_v)
    count = 0.0
    for i in range(1, len(m)):
        if not (valid[i] and valid[i - 1]):
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
    metadata = _metadata("intra_limit_reopen_count", "封板打开（open）/ 重新封板（reseal）次数。", ["close", "high_limit", "low_limit", "side", "transition"], unit="count")
    metadata.tags.append("allow_panel_broadcast")

    def _calculate_series(self, close, high_limit=None, low_limit=None, side="up", transition="open", **_):
        frame = _as_panel(close)
        lim = _broadcast_daily_limits(frame, high_limit if side == "up" else low_limit)
        out = {}
        for inst in frame.columns:
            # R6-188: NO dropna — the reopen state machine must run on the
            # OFFICIAL minute grid so a missing minute is a boundary, not a
            # silent reconnection (``1 -> [missing] -> 0`` must not count as a
            # 1->0 adjacent transition).
            joined = pd.concat([frame[inst], lim[inst]], axis=1, keys=["c", "l"])
            joined["day"] = joined.index.normalize()
            per_day = {}
            for day, group in joined.groupby("day"):
                per_day[day] = _limit_reopen_count(
                    np.asarray(group["c"], dtype=float),
                    np.asarray(group["l"], dtype=float), side, transition,
                )
            out[inst] = pd.Series(per_day, dtype=float)
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

_surface.extend_extended_only({
        "intra_segment_return", "intra_segment_volume_share", "intra_segment_amount_share",
        "intra_segment_vwap_deviation", "intra_segment_realized_vol",
        "intra_realized_variance", "intra_realized_semivariance", "intra_bipower_variation",
        "intra_jump_ratio", "intra_path_efficiency", "intra_high_time", "intra_low_time",
        "intra_vwap_above_ratio", "intra_vwap_cross_count", "intra_concentration",
        "intra_entropy", "intra_signed_imbalance_proxy", "intra_return_activity_corr",
        "intra_amihud", "intra_kyle_lambda_proxy", "intra_extreme_bar_return",
        "intra_lunch_gap_return", "intra_limit_first_hit_time", "intra_limit_duration",
        "intra_limit_reopen_count",
    })
