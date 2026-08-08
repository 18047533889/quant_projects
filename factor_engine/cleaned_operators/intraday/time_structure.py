# -*- coding: utf-8 -*-
"""Intraday time-slot structure, cross-day slot memory, and profile distances.

Minute panels in, daily panels out.  Interval operators are parameterised by
``start_minute``/``end_minute`` (minute-of-day, A-share session: 570=09:30,
690=11:30, 780=13:00, 900=15:00).  Same-slot / profile operators align minutes
across days by minute-of-day and only use current plus past days (causal).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from cleaned_operators.base import SeriesOperator, register_operator
from cleaned_operators.intraday._core import (
    _EPS,
    daily_agg,
    daily_agg_two,
    daily_agg_three,
    log_returns,
    metadata,
    minute_of_day,
    np_errstate,
    register_surface,
    require_same_session_grid,
    safe_div,
)

_CANONICALS: list[str] = []


def _interval_mask(times: np.ndarray, start_minute: int, end_minute: int) -> np.ndarray:
    minutes = minute_of_day(times)
    return (minutes >= int(start_minute)) & (minutes <= int(end_minute))


def _interval_return(close_v: np.ndarray, times: np.ndarray, start: int, end: int) -> float:
    mask = _interval_mask(times, start, end)
    sel = close_v[mask]
    finite = sel[np.isfinite(sel)]
    if len(finite) < 2:
        return np.nan
    return float(finite[-1] / finite[0] - 1.0)


@register_operator(
    name="intra_interval_return",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_interval_return",
    source="intraday.time_structure",
    backend="pandas_numpy",
    status="experimental",
)
class IntraIntervalReturn(SeriesOperator):
    """指定分钟区间收益：区间末价/区间首价 - 1。"""

    metadata = metadata(
        "intra_interval_return", "指定 [start_minute, end_minute] 区间收益。",
        ["close", "start_minute", "end_minute"], unit="return",
    )

    def _calculate_series(self, close, start_minute=570, end_minute=900, session_tz=None, **_):
        s, e = int(start_minute), int(end_minute)
        if s < 0 or e < s or e > 1440:
            raise ValueError(f"invalid interval [{s}, {e}]")
        from cleaned_operators.intraday._core import session_local
        return daily_agg(session_local(close, session_tz), lambda v, t: _interval_return(v, t, s, e))


def _interval_share(vals: np.ndarray, times: np.ndarray, start: int, end: int) -> float:
    # P1-105: ``nansum`` treats a NaN (unknown) minute as a zero-value minute,
    # silently deflating the share.  Sum over FINITE values only; unknown bars
    # contribute to neither the interval nor the denominator.
    mask = _interval_mask(times, start, end)
    seg = float(np.nansum(vals[mask] * np.isfinite(vals[mask])))
    total = float(np.nansum(vals * np.isfinite(vals)))
    return seg / total if total > _EPS else np.nan


@register_operator(
    name="intra_interval_volume_share",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_interval_volume_share",
    source="intraday.time_structure",
    backend="pandas_numpy",
    status="experimental",
)
class IntraIntervalVolumeShare(SeriesOperator):
    """指定区间成交量占全天比例。"""

    metadata = metadata(
        "intra_interval_volume_share", "区间成交量占比。", ["volume", "start_minute", "end_minute"], unit="ratio",
    )

    def _calculate_series(self, volume, start_minute=570, end_minute=900, session_tz=None, **_):
        s, e = int(start_minute), int(end_minute)
        from cleaned_operators.intraday._core import session_local
        return daily_agg(session_local(volume, session_tz), lambda v, t: _interval_share(v, t, s, e))


@register_operator(
    name="intra_interval_amount_share",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_interval_amount_share",
    source="intraday.time_structure",
    backend="pandas_numpy",
    status="experimental",
)
class IntraIntervalAmountShare(SeriesOperator):
    """指定区间成交额占全天比例。"""

    metadata = metadata(
        "intra_interval_amount_share", "区间成交额占比。", ["amount", "start_minute", "end_minute"], unit="ratio",
    )

    def _calculate_series(self, amount, start_minute=570, end_minute=900, session_tz=None, **_):
        s, e = int(start_minute), int(end_minute)
        from cleaned_operators.intraday._core import session_local
        return daily_agg(session_local(amount, session_tz), lambda v, t: _interval_share(v, t, s, e))


def _interval_rv(close_v: np.ndarray, times: np.ndarray, start: int, end: int) -> float:
    mask = _interval_mask(times, start, end)
    r = log_returns(close_v[mask])
    r_finite = r[np.isfinite(r)]
    if r_finite.size < 2:
        return np.nan
    # P1-105: sum over FINITE squared returns only — a NaN bar is an unknown
    # return, not a zero-return, and must not be counted as ``0``.
    with np_errstate():
        return float(np.sum(r_finite * r_finite))


@register_operator(
    name="intra_interval_realized_variance",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_interval_realized_variance",
    source="intraday.time_structure",
    backend="pandas_numpy",
    status="experimental",
)
class IntraIntervalRealizedVariance(SeriesOperator):
    """指定区间已实现方差。"""

    metadata = metadata(
        "intra_interval_realized_variance", "区间已实现方差。", ["close", "start_minute", "end_minute"], unit="variance",
    )

    def _calculate_series(self, close, start_minute=570, end_minute=900, session_tz=None, **_):
        s, e = int(start_minute), int(end_minute)
        from cleaned_operators.intraday._core import session_local
        return daily_agg(session_local(close, session_tz), lambda v, t: _interval_rv(v, t, s, e))


def _interval_vwap_dev(close_v, amt_v, vol_v, times, start, end):
    mask = _interval_mask(times, start, end)
    c = close_v[mask]
    # P1-105: sum FINITE amount/volume only — an unknown (NaN) bar is not a
    # zero-amount bar and must not drag the VWAP toward 0.
    a = float(np.sum(amt_v[mask][np.isfinite(amt_v[mask])]))
    v = float(np.sum(vol_v[mask][np.isfinite(vol_v[mask])]))
    finite_c = c[np.isfinite(c)]
    if len(finite_c) == 0 or v <= _EPS:
        return np.nan
    return float(finite_c[-1] / (a / v) - 1.0)


@register_operator(
    name="intra_interval_vwap_deviation",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_interval_vwap_deviation",
    source="intraday.time_structure",
    backend="pandas_numpy",
    status="experimental",
)
class IntraIntervalVwapDeviation(SeriesOperator):
    """区间末价相对区间累计 VWAP 的偏差。"""

    metadata = metadata(
        "intra_interval_vwap_deviation", "区间 VWAP 偏离。",
        ["close", "amount", "volume", "start_minute", "end_minute"], unit="ratio",
    )

    def _calculate_series(self, close, amount, volume, start_minute=570, end_minute=900, session_tz=None, **_):
        s, e = int(start_minute), int(end_minute)
        from cleaned_operators.intraday._core import session_local
        frame = session_local(close, session_tz)
        amt = session_local(amount, session_tz)
        vol = session_local(volume, session_tz)
        # P0-08: the concat+dropna below would silently compress a mismatched
        # session grid; require the same grid first, then keep the dropna (it
        # only removes rows where the PRIMARY column is NaN).
        require_same_session_grid(frame, amt, vol)
        out = {}
        for inst in frame.columns:
            joined = pd.concat([frame[inst], amt[inst], vol[inst]], axis=1, keys=["c", "a", "v"]).dropna(subset=["c"])
            joined["day"] = joined.index.normalize()
            per_day = {}
            for day, group in joined.groupby("day"):
                per_day[day] = _interval_vwap_dev(
                    np.asarray(group["c"], dtype=float),
                    np.asarray(group["a"], dtype=float),
                    np.asarray(group["v"], dtype=float),
                    np.asarray(group.index, dtype="datetime64[ns]"),
                    s, e,
                )
            out[inst] = pd.Series(per_day, dtype=float)
        return pd.DataFrame(out).sort_index()


def _interval_illiq(close_v: np.ndarray, amt_v: np.ndarray, times: np.ndarray, start: int, end: int) -> float:
    mask = _interval_mask(times, start, end)
    r = log_returns(close_v[mask])
    a = amt_v[mask]
    # P1-105: an unknown (NaN) amount is NOT a zero-amount bar — pair only the
    # minutes where BOTH the return and the amount are finite.
    ok = np.isfinite(r) & np.isfinite(a)
    r_ok = r[ok]
    a_ok = a[ok]
    if len(r_ok) == 0:
        return np.nan
    denom = np.maximum(a_ok, _EPS)
    ratio = np.abs(r_ok) / denom
    return float(np.mean(ratio))


@register_operator(
    name="intra_interval_illiquidity",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_interval_illiquidity",
    source="intraday.time_structure",
    backend="pandas_numpy",
    status="experimental",
)
class IntraIntervalIlliquidity(SeriesOperator):
    """区间内 Amihud 非流动性 mean(|r|/max(amount,eps))。"""

    metadata = metadata(
        "intra_interval_illiquidity", "区间 Amihud 非流动性。",
        ["close", "amount", "start_minute", "end_minute"], unit="illiquidity",
    )

    def _calculate_series(self, close, amount, start_minute=570, end_minute=900, session_tz=None, **_):
        s, e = int(start_minute), int(end_minute)
        from cleaned_operators.intraday._core import session_local
        frame = session_local(close, session_tz)
        amt = session_local(amount, session_tz)
        # P0-08: fail closed on a session-grid mismatch instead of splicing the
        # axis; the dropna below only removes rows where the PRIMARY column is NaN.
        require_same_session_grid(frame, amt)
        out = {}
        for inst in frame.columns:
            joined = pd.concat([frame[inst], amt[inst]], axis=1, keys=["c", "a"]).dropna(subset=["c"])
            joined["day"] = joined.index.normalize()
            per_day = {}
            for day, group in joined.groupby("day"):
                per_day[day] = _interval_illiq(
                    np.asarray(group["c"], dtype=float),
                    np.asarray(group["a"], dtype=float),
                    np.asarray(group.index, dtype="datetime64[ns]"),
                    s, e,
                )
            out[inst] = pd.Series(per_day, dtype=float)
        return pd.DataFrame(out).sort_index()


# ---------------------------------------------------------------------------
# § Same-slot cross-day memory and profile distances
# ---------------------------------------------------------------------------

def _slot_matrix(values: pd.Series) -> pd.DataFrame:
    """Turn a per-stock minute series into a day x minute-of-day matrix."""
    frame = values.to_frame("v")
    frame["slot"] = minute_of_day(frame.index.to_numpy(dtype="datetime64[ns]"))
    frame["day"] = frame.index.normalize()
    return frame.pivot_table(index="day", columns="slot", values="v", aggfunc="mean")


def _same_slot_score(close: pd.DataFrame, window: int, reverse: bool) -> pd.DataFrame:
    w = max(2, int(window))
    out: dict[str, pd.Series] = {}
    for inst in close.columns:
        r = pd.Series(log_returns(close[inst].to_numpy(dtype=float)), index=close.index)
        mat = _slot_matrix(r)
        hist = mat.shift(1).rolling(w, min_periods=max(2, w // 2)).mean()
        if reverse:
            prod = mat * hist
            prod = prod.where(np.sign(mat) != np.sign(hist))
            score = -prod.sum(axis=1, min_count=1)
        else:
            score = (mat * hist).sum(axis=1, min_count=1)
        out[inst] = score
    return pd.DataFrame(out).sort_index()


@register_operator(
    name="intra_same_slot_momentum",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_same_slot_momentum",
    source="intraday.time_structure",
    backend="pandas_numpy",
    status="experimental",
)
class IntraSameSlotMomentum(SeriesOperator):
    """同日段跨日延续：Σ_m r_t,m * mean_k(r_{t-k,m})。"""

    metadata = metadata(
        "intra_same_slot_momentum", "同日段跨日延续得分。", ["close", "window"], unit="level", cost=7,
    )

    def _calculate_series(self, close, window=20, session_tz=None, **_):
        from cleaned_operators.intraday._core import session_local
        return _same_slot_score(session_local(close, session_tz), int(window), reverse=False)


@register_operator(
    name="intra_same_slot_reversal",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_same_slot_reversal",
    source="intraday.time_structure",
    backend="pandas_numpy",
    status="experimental",
)
class IntraSameSlotReversal(SeriesOperator):
    """同日段反向匹配：只统计与历史同槽位符号相反的槽位。"""

    metadata = metadata(
        "intra_same_slot_reversal", "同日段反向得分。", ["close", "window"], unit="level", cost=7,
    )

    def _calculate_series(self, close, window=20, session_tz=None, **_):
        from cleaned_operators.intraday._core import session_local
        return _same_slot_score(session_local(close, session_tz), int(window), reverse=True)


def _profile_cosine(mat: pd.DataFrame, window: int) -> pd.Series:
    w = max(2, int(window))
    hist = mat.shift(1).rolling(w, min_periods=max(2, w // 2)).mean()
    out: dict[pd.Timestamp, float] = {}
    for day in mat.index:
        a = mat.loc[day].to_numpy(dtype=float)
        b = hist.loc[day].to_numpy(dtype=float)
        valid = np.isfinite(a) & np.isfinite(b)
        if valid.sum() < 2:
            out[day] = np.nan
            continue
        va, vb = a[valid], b[valid]
        na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
        if na <= _EPS or nb <= _EPS:
            out[day] = np.nan
            continue
        out[day] = float(np.dot(va, vb) / (na * nb))
    return pd.Series(out, dtype=float)


def _profile_jsd(mat: pd.DataFrame, window: int) -> pd.Series:
    w = max(2, int(window))
    hist = mat.shift(1).rolling(w, min_periods=max(2, w // 2)).mean()
    out: dict[pd.Timestamp, float] = {}
    for day in mat.index:
        a = mat.loc[day].to_numpy(dtype=float)
        b = hist.loc[day].to_numpy(dtype=float)
        valid = np.isfinite(a) & np.isfinite(b)
        if valid.sum() < 2:
            out[day] = np.nan
            continue
        va, vb = a[valid], b[valid]
        pa = np.maximum(va, 0.0) + 1e-12
        pb = np.maximum(vb, 0.0) + 1e-12
        pa /= pa.sum()
        pb /= pb.sum()
        m = 0.5 * (pa + pb)
        with np_errstate():
            kl = float(np.sum(pa * np.log(pa / m)) + np.sum(pb * np.log(pb / m)))
        out[day] = 0.5 * kl
    return pd.Series(out, dtype=float)


def _profile_emd(mat: pd.DataFrame, window: int) -> pd.Series:
    w = max(2, int(window))
    hist = mat.shift(1).rolling(w, min_periods=max(2, w // 2)).mean()
    out: dict[pd.Timestamp, float] = {}
    for day in mat.index:
        a = mat.loc[day].to_numpy(dtype=float)
        b = hist.loc[day].to_numpy(dtype=float)
        valid = np.isfinite(a) & np.isfinite(b)
        if valid.sum() < 2:
            out[day] = np.nan
            continue
        va, vb = a[valid], b[valid]
        pa = np.maximum(va, 0.0) + 1e-12
        pb = np.maximum(vb, 0.0) + 1e-12
        pa /= pa.sum()
        pb /= pb.sum()
        cdf_a = np.cumsum(pa)
        cdf_b = np.cumsum(pb)
        out[day] = float(np.sum(np.abs(cdf_a - cdf_b)))
    return pd.Series(out, dtype=float)


def _within_day_log_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Per-column within-day log returns.

    The return at the first bar of each trading day is NaN so overnight gaps do
    not pollute the intraday return profile.  (The historic
    ``intra_return_profile_cosine`` fed whatever panel the caller passed
    straight into the profile kernel, so passing Close produced a *price-level*
    curve similarity instead of a return-curve one; these ops compute the
    returns explicitly.)
    """
    v = close.to_numpy(dtype=float)
    days = np.asarray(pd.DatetimeIndex(close.index).normalize())
    out = np.full_like(v, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(1, len(close.index)):
            if days[i] == days[i - 1]:
                out[i] = np.log(v[i] / v[i - 1])
    return pd.DataFrame(out, index=close.index, columns=close.columns)


def _return_profile_cosine(close: pd.DataFrame, window: int, absolute: bool) -> pd.DataFrame:
    rets = _within_day_log_returns(close)
    if absolute:
        rets = rets.abs()
    return _profile_daily(rets, int(window), _profile_cosine)


def _mk_profile_close_op(name: str, description: str, absolute: bool):
    @register_operator(
        name=name,
        category="intraday_microstructure",
        business_category="intraday_microstructure",
        canonical=name,
        source="intraday.time_structure",
        backend="pandas_numpy",
        status="experimental",
    )
    class _ProfileCloseOp(SeriesOperator):
        metadata = metadata(name, description, ["close", "window"], unit="cosine", cost=7)

        def _calculate_series(self, close, window=20, session_tz=None, **_):
            from cleaned_operators.intraday._core import session_local
            return _return_profile_cosine(session_local(close, session_tz), int(window), absolute)

    return _ProfileCloseOp


_mk_profile_close_op(
    "intra_signed_return_profile_cosine",
    "分钟有符号收益曲线与历史均值曲线余弦相似度（内部计算日内收益）。",
    False,
)
_mk_profile_close_op(
    "intra_abs_return_profile_cosine",
    "分钟绝对收益曲线与历史均值曲线余弦相似度（内部计算日内收益）。",
    True,
)


def _profile_daily(values: pd.DataFrame, window: int, fn: Callable[[pd.DataFrame, int], pd.Series]) -> pd.DataFrame:
    out: dict[str, pd.Series] = {}
    for inst in values.columns:
        mat = _slot_matrix(values[inst])
        out[inst] = fn(mat, int(window))
    return pd.DataFrame(out).sort_index()


def _mk_profile_op(name: str, description: str, unit: str, fn):
    @register_operator(
        name=name,
        category="intraday_microstructure",
        business_category="intraday_microstructure",
        canonical=name,
        source="intraday.time_structure",
        backend="pandas_numpy",
        status="experimental",
    )
    class _ProfileOp(SeriesOperator):
        metadata = metadata(name, description, ["x", "window"], unit=unit, cost=7)

        def _calculate_series(self, x, window=20, session_tz=None, **_):
            from cleaned_operators.intraday._core import session_local
            return _profile_daily(session_local(x, session_tz), int(window), fn)

    return _ProfileOp


_mk_profile_op("intra_return_profile_cosine", "分钟收益曲线与历史均值曲线余弦相似度。", "cosine", _profile_cosine)
_mk_profile_op("intra_volume_profile_cosine", "分钟成交量占比曲线余弦相似度。", "cosine", _profile_cosine)
_mk_profile_op("intra_amount_profile_cosine", "分钟成交额占比曲线余弦相似度。", "cosine", _profile_cosine)
_mk_profile_op("intra_volume_profile_jsd", "成交量分布与历史基准 JSD。", "jsd", _profile_jsd)
_mk_profile_op("intra_amount_profile_jsd", "成交额分布与历史基准 JSD。", "jsd", _profile_jsd)
_mk_profile_op("intra_profile_earth_mover_distance", "分布与历史基准 Wasserstein 距离（1D CDF 差）。", "distance", _profile_emd)


_CANONICALS.extend(
    [
        "intra_interval_return",
        "intra_interval_volume_share",
        "intra_interval_amount_share",
        "intra_interval_realized_variance",
        "intra_interval_vwap_deviation",
        "intra_interval_illiquidity",
        "intra_same_slot_momentum",
        "intra_same_slot_reversal",
        "intra_return_profile_cosine",
        "intra_volume_profile_cosine",
        "intra_amount_profile_cosine",
        "intra_volume_profile_jsd",
        "intra_amount_profile_jsd",
        "intra_profile_earth_mover_distance",
        "intra_signed_return_profile_cosine",
        "intra_abs_return_profile_cosine",
    ]
)

register_surface(_CANONICALS)
