# -*- coding: utf-8 -*-
"""Intraday VWAP-path and drawdown / recovery operators (P0).

Minute Close (and Amount / Volume for VWAP terms) panels in, daily panels out.
Cumulative VWAP is computed from the day's own amount/volume only; price-path
operators use the minute close path.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import SeriesOperator, register_operator
from cleaned_operators.intraday._core import (
    _EPS,
    daily_agg,
    daily_agg_three,
    metadata,
    np_errstate,
    register_surface,
)

_CANONICALS: list[str] = []


def _cum_vwap(close_v: np.ndarray, amt_v: np.ndarray, vol_v: np.ndarray) -> np.ndarray:
    vol = np.where(np.isfinite(vol_v), vol_v, 0.0)
    amt = np.where(np.isfinite(amt_v), amt_v, 0.0)
    cum_v = np.cumsum(vol)
    cum_a = np.cumsum(amt)
    with np_errstate():
        out = np.where(cum_v > _EPS, cum_a / cum_v, np.nan)
    return out


def _path_slope(cum_vwap: np.ndarray, degree: int, coeff_idx: int) -> float:
    finite = np.isfinite(cum_vwap)
    y = cum_vwap[finite]
    n = len(y)
    if n < degree + 2:
        return np.nan
    t = np.linspace(0.0, 1.0, n)
    design = np.column_stack([t ** k for k in range(degree + 1)])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(beta[coeff_idx])


def _vwap_path_common(close_v, amt_v, vol_v, degree: int, coeff_idx: int) -> float:
    c = close_v[np.isfinite(close_v)]
    if len(c) < 2:
        return np.nan
    cv = _cum_vwap(c, amt_v, vol_v)
    cv = cv[: len(c)]
    return _path_slope(cv, degree, coeff_idx)


@register_operator(
    name="intra_vwap_path_slope",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_vwap_path_slope",
    source="intraday.vwap_path",
    backend="pandas_numpy",
    status="experimental",
)
class IntraVwapPathSlope(SeriesOperator):
    """累计 VWAP 路径对标准化时间线性回归斜率。"""

    metadata = metadata(
        "intra_vwap_path_slope", "累计 VWAP 路径斜率。", ["close", "amount", "volume"], unit="price",
    )

    def _calculate_series(self, close, amount, volume, **_):
        return daily_agg_three(close, amount, volume, lambda a, b, c: _vwap_path_common(a, b, c, 1, 1))


@register_operator(
    name="intra_vwap_path_curvature",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_vwap_path_curvature",
    source="intraday.vwap_path",
    backend="pandas_numpy",
    status="experimental",
)
class IntraVwapPathCurvature(SeriesOperator):
    """累计 VWAP 路径对时间二次回归的二次项系数。"""

    metadata = metadata(
        "intra_vwap_path_curvature", "累计 VWAP 路径曲率。", ["close", "amount", "volume"], unit="price",
    )

    def _calculate_series(self, close, amount, volume, **_):
        return daily_agg_three(close, amount, volume, lambda a, b, c: _vwap_path_common(a, b, c, 2, 2))


def _vwap_excursion(close_v, amt_v, vol_v, side: str) -> float:
    c = close_v[np.isfinite(close_v)]
    if len(c) < 2:
        return np.nan
    cv = _cum_vwap(c, amt_v, vol_v)
    cv = cv[: len(c)]
    valid = np.isfinite(cv) & (cv > _EPS)
    if valid.sum() == 0:
        return np.nan
    dev = c[valid] / cv[valid] - 1.0
    return float(np.max(dev)) if side == "max" else float(np.min(dev))


@register_operator(
    name="intra_price_vwap_max_positive_excursion",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_price_vwap_max_positive_excursion",
    source="intraday.vwap_path",
    backend="pandas_numpy",
    status="experimental",
)
class IntraPriceVwapMaxPositiveExcursion(SeriesOperator):
    """价格相对累计 VWAP 的最大正偏离 max(Close/CumVWAP-1)。"""

    metadata = metadata(
        "intra_price_vwap_max_positive_excursion", "相对累计 VWAP 最大正偏离。", ["close", "amount", "volume"], unit="ratio",
    )

    def _calculate_series(self, close, amount, volume, **_):
        return daily_agg_three(close, amount, volume, lambda a, b, c: _vwap_excursion(a, b, c, "max"))


@register_operator(
    name="intra_price_vwap_max_negative_excursion",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_price_vwap_max_negative_excursion",
    source="intraday.vwap_path",
    backend="pandas_numpy",
    status="experimental",
)
class IntraPriceVwapMaxNegativeExcursion(SeriesOperator):
    """价格相对累计 VWAP 的最大负偏离 min(Close/CumVWAP-1)。"""

    metadata = metadata(
        "intra_price_vwap_max_negative_excursion", "相对累计 VWAP 最大负偏离。", ["close", "amount", "volume"], unit="ratio",
    )

    def _calculate_series(self, close, amount, volume, **_):
        return daily_agg_three(close, amount, volume, lambda a, b, c: _vwap_excursion(a, b, c, "min"))


def _time_above_vwap(close_v, amt_v, vol_v) -> float:
    c = close_v[np.isfinite(close_v)]
    if len(c) < 2:
        return np.nan
    cv = _cum_vwap(c, amt_v, vol_v)[: len(c)]
    valid = np.isfinite(cv)
    if valid.sum() == 0:
        return np.nan
    return float(np.mean(c[valid] > cv[valid]))


@register_operator(
    name="intra_time_above_vwap",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_time_above_vwap",
    source="intraday.vwap_path",
    backend="pandas_numpy",
    status="experimental",
)
class IntraTimeAboveVwap(SeriesOperator):
    """全天价格高于累计 VWAP 的分钟占比。"""

    metadata = metadata("intra_time_above_vwap", "高于累计 VWAP 的分钟比例。", ["close", "amount", "volume"], unit="ratio")

    def _calculate_series(self, close, amount, volume, **_):
        return daily_agg_three(close, amount, volume, lambda a, b, c: _time_above_vwap(a, b, c))


def _longest_streak(close_v, amt_v, vol_v, side: str) -> float:
    c = close_v[np.isfinite(close_v)]
    if len(c) < 2:
        return np.nan
    cv = _cum_vwap(c, amt_v, vol_v)[: len(c)]
    valid = np.isfinite(cv)
    if valid.sum() == 0:
        return np.nan
    above = c[valid] > cv[valid]
    if side == "below":
        above = ~above
    best = cur = 0
    for flag in above:
        cur = cur + 1 if flag else 0
        best = max(best, cur)
    return float(best)


@register_operator(
    name="intra_longest_above_vwap_streak",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_longest_above_vwap_streak",
    source="intraday.vwap_path",
    backend="pandas_numpy",
    status="experimental",
)
class IntraLongestAboveVwapStreak(SeriesOperator):
    """连续高于累计 VWAP 的最长分钟数。"""

    metadata = metadata(
        "intra_longest_above_vwap_streak", "高于 VWAP 最长连续分钟数。", ["close", "amount", "volume"], unit="count",
    )

    def _calculate_series(self, close, amount, volume, **_):
        return daily_agg_three(close, amount, volume, lambda a, b, c: _longest_streak(a, b, c, "above"))


@register_operator(
    name="intra_longest_below_vwap_streak",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_longest_below_vwap_streak",
    source="intraday.vwap_path",
    backend="pandas_numpy",
    status="experimental",
)
class IntraLongestBelowVwapStreak(SeriesOperator):
    """连续低于累计 VWAP 的最长分钟数。"""

    metadata = metadata(
        "intra_longest_below_vwap_streak", "低于 VWAP 最长连续分钟数。", ["close", "amount", "volume"], unit="count",
    )

    def _calculate_series(self, close, amount, volume, **_):
        return daily_agg_three(close, amount, volume, lambda a, b, c: _longest_streak(a, b, c, "below"))


def _vwap_reversion_speed(close_v, amt_v, vol_v) -> float:
    c = close_v[np.isfinite(close_v)]
    if len(c) < 5:
        return np.nan
    cv = _cum_vwap(c, amt_v, vol_v)[: len(c)]
    valid = np.isfinite(cv)
    d = (c[valid] / cv[valid] - 1.0)[1:]
    dprev = (c[valid] / cv[valid] - 1.0)[:-1]
    if len(d) < 3 or np.std(dprev) <= _EPS:
        return np.nan
    b = float(np.cov(d, dprev)[0, 1] / np.var(dprev))
    return b


@register_operator(
    name="intra_vwap_reversion_speed",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_vwap_reversion_speed",
    source="intraday.vwap_path",
    backend="pandas_numpy",
    status="experimental",
)
class IntraVwapReversionSpeed(SeriesOperator):
    """价格偏离累计 VWAP 后的一阶自回归系数（负值表示回归）。"""

    metadata = metadata(
        "intra_vwap_reversion_speed", "VWAP 偏离 AR(1) 系数。", ["close", "amount", "volume"], unit="level",
    )

    def _calculate_series(self, close, amount, volume, **_):
        return daily_agg_three(close, amount, volume, lambda a, b, c: _vwap_reversion_speed(a, b, c))


def _max_drawdown(close_v: np.ndarray, side: str) -> float:
    finite = close_v[np.isfinite(close_v)]
    if len(finite) < 2:
        return np.nan
    with np_errstate():
        if side == "down":
            running = np.maximum.accumulate(finite)
            path = finite / running - 1.0
            return float(np.min(path))
        running = np.minimum.accumulate(finite)
        path = finite / running - 1.0
        return float(np.max(path))


@register_operator(
    name="intra_max_drawdown",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_max_drawdown",
    source="intraday.vwap_path",
    backend="pandas_numpy",
    status="experimental",
)
class IntraMaxDrawdown(SeriesOperator):
    """分钟价格路径最大回撤（负值）。"""

    metadata = metadata("intra_max_drawdown", "日内最大回撤。", ["close"], unit="ratio")

    def _calculate_series(self, close, **_):
        return daily_agg(close, lambda v, t: _max_drawdown(v, "down"))


@register_operator(
    name="intra_max_drawup",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_max_drawup",
    source="intraday.vwap_path",
    backend="pandas_numpy",
    status="experimental",
)
class IntraMaxDrawup(SeriesOperator):
    """分钟价格路径最大上涨段。"""

    metadata = metadata("intra_max_drawup", "日内最大上涨段。", ["close"], unit="ratio")

    def _calculate_series(self, close, **_):
        return daily_agg(close, lambda v, t: _max_drawdown(v, "up"))


def _drawdown_depth(close_v: np.ndarray) -> float:
    finite = close_v[np.isfinite(close_v)]
    if len(finite) < 2:
        return np.nan
    peak_idx = int(np.argmax(finite))
    trough_idx = int(np.argmin(finite[peak_idx:]) + peak_idx)
    if peak_idx == trough_idx or finite[peak_idx] <= _EPS:
        return np.nan
    return float(finite[trough_idx] / finite[peak_idx] - 1.0)


def _drawdown_duration(close_v: np.ndarray) -> float:
    finite = close_v[np.isfinite(close_v)]
    n = len(finite)
    if n < 2:
        return np.nan
    peak_idx = int(np.argmax(finite))
    trough_idx = int(np.argmin(finite[peak_idx:]) + peak_idx)
    if peak_idx == trough_idx:
        return 0.0
    return float(trough_idx - peak_idx)


def _drawdown_recovery_half_life(close_v: np.ndarray) -> float:
    finite = close_v[np.isfinite(close_v)]
    n = len(finite)
    if n < 2:
        return np.nan
    peak_idx = int(np.argmax(finite))
    trough_idx = int(np.argmin(finite[peak_idx:]) + peak_idx)
    peak, trough = finite[peak_idx], finite[trough_idx]
    if trough <= _EPS or peak <= trough:
        return np.nan
    halfway = trough + 0.5 * (peak - trough)
    recovered = np.flatnonzero(finite[trough_idx:] >= halfway)
    if len(recovered) == 0:
        return np.nan
    return float(recovered[0])


@register_operator(
    name="intra_drawdown_depth",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_drawdown_depth",
    source="intraday.vwap_path",
    backend="pandas_numpy",
    status="experimental",
)
class IntraDrawdownDepth(SeriesOperator):
    """最大回撤深度：峰到谷（负值）。"""

    metadata = metadata("intra_drawdown_depth", "最大回撤深度。", ["close"], unit="ratio")

    def _calculate_series(self, close, **_):
        return daily_agg(close, lambda v, t: _drawdown_depth(v))


@register_operator(
    name="intra_drawdown_duration",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_drawdown_duration",
    source="intraday.vwap_path",
    backend="pandas_numpy",
    status="experimental",
)
class IntraDrawdownDuration(SeriesOperator):
    """最大回撤持续分钟数（峰到谷）。"""

    metadata = metadata("intra_drawdown_duration", "最大回撤持续期。", ["close"], unit="count")

    def _calculate_series(self, close, **_):
        return daily_agg(close, lambda v, t: _drawdown_duration(v))


@register_operator(
    name="intra_drawdown_recovery_half_life",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_drawdown_recovery_half_life",
    source="intraday.vwap_path",
    backend="pandas_numpy",
    status="experimental",
)
class IntraDrawdownRecoveryHalfLife(SeriesOperator):
    """回撤后恢复到一半深度所需分钟数。"""

    metadata = metadata("intra_drawdown_recovery_half_life", "回撤半恢复期。", ["close"], unit="count")

    def _calculate_series(self, close, **_):
        return daily_agg(close, lambda v, t: _drawdown_recovery_half_life(v))


_CANONICALS.extend(
    [
        "intra_vwap_path_slope",
        "intra_vwap_path_curvature",
        "intra_price_vwap_max_positive_excursion",
        "intra_price_vwap_max_negative_excursion",
        "intra_time_above_vwap",
        "intra_longest_above_vwap_streak",
        "intra_longest_below_vwap_streak",
        "intra_vwap_reversion_speed",
        "intra_max_drawdown",
        "intra_max_drawup",
        "intra_drawdown_depth",
        "intra_drawdown_duration",
        "intra_drawdown_recovery_half_life",
    ]
)

register_surface(_CANONICALS)
