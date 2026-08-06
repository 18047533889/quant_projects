# -*- coding: utf-8 -*-
"""Realized higher-order moments and jump decomposition (P0).

Minute Close panels in, daily panels out.  All kernels are causal and PIT-safe;
empty / degenerate days yield NaN.  Threshold-based jump classification uses a
per-bar bipower-free scale ``sqrt(RV / N)`` so the jump split is monotone in
the day's own realized variance and needs no auxiliary dataset.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from cleaned_operators.base import SeriesOperator, register_operator
from cleaned_operators.intraday._core import (
    _EPS,
    daily_agg,
    log_returns,
    metadata,
    np_errstate,
    register_surface,
    tripower_scale,
)

_CANONICALS: list[str] = []


def _realized_var(r: np.ndarray) -> float:
    with np_errstate():
        return float(np.nansum(r * r))


def _bipower_var(r: np.ndarray) -> float:
    finite = r[np.isfinite(r)]
    if len(finite) < 2:
        return np.nan
    with np_errstate():
        return float((np.pi / 2.0) * np.nansum(np.abs(finite[1:]) * np.abs(finite[:-1])))


def _jump_mask(r: np.ndarray, threshold_scale: float) -> np.ndarray:
    """Per-bar significant-jump mask ``|r| > threshold_scale * sqrt(RV/N)``.

    ``threshold_scale`` is measured in per-bar standard deviations of the day's
    own realized volatility; the default 3.0 yields roughly one jump bar per
    trading day for an i.i.d. Gaussian minute process (0.27% two-sided tail).
    """
    finite = r[np.isfinite(r)]
    n = len(finite)
    mask = np.zeros(len(r), dtype=bool)
    if n < 2:
        return mask
    rv = float(np.sum(finite * finite))
    if not np.isfinite(rv) or rv <= _EPS:
        return mask
    per_bar_vol = float(np.sqrt(rv / n))
    if per_bar_vol <= _EPS:
        return mask
    thresh = float(threshold_scale) * per_bar_vol
    with np_errstate():
        mask = np.abs(r) > thresh
    return mask


# ---------------------------------------------------------------------------
# § Realized moments
# ---------------------------------------------------------------------------


def _realized_skewness(close_v: np.ndarray) -> float:
    r = log_returns(close_v)
    r2 = np.nansum(r * r)
    if not np.isfinite(r2) or r2 <= _EPS:
        return np.nan
    n = int(np.sum(np.isfinite(r)))
    r3 = np.nansum(r ** 3)
    return float(np.sqrt(n) * r3 / r2 ** 1.5)


@register_operator(
    name="intra_realized_skewness",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_realized_skewness",
    source="intraday.higher_moments",
    backend="pandas_numpy",
    status="experimental",
)
class IntraRealizedSkewness(SeriesOperator):
    """日内已实现偏度 sqrt(N)*sum(r^3)/(sum(r^2))^(3/2)。"""

    metadata = metadata("intra_realized_skewness", "日内已实现偏度 RSK。", ["close"], unit="level")

    def _calculate_series(self, close, **_):
        return daily_agg(close, lambda v, t: _realized_skewness(v))


def _realized_kurtosis(close_v: np.ndarray) -> float:
    r = log_returns(close_v)
    r2 = np.nansum(r * r)
    if not np.isfinite(r2) or r2 <= _EPS:
        return np.nan
    n = int(np.sum(np.isfinite(r)))
    r4 = np.nansum(r ** 4)
    return float(n * r4 / (r2 * r2))


@register_operator(
    name="intra_realized_kurtosis",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_realized_kurtosis",
    source="intraday.higher_moments",
    backend="pandas_numpy",
    status="experimental",
)
class IntraRealizedKurtosis(SeriesOperator):
    """日内已实现峰度 N*sum(r^4)/(sum(r^2))^2。"""

    metadata = metadata("intra_realized_kurtosis", "日内已实现峰度 RKT。", ["close"], unit="level")

    def _calculate_series(self, close, **_):
        return daily_agg(close, lambda v, t: _realized_kurtosis(v))


def _realized_quarticity(close_v: np.ndarray) -> float:
    r = log_returns(close_v)
    n = int(np.sum(np.isfinite(r)))
    if n < 2:
        return np.nan
    return float(n / 3.0 * np.nansum(r ** 4))


@register_operator(
    name="intra_realized_quarticity",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_realized_quarticity",
    source="intraday.higher_moments",
    backend="pandas_numpy",
    status="experimental",
)
class IntraRealizedQuarticity(SeriesOperator):
    """日内已实现四次变差 RQ = N/3 * sum(r^4)。"""

    metadata = metadata("intra_realized_quarticity", "日内已实现四次变差。", ["close"], unit="quarticity")

    def _calculate_series(self, close, **_):
        return daily_agg(close, lambda v, t: _realized_quarticity(v))


def _tripower_quarticity(close_v: np.ndarray) -> float:
    r = log_returns(close_v)
    finite = r[np.isfinite(r)]
    if len(finite) < 4:
        return np.nan
    with np_errstate():
        prod = np.abs(finite[2:]) ** (4.0 / 3.0) * np.abs(finite[1:-1]) ** (4.0 / 3.0) * np.abs(finite[:-2]) ** (4.0 / 3.0)
        total = float(np.nansum(prod))
    n = len(finite)
    return tripower_scale() * n * total


@register_operator(
    name="intra_tripower_quarticity",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_tripower_quarticity",
    source="intraday.higher_moments",
    backend="pandas_numpy",
    status="experimental",
)
class IntraTripowerQuarticity(SeriesOperator):
    """日内三次幂四次变差（跳跃稳健）。"""

    metadata = metadata("intra_tripower_quarticity", "Tripower quarticity 估计。", ["close"], unit="quarticity")

    def _calculate_series(self, close, **_):
        return daily_agg(close, lambda v, t: _tripower_quarticity(v))


# ---------------------------------------------------------------------------
# § Continuous / jump variance decomposition
# ---------------------------------------------------------------------------

def _continuous_and_jump(close_v: np.ndarray) -> tuple[float, float]:
    r = log_returns(close_v)
    rv = _realized_var(r)
    bv = _bipower_var(r)
    if not np.isfinite(rv) or not np.isfinite(bv) or rv <= _EPS:
        return np.nan, np.nan
    cont = float(min(rv, bv))
    jump = float(max(rv - bv, 0.0))
    return cont, jump


@register_operator(
    name="intra_continuous_variance",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_continuous_variance",
    source="intraday.higher_moments",
    backend="pandas_numpy",
    status="experimental",
)
class IntraContinuousVariance(SeriesOperator):
    """日内连续方差分量 min(RV, BV)。"""

    metadata = metadata("intra_continuous_variance", "日内连续方差分量 min(RV,BV)。", ["close"], unit="variance")

    def _calculate_series(self, close, **_):
        def _fn(v, t):
            c, _j = _continuous_and_jump(v)
            return c

        return daily_agg(close, _fn)


@register_operator(
    name="intra_jump_variation",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_jump_variation",
    source="intraday.higher_moments",
    backend="pandas_numpy",
    status="experimental",
)
class IntraJumpVariation(SeriesOperator):
    """日内跳跃方差分量 max(RV-BV, 0)。"""

    metadata = metadata("intra_jump_variation", "日内跳跃方差分量 max(RV-BV,0)。", ["close"], unit="variance")

    def _calculate_series(self, close, **_):
        def _fn(v, t):
            c, j = _continuous_and_jump(v)
            return j

        return daily_agg(close, _fn)


def _signed_jump_stats(close_v: np.ndarray, threshold_scale: float) -> tuple[float, float, float, float, float, float, float]:
    """Return (pos_jv, neg_jv, count, concentration, first_t, last_t, cluster_cv)."""
    r = log_returns(close_v)
    mask = _jump_mask(r, threshold_scale)
    rj = r[mask]
    if len(rj) == 0:
        return np.nan, np.nan, 0.0, np.nan, np.nan, np.nan, np.nan
    with np_errstate():
        pos = float(np.nansum(rj[rj > 0] ** 2))
        neg = float(np.nansum(rj[rj < 0] ** 2))
    total = pos + neg
    concentration = np.nan
    if total > _EPS:
        w = rj * rj / total
        concentration = float(np.sum(w * w))
    idx = np.flatnonzero(mask)
    n_bars = int(np.sum(np.isfinite(r)))
    first_t = float(idx[0]) / (n_bars - 1) if n_bars > 1 else np.nan
    last_t = float(idx[-1]) / (n_bars - 1) if n_bars > 1 else np.nan
    cluster_cv = np.nan
    if len(idx) > 2:
        gaps = np.diff(idx).astype(float)
        mean_gap = float(np.mean(gaps))
        if mean_gap > _EPS:
            cluster_cv = float(np.std(gaps) / mean_gap)
    return pos, neg, float(len(idx)), concentration, first_t, last_t, cluster_cv


@register_operator(
    name="intra_positive_jump_variation",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_positive_jump_variation",
    source="intraday.higher_moments",
    backend="pandas_numpy",
    status="experimental",
)
class IntraPositiveJumpVariation(SeriesOperator):
    """日内正向跳跃平方和（阈值判定）。"""

    metadata = metadata(
        "intra_positive_jump_variation", "日内正跳跃方差分量。", ["close", "threshold_scale"], unit="variance"
    )

    def _calculate_series(self, close, threshold_scale=3.0, **_):
        ts = float(threshold_scale)

        def _fn(v, t):
            return _signed_jump_stats(v, ts)[0]

        return daily_agg(close, _fn)


@register_operator(
    name="intra_negative_jump_variation",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_negative_jump_variation",
    source="intraday.higher_moments",
    backend="pandas_numpy",
    status="experimental",
)
class IntraNegativeJumpVariation(SeriesOperator):
    """日内负向跳跃平方和（阈值判定）。"""

    metadata = metadata(
        "intra_negative_jump_variation", "日内负跳跃方差分量。", ["close", "threshold_scale"], unit="variance"
    )

    def _calculate_series(self, close, threshold_scale=3.0, **_):
        ts = float(threshold_scale)

        def _fn(v, t):
            return _signed_jump_stats(v, ts)[1]

        return daily_agg(close, _fn)


@register_operator(
    name="intra_signed_jump_ratio",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_signed_jump_ratio",
    source="intraday.higher_moments",
    backend="pandas_numpy",
    status="experimental",
)
class IntraSignedJumpRatio(SeriesOperator):
    """有符号跳跃比 (posJV-negJV)/(jump_variation+eps)。"""

    metadata = metadata(
        "intra_signed_jump_ratio", "有符号跳跃比。", ["close", "threshold_scale"], unit="ratio"
    )

    def _calculate_series(self, close, threshold_scale=3.0, **_):
        ts = float(threshold_scale)

        def _fn(v, t):
            pos, neg, *_ = _signed_jump_stats(v, ts)
            if not np.isfinite(pos) or not np.isfinite(neg):
                return np.nan
            denom = pos + neg + _EPS
            return float((pos - neg) / denom)

        return daily_agg(close, _fn)


@register_operator(
    name="intra_jump_count",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_jump_count",
    source="intraday.higher_moments",
    backend="pandas_numpy",
    status="experimental",
)
class IntraJumpCount(SeriesOperator):
    """日内跳跃分钟数量（阈值判定）。"""

    metadata = metadata("intra_jump_count", "日内跳跃分钟数。", ["close", "threshold_scale"], unit="count")

    def _calculate_series(self, close, threshold_scale=3.0, **_):
        ts = float(threshold_scale)

        def _fn(v, t):
            return _signed_jump_stats(v, ts)[2]

        return daily_agg(close, _fn)


@register_operator(
    name="intra_jump_concentration",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_jump_concentration",
    source="intraday.higher_moments",
    backend="pandas_numpy",
    status="experimental",
)
class IntraJumpConcentration(SeriesOperator):
    """跳跃集中度 sum(jump_share_i^2)，单根集中为 1、分散趋近 0。"""

    metadata = metadata(
        "intra_jump_concentration", "跳跃平方份额 HHI。", ["close", "threshold_scale"], unit="hhi"
    )

    def _calculate_series(self, close, threshold_scale=3.0, **_):
        ts = float(threshold_scale)

        def _fn(v, t):
            return _signed_jump_stats(v, ts)[3]

        return daily_agg(close, _fn)


@register_operator(
    name="intra_jump_first_time",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_jump_first_time",
    source="intraday.higher_moments",
    backend="pandas_numpy",
    status="experimental",
)
class IntraJumpFirstTime(SeriesOperator):
    """首次跳跃的标准化时点（0=首根, 1=末根）。"""

    metadata = metadata(
        "intra_jump_first_time", "首次跳跃位置/有效分钟数。", ["close", "threshold_scale"], unit="position"
    )

    def _calculate_series(self, close, threshold_scale=3.0, **_):
        ts = float(threshold_scale)

        def _fn(v, t):
            return _signed_jump_stats(v, ts)[4]

        return daily_agg(close, _fn)


@register_operator(
    name="intra_jump_last_time",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_jump_last_time",
    source="intraday.higher_moments",
    backend="pandas_numpy",
    status="experimental",
)
class IntraJumpLastTime(SeriesOperator):
    """末次跳跃的标准化时点（0=首根, 1=末根）。"""

    metadata = metadata(
        "intra_jump_last_time", "末次跳跃位置/有效分钟数。", ["close", "threshold_scale"], unit="position"
    )

    def _calculate_series(self, close, threshold_scale=3.0, **_):
        ts = float(threshold_scale)

        def _fn(v, t):
            return _signed_jump_stats(v, ts)[5]

        return daily_agg(close, _fn)


@register_operator(
    name="intra_jump_clustering",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_jump_clustering",
    source="intraday.higher_moments",
    backend="pandas_numpy",
    status="experimental",
)
class IntraJumpClustering(SeriesOperator):
    """跳跃事件间隔变异系数（>=3 次跳跃时定义）。"""

    metadata = metadata(
        "intra_jump_clustering", "跳跃间隔 CV。", ["close", "threshold_scale"], unit="level"
    )

    def _calculate_series(self, close, threshold_scale=3.0, **_):
        ts = float(threshold_scale)

        def _fn(v, t):
            return _signed_jump_stats(v, ts)[6]

        return daily_agg(close, _fn)


_CANONICALS.extend(
    [
        "intra_realized_skewness",
        "intra_realized_kurtosis",
        "intra_realized_quarticity",
        "intra_tripower_quarticity",
        "intra_continuous_variance",
        "intra_jump_variation",
        "intra_positive_jump_variation",
        "intra_negative_jump_variation",
        "intra_signed_jump_ratio",
        "intra_jump_count",
        "intra_jump_concentration",
        "intra_jump_first_time",
        "intra_jump_last_time",
        "intra_jump_clustering",
    ]
)

register_surface(_CANONICALS)
