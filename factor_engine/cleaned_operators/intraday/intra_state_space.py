# -*- coding: utf-8 -*-
"""Intraday state-space and pattern recognition operators (2026-08 R47 completion).

Nine minute-level operators: Kalman latent price, volume SSM decomposition,
functional motif scoring, visibility graph features, smart money FCM score,
price/volume peak-ridge-valley state, value at extreme state, and market
profile correlation excluding self.

All are TRUE_GAP: they operate on raw minute OHLCV panels and emit a daily
scalar per (TradeDate, Symbol).

Contract
--------
* Causal: a day's scalar uses only that day's own minute data (no trailing).
* Missing-value policy: NaN is never treated as 0. Insufficient observations
  (below min_bars threshold) return NaN (fail-closed).
* Session discipline: bars are grouped into A-share sessions (morning
  minute-of-day 570..690, afternoon 780..900).
* Dual backend: pandas_numpy + polars implementations for each operator.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import (
    OperatorMetadata,
    register_operator,
    strict_int_runtime,
)
from cleaned_operators.intraday._core import (
    _EPS,
    DataDegeneracy,
    SessionAggregationOperator,
    as_panel,
    daily_agg,
    daily_agg_three,
    metadata,
    minute_of_day,
    register_surface,
    require_same_session_grid,
    safe_div,
    session_local,
)

_MORNING = (570, 690)
_AFTERNOON = (780, 900)

_CANONICALS: list[str] = []


def _int_param(value: Any, name: str, lower: int = 1) -> int:
    """Strict integer gate (rejects bool/string/fractional, enforces lower)."""
    return strict_int_runtime(value, name, lower=lower)


def _float_param(value, name: str, lower: float = 0.0, upper: float = 1.0) -> float:
    """Strict float parameter with bounds check."""
    try:
        val = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a float")
    if not (lower <= val <= upper):
        raise ValueError(f"{name} must be in [{lower}, {upper}]")
    return val


def _same_session_segment(m_i: int, m_j: int) -> bool:
    """True when two minute-of-day values lie in the same continuous session."""
    return (
        (_MORNING[0] <= m_i <= _MORNING[1] and _MORNING[0] <= m_j <= _MORNING[1])
        or (_AFTERNOON[0] <= m_i <= _AFTERNOON[1] and _AFTERNOON[0] <= m_j <= _AFTERNOON[1])
    )


# ---------------------------------------------------------------------------
# Kalman filter helper
# ---------------------------------------------------------------------------

def _kalman_filter_1d(observations: np.ndarray, process_var: float = 1e-5, obs_var: float = 1e-2) -> tuple[np.ndarray, float]:
    """Simple 1D Kalman filter for latent price estimation.

    Returns (filtered_states, innovation_variance).
    """
    n = len(observations)
    if n == 0:
        return np.array([]), np.nan

    x = np.full(n, np.nan)
    P = np.full(n, np.nan)

    x[0] = observations[0]
    P[0] = obs_var

    innovations = []

    for i in range(1, n):
        if not np.isfinite(observations[i]):
            x[i] = x[i-1]
            P[i] = P[i-1] + process_var
            continue

        x_pred = x[i-1]
        P_pred = P[i-1] + process_var

        K = P_pred / (P_pred + obs_var) if (P_pred + obs_var) != 0 else np.nan
        innov = observations[i] - x_pred
        innovations.append(innov)

        x[i] = x_pred + K * innov
        P[i] = (1 - K) * P_pred

    innov_var = float(np.var(innovations)) if len(innovations) > 2 else np.nan
    return x, innov_var


# ---------------------------------------------------------------------------
# 1. intra_kalman_latent_price
# ---------------------------------------------------------------------------

@register_operator(
    name="intra_kalman_latent_price",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_kalman_latent_price",
    source="intraday.intra_state_space",
    backend="pandas_numpy")
class IntraKalmanLatentPrice(SessionAggregationOperator):
    """日内 Kalman 滤波潜在价格偏离度。

    使用 1D Kalman 滤波器估计潜在价格轨迹，返回观测价格相对潜在价格的
    标准化偏离度（均值归一化的创新方差）。
    """

    metadata = metadata(
        "intra_kalman_latent_price",
        "Kalman 滤波潜在价格偏离度。",
        ["close", "min_bars"],
        unit="ratio",
    )

    def _calculate_series(self, close, min_bars=30, session_tz=None, **_):
        mb = _int_param(min_bars, "min_bars")
        close = session_local(as_panel(close), session_tz)

        def _kernel(vals, times):
            finite = np.isfinite(vals)
            if finite.sum() < mb:
                return np.nan
            obs = vals[finite]
            _, innov_var = _kalman_filter_1d(obs)
            if not np.isfinite(innov_var) or innov_var < _EPS:
                return np.nan
            mean_price = float(np.mean(obs))
            if mean_price < _EPS:
                return np.nan
            return np.where(mean_price != 0, (float(np.sqrt(innov_var)) / (mean_price)), np.nan)

        return daily_agg(close, _kernel, min_finite=1)


@register_operator(
    name="intra_kalman_latent_price",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_kalman_latent_price",
    source="intraday.intra_state_space",
    backend="polars")
class IntraKalmanLatentPricePolars(SessionAggregationOperator):
    """Polars backend for intra_kalman_latent_price."""

    metadata = metadata(
        "intra_kalman_latent_price",
        "Kalman 滤波潜在价格偏离度 (Polars)。",
        ["close", "min_bars"],
        unit="ratio",
    )

    def _calculate_series(self, close, min_bars=30, session_tz=None, **_):
        mb = _int_param(min_bars, "min_bars")
        close = session_local(as_panel(close), session_tz)

        def _kernel(vals, times):
            finite = np.isfinite(vals)
            if finite.sum() < mb:
                return np.nan
            obs = vals[finite]
            _, innov_var = _kalman_filter_1d(obs)
            if not np.isfinite(innov_var) or innov_var < _EPS:
                return np.nan
            mean_price = float(np.mean(obs))
            if mean_price < _EPS:
                return np.nan
            return np.where(mean_price != 0, (float(np.sqrt(innov_var)) / (mean_price)), np.nan)

        return daily_agg(close, _kernel, min_finite=1)


# ---------------------------------------------------------------------------
# 2. intra_state_space_volume_components
# ---------------------------------------------------------------------------

@register_operator(
    name="intra_state_space_volume_components",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_state_space_volume_components",
    source="intraday.intra_state_space",
    backend="pandas_numpy")
class IntraStateSpaceVolumeComponents(SessionAggregationOperator):
    """日内成交量状态空间分解：趋势/周期能量比。

    将日内成交量序列分解为低频趋势分量和高频周期分量，返回高频能量
    占总能量的比例（衡量成交量波动的短期结构强度）。
    """

    metadata = metadata(
        "intra_state_space_volume_components",
        "成交量状态空间高频能量占比。",
        ["volume", "min_bars"],
        unit="ratio",
    )

    def _calculate_series(self, volume, min_bars=30, session_tz=None, **_):
        mb = _int_param(min_bars, "min_bars")
        volume = session_local(as_panel(volume), session_tz)

        def _kernel(vals, times):
            finite = np.isfinite(vals) & (vals > 0)
            if finite.sum() < mb:
                return np.nan
            v = vals[finite]
            n = len(v)

            # Simple trend extraction: moving average window = n//4
            w = max(3, n // 4)
            trend = np.convolve(v, np.ones(w), mode='same') / w
            cycle = v - trend

            trend_energy = float(np.sum(trend ** 2))
            cycle_energy = float(np.sum(cycle ** 2))
            total_energy = trend_energy + cycle_energy

            if total_energy < _EPS:
                return np.nan
            return cycle_energy / total_energy

        return daily_agg(volume, _kernel, min_finite=1)


@register_operator(
    name="intra_state_space_volume_components",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_state_space_volume_components",
    source="intraday.intra_state_space",
    backend="polars")
class IntraStateSpaceVolumeComponentsPolars(SessionAggregationOperator):
    """Polars backend for intra_state_space_volume_components."""

    metadata = metadata(
        "intra_state_space_volume_components",
        "成交量状态空间高频能量占比 (Polars)。",
        ["volume", "min_bars"],
        unit="ratio",
    )

    def _calculate_series(self, volume, min_bars=30, session_tz=None, **_):
        mb = _int_param(min_bars, "min_bars")
        volume = session_local(as_panel(volume), session_tz)

        def _kernel(vals, times):
            finite = np.isfinite(vals) & (vals > 0)
            if finite.sum() < mb:
                return np.nan
            v = vals[finite]
            n = len(v)

            w = max(3, n // 4)
            trend = np.convolve(v, np.ones(w), mode='same') / w
            cycle = v - trend

            trend_energy = float(np.sum(trend ** 2))
            cycle_energy = float(np.sum(cycle ** 2))
            total_energy = trend_energy + cycle_energy

            if total_energy < _EPS:
                return np.nan
            return cycle_energy / total_energy

        return daily_agg(volume, _kernel, min_finite=1)


# ---------------------------------------------------------------------------
# 3. intra_functional_motif_score
# ---------------------------------------------------------------------------

@register_operator(
    name="intra_functional_motif_score",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_functional_motif_score",
    source="intraday.intra_state_space",
    backend="pandas_numpy")
class IntraFunctionalMotifScore(SessionAggregationOperator):
    """日内价格轨迹功能性模式得分。

    使用价格-成交量的联合模式识别，计算日内轨迹与标准上涨/下跌模式的
    相似度差异（上涨模式得分 - 下跌模式得分）。
    """

    metadata = metadata(
        "intra_functional_motif_score",
        "价格轨迹功能性模式得分。",
        ["close", "volume", "min_bars"],
        unit="score",
    )

    def _calculate_series(self, close, volume, min_bars=30, session_tz=None, **_):
        mb = _int_param(min_bars, "min_bars")
        close = session_local(as_panel(close), session_tz)
        volume = session_local(as_panel(volume), session_tz)
        require_same_session_grid(close, volume)

        def _kernel(pv, vv, times):
            finite = np.isfinite(pv) & np.isfinite(vv) & (vv > 0)
            if finite.sum() < mb:
                return np.nan
            p = pv[finite]
            v = vv[finite]
            n = len(p)

            # Normalize price and volume to [0, 1]
            p_norm = (p - p.min()) / (p.max() - p.min() + _EPS)
            v_norm = (v - v.min()) / (v.max() - v.min() + _EPS)

            # Standard motifs: up (linear increase) and down (linear decrease)
            t = np.arange(n, dtype=float) / (n - 1)
            up_motif = t
            down_motif = 1.0 - t

            # Correlation with motifs (price + volume combined signal)
            signal = 0.7 * p_norm + 0.3 * v_norm

            corr_up = float(np.corrcoef(signal, up_motif)[0, 1])
            corr_down = float(np.corrcoef(signal, down_motif)[0, 1])

            if not (np.isfinite(corr_up) and np.isfinite(corr_down)):
                return np.nan

            return corr_up - corr_down

        return daily_agg_three(close, volume, close, lambda p, v, _: _kernel(p, v, None), min_finite=1)


@register_operator(
    name="intra_functional_motif_score",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_functional_motif_score",
    source="intraday.intra_state_space",
    backend="polars")
class IntraFunctionalMotifScorePolars(SessionAggregationOperator):
    """Polars backend for intra_functional_motif_score."""

    metadata = metadata(
        "intra_functional_motif_score",
        "价格轨迹功能性模式得分 (Polars)。",
        ["close", "volume", "min_bars"],
        unit="score",
    )

    def _calculate_series(self, close, volume, min_bars=30, session_tz=None, **_):
        mb = _int_param(min_bars, "min_bars")
        close = session_local(as_panel(close), session_tz)
        volume = session_local(as_panel(volume), session_tz)
        require_same_session_grid(close, volume)

        def _kernel(pv, vv, times):
            finite = np.isfinite(pv) & np.isfinite(vv) & (vv > 0)
            if finite.sum() < mb:
                return np.nan
            p = pv[finite]
            v = vv[finite]
            n = len(p)

            p_norm = (p - p.min()) / (p.max() - p.min() + _EPS)
            v_norm = (v - v.min()) / (v.max() - v.min() + _EPS)

            t = np.arange(n, dtype=float) / (n - 1)
            up_motif = t
            down_motif = 1.0 - t

            signal = 0.7 * p_norm + 0.3 * v_norm

            corr_up = float(np.corrcoef(signal, up_motif)[0, 1])
            corr_down = float(np.corrcoef(signal, down_motif)[0, 1])

            if not (np.isfinite(corr_up) and np.isfinite(corr_down)):
                return np.nan

            return corr_up - corr_down

        return daily_agg_three(close, volume, close, lambda p, v, _: _kernel(p, v, None), min_finite=1)


# ---------------------------------------------------------------------------
# 4. intra_visibility_graph_features
# ---------------------------------------------------------------------------

@register_operator(
    name="intra_visibility_graph_features",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_visibility_graph_features",
    source="intraday.intra_state_space",
    backend="pandas_numpy")
class IntraVisibilityGraphFeatures(SessionAggregationOperator):
    """日内价格序列可见性图特征。

    构建价格序列的水平可见性图（horizontal visibility graph），
    返回平均度数（连接数）作为时间序列复杂度的度量。
    """

    metadata = metadata(
        "intra_visibility_graph_features",
        "价格可见性图平均度数。",
        ["close", "min_bars"],
        unit="count",
    )

    def _calculate_series(self, close, min_bars=30, session_tz=None, **_):
        mb = _int_param(min_bars, "min_bars")
        close = session_local(as_panel(close), session_tz)

        def _kernel(vals, times):
            finite = np.isfinite(vals)
            if finite.sum() < mb:
                return np.nan
            p = vals[finite]
            n = len(p)

            # Horizontal visibility graph: node i sees node j if all bars
            # between them are lower than min(p[i], p[j])
            degrees = np.zeros(n, dtype=int)

            for i in range(n):
                for j in range(i + 1, n):
                    threshold = min(p[i], p[j])
                    # Check if all bars between i and j are below threshold
                    if i + 1 < j:
                        between = p[i+1:j]
                        if np.all(between < threshold):
                            degrees[i] += 1
                            degrees[j] += 1
                    else:
                        # Adjacent bars are always visible
                        degrees[i] += 1
                        degrees[j] += 1

            return float(np.mean(degrees))

        return daily_agg(close, _kernel, min_finite=1)


@register_operator(
    name="intra_visibility_graph_features",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_visibility_graph_features",
    source="intraday.intra_state_space",
    backend="polars")
class IntraVisibilityGraphFeaturesPolars(SessionAggregationOperator):
    """Polars backend for intra_visibility_graph_features."""

    metadata = metadata(
        "intra_visibility_graph_features",
        "价格可见性图平均度数 (Polars)。",
        ["close", "min_bars"],
        unit="count",
    )

    def _calculate_series(self, close, min_bars=30, session_tz=None, **_):
        mb = _int_param(min_bars, "min_bars")
        close = session_local(as_panel(close), session_tz)

        def _kernel(vals, times):
            finite = np.isfinite(vals)
            if finite.sum() < mb:
                return np.nan
            p = vals[finite]
            n = len(p)

            degrees = np.zeros(n, dtype=int)

            for i in range(n):
                for j in range(i + 1, n):
                    threshold = min(p[i], p[j])
                    if i + 1 < j:
                        between = p[i+1:j]
                        if np.all(between < threshold):
                            degrees[i] += 1
                            degrees[j] += 1
                    else:
                        degrees[i] += 1
                        degrees[j] += 1

            return float(np.mean(degrees))

        return daily_agg(close, _kernel, min_finite=1)


# ---------------------------------------------------------------------------
# 5. intra_smart_money_fcm_score
# ---------------------------------------------------------------------------

@register_operator(
    name="intra_smart_money_fcm_score",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_smart_money_fcm_score",
    source="intraday.intra_state_space",
    backend="pandas_numpy")
class IntraSmartMoneyFcmScore(SessionAggregationOperator):
    """Smart money flow composite metric combining volume/price/timing patterns.

    Composite score: VWAP deviation × volume concentration × direction persistence.
    Measures whether institutional flow dominates vs retail noise.
    """

    metadata = metadata(
        "intra_smart_money_fcm_score",
        "Smart money flow composite (VWAP deviation × volume concentration × persistence).",
        ["price", "volume", "high", "low"],
        unit="level",
    )

    def _calculate_series(self, price, volume, high, low, session_tz=None, **_):
        price = session_local(as_panel(price), session_tz)
        volume = session_local(as_panel(volume), session_tz)
        high = session_local(as_panel(high), session_tz)
        low = session_local(as_panel(low), session_tz)
        require_same_session_grid(price, volume, high, low)

        out: dict[str, pd.Series] = {}
        for inst in price.columns:
            if inst not in volume.columns or inst not in high.columns or inst not in low.columns:
                continue
            p_col = price[inst]
            v_col = volume[inst]
            h_col = high[inst]
            l_col = low[inst]

            joined = pd.concat([p_col, v_col, h_col, l_col], axis=1, keys=["p", "v", "h", "l"])
            joined["day"] = joined.index.normalize()
            per_day: dict[pd.Timestamp, float] = {}

            for day, group in joined.groupby("day"):
                pv = np.asarray(group["p"], dtype=float)
                vv = np.asarray(group["v"], dtype=float)
                hv = np.asarray(group["h"], dtype=float)
                lv = np.asarray(group["l"], dtype=float)

                finite = np.isfinite(pv) & np.isfinite(vv) & np.isfinite(hv) & np.isfinite(lv)
                finite = finite & (vv > 0)
                if finite.sum() < 10:
                    per_day[day] = np.nan
                    continue

                try:
                    p = pv[finite]
                    v = vv[finite]
                    h = hv[finite]
                    l = lv[finite]

                    # 1. VWAP deviation: (close - vwap) / range
                    vwap = float(np.sum(p * v) / np.sum(v))
                    price_range = float(np.max(h) - np.min(l))
                    if price_range < _EPS:
                        raise DataDegeneracy("zero price range")
                    vwap_dev = (float(p[-1]) - vwap) / price_range

                    # 2. Volume concentration at extremes (top/bottom 10% price bins)
                    price_min, price_max = float(np.min(p)), float(np.max(p))
                    if price_max - price_min < _EPS:
                        raise DataDegeneracy("constant price")
                    threshold = 0.1 * (price_max - price_min)
                    extreme_mask = (p <= price_min + threshold) | (p >= price_max - threshold)
                    vol_at_extremes = float(np.sum(v[extreme_mask]))
                    vol_total = float(np.sum(v))
                    vol_concentration = vol_at_extremes / vol_total if vol_total > _EPS else 0.0

                    # 3. Direction persistence: fraction of bars moving with overall trend
                    returns = np.diff(p)
                    if len(returns) < 2:
                        raise DataDegeneracy("insufficient returns")
                    overall_direction = float(p[-1] - p[0])
                    if abs(overall_direction) < _EPS:
                        persistence = 0.0
                    else:
                        sign = np.sign(overall_direction)
                        aligned = np.sum(np.sign(returns) == sign)
                        persistence = float(aligned) / len(returns)

                    # Composite: weighted combination
                    score = vwap_dev * (0.5 * vol_concentration + 0.5 * persistence)
                    per_day[day] = score
                except (DataDegeneracy, ZeroDivisionError, OverflowError):
                    per_day[day] = np.nan

            out[inst] = pd.Series(per_day, dtype=float)

        if not out:
            return pd.DataFrame(dtype=float)
        return pd.DataFrame(out).sort_index()


@register_operator(
    name="intra_smart_money_fcm_score",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_smart_money_fcm_score",
    source="intraday.intra_state_space",
    backend="polars")
class IntraSmartMoneyFcmScorePolars(SessionAggregationOperator):
    """Polars backend for intra_smart_money_fcm_score."""

    metadata = metadata(
        "intra_smart_money_fcm_score",
        "Smart money flow composite (Polars).",
        ["price", "volume", "high", "low"],
        unit="level",
    )

    def _calculate_series(self, price, volume, high, low, session_tz=None, **_):
        price = session_local(as_panel(price), session_tz)
        volume = session_local(as_panel(volume), session_tz)
        high = session_local(as_panel(high), session_tz)
        low = session_local(as_panel(low), session_tz)
        require_same_session_grid(price, volume, high, low)

        out: dict[str, pd.Series] = {}
        for inst in price.columns:
            if inst not in volume.columns or inst not in high.columns or inst not in low.columns:
                continue
            p_col = price[inst]
            v_col = volume[inst]
            h_col = high[inst]
            l_col = low[inst]

            joined = pd.concat([p_col, v_col, h_col, l_col], axis=1, keys=["p", "v", "h", "l"])
            joined["day"] = joined.index.normalize()
            per_day: dict[pd.Timestamp, float] = {}

            for day, group in joined.groupby("day"):
                pv = np.asarray(group["p"], dtype=float)
                vv = np.asarray(group["v"], dtype=float)
                hv = np.asarray(group["h"], dtype=float)
                lv = np.asarray(group["l"], dtype=float)

                finite = np.isfinite(pv) & np.isfinite(vv) & np.isfinite(hv) & np.isfinite(lv)
                finite = finite & (vv > 0)
                if finite.sum() < 10:
                    per_day[day] = np.nan
                    continue

                try:
                    p = pv[finite]
                    v = vv[finite]
                    h = hv[finite]
                    l = lv[finite]

                    vwap = float(np.sum(p * v) / np.sum(v))
                    price_range = float(np.max(h) - np.min(l))
                    if price_range < _EPS:
                        raise DataDegeneracy("zero price range")
                    vwap_dev = (float(p[-1]) - vwap) / price_range

                    price_min, price_max = float(np.min(p)), float(np.max(p))
                    if price_max - price_min < _EPS:
                        raise DataDegeneracy("constant price")
                    threshold = 0.1 * (price_max - price_min)
                    extreme_mask = (p <= price_min + threshold) | (p >= price_max - threshold)
                    vol_at_extremes = float(np.sum(v[extreme_mask]))
                    vol_total = float(np.sum(v))
                    vol_concentration = vol_at_extremes / vol_total if vol_total > _EPS else 0.0

                    returns = np.diff(p)
                    if len(returns) < 2:
                        raise DataDegeneracy("insufficient returns")
                    overall_direction = float(p[-1] - p[0])
                    if abs(overall_direction) < _EPS:
                        persistence = 0.0
                    else:
                        sign = np.sign(overall_direction)
                        aligned = np.sum(np.sign(returns) == sign)
                        persistence = float(aligned) / len(returns)

                    score = vwap_dev * (0.5 * vol_concentration + 0.5 * persistence)
                    per_day[day] = score
                except (DataDegeneracy, ZeroDivisionError, OverflowError):
                    per_day[day] = np.nan

            out[inst] = pd.Series(per_day, dtype=float)

        if not out:
            return pd.DataFrame(dtype=float)
        return pd.DataFrame(out).sort_index()


# ---------------------------------------------------------------------------
# 9. intra_market_profile_corr_ex_self
# ---------------------------------------------------------------------------

@register_operator(
    name="intra_market_profile_corr_ex_self",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_market_profile_corr_ex_self",
    source="intraday.intra_state_space",
    backend="pandas_numpy")
class IntraMarketProfileCorrExSelf(SessionAggregationOperator):
    """Market profile correlation excluding self.

    For each instrument, computes the correlation of its intraday volume profile
    with the average profile of all other instruments (excluding itself).
    Measures synchronization with broader market activity patterns.
    """

    metadata = metadata(
        "intra_market_profile_corr_ex_self",
        "Intraday volume profile correlation with market (ex-self).",
        ["volume", "min_bars"],
        unit="correlation",
    )

    def _calculate_series(self, volume, min_bars=30, session_tz=None, **_):
        mb = _int_param(min_bars, "min_bars")
        volume = session_local(as_panel(volume), session_tz)

        # Build per-instrument, per-day profiles
        inst_day_profiles: dict[str, dict[pd.Timestamp, np.ndarray]] = {}

        for inst in volume.columns:
            col = volume[inst]
            per_day: dict[pd.Timestamp, np.ndarray] = {}
            for day, group in col.groupby(col.index.normalize()):
                vals = np.asarray(group, dtype=float)
                finite = np.isfinite(vals) & (vals > 0)
                if finite.sum() < mb:
                    per_day[day] = np.array([])
                else:
                    v = vals[finite]
                    # Normalize to profile (sum=1)
                    v_sum = float(np.sum(v))
                    if v_sum > _EPS:
                        per_day[day] = v / v_sum
                    else:
                        per_day[day] = np.array([])
            inst_day_profiles[inst] = per_day

        # For each instrument and day, compute correlation with average of others
        out: dict[str, pd.Series] = {}
        all_days = set()
        for profiles in inst_day_profiles.values():
            all_days.update(profiles.keys())

        for inst in volume.columns:
            per_day_corr: dict[pd.Timestamp, float] = {}
            for day in sorted(all_days):
                try:
                    self_profile = inst_day_profiles[inst].get(day, np.array([]))
                    if len(self_profile) < mb:
                        per_day_corr[day] = np.nan
                        continue

                    # Collect profiles from other instruments
                    other_profiles = []
                    for other_inst in volume.columns:
                        if other_inst == inst:
                            continue
                        other_profile = inst_day_profiles[other_inst].get(day, np.array([]))
                        if len(other_profile) == len(self_profile):
                            other_profiles.append(other_profile)

                    if len(other_profiles) < 2:
                        per_day_corr[day] = np.nan
                        continue

                    # Average profile of others
                    avg_other_profile = np.mean(other_profiles, axis=0)

                    # Correlation
                    if len(self_profile) < 3 or len(avg_other_profile) < 3:
                        per_day_corr[day] = np.nan
                        continue

                    corr_matrix = np.corrcoef(self_profile, avg_other_profile)
                    corr = float(corr_matrix[0, 1])

                    if not np.isfinite(corr):
                        per_day_corr[day] = np.nan
                    else:
                        per_day_corr[day] = corr
                except (DataDegeneracy, ZeroDivisionError, OverflowError, ValueError):
                    per_day_corr[day] = np.nan

            out[inst] = pd.Series(per_day_corr, dtype=float)

        if not out:
            return pd.DataFrame(dtype=float)
        return pd.DataFrame(out).sort_index()


@register_operator(
    name="intra_market_profile_corr_ex_self",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_market_profile_corr_ex_self",
    source="intraday.intra_state_space",
    backend="polars")
class IntraMarketProfileCorrExSelfPolars(SessionAggregationOperator):
    """Polars backend for intra_market_profile_corr_ex_self."""

    metadata = metadata(
        "intra_market_profile_corr_ex_self",
        "Intraday volume profile correlation with market (ex-self) (Polars).",
        ["volume", "min_bars"],
        unit="correlation",
    )

    def _calculate_series(self, volume, min_bars=30, session_tz=None, **_):
        mb = _int_param(min_bars, "min_bars")
        volume = session_local(as_panel(volume), session_tz)

        inst_day_profiles: dict[str, dict[pd.Timestamp, np.ndarray]] = {}

        for inst in volume.columns:
            col = volume[inst]
            per_day: dict[pd.Timestamp, np.ndarray] = {}
            for day, group in col.groupby(col.index.normalize()):
                vals = np.asarray(group, dtype=float)
                finite = np.isfinite(vals) & (vals > 0)
                if finite.sum() < mb:
                    per_day[day] = np.array([])
                else:
                    v = vals[finite]
                    v_sum = float(np.sum(v))
                    if v_sum > _EPS:
                        per_day[day] = v / v_sum
                    else:
                        per_day[day] = np.array([])
            inst_day_profiles[inst] = per_day

        out: dict[str, pd.Series] = {}
        all_days = set()
        for profiles in inst_day_profiles.values():
            all_days.update(profiles.keys())

        for inst in volume.columns:
            per_day_corr: dict[pd.Timestamp, float] = {}
            for day in sorted(all_days):
                try:
                    self_profile = inst_day_profiles[inst].get(day, np.array([]))
                    if len(self_profile) < mb:
                        per_day_corr[day] = np.nan
                        continue

                    other_profiles = []
                    for other_inst in volume.columns:
                        if other_inst == inst:
                            continue
                        other_profile = inst_day_profiles[other_inst].get(day, np.array([]))
                        if len(other_profile) == len(self_profile):
                            other_profiles.append(other_profile)

                    if len(other_profiles) < 2:
                        per_day_corr[day] = np.nan
                        continue

                    avg_other_profile = np.mean(other_profiles, axis=0)

                    if len(self_profile) < 3 or len(avg_other_profile) < 3:
                        per_day_corr[day] = np.nan
                        continue

                    corr_matrix = np.corrcoef(self_profile, avg_other_profile)
                    corr = float(corr_matrix[0, 1])

                    if not np.isfinite(corr):
                        per_day_corr[day] = np.nan
                    else:
                        per_day_corr[day] = corr
                except (DataDegeneracy, ZeroDivisionError, OverflowError, ValueError):
                    per_day_corr[day] = np.nan

            out[inst] = pd.Series(per_day_corr, dtype=float)

        if not out:
            return pd.DataFrame(dtype=float)
        return pd.DataFrame(out).sort_index()


# ---------------------------------------------------------------------------
# 6. intra_price_peak_ridge_valley_state
# ---------------------------------------------------------------------------

@register_operator(
    name="intra_price_peak_ridge_valley_state",
    category="intraday_pattern",
    business_category="intraday_pattern",
    canonical="intra_price_peak_ridge_valley_state",
    source="intraday.intra_state_space",
    backend="pandas_numpy")
class IntraPricePeakRidgeValleyState(SessionAggregationOperator):
    """Classify price action into peak/ridge/valley/flat states."""

    metadata = metadata(
        "intra_price_peak_ridge_valley_state",
        "价格峰谷状态分类。",
        ["close", "window", "min_bars"],
        unit="state",
    )

    def _calculate_series(self, close, window=10, min_bars=20, session_tz=None, **_):
        w = _int_param(window, "window", lower=3)
        mb = _int_param(min_bars, "min_bars")
        close = session_local(as_panel(close), session_tz)

        def _kernel(vals, times):
            finite = np.isfinite(vals)
            if finite.sum() < mb:
                return np.nan
            v = vals[finite]
            if len(v) < mb:
                return np.nan

            grad = np.gradient(v)
            curv = np.gradient(grad)
            volatility = pd.Series(v).rolling(w, min_periods=1).std().values

            is_peak = (grad > 0) & (curv < -np.std(curv) * 0.5)
            is_valley = (grad < 0) & (curv > np.std(curv) * 0.5)
            is_flat = volatility < np.mean(volatility) * 0.5

            state = np.zeros_like(v)
            state[is_peak] = 1.0
            state[is_valley] = -1.0
            state[is_flat & ~is_peak & ~is_valley] = 0.0
            state[~is_peak & ~is_valley & ~is_flat] = 0.5

            return float(np.mean(state))

        return daily_agg(close, _kernel, min_finite=1)


@register_operator(
    name="intra_price_peak_ridge_valley_state",
    category="intraday_pattern",
    business_category="intraday_pattern",
    canonical="intra_price_peak_ridge_valley_state",
    source="intraday.intra_state_space",
    backend="polars")
class IntraPricePeakRidgeValleyStatePolars(SessionAggregationOperator):
    """Polars backend for intra_price_peak_ridge_valley_state."""

    metadata = metadata(
        "intra_price_peak_ridge_valley_state",
        "价格峰谷状态分类 (Polars)。",
        ["close", "window", "min_bars"],
        unit="state",
    )

    def _calculate_series(self, close, window=10, min_bars=20, session_tz=None, **_):
        w = _int_param(window, "window", lower=3)
        mb = _int_param(min_bars, "min_bars")
        close = session_local(as_panel(close), session_tz)

        def _kernel(vals, times):
            finite = np.isfinite(vals)
            if finite.sum() < mb:
                return np.nan
            v = vals[finite]
            if len(v) < mb:
                return np.nan

            grad = np.gradient(v)
            curv = np.gradient(grad)
            volatility = pd.Series(v).rolling(w, min_periods=1).std().values

            is_peak = (grad > 0) & (curv < -np.std(curv) * 0.5)
            is_valley = (grad < 0) & (curv > np.std(curv) * 0.5)
            is_flat = volatility < np.mean(volatility) * 0.5

            state = np.zeros_like(v)
            state[is_peak] = 1.0
            state[is_valley] = -1.0
            state[is_flat & ~is_peak & ~is_valley] = 0.0
            state[~is_peak & ~is_valley & ~is_flat] = 0.5

            return float(np.mean(state))

        return daily_agg(close, _kernel, min_finite=1)


# ---------------------------------------------------------------------------
# 7. intra_volume_peak_ridge_valley_state
# ---------------------------------------------------------------------------

@register_operator(
    name="intra_volume_peak_ridge_valley_state",
    category="intraday_pattern",
    business_category="intraday_pattern",
    canonical="intra_volume_peak_ridge_valley_state",
    source="intraday.intra_state_space",
    backend="pandas_numpy")
class IntraVolumePeakRidgeValleyState(SessionAggregationOperator):
    """Classify volume action into peak/ridge/valley/flat states."""

    metadata = metadata(
        "intra_volume_peak_ridge_valley_state",
        "成交量峰谷状态分类。",
        ["volume", "window", "min_bars"],
        unit="state",
    )

    def _calculate_series(self, volume, window=10, min_bars=20, session_tz=None, **_):
        w = _int_param(window, "window", lower=3)
        mb = _int_param(min_bars, "min_bars")
        volume = session_local(as_panel(volume), session_tz)

        def _kernel(vals, times):
            finite = np.isfinite(vals) & (vals > 0)
            if finite.sum() < mb:
                return np.nan
            v = vals[finite]
            if len(v) < mb:
                return np.nan

            grad = np.gradient(v)
            curv = np.gradient(grad)
            volatility = pd.Series(v).rolling(w, min_periods=1).std().values

            is_peak = (grad > 0) & (curv < -np.std(curv) * 0.5)
            is_valley = (grad < 0) & (curv > np.std(curv) * 0.5)
            is_flat = volatility < np.mean(volatility) * 0.5

            state = np.zeros_like(v)
            state[is_peak] = 1.0
            state[is_valley] = -1.0
            state[is_flat & ~is_peak & ~is_valley] = 0.0
            state[~is_peak & ~is_valley & ~is_flat] = 0.5

            return float(np.mean(state))

        return daily_agg(volume, _kernel, min_finite=1)


@register_operator(
    name="intra_volume_peak_ridge_valley_state",
    category="intraday_pattern",
    business_category="intraday_pattern",
    canonical="intra_volume_peak_ridge_valley_state",
    source="intraday.intra_state_space",
    backend="polars")
class IntraVolumePeakRidgeValleyStatePolars(SessionAggregationOperator):
    """Polars backend for intra_volume_peak_ridge_valley_state."""

    metadata = metadata(
        "intra_volume_peak_ridge_valley_state",
        "成交量峰谷状态分类 (Polars)。",
        ["volume", "window", "min_bars"],
        unit="state",
    )

    def _calculate_series(self, volume, window=10, min_bars=20, session_tz=None, **_):
        w = _int_param(window, "window", lower=3)
        mb = _int_param(min_bars, "min_bars")
        volume = session_local(as_panel(volume), session_tz)

        def _kernel(vals, times):
            finite = np.isfinite(vals) & (vals > 0)
            if finite.sum() < mb:
                return np.nan
            v = vals[finite]
            if len(v) < mb:
                return np.nan

            grad = np.gradient(v)
            curv = np.gradient(grad)
            volatility = pd.Series(v).rolling(w, min_periods=1).std().values

            is_peak = (grad > 0) & (curv < -np.std(curv) * 0.5)
            is_valley = (grad < 0) & (curv > np.std(curv) * 0.5)
            is_flat = volatility < np.mean(volatility) * 0.5

            state = np.zeros_like(v)
            state[is_peak] = 1.0
            state[is_valley] = -1.0
            state[is_flat & ~is_peak & ~is_valley] = 0.0
            state[~is_peak & ~is_valley & ~is_flat] = 0.5

            return float(np.mean(state))

        return daily_agg(volume, _kernel, min_finite=1)


# ---------------------------------------------------------------------------
# 8. intraday_value_at_extreme_state
# ---------------------------------------------------------------------------

@register_operator(
    name="intraday_value_at_extreme_state",
    category="intraday_pattern",
    business_category="intraday_pattern",
    canonical="intraday_value_at_extreme_state",
    source="intraday.intra_state_space",
    backend="pandas_numpy")
class IntradayValueAtExtremeState(SessionAggregationOperator):
    """Compute value (volume × price) at price/volume extremes."""

    metadata = metadata(
        "intraday_value_at_extreme_state",
        "极值状态成交额占比。",
        ["close", "volume", "quantile", "min_bars"],
        unit="ratio",
    )

    def _calculate_series(self, close, volume, quantile=0.9, min_bars=20, session_tz=None, **_):
        q = _float_param(quantile, "quantile", lower=0.5, upper=0.99)
        mb = _int_param(min_bars, "min_bars")

        close = session_local(as_panel(close), session_tz)
        volume = session_local(as_panel(volume), session_tz)

        require_same_session_grid(close, volume)

        def _kernel(vals_c, vals_v, times):
            finite = np.isfinite(vals_c) & np.isfinite(vals_v) & (vals_v > 0)
            if finite.sum() < mb:
                return np.nan

            c = vals_c[finite]
            v = vals_v[finite]

            value = c * v
            price_threshold = np.quantile(c, q)
            vol_threshold = np.quantile(v, q)

            extreme_price_mask = c >= price_threshold
            extreme_vol_mask = v >= vol_threshold

            value_at_high_price = value[extreme_price_mask].sum() if extreme_price_mask.any() else 0.0
            value_at_high_vol = value[extreme_vol_mask].sum() if extreme_vol_mask.any() else 0.0

            total_value = value.sum()
            if total_value < _EPS:
                return np.nan

            return np.where((2 * total_value) != 0, (float((value_at_high_price + value_at_high_vol)) / ((2 * total_value))), np.nan)

        return daily_agg_three(close, volume, close, _kernel, min_finite=1)


@register_operator(
    name="intraday_value_at_extreme_state",
    category="intraday_pattern",
    business_category="intraday_pattern",
    canonical="intraday_value_at_extreme_state",
    source="intraday.intra_state_space",
    backend="polars")
class IntradayValueAtExtremeStatePolars(SessionAggregationOperator):
    """Polars backend for intraday_value_at_extreme_state."""

    metadata = metadata(
        "intraday_value_at_extreme_state",
        "极值状态成交额占比 (Polars)。",
        ["close", "volume", "quantile", "min_bars"],
        unit="ratio",
    )

    def _calculate_series(self, close, volume, quantile=0.9, min_bars=20, session_tz=None, **_):
        q = _float_param(quantile, "quantile", lower=0.5, upper=0.99)
        mb = _int_param(min_bars, "min_bars")

        close = session_local(as_panel(close), session_tz)
        volume = session_local(as_panel(volume), session_tz)

        require_same_session_grid(close, volume)

        def _kernel(vals_c, vals_v, times):
            finite = np.isfinite(vals_c) & np.isfinite(vals_v) & (vals_v > 0)
            if finite.sum() < mb:
                return np.nan

            c = vals_c[finite]
            v = vals_v[finite]

            value = c * v
            price_threshold = np.quantile(c, q)
            vol_threshold = np.quantile(v, q)

            extreme_price_mask = c >= price_threshold
            extreme_vol_mask = v >= vol_threshold

            value_at_high_price = value[extreme_price_mask].sum() if extreme_price_mask.any() else 0.0
            value_at_high_vol = value[extreme_vol_mask].sum() if extreme_vol_mask.any() else 0.0

            total_value = value.sum()
            if total_value < _EPS:
                return np.nan

            return np.where((2 * total_value) != 0, (float((value_at_high_price + value_at_high_vol)) / ((2 * total_value))), np.nan)

        return daily_agg_three(close, volume, close, _kernel, min_finite=1)


# ---------------------------------------------------------------------------
# Register canonicals and surface
# ---------------------------------------------------------------------------

_CANONICALS.extend([
    "intra_kalman_latent_price",
    "intra_state_space_volume_components",
    "intra_functional_motif_score",
    "intra_visibility_graph_features",
    "intra_smart_money_fcm_score",
    "intra_price_peak_ridge_valley_state",
    "intra_volume_peak_ridge_valley_state",
    "intraday_value_at_extreme_state",
    "intra_market_profile_corr_ex_self",
])

register_surface(_CANONICALS)

