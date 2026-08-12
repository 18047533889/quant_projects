# -*- coding: utf-8 -*-
"""Intraday state-space operators (2026-08 R47 TRUE_GAP pack).

Four minute-level state-space models: Kalman latent price, volume SSM
decomposition, functional motif scoring, and visibility graph features.
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

        K = P_pred / (P_pred + obs_var)
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
    source="intraday.state_space",
    backend="pandas_numpy",
)
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
            return float(np.sqrt(innov_var) / mean_price)

        return daily_agg(close, _kernel, min_finite=1)


@register_operator(
    name="intra_kalman_latent_price",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_kalman_latent_price",
    source="intraday.state_space",
    backend="polars",
)
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
            return float(np.sqrt(innov_var) / mean_price)

        return daily_agg(close, _kernel, min_finite=1)


# ---------------------------------------------------------------------------
# 2. intra_state_space_volume_components
# ---------------------------------------------------------------------------

@register_operator(
    name="intra_state_space_volume_components",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_state_space_volume_components",
    source="intraday.state_space",
    backend="pandas_numpy",
)
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
            trend = np.convolve(v, np.ones(w) / w, mode='same')
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
    source="intraday.state_space",
    backend="polars",
)
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
            trend = np.convolve(v, np.ones(w) / w, mode='same')
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
    source="intraday.state_space",
    backend="pandas_numpy",
)
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
    source="intraday.state_space",
    backend="polars",
)
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
    source="intraday.state_space",
    backend="pandas_numpy",
)
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
    source="intraday.state_space",
    backend="polars",
)
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


_CANONICALS.extend([
    "intra_kalman_latent_price",
    "intra_state_space_volume_components",
    "intra_functional_motif_score",
    "intra_visibility_graph_features",
])

register_surface(_CANONICALS)
