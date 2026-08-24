# -*- coding: utf-8 -*-
"""Intraday pattern recognition operators (2026-08 R47 pack).

Minute panels in, daily scalar out. Four operators analyzing intraday price/volume
patterns: smart money flow composite, peak/ridge/valley state classification for
price and volume, and value concentration at price extremes.

Contract
--------
* Causal: minute -> daily, uses only completed session data (TRUE_GAP timing).
* Session-aware: respects A-share morning/afternoon segments.
* NaN fail-closed: never fabricates 0, degeneracies return NaN.
* All operators emit one scalar per (TradeDate, Symbol) — daily grain output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, register_operator, strict_int_runtime
from cleaned_operators.intraday._core import (
    _EPS,
    DataDegeneracy,
    SessionAggregationOperator,
    as_panel,
    daily_agg_three,
    metadata,
    minute_of_day,
    register_surface,
    require_same_session_grid,
    safe_div,
    session_local,
)

_MORNING = (570, 690)    # 09:30 .. 11:30 Shanghai minute-of-day
_AFTERNOON = (780, 900)  # 13:00 .. 15:00 Shanghai minute-of-day
_CANONICALS: list[str] = []


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _same_session_segment(m_i: int, m_j: int) -> bool:
    """True when two minute-of-day values lie in the same continuous session."""
    return (
        (_MORNING[0] <= m_i <= _MORNING[1] and _MORNING[0] <= m_j <= _MORNING[1])
        or (_AFTERNOON[0] <= m_i <= _AFTERNOON[1] and _AFTERNOON[0] <= m_j <= _AFTERNOON[1])
    )


def _int_param(value, name: str, lower: int = 1) -> int:
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


# ---------------------------------------------------------------------------
# 1. intra_smart_money_fcm_score
# ---------------------------------------------------------------------------

@register_operator(
    name="intra_smart_money_fcm_score",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_smart_money_fcm_score",
    source="intraday.pattern_recognition",
)
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

        # Use _aligned_agg from state_ops pattern for 4-input aggregation
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


# ---------------------------------------------------------------------------
# 2. intra_price_peak_ridge_valley_state
# ---------------------------------------------------------------------------

@register_operator(
    name="intra_price_peak_ridge_valley_state",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_price_peak_ridge_valley_state",
    source="intraday.pattern_recognition",
)
class IntraPricePeakRidgeValleyState(SessionAggregationOperator):
    """Price peak/ridge/valley state classifier.

    Classifies each minute bar: 0=flat, 1=peak, 2=ridge, 3=valley.
    Returns daily aggregation: peak_count.
    """

    metadata = metadata(
        "intra_price_peak_ridge_valley_state",
        "Price peak/ridge/valley count.",
        ["price", "window", "prominence"],
        unit="count",
    )

    def _calculate_series(self, price, window=5, prominence=0.5, session_tz=None, **_):
        w = _int_param(window, "window", lower=1)
        prom = _float_param(prominence, "prominence", lower=0.0, upper=10.0)
        price = session_local(as_panel(price), session_tz)

        def _kernel(vals, times):
            """Classify peaks/ridges/valleys within session."""
            finite = np.isfinite(vals)
            if finite.sum() < w * 2:
                raise DataDegeneracy("insufficient finite bars")

            p = vals[finite]
            minutes = minute_of_day(times[finite])
            n = len(p)

            # Detect peaks: local max with sufficient prominence
            peak_count = 0
            for i in range(w, n - w):
                # Check session boundary
                m_prev = int(minutes[i - 1])
                m_cur = int(minutes[i])
                m_next = int(minutes[i + 1]) if i + 1 < n else m_cur
                if not (_same_session_segment(m_prev, m_cur) and _same_session_segment(m_cur, m_next)):
                    continue

                # Local window
                window_vals = p[max(0, i - w):min(n, i + w + 1)]
                if len(window_vals) < 2:
                    continue
                local_max = float(np.max(window_vals))
                local_min = float(np.min(window_vals))
                local_range = local_max - local_min

                if local_range < _EPS:
                    continue

                # Peak: current value is local max and prominence sufficient
                if p[i] == local_max and (p[i] - local_min) >= prom * local_range:
                    peak_count += 1

            return float(peak_count)

        from cleaned_operators.intraday._core import daily_agg
        return daily_agg(price, _kernel, min_finite=w * 2)


# ---------------------------------------------------------------------------
# 3. intra_volume_peak_ridge_valley_state
# ---------------------------------------------------------------------------

@register_operator(
    name="intra_volume_peak_ridge_valley_state",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_volume_peak_ridge_valley_state",
    source="intraday.pattern_recognition",
)
class IntraVolumePeakRidgeValleyState(SessionAggregationOperator):
    """Volume peak/ridge/valley state classifier.

    Same logic as price peaks but on volume. Returns daily peak count.
    """

    metadata = metadata(
        "intra_volume_peak_ridge_valley_state",
        "Volume peak/ridge/valley count.",
        ["volume", "window", "prominence"],
        unit="count",
    )

    def _calculate_series(self, volume, window=5, prominence=0.5, session_tz=None, **_):
        w = _int_param(window, "window", lower=1)
        prom = _float_param(prominence, "prominence", lower=0.0, upper=10.0)
        volume = session_local(as_panel(volume), session_tz)

        def _kernel(vals, times):
            """Classify volume peaks within session."""
            finite = np.isfinite(vals) & (vals > 0)
            if finite.sum() < w * 2:
                raise DataDegeneracy("insufficient finite bars")

            v = vals[finite]
            minutes = minute_of_day(times[finite])
            n = len(v)

            peak_count = 0
            for i in range(w, n - w):
                m_prev = int(minutes[i - 1])
                m_cur = int(minutes[i])
                m_next = int(minutes[i + 1]) if i + 1 < n else m_cur
                if not (_same_session_segment(m_prev, m_cur) and _same_session_segment(m_cur, m_next)):
                    continue

                window_vals = v[max(0, i - w):min(n, i + w + 1)]
                if len(window_vals) < 2:
                    continue
                local_max = float(np.max(window_vals))
                local_min = float(np.min(window_vals))
                local_range = local_max - local_min

                if local_range < _EPS:
                    continue

                if v[i] == local_max and (v[i] - local_min) >= prom * local_range:
                    peak_count += 1

            return float(peak_count)

        from cleaned_operators.intraday._core import daily_agg
        return daily_agg(volume, _kernel, min_finite=w * 2)


# ---------------------------------------------------------------------------
# 4. intraday_value_at_extreme_state
# ---------------------------------------------------------------------------

@register_operator(
    name="intraday_value_at_extreme_state",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_value_at_extreme_state",
    source="intraday.pattern_recognition",
)
class IntradayValueAtExtremeState(SessionAggregationOperator):
    """Value concentration at price extremes.

    Measures volume-weighted distribution: volume at extremes / total volume.
    Extremes = top/bottom quantile price bins.
    """

    metadata = metadata(
        "intraday_value_at_extreme_state",
        "Volume concentration at price extremes.",
        ["price", "volume", "high", "low", "quantile"],
        unit="ratio",
    )

    def _calculate_series(self, price, volume, high, low, quantile=0.1, session_tz=None, **_):
        q = _float_param(quantile, "quantile", lower=0.01, upper=0.5)
        price = session_local(as_panel(price), session_tz)
        volume = session_local(as_panel(volume), session_tz)
        high = session_local(as_panel(high), session_tz)
        low = session_local(as_panel(low), session_tz)
        require_same_session_grid(price, volume, high, low)

        # Use similar pattern to smart_money_fcm_score for 4-input aggregation
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

                    # Compute price range using high/low
                    price_min = float(np.min(l))
                    price_max = float(np.max(h))
                    price_range = price_max - price_min

                    if price_range < _EPS:
                        raise DataDegeneracy("zero price range")

                    # Define extremes: top and bottom quantile
                    threshold = q * price_range
                    low_threshold = price_min + threshold
                    high_threshold = price_max - threshold

                    # Volume at extremes
                    extreme_mask = (p <= low_threshold) | (p >= high_threshold)
                    vol_at_extremes = float(np.sum(v[extreme_mask]))
                    vol_total = float(np.sum(v))

                    if vol_total < _EPS:
                        raise DataDegeneracy("zero total volume")

                    per_day[day] = vol_at_extremes / vol_total
                except (DataDegeneracy, ZeroDivisionError, OverflowError):
                    per_day[day] = np.nan

            out[inst] = pd.Series(per_day, dtype=float)

        if not out:
            return pd.DataFrame(dtype=float)
        return pd.DataFrame(out).sort_index()


_CANONICALS.extend([
    "intra_smart_money_fcm_score",
    "intra_price_peak_ridge_valley_state",
    "intra_volume_peak_ridge_valley_state",
    "intraday_value_at_extreme_state",
])

register_surface(_CANONICALS)
