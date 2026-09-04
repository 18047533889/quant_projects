# -*- coding: utf-8 -*-
"""TRUE_GAP intraday operators batch 3 (2026-08-13).

Implements session-aware intraday operators with EOD completion principle:
1. intra_session_mean_reversion - session mean reversion speed
2. intra_price_delay - intraday price delay score
3. intra_volume_imbalance - session volume imbalance measure

Contract
--------
* Session-aware: respects trading day boundaries and session structure
* EOD completion: signals complete at close, available next trading day
* PIT-safe: strictly causal, uses only completed data
* TRUE_GAP semantics: honors actual trading calendar structure
* Note: intra_same_slot_zscore exists in polars_native/intraday_batch3.py
* Note: intra_realized_variance exists in microstructure/intraday_agg.py
* Note: intra_state_vwap exists in intraday/state_ops.py
* Note: intra_smart_money_vwap_ratio exists in intraday/smart_money.py

All operators: minute -> daily aggregation, scope=session_intraday.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import register_operator
from factor_engine.cleaned_operators.intraday._core import (
    DataDegeneracy,
    SessionAggregationOperator,
    _EPS,
    as_panel,
    daily_agg,
    daily_agg_two,
    daily_agg_three,
    metadata,
    minute_of_day,
    np_errstate,
    register_surface,
)

__all__ = [
    "IntraSessionMeanReversion",
    "IntraPriceDelay",
    "IntraVolumeImbalance",
]

_CANONICALS = [
    "intra_session_mean_reversion",
    "intra_price_delay",
    "intra_volume_imbalance",
]


# ---------------------------------------------------------------------------
# 1. intra_session_mean_reversion: session mean reversion speed
# ---------------------------------------------------------------------------

def _session_mean_reversion_kernel(vals: np.ndarray, times: np.ndarray) -> float:
    """Compute intraday mean reversion speed to session VWAP.

    TRUE_GAP: measures how quickly prices revert to session VWAP,
    respecting trading session structure (no lunch/overnight bridge).

    Returns: negative of lag-1 autocorrelation of VWAP deviations.
    """
    finite = vals[np.isfinite(vals)]
    if len(finite) < 5:
        raise DataDegeneracy("insufficient data for mean reversion")

    # Compute session VWAP (equal-weighted as proxy)
    vwap = float(np.mean(finite))

    # Deviations from VWAP
    deviations = finite - vwap

    if np.std(deviations) <= _EPS:
        raise DataDegeneracy("constant deviations")

    # Lag-1 autocorrelation of deviations
    with np_errstate():
        corr = float(np.corrcoef(deviations[:-1], deviations[1:])[0, 1])

    if not np.isfinite(corr):
        raise DataDegeneracy("undefined correlation")

    # Negative autocorrelation = mean reversion
    # Return -corr so positive values indicate mean reversion
    mean_reversion = -corr

    return float(mean_reversion)


@register_operator(
    name="intra_session_mean_reversion",
    category="intraday_microstructure",
    business_category="intraday_true_gap",
    canonical="intra_session_mean_reversion",
    source="intraday.true_gap_batch3",
    backend="pandas_numpy",
    status="extended",
)
class IntraSessionMeanReversion(SessionAggregationOperator):
    """Session mean reversion speed (TRUE_GAP).

    Measures how quickly prices revert to session VWAP. Positive values
    indicate mean-reverting behavior. Respects calendar boundaries.
    """

    metadata = metadata(
        "intra_session_mean_reversion",
        "Session mean reversion speed (VWAP deviation autocorrelation).",
        ["close"],
        unit="correlation",
        cost=5,
        extra_tags=["true_gap", "session_aware", "microstructure"],
    )

    def _calculate_series(self, close, **_):
        """Calculate mean reversion from minute close prices."""
        return daily_agg(
            close,
            _session_mean_reversion_kernel,  # raw kernel: carries __vec__ (PERF-2)
            min_finite=5,
        )


# ---------------------------------------------------------------------------
# 2. intra_price_delay: intraday price delay score
# ---------------------------------------------------------------------------

def _price_delay_kernel(vals: np.ndarray, volume: np.ndarray, times: np.ndarray) -> float:
    """Compute intraday price delay (information incorporation speed).

    TRUE_GAP: measures how quickly prices adjust to volume shocks within
    the session, respecting actual trading timeline.

    Returns: lag-1 autocorrelation of volume-weighted returns.
    """
    finite_mask = np.isfinite(vals) & np.isfinite(volume) & (volume > 0)
    if np.sum(finite_mask) < 5:
        raise DataDegeneracy("insufficient data for price delay")

    prices = vals[finite_mask]
    vols = volume[finite_mask]

    # Compute log returns
    with np_errstate():
        log_rets = np.diff(np.log(prices))

    if len(log_rets) < 4:
        raise DataDegeneracy("insufficient returns for delay")

    # Volume-weight the returns
    vol_weights = (vols[1:]) / (np.sum(vols[1:]) + _EPS) if (np.sum(vols[1:]) + _EPS) > 1e-10 else np.nan
    weighted_rets = log_rets * vol_weights

    # Lag-1 autocorrelation as delay proxy
    if np.std(weighted_rets) <= _EPS:
        raise DataDegeneracy("constant weighted returns")

    with np_errstate():
        corr = float(np.corrcoef(weighted_rets[:-1], weighted_rets[1:])[0, 1])

    if not np.isfinite(corr):
        raise DataDegeneracy("undefined correlation")

    # Higher autocorrelation = slower price adjustment = higher delay
    delay_score = float(corr)

    return delay_score


@register_operator(
    name="intra_price_delay",
    category="intraday_microstructure",
    business_category="intraday_true_gap",
    canonical="intra_price_delay",
    source="intraday.true_gap_batch3",
    backend="pandas_numpy",
    status="extended",
)
class IntraPriceDelay(SessionAggregationOperator):
    """Intraday price delay score (TRUE_GAP).

    Measures information incorporation speed within trading session via
    volume-weighted return autocorrelation. Respects calendar boundaries.
    """

    metadata = metadata(
        "intra_price_delay",
        "Intraday price delay (information incorporation speed).",
        ["close", "volume"],
        unit="correlation",
        cost=6,
        extra_tags=["true_gap", "session_aware", "microstructure"],
    )

    def _calculate_series(self, close, volume, **_):
        """Calculate price delay from minute close and volume."""
        return daily_agg_two(
            close,
            volume,
            _price_delay_kernel,  # raw kernel: carries __vec__; daily_agg_two binds times=None
            min_finite=5,
        )


# ---------------------------------------------------------------------------
# 4. intra_volume_imbalance: session volume imbalance
# ---------------------------------------------------------------------------

def _volume_imbalance_kernel(
    vals: np.ndarray, volume: np.ndarray, times: np.ndarray
) -> float:
    """Compute intraday volume imbalance (buy vs sell pressure proxy).

    TRUE_GAP: measures volume distribution around session VWAP, respecting
    actual trading session structure.

    Returns: (volume_above_vwap - volume_below_vwap) / total_volume
    """
    finite_mask = np.isfinite(vals) & np.isfinite(volume) & (volume > 0)
    if np.sum(finite_mask) < 3:
        raise DataDegeneracy("insufficient data for volume imbalance")

    prices = vals[finite_mask]
    vols = volume[finite_mask]

    # Compute session VWAP
    total_vol = float(np.sum(vols))
    if total_vol <= _EPS:
        raise DataDegeneracy("zero total volume")

    vwap = (float(np.sum(prices * vols)) / total_vol) if total_vol > 1e-10 else np.nan

    # Volume above and below VWAP
    above_mask = prices > vwap
    below_mask = prices < vwap

    vol_above = float(np.sum(vols[above_mask]))
    vol_below = float(np.sum(vols[below_mask]))

    # Imbalance: positive = more volume above VWAP (buying pressure)
    imbalance = ((vol_above - vol_below)) / total_vol if total_vol > 1e-10 else np.nan

    return float(imbalance)


@register_operator(
    name="intra_volume_imbalance",
    category="intraday_microstructure",
    business_category="intraday_true_gap",
    canonical="intra_volume_imbalance",
    source="intraday.true_gap_batch3",
    backend="pandas_numpy",
    status="extended",
)
class IntraVolumeImbalance(SessionAggregationOperator):
    """Session volume imbalance (TRUE_GAP).

    Measures buy vs sell pressure via volume distribution around VWAP within
    trading session. Respects calendar structure (no overnight/lunch bridge).
    """

    metadata = metadata(
        "intra_volume_imbalance",
        "Session volume imbalance (volume above vs below VWAP).",
        ["close", "volume"],
        unit="ratio",
        cost=5,
        extra_tags=["true_gap", "session_aware", "microstructure"],
    )

    def _calculate_series(self, close, volume, **_):
        """Calculate volume imbalance from minute close and volume."""
        return daily_agg_two(
            close,
            volume,
            _volume_imbalance_kernel,  # raw kernel: carries __vec__; daily_agg_two binds times=None
            min_finite=3,
        )


# Register all operators to extended surface
register_surface(_CANONICALS)
