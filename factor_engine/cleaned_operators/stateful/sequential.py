# -*- coding: utf-8 -*-
"""Sequential detection / conditional-memory primitives (2026-08 CTA pack).

* ``ts_cusum_pressure``    — recursive two-sided CUSUM of standardized shocks
  against a strictly-past baseline (median / scaled MAD).  Unlike the windowed
  ``ts_cusum_break_score`` (max |running cumsum| inside one window), this keeps
  the S⁺ / S⁻ accumulation across the whole history with a drift allowance.
* ``ts_rank_if``           — trailing-window percentile rank of the current
  value among the values where a condition held (completes the ``*_if`` family).
* ``state_ewm_if``         — exponential moving average updated only on
  condition rows; otherwise the memory is carried unchanged.
* ``ts_lag_of_peak_corr``  — lag (0..max_lag) with the strongest absolute
  trailing correlation ``Corr(x_s, y_{s-k})`` (k ≥ 0 only), normalised by
  ``max_lag``.  ``ts_best_lag_corr`` exposes the peak magnitude, not the lag.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like
from cleaned_operators.stateful._common import metadata

_EPS = 1e-12


@register_operator(
    name="ts_cusum_pressure",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_cusum_pressure",
    source="stateful.sequential",
)
class TsCusumPressure(SeriesOperator):
    """Recursive two-sided CUSUM pressure of standardized shocks.

    Baseline (median ``mu``, scaled MAD ``s = 1.4826*MAD``) is estimated from
    the strictly-past window ``[t-W, t-1]``; ``z_t = (x_t - mu)/(s + eps)``;
    ``S⁺ = max(0, S⁺ + z - drift)`` and ``S⁻ = min(0, S⁻ + z + drift)``; the
    output is ``S⁺ + S⁻``.  Large magnitude means a sustained one-direction
    deviation is accumulating; ~0 means no persistent drift.  A NaN input or an
    undersized baseline emits NaN and re-baselines the recursion.
    """

    metadata = metadata(
        "ts_cusum_pressure",
        "递归双端 CUSUM: 标准化冲击对历史基线的累积压力。",
        ["x", "reference_window", "drift", "min_periods"],
        domain="price_volume",
        unit="level",
        category="time_series_regression",
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        reference_window: int = 20,
        drift: float = 0.5,
        min_periods: Any = None,
        **_: Any,
    ) -> pd.DataFrame:
        w = max(2, int(reference_window))
        k = float(drift)
        mp = int(min_periods) if min_periods is not None else max(5, w // 2)
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            sp = 0.0
            sm = 0.0
            for row in range(rows):
                xt = xv[row, col]
                if not np.isfinite(xt):
                    out[row, col] = np.nan
                    sp = 0.0
                    sm = 0.0
                    continue
                lo = max(0, row - w)
                seg = xv[lo:row, col]  # strictly past
                valid = seg[np.isfinite(seg)]
                if valid.size < mp:
                    out[row, col] = np.nan
                    sp = 0.0
                    sm = 0.0
                    continue
                med = float(np.median(valid))
                mad = float(np.median(np.abs(valid - med)))
                scale = 1.4826 * mad + _EPS
                z = (float(xt) - med) / scale
                sp = max(0.0, sp + z - k)
                sm = min(0.0, sm + z + k)
                out[row, col] = sp + sm
        return frame_like(x, out)


@register_operator(
    name="ts_rank_if",
    category="time_series_condition",
    business_category="time_series_condition",
    canonical="ts_rank_if",
    source="stateful.sequential",
)
class TsRankIf(SeriesOperator):
    """Trailing-window percentile rank of the current value among the values
    where the condition held.

    The reference set is ``{x_s : s in [t-w+1, t], condition_s true}``.  The
    current value ``x_t`` is ranked against that set (rank with average tie
    handling, output in ``[0, 1]``).  NaN where the reference set is too small
    or the current value is missing.
    """

    metadata = metadata(
        "ts_rank_if",
        "条件成立子窗口内当前值的百分位排名。",
        ["x", "condition", "window", "min_periods"],
        domain="price_volume",
        unit="ratio",
        category="time_series_condition",
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        condition: pd.DataFrame,
        window: int = 20,
        min_periods: int = 5,
        **_: Any,
    ) -> pd.DataFrame:
        w = max(2, int(window))
        mp = max(2, int(min_periods))
        xv = x.to_numpy(dtype=float)
        cv = condition.to_numpy(dtype=float)
        truth = np.isfinite(cv) & (cv != 0.0)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                xt = xv[row, col]
                if not np.isfinite(xt):
                    continue
                lo = max(0, row - w + 1)
                sel = xv[lo : row + 1, col][truth[lo : row + 1, col]]
                sel = sel[np.isfinite(sel)]
                if sel.size < mp:
                    continue
                less = float(np.sum(sel < xt))
                equal = float(np.sum(sel == xt))
                out[row, col] = (less + 0.5 * equal) / sel.size
        return frame_like(x, out)


@register_operator(
    name="state_ewm_if",
    category="time_series_state",
    business_category="time_series_state",
    canonical="state_ewm_if",
    source="stateful.sequential",
)
class StateEwmIf(SeriesOperator):
    """Exponential moving average updated only while the condition holds.

    ``Y_t = alpha * x_t + (1 - alpha) * Y_{t-1}`` when ``condition_t`` is true;
    otherwise ``Y_t = Y_{t-1}`` (memory carried).  ``alpha`` derives from
    ``half_life``.  A NaN input emits NaN and re-baselines the memory.
    """

    metadata = metadata(
        "state_ewm_if",
        "仅条件成立时更新的指数移动记忆。",
        ["x", "condition", "half_life"],
        domain="price_volume",
        unit="level",
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        condition: pd.DataFrame,
        half_life: float = 10.0,
        **_: Any,
    ) -> pd.DataFrame:
        hl = float(half_life)
        alpha = 1.0 - np.exp(-np.log(2.0) / hl) if hl > 0.0 else 1.0
        xv = x.to_numpy(dtype=float)
        cv = condition.to_numpy(dtype=float)
        truth = np.isfinite(cv) & (cv != 0.0)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            state = np.nan
            for row in range(rows):
                if not (np.isfinite(xv[row, col]) and np.isfinite(cv[row, col])):
                    out[row, col] = np.nan
                    state = np.nan
                    continue
                if truth[row, col]:
                    if not np.isfinite(state):
                        state = float(xv[row, col])
                    else:
                        state = alpha * float(xv[row, col]) + (1.0 - alpha) * state
                out[row, col] = state
        return frame_like(x, out)


@register_operator(
    name="ts_lag_of_peak_corr",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_lag_of_peak_corr",
    source="stateful.sequential",
)
class TsLagOfPeakCorr(SeriesOperator):
    """Normalised lag (0..max_lag) with the strongest absolute trailing
    correlation ``Corr(x_s, y_{s-k})`` for ``k >= 0`` (only past ``y``).

    Output ``k* / max_lag`` in ``[0, 1]``; low means the relationship is
    immediate, high means it is delayed.  NaN when no lag meets the minimum
    sample size.
    """

    metadata = metadata(
        "ts_lag_of_peak_corr",
        "最强绝对滞后相关的滞后阶(归一化)。",
        ["x", "y", "window", "max_lag", "min_periods"],
        domain="price_volume",
        unit="ratio",
        category="time_series_risk",
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        window: int = 20,
        max_lag: int = 5,
        min_periods: Any = None,
        **_: Any,
    ) -> pd.DataFrame:
        w = max(3, int(window))
        ml = max(1, int(max_lag))
        mp = int(min_periods) if min_periods is not None else max(ml + 2, w // 2)
        xv = x.to_numpy(dtype=float)
        yv = y.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                best: tuple[float, int] | None = None
                for kk in range(ml + 1):
                    # pairs (x_s, y_{s-kk}) for s in [max(lo, kk), row]
                    s_lo = max(lo, kk)
                    if s_lo > row:
                        continue
                    xs = xv[s_lo : row + 1, col]
                    ys = yv[s_lo - kk : row + 1 - kk, col]
                    valid = np.isfinite(xs) & np.isfinite(ys)
                    if int(valid.sum()) < mp:
                        continue
                    a = xs[valid]
                    b = ys[valid]
                    if a.size < 2 or np.std(a) <= 0.0 or np.std(b) <= 0.0:
                        continue
                    corr = float(np.corrcoef(a, b)[0, 1])
                    if best is None or abs(corr) > abs(best[0]):
                        best = (corr, kk)
                if best is not None:
                    out[row, col] = best[1] / float(ml)
        return frame_like(x, out)


def _register_surface() -> None:
    from cleaned_operators.stateful._common import register_stateful_surface

    register_stateful_surface(
        [
            "ts_cusum_pressure", "ts_rank_if", "state_ewm_if",
            "ts_lag_of_peak_corr",
        ]
    )


_register_surface()
