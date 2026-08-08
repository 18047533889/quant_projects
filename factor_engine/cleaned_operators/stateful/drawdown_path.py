# -*- coding: utf-8 -*-
"""Drawdown-path / recovery primitives (2026-08 CTA pack).

* ``ts_recovery_fraction`` — position of the current price between the trough
  (since the trailing peak) and the peak itself: 0 = still at the low, 0.5 =
  half recovered, 1 = fully recovered / new high.  Distinguishes "just fell
  to -10%" from "fell -30% then bounced to -10%".
* ``ts_current_drawdown_area`` — trailing sum of daily drawdown depths from the
  running peak (depth x duration integral).

Both are trailing-window and prefix-causal; they only use rows ``<= t``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like
from cleaned_operators.stateful._common import metadata

_EPS = 1e-12


def _trailing_contiguous(vals: np.ndarray) -> np.ndarray:
    """Most recent suffix of consecutive finite values ending at the last row.

    A missing observation *breaks* the path: rows on either side of a NaN are
    never treated as adjacent trading days (no time-axis compression), and a
    NaN at the current row yields an empty block so the operator emits NaN
    instead of re-using the last valid price.
    """
    n = len(vals)
    if n == 0 or not np.isfinite(vals[-1]):
        return np.empty(0, dtype=float)
    i = n - 1
    while i >= 0 and np.isfinite(vals[i]):
        i -= 1
    return vals[i + 1 :]


@register_operator(
    name="ts_recovery_fraction",
    category="downside_risk",
    business_category="downside_risk",
    canonical="ts_recovery_fraction",
    source="stateful.drawdown_path",
)
class TsRecoveryFraction(SeriesOperator):
    """Current price's recovery progress between the trailing peak and the
    trough after that peak.

    ``RF = clip((x_t - T) / (P - T + eps), 0, 1)`` where ``P`` is the trailing
    running max and ``T`` is the minimum from ``P``'s position to today.  NaN
    when fewer than two valid positive observations are in the window.
    """

    metadata = metadata(
        "ts_recovery_fraction",
        "当前价格在峰谷之间的修复进度 [0,1]。",
        ["x", "window"],
        domain="price_volume",
        unit="ratio",
        category="downside_risk",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, **_: Any) -> pd.DataFrame:
        w = max(2, int(window))
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                seg = _trailing_contiguous(xv[lo : row + 1, col])
                # P0-006/007: a NaN current value yields an empty block -> NaN,
                # never a stale last-valid recovery fraction; gaps never bridge.
                # P1-26: a non-positive price inside the active segment must
                # break the path — deleting it and re-connecting (100 -> 90 from
                # [100, 0, 90]) silently fabricates a valid sequence.
                if seg.size < 2 or not np.all(seg > 0.0):
                    continue
                vals = seg
                p = float(np.max(vals))
                p_pos = int(np.argmax(vals))
                t = float(np.min(vals[p_pos:]))
                if p - t <= _EPS:
                    out[row, col] = 1.0
                    continue
                rf = (float(vals[-1]) - t) / (p - t + _EPS)
                out[row, col] = float(np.clip(rf, 0.0, 1.0))
        return frame_like(x, out)


@register_operator(
    name="ts_current_drawdown_area",
    category="downside_risk",
    business_category="downside_risk",
    canonical="ts_current_drawdown_area",
    source="stateful.drawdown_path",
)
class TsCurrentDrawdownArea(SeriesOperator):
    """Sum of daily drawdown depths of the *current* (not-yet-recovered)
    drawdown episode (drawdown depth x duration integral).

    P1-27: the earlier implementation summed every daily depth inside the
    trailing window, so a drawdown that had fully recovered still contributed
    area — that is a *trailing* drawdown area.  The current-episode semantic
    resets at the last new high: only rows from that peak onward count, so a
    fully recovered series has area 0.  NaN when fewer than two valid positive
    observations are available or the path contains a non-positive price
    (P1-26).
    """

    metadata = metadata(
        "ts_current_drawdown_area",
        "当前未收复回撤的每日深度之和(自最后一个新高起)。",
        ["x", "window"],
        domain="price_volume",
        unit="ratio",
        category="downside_risk",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, **_: Any) -> pd.DataFrame:
        w = max(2, int(window))
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                seg = _trailing_contiguous(xv[lo : row + 1, col])
                # P0-007: no time-axis compression — the running peak and the
                # depth sum run over the contiguous block only, so the duration
                # dimension is real; a NaN current row emits NaN.  P1-26: a
                # non-positive price breaks the path.
                if seg.size < 2 or not np.all(seg > 0.0):
                    continue
                vals = seg
                running_max = np.maximum.accumulate(vals)
                # Last new-high position = last index where the running max
                # increased (or the first row).  Only rows from there on are the
                # current, not-yet-recovered episode.
                new_high = np.r_[True, running_max[1:] > running_max[:-1]]
                last_peak = int(np.flatnonzero(new_high)[-1])
                dd = 1.0 - vals / running_max
                out[row, col] = float(np.sum(dd[last_peak:]))
        return frame_like(x, out)


def _register_surface() -> None:
    from cleaned_operators.stateful._common import register_stateful_surface

    register_stateful_surface(["ts_recovery_fraction", "ts_current_drawdown_area"])


_register_surface()
