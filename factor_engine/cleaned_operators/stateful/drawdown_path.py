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
                seg = xv[lo : row + 1, col]
                valid = np.isfinite(seg) & (seg > 0.0)
                if int(valid.sum()) < 2:
                    continue
                vals = seg[valid]
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
    """Trailing sum of daily drawdown depths from the running peak
    (drawdown depth x duration integral).

    For each valid positive observation ``DD_s = 1 - x_s / running_max_s``;
    the output is the sum of ``DD_s`` over the trailing window.  NaN when fewer
    than two valid positive observations are available.
    """

    metadata = metadata(
        "ts_current_drawdown_area",
        "窗口内每日回撤深度之和(深度×时长积分)。",
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
                seg = xv[lo : row + 1, col]
                valid = np.isfinite(seg) & (seg > 0.0)
                if int(valid.sum()) < 2:
                    continue
                vals = seg[valid]
                running_max = np.maximum.accumulate(vals)
                dd = 1.0 - vals / running_max
                out[row, col] = float(np.sum(dd))
        return frame_like(x, out)


def _register_surface() -> None:
    from cleaned_operators.stateful._common import register_stateful_surface

    register_stateful_surface(["ts_recovery_fraction", "ts_current_drawdown_area"])


_register_surface()
