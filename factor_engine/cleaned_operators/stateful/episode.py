# -*- coding: utf-8 -*-
"""Dynamic-episode / intrinsic-time primitives (2026-08 CTA pack).

* ``state_since_reduce``       — reduce ``x`` over the current episode
  ``[tau, t]`` where ``tau`` is the last reset (dynamic episode window, unlike
  a fixed rolling window).
* ``directional_change_state`` — binary/ternary Directional-Change event state:
  +1 while an up event is running, -1 while a down event runs, 0 before the
  first event confirms.
* ``directional_change_extent`` — normalized progress of the current DC event
  (signed by direction; magnitude = number of ``threshold`` moves since the
  event's origin, i.e. overshoot accumulates from the event start).
* ``state_since_trend_tstat``  — OLS slope t-statistic of ``x`` over the
  current episode since the last reset.

All are forward per-column recursions (prefix-causal, PIT).  A NaN input emits
NaN and re-baselines the recursion state (``missing_policy="break"``).
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
    name="state_since_reduce",
    category="time_series_state",
    business_category="time_series_state",
    canonical="state_since_reduce",
    source="stateful.episode",
)
class StateSinceReduce(SeriesOperator):
    """Reduce ``x`` over the current episode since the last reset.

    ``mode`` in {"sum", "mean", "count", "last"}.  Reset rows (and NaN reset
    input) emit NaN and start a new episode; a NaN ``x`` emits NaN that day but
    does not break the episode accumulation (the day exists, the value is
    missing).  Output is NaN until the episode has ``min_episode`` valid rows.
    """

    metadata = metadata(
        "state_since_reduce",
        "自上次 reset 以来的 episode 内 x 的累计/均值/计数/末值。mode 单位: sum/mean/last=level, count=count。",
        ["x", "reset_condition", "mode", "min_episode"],
        domain="price_volume",
        unit="level",
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        reset_condition: pd.DataFrame,
        mode: str = "sum",
        min_episode: int = 2,
        **_: Any,
    ) -> pd.DataFrame:
        xv = x.to_numpy(dtype=float)
        rv = reset_condition.to_numpy(dtype=float)
        m = str(mode).lower()
        if m not in {"sum", "mean", "count", "last"}:
            raise ValueError(f"mode must be one of sum/mean/count/last, got {mode!r}")
        min_e = max(1, int(min_episode))
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            acc = 0.0
            cnt = 0
            last = np.nan
            for row in range(rows):
                if not np.isfinite(rv[row, col]):
                    out[row, col] = np.nan
                    acc = 0.0
                    cnt = 0
                    last = np.nan
                    continue
                if bool(rv[row, col] != 0.0):
                    out[row, col] = np.nan
                    acc = 0.0
                    cnt = 0
                    last = np.nan
                    continue
                xt = xv[row, col]
                if not np.isfinite(xt):
                    # P1-73: a missing x emits NaN that day (the docstring
                    # contract) instead of continuing to print the old episode
                    # statistics.  The episode accumulation itself is not
                    # broken — acc/cnt/last are carried forward unchanged.
                    out[row, col] = np.nan
                    continue
                acc = acc + float(xt)
                cnt = cnt + 1
                last = float(xt)
                if cnt < min_e:
                    out[row, col] = np.nan
                    continue
                if m == "sum":
                    out[row, col] = acc
                elif m == "mean":
                    out[row, col] = acc / float(cnt)
                elif m == "count":
                    out[row, col] = float(cnt)
                else:  # last
                    out[row, col] = last
        return frame_like(x, out)


def _dc_states_and_extents(price: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    rows = price.shape[0]
    state = np.full(rows, np.nan, dtype=float)
    extent = np.full(rows, np.nan, dtype=float)
    direction = 0
    # P0-25: two distinct state variables.  ``origin`` is the price level where
    # the current event *started* (the extent reference, so overshoot
    # accumulates from the event origin); ``extreme`` is the running extreme of
    # the current direction (peak for up, trough for down) and only drives the
    # reversal test.  Reusing one variable for both made a new high reset the
    # extent to ~0 (extent = (p/turning - 1)/threshold with turning = p).
    origin = np.nan
    extreme = np.nan
    first = np.nan
    for row in range(rows):
        p = price[row]
        if not np.isfinite(p):
            state[row] = np.nan
            extent[row] = np.nan
            direction = 0
            origin = np.nan
            extreme = np.nan
            first = np.nan
            continue
        if direction == 0:
            if not np.isfinite(first):
                first = p
                state[row] = 0.0
                extent[row] = 0.0
                continue
            if p >= first * (1.0 + threshold):
                direction = 1
                origin = first
                extreme = p
            elif p <= first * (1.0 - threshold):
                direction = -1
                origin = first
                extreme = p
            state[row] = float(direction)
            if direction == 0:
                extent[row] = 0.0
            else:
                extent[row] = (p / origin - 1.0) / threshold
            continue
        if direction == 1:
            if p > extreme:
                extreme = p
            if p <= extreme * (1.0 - threshold):
                # reversal: a down event starts from the running peak
                direction = -1
                origin = extreme
                extreme = p
            state[row] = float(direction)
            extent[row] = (p / origin - 1.0) / threshold
            continue
        # direction == -1
        if p < extreme:
            extreme = p
        if p >= extreme * (1.0 + threshold):
            # reversal: an up event starts from the running trough
            direction = 1
            origin = extreme
            extreme = p
        state[row] = float(direction)
        extent[row] = (p / origin - 1.0) / threshold
    return state, extent


@register_operator(
    name="directional_change_state",
    category="time_series_state",
    business_category="time_series_state",
    canonical="directional_change_state",
    source="stateful.episode",
)
class DirectionalChangeState(SeriesOperator):
    """Directional-Change event state: +1 up event / -1 down event / 0 none.

    An up event runs while price keeps making new highs and ends when it falls
    ``threshold`` below the running peak (a down event then starts).  NaN price
    emits NaN and re-baselines the DC state machine.
    """

    metadata = metadata(
        "directional_change_state",
        "方向变化事件状态: +1 上行事件 / -1 下行事件 / 0 未确认。",
        ["price", "threshold"],
        domain="price_volume",
        unit="state",
    )

    def _calculate_series(self, price: pd.DataFrame, threshold: float = 0.03, **_: Any) -> pd.DataFrame:
        th = float(threshold)
        if th <= 0.0:
            raise ValueError("threshold must be > 0")
        pv = price.to_numpy(dtype=float)
        rows, cols = pv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            st, _ = _dc_states_and_extents(pv[:, col], th)
            out[:, col] = st
        return frame_like(price, out)


@register_operator(
    name="directional_change_extent",
    category="time_series_state",
    business_category="time_series_state",
    canonical="directional_change_extent",
    source="stateful.episode",
)
class DirectionalChangeExtent(SeriesOperator):
    """Normalized progress of the current Directional-Change event.

    Signed by direction: for an up event ``(price/origin - 1)/threshold``, for
    a down event the same ratio (negative), where ``origin`` is the price level
    at which the event started.  Magnitude = number of ``threshold`` moves since
    the event's origin (1.0 at confirmation, then overshoot accumulates; a new
    running high inside an up event does *not* reset the extent — P0-25).
    """

    metadata = metadata(
        "directional_change_extent",
        "方向变化事件进度: 距转折点多深(按 threshold 归一)。",
        ["price", "threshold"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(self, price: pd.DataFrame, threshold: float = 0.03, **_: Any) -> pd.DataFrame:
        th = float(threshold)
        if th <= 0.0:
            raise ValueError("threshold must be > 0")
        pv = price.to_numpy(dtype=float)
        rows, cols = pv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            _, ext = _dc_states_and_extents(pv[:, col], th)
            out[:, col] = ext
        return frame_like(price, out)


@register_operator(
    name="state_since_trend_tstat",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="state_since_trend_tstat",
    source="stateful.episode",
)
class StateSinceTrendTstat(SeriesOperator):
    """OLS slope t-statistic of ``x`` over the current episode since the last
    reset, computed incrementally over the episode.

    The regression time axis is the *physical* bar offset within the episode
    (P1-74): a NaN ``x`` row advances the clock but does not contribute a point,
    so a gap between valid observations is never compressed into an adjacent
    pair.  Requires at least ``min_obs`` valid rows in the episode.  Reset rows
    emit NaN and start a new episode; once the episode exceeds ``max_age`` rows
    it is censored and emits NaN until a real state reset (P1-75) rather than
    silently starting a synthetic fresh episode.
    """

    metadata = metadata(
        "state_since_trend_tstat",
        "episode 内 OLS 斜率 t 统计(事件以来趋势强度)。",
        ["x", "reset_condition", "min_obs", "max_age"],
        domain="price_volume",
        unit="level",
        category="time_series_regression",
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        reset_condition: pd.DataFrame,
        min_obs: int = 5,
        max_age: Any = None,
        **_: Any,
    ) -> pd.DataFrame:
        xv = x.to_numpy(dtype=float)
        rv = reset_condition.to_numpy(dtype=float)
        min_o = max(3, int(min_obs))
        cap = None if max_age is None else int(max_age)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            Ss = 0.0
            Sy = 0.0
            Ssy = 0.0
            Ss2 = 0.0
            Syy = 0.0
            cnt = 0
            age = 0
            censored = False
            for row in range(rows):
                age += 1
                if not np.isfinite(rv[row, col]):
                    out[row, col] = np.nan
                    Ss = Sy = Ssy = Ss2 = Syy = 0.0
                    cnt = 0
                    age = 0
                    censored = False
                    continue
                if bool(rv[row, col] != 0.0):
                    out[row, col] = np.nan
                    Ss = Sy = Ssy = Ss2 = Syy = 0.0
                    cnt = 0
                    age = 0
                    censored = False
                    continue
                if censored:
                    # P1-75: once max_age is exceeded the episode is censored —
                    # stay NaN until a genuine reset, never start a synthetic
                    # fresh episode on the very next bar.
                    out[row, col] = np.nan
                    continue
                if cap is not None and age > cap:
                    out[row, col] = np.nan
                    censored = True
                    continue
                xt = xv[row, col]
                if np.isfinite(xt):
                    # P1-74: physical bar offset within the episode (age), not
                    # the count of valid x rows — missing x rows must advance
                    # the clock so the regression never compresses time.
                    s = float(age)
                    xf = float(xt)
                    Ss += s
                    Sy += xf
                    Ssy += s * xf
                    Ss2 += s * s
                    Syy += xf * xf
                    cnt += 1
                if cnt < min_o:
                    out[row, col] = np.nan
                    continue
                n = float(cnt)
                denom = n * Ss2 - Ss * Ss
                if denom <= 0.0:
                    out[row, col] = np.nan
                    continue
                b = (n * Ssy - Ss * Sy) / denom
                a = (Sy - b * Ss) / n
                sse = Syy - a * Sy - b * Ssy
                if sse < 0.0:
                    sse = 0.0
                mse = sse / (n - 2.0) if n > 2.0 else np.nan
                if not np.isfinite(mse) or mse <= 0.0:
                    out[row, col] = np.nan
                    continue
                se_b = np.sqrt(mse / (Ss2 - Ss * Ss / n))
                out[row, col] = b / (se_b + _EPS)
        return frame_like(x, out)


def _register_surface() -> None:
    from cleaned_operators.stateful._common import register_stateful_surface

    register_stateful_surface(
        [
            "state_since_reduce", "directional_change_state",
            "directional_change_extent", "state_since_trend_tstat",
        ]
    )


_register_surface()
