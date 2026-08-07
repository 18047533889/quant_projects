# -*- coding: utf-8 -*-
"""Stateful rule-control primitives (2026-08 CTA pack).

* ``state_latch``      — set/reset SR latch (reset priority), regime memory.
* ``state_hold``       — snapshot-and-remember: keep the last value seen on an
                         update condition until reset (recursive memory, unlike
                         the rolling-window ``ts_last_if``).
* ``state_slew_limit`` — rate limiter: output moves at most ``limit`` per row
                         toward the target (slow-adjustment signal control).
* ``state_deadband``   — continuous hysteresis: changes smaller than ``band``
                         are ignored; larger changes move by the excess only.

All four are forward per-column recursions (prefix-causal, PIT).  A NaN input
emits NaN and re-baselines the internal state (``missing_policy="break"``): a
missing observation is never treated as False / no-change.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like
from cleaned_operators.stateful._common import metadata, panel_or_scalar

_EPS = 1e-12


@register_operator(
    name="state_latch",
    category="time_series_state",
    business_category="time_series_state",
    canonical="state_latch",
    source="stateful.rule_language",
)
class StateLatch(SeriesOperator):
    """Set/reset SR latch: ``reset`` wins, then ``set``, else carry.

    ``initial_state`` (0 or 1) is the neutral value the latch returns before
    any event and the value a NaN input re-baselines to.
    """

    metadata = metadata(
        "state_latch",
        "SR 锁存状态: reset 优先, 其次 set, 否则保持。",
        ["set_condition", "reset_condition", "initial_state"],
        domain="trading_state",
        unit="state",
    )

    def _calculate_series(
        self,
        set_condition: pd.DataFrame,
        reset_condition: pd.DataFrame,
        initial_state: float = 0.0,
        **_: Any,
    ) -> pd.DataFrame:
        sc = set_condition.to_numpy(dtype=float)
        rc = reset_condition.to_numpy(dtype=float)
        init = 1.0 if float(initial_state) != 0.0 else 0.0
        rows, cols = sc.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            state = init
            for row in range(rows):
                if not (np.isfinite(sc[row, col]) and np.isfinite(rc[row, col])):
                    out[row, col] = np.nan
                    state = init
                    continue
                if bool(rc[row, col] != 0.0):
                    state = 0.0
                elif bool(sc[row, col] != 0.0):
                    state = 1.0
                out[row, col] = state
        return frame_like(set_condition, out)


@register_operator(
    name="state_hold",
    category="time_series_state",
    business_category="time_series_state",
    canonical="state_hold",
    source="stateful.rule_language",
)
class StateHold(SeriesOperator):
    """Recursive snapshot memory.

    ``Y_t = x_t`` when ``update_condition`` holds and ``x_t`` is finite;
    ``Y_t = NaN`` on ``reset_condition`` (or when not yet updated); otherwise
    ``Y_t = Y_{t-1}`` (the last snapshot is remembered).
    """

    metadata = metadata(
        "state_hold",
        "条件触发时记录 x, 之后一直保持最近一次记录值, reset 清空。",
        ["value", "update_condition", "reset_condition"],
        domain="trading_state",
        unit="level",
    )

    def _calculate_series(
        self,
        value: pd.DataFrame,
        update_condition: pd.DataFrame,
        reset_condition: Any = None,
        **_: Any,
    ) -> pd.DataFrame:
        vv = value.to_numpy(dtype=float)
        uc = update_condition.to_numpy(dtype=float)
        rv = None if reset_condition is None else reset_condition.to_numpy(dtype=float)
        rows, cols = vv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            held = np.nan
            for row in range(rows):
                u_ok = bool(np.isfinite(uc[row, col]))
                r_ok = True if rv is None else bool(np.isfinite(rv[row, col]))
                if not (u_ok and r_ok):
                    out[row, col] = np.nan
                    held = np.nan
                    continue
                if rv is not None and bool(rv[row, col] != 0.0):
                    held = np.nan
                elif bool(uc[row, col] != 0.0) and np.isfinite(vv[row, col]):
                    held = float(vv[row, col])
                out[row, col] = held
        return frame_like(value, out)


@register_operator(
    name="state_slew_limit",
    category="time_series_state",
    business_category="time_series_state",
    canonical="state_slew_limit",
    source="stateful.rule_language",
)
class StateSlewLimit(SeriesOperator):
    """Rate limiter: output moves at most ``limit`` per row toward target.

    ``Y_t = Y_{t-1} + clip(X_t - Y_{t-1}, -L_t, L_t)``.  ``limit`` may be a
    scalar or a per-row panel.  The first finite observation initialises the
    output directly.
    """

    metadata = metadata(
        "state_slew_limit",
        "输出每行最多向目标移动 limit, 实现慢调整信号控制。",
        ["x", "limit"],
        domain="trading_state",
        unit="level",
    )

    def _calculate_series(self, x: pd.DataFrame, limit: Any = 0.01, **_: Any) -> pd.DataFrame:
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            prev = np.nan
            for row in range(rows):
                target = xv[row, col]
                lim = panel_or_scalar(limit, row)
                if not np.isfinite(target) or not np.isfinite(lim) or lim < 0.0:
                    out[row, col] = np.nan
                    prev = np.nan
                    continue
                if not np.isfinite(prev):
                    prev = target
                    out[row, col] = prev
                    continue
                delta = float(target) - prev
                prev = prev + float(np.clip(delta, -lim, lim))
                out[row, col] = prev
        return frame_like(x, out)


@register_operator(
    name="state_deadband",
    category="time_series_state",
    business_category="time_series_state",
    canonical="state_deadband",
    source="stateful.rule_language",
)
class StateDeadband(SeriesOperator):
    """Continuous hysteresis: ignore small changes, move by the excess only.

    With ``d = X_t - Y_{t-1}``: ``Y_t = Y_{t-1}`` when ``|d| <= band``, else
    ``Y_t = Y_{t-1} + sign(d) * (|d| - band)``.  ``band`` may be a scalar or a
    per-row panel.  The first finite observation initialises the output.
    """

    metadata = metadata(
        "state_deadband",
        "连续迟滞: 变化小于 band 不动, 超过部分才移动。",
        ["x", "band"],
        domain="trading_state",
        unit="level",
    )

    def _calculate_series(self, x: pd.DataFrame, band: Any = 0.0, **_: Any) -> pd.DataFrame:
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            prev = np.nan
            for row in range(rows):
                target = xv[row, col]
                b = panel_or_scalar(band, row)
                if not np.isfinite(target) or not np.isfinite(b) or b < 0.0:
                    out[row, col] = np.nan
                    prev = np.nan
                    continue
                if not np.isfinite(prev):
                    prev = target
                    out[row, col] = prev
                    continue
                d = float(target) - prev
                if abs(d) <= b:
                    out[row, col] = prev
                    continue
                prev = prev + float(np.sign(d) * (abs(d) - b))
                out[row, col] = prev
        return frame_like(x, out)


def _register_surface() -> None:
    from cleaned_operators.stateful._common import register_stateful_surface

    register_stateful_surface(
        ["state_latch", "state_hold", "state_slew_limit", "state_deadband"]
    )


_register_surface()
