# -*- coding: utf-8 -*-
"""Event-language primitives (2026-08 CTA pack).

* ``event_refractory`` — a condition that fires at most once per ``cooldown``
  rows: once an event is accepted, identical events inside the cooldown are
  suppressed, collapsing a multi-day phenomenon into one episode.
* ``cross_event``     — directional crossing detector (``x`` crosses ``y`` up
  or down) on the previous row; ``delay``-style composition but exposed as a
  fused primitive for CTA / rule search.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like
from cleaned_operators.stateful._common import (
    assert_condition_bool,
    metadata,
    truth_mask,
)

_EPS = 1e-12


@register_operator(
    name="event_refractory",
    category="time_series_event",
    business_category="time_series_event",
    canonical="event_refractory",
    source="stateful.events",
)
class EventRefractory(SeriesOperator):
    """Accept a condition event at most once per ``cooldown`` rows.

    An event at row ``t`` is accepted iff ``condition_t`` holds and
    ``t - tau* > cooldown`` where ``tau*`` is the last *accepted* event.  The
    first event is always accepted.  A NaN condition emits NaN and re-baselines
    the accepted-event memory (``missing_policy="break"``).
    """

    metadata = metadata(
        "event_refractory",
        "事件冷却: 接受一次后在 cooldown 天内忽略同类事件。",
        ["condition", "cooldown"],
        domain="trading_state",
        unit="state",
        category="time_series_event",
    )

    def _calculate_series(self, condition: pd.DataFrame, cooldown: float = 5, **_: Any) -> pd.DataFrame:
        assert_condition_bool(condition)
        cv = condition.to_numpy(dtype=float)
        cd = max(0, int(cooldown))
        rows, cols = cv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            last_accepted = None
            for row in range(rows):
                if not np.isfinite(cv[row, col]):
                    # NaN event state is unknown: it must not fire and it
                    # re-baselines the accepted-event memory.
                    out[row, col] = np.nan
                    last_accepted = None
                    continue
                truth = cv[row, col] == 1.0
                if truth and (last_accepted is None or row - last_accepted > cd):
                    out[row, col] = 1.0
                    last_accepted = row
                else:
                    out[row, col] = 0.0
        return frame_like(condition, out)


@register_operator(
    name="cross_event",
    category="time_series_event",
    business_category="time_series_event",
    canonical="cross_event",
    source="stateful.events",
)
class CrossEvent(SeriesOperator):
    """Directional crossing detector.

    ``direction="up"``: 1 where ``x_t > y_t`` and ``x_{t-1} <= y_{t-1}``.
    ``direction="down"``: 1 where ``x_t < y_t`` and ``x_{t-1} >= y_{t-1}``.
    ``direction`` must be ``"up"`` or ``"down"`` (anything else raises
    ValueError, P1-70).  NaN in either series (current or previous row) emits
    NaN — a missing observation is never read as "no cross"; row 0 has no
    predecessor so it is also NaN.
    """

    metadata = metadata(
        "cross_event",
        "上穿/下穿检测: x 相对 y 的方向性穿越。",
        ["x", "y", "direction"],
        domain="price_volume",
        unit="state",
        category="time_series_event",
    )

    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, direction: str = "up", **_: Any) -> pd.DataFrame:
        xv = x.to_numpy(dtype=float)
        yv = y.to_numpy(dtype=float)
        d = str(direction).lower()
        if d not in {"up", "down"}:
            raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")
        up = d == "up"
        rows, cols = xv.shape
        # P1-70: a missing current/previous value emits NaN, never 0 (which
        # would be indistinguishable from a genuine "no cross").
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(1, rows):
                cur_ok = np.isfinite(xv[row, col]) and np.isfinite(yv[row, col])
                prev_ok = np.isfinite(xv[row - 1, col]) and np.isfinite(yv[row - 1, col])
                if not (cur_ok and prev_ok):
                    continue
                if up:
                    crossed = bool(
                        xv[row, col] > yv[row, col] and xv[row - 1, col] <= yv[row - 1, col]
                    )
                else:
                    crossed = bool(
                        xv[row, col] < yv[row, col] and xv[row - 1, col] >= yv[row - 1, col]
                    )
                out[row, col] = 1.0 if crossed else 0.0
        return frame_like(x, out)


def _register_surface() -> None:
    from cleaned_operators.stateful._common import register_stateful_surface

    register_stateful_surface(["event_refractory", "cross_event"])


_register_surface()
