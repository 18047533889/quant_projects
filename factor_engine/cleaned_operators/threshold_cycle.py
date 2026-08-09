# -*- coding: utf-8 -*-
"""Threshold-hysteresis cycle operators (2026-08 geometry/math expansion).

A scalar series ``x`` is converted into a two-state machine with *hysteresis*:
``L`` when ``x <= lower``, ``U`` when ``x >= upper``, otherwise the previous
state is held.  This forbids chattering around a single threshold and turns a
meandering series into a well-defined sequence of alternating ``L`` / ``U``
runs separated by state transitions.  A *completed cycle* is an ``L→U→L`` or
``U→L→U`` round trip (two consecutive transitions) that lies fully inside the
trailing window.

The family characterises the temporal rhythm of those cycles:

* ``ts_threshold_cycle_period``    — median (in bars) of completed cycle legs.
* ``ts_threshold_cycle_asymmetry`` — ratio of the median up-leg (``L→U``)
  duration to the median down-leg (``U→L``) duration, mapped to ``[-1, 1]``.

Both operators are trailing-window, per-column, deterministic and NaN-safe.
A non-finite ``x`` censors the state machine: the L/U state becomes UNKNOWN
and any cycle/leg whose span would cross the gap is broken (never measured as
if the data were continuous); the output row is NaN.  Degenerate windows (no
completed cycle / no up- or down-legs) also emit NaN.  ``upper <= lower``
raises ``ValueError``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like, register_polars_bridge

_EPS = 1e-12


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="threshold_cycle",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "threshold_cycle", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:threshold_hysteresis",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


# ---------------------------------------------------------------------------
# shared kernels
# ---------------------------------------------------------------------------
def _state_series(x: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Two-state hysteresis panel column: ``1`` = U, ``0`` = L, ``-1`` = UNKNOWN.

    Until the series first crosses a threshold the state is UNKNOWN — a value
    sitting inside the deadband at the window start must not be coerced to L
    (P0-13); that artificial initialisation made the first "leg" depend on
    where the window happened to start.

    A non-finite ``x`` is *not* allowed to hold the previous L/U state (R4-05):
    a missing bar means the machine's state is unknown from there on, so it is
    set to UNKNOWN.  ``_legs``/``_full_cycles`` then treat any span that crosses
    an UNKNOWN row as censored, so a cycle duration can never straddle a data
    gap.
    """
    n = len(x)
    s = np.full(n, -1, dtype=np.int8)
    for t in range(n):
        xt = x[t]
        if not np.isfinite(xt):
            s[t] = -1
        elif xt <= lower:
            s[t] = 0
        elif xt >= upper:
            s[t] = 1
        else:
            s[t] = s[t - 1] if t > 0 else -1
    return s


def _legs(s: np.ndarray) -> list[tuple[int, int, int]]:
    """Consecutive transition pairs ``(a, b, is_up)`` between *defined* states.

    A leg runs from transition row ``a`` to the next transition row ``b``;
    ``is_up == 1`` when the entered state is ``U`` (an ``L→U`` up-leg, i.e. the
    U-run), ``is_up == 0`` when it is ``L`` (a ``U→L`` down-leg, the L-run).
    Duration of the leg is ``b - a`` bars.  Transitions involving the UNKNOWN
    state are ignored: a run whose start is censored by the window is not
    measured as a phantom leg.  A leg whose span crosses an UNKNOWN row (a
    missing ``x``) is also censored — R4-05: a gap must never be silently
    bridged as if the state machine had stayed defined.
    """
    n = len(s)
    tr = [
        t for t in range(1, n)
        if s[t] != s[t - 1] and s[t] in (0, 1) and s[t - 1] in (0, 1)
    ]
    out: list[tuple[int, int, int]] = []
    for i in range(len(tr) - 1):
        a, b = tr[i], tr[i + 1]
        if np.any(s[a + 1:b] == -1):
            continue
        out.append((a, b, int(s[a])))
    return out


def _full_cycles(s: np.ndarray) -> list[tuple[int, int, int]]:
    """Completed round trips ``(a, c, dur)``.

    States alternate after every transition, so any two consecutive defined
    transitions ``tr[i]``/``tr[i+2]`` delimit a full ``L→U→L`` or ``U→L→U``
    cycle; its duration is ``tr[i+2] - tr[i]`` bars (P0-12 — the previous
    "period" actually measured a single leg/half-cycle).  A cycle whose span
    crosses an UNKNOWN row is censored (R4-05) — the gap breaks the round trip.
    """
    n = len(s)
    tr = [
        t for t in range(1, n)
        if s[t] != s[t - 1] and s[t] in (0, 1) and s[t - 1] in (0, 1)
    ]
    out: list[tuple[int, int, int]] = []
    for i in range(len(tr) - 2):
        a, c = tr[i], tr[i + 2]
        if np.any(s[a + 1:c] == -1):
            continue
        out.append((a, c, c - a))
    return out


def _cycle_period_series(x2d: np.ndarray, lower: float, upper: float, window: int) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        cycles = _full_cycles(_state_series(x2d[:, c], lower, upper))
        for r in range(rows):
            if not np.isfinite(x2d[r, c]):
                continue  # missing x -> unknown state -> NaN (R4-05)
            lo = max(0, r - w + 1)
            dur = [d for (a, cc, d) in cycles if a >= lo and cc <= r]
            if dur:
                out[r, c] = float(np.median(dur))
    return out


def _cycle_asymmetry_series(x2d: np.ndarray, lower: float, upper: float, window: int) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        legs = _legs(_state_series(x2d[:, c], lower, upper))
        for r in range(rows):
            if not np.isfinite(x2d[r, c]):
                continue  # missing x -> unknown state -> NaN (R4-05)
            lo = max(0, r - w + 1)
            up: list[int] = []
            down: list[int] = []
            for (a, b, is_up) in legs:
                if a >= lo and b <= r:
                    (up if is_up else down).append(b - a)
            if not up or not down:
                continue
            tlu = float(np.median(up))  # median L→U leg (U-run) duration
            tul = float(np.median(down))  # median U→L leg (L-run) duration
            out[r, c] = (tlu - tul) / (tlu + tul + _EPS)
    return out


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------
@register_operator(
    name="ts_threshold_cycle_period",
    category="threshold_cycle",
    business_category="threshold_cycle",
    canonical="ts_threshold_cycle_period",
    source="threshold_cycle",
)
class TsThresholdCyclePeriod(SeriesOperator):
    """滞回区间状态机下, 窗口内完整周期 (L→U→L 或 U→L→U) 时长的中位数。

    大 → 状态切换缓慢(慢周期); 小 → 频繁穿越阈值(快周期)。 没有完整周期时
    输出 NaN。 完整周期时长 = 相邻两次同向穿越之间的 bar 数。 P2。
    """

    metadata = _metadata(
        "ts_threshold_cycle_period",
        "滞回状态机中完整周期时长(bar)的中位数。",
        ["x", "lower", "upper", "window"],
        unit="bars",
        cost=3,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        lower: float,
        upper: float,
        window: int = 120,
        **_: Any,
    ) -> pd.DataFrame:
        if not upper > lower:
            raise ValueError("upper must be > lower")
        w = int(window)
        if w < 2:
            raise ValueError("window must be >= 2")
        return frame_like(
            x,
            _cycle_period_series(x.to_numpy(dtype=float), float(lower), float(upper), w),
        )


@register_operator(
    name="ts_threshold_cycle_asymmetry",
    category="threshold_cycle",
    business_category="threshold_cycle",
    canonical="ts_threshold_cycle_asymmetry",
    source="threshold_cycle",
)
class TsThresholdCycleAsymmetry(SeriesOperator):
    """上冲腿与下挫腿时长不对称度: (TLU-TUL)/(TLU+TUL+eps)。

    正 → 上行(U 态)持续比下行(L 态)更久(慢涨急跌的反面); 负 → 下行更持久。
    TLU/TUL 任一缺失(窗口内缺某类腿)时输出 NaN。 P2。
    """

    metadata = _metadata(
        "ts_threshold_cycle_asymmetry",
        "滞回周期中 L→U 与 U→L 腿长中位数的归一化不对称度。",
        ["x", "lower", "upper", "window"],
        unit="ratio",
        cost=3,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        lower: float,
        upper: float,
        window: int = 120,
        **_: Any,
    ) -> pd.DataFrame:
        if not upper > lower:
            raise ValueError("upper must be > lower")
        w = int(window)
        if w < 2:
            raise ValueError("window must be >= 2")
        return frame_like(
            x,
            _cycle_asymmetry_series(x.to_numpy(dtype=float), float(lower), float(upper), w),
        )


_NEW_CANONICALS = (
    "ts_threshold_cycle_period",
    "ts_threshold_cycle_asymmetry",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
