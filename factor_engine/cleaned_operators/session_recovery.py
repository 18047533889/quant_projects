# -*- coding: utf-8 -*-
"""Intraday shock-recovery / resilience (2026-08 V2, P1, minute → daily).

``session_event_recovery_score`` measures how fast the day's event shocks get
absorbed.  For each event minute ``s`` the base ``B_s = x_{s-1}`` and the shock
``A_s = |x_s - B_s|``; the first ``k ∈ [1, horizon]`` with
``|x_{s+k} - B_s| ≤ residual_fraction * A_s`` gives ``τ_s`` (else ``H+1``).
The daily score is ``median_s τ_s / (H+1)``.

Low = shocks absorbed quickly (liquidity resilient); high = shock impact
persists (fragile).  ``event`` is a user-supplied 0/1 minute indicator (return
shock / volume shock / limit approach / imbalance shock); the operator does not
hard-code what an event is.

This is an EOD daily factor: it may use the whole day's minute data, so its
policy carries ``available_at = session_close`` and it must never be used as a
mid-session real-time minute signal.  Deterministic and prefix-causal.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.microstructure.intraday_agg import _as_panel

_EPS = 1e-12


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="intraday_microstructure",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "intraday", "daily_agg", "minute", "pit_safe", "causal", "typed_v2",
            "deterministic", "session_close",
            f"signature:{','.join(params)}->series", "domain:price_volume",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _recovery_day(x: np.ndarray, event: np.ndarray, horizon: int, residual_fraction: float) -> float:
    H = max(1, int(horizon))
    c = max(_EPS, float(residual_fraction))
    n = x.shape[0]
    taus: list[float] = []
    for s in range(1, n):
        if not np.isfinite(event[s]) or event[s] == 0.0:
            continue
        b = x[s - 1]
        cur = x[s]
        if not np.isfinite(b) or not np.isfinite(cur):
            continue
        a = abs(cur - b)
        if a <= _EPS:
            continue
        tau = H + 1
        for k in range(1, H + 1):
            if s + k >= n:
                break
            v = x[s + k]
            if not np.isfinite(v):
                continue
            if abs(v - b) <= c * a:
                tau = k
                break
        taus.append(float(tau))
    if not taus:
        return np.nan
    return float(np.median(taus) / (H + 1))


@register_operator(
    name="session_event_recovery_score",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="session_event_recovery_score",
    source="session_recovery",
)
class SessionEventRecoveryScore(SeriesOperator):
    """日内事件冲击恢复得分（EOD，``available_at=session_close``）。

    对每个事件分钟 s，测价格回到冲击前基线的首个 horizon 内位置；输出
    ``median_s τ_s/(H+1)`` ∈ (0,1]。低 = 冲击快速被吸收；高 = 影响持续。
    PIT 安全仅限日终因子，禁止盘中实时信号。
    """

    metadata = _metadata(
        "session_event_recovery_score",
        "日内事件冲击恢复得分 median τ/(H+1)（EOD 因子）。",
        ["x", "event", "horizon", "residual_fraction"],
        unit="ratio",
        cost=4,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        event: pd.DataFrame,
        horizon: int = 10,
        residual_fraction: float = 0.25,
        **_: Any,
    ) -> pd.DataFrame:
        x = _as_panel(x)
        event = _as_panel(event)
        out: dict[str, pd.Series] = {}
        for inst in x.columns:
            col = x[inst]
            ev = event[inst] if inst in event.columns else None
            per_day: dict[pd.Timestamp, float] = {}
            for day, group in col.groupby(col.index.normalize()):
                vals = np.asarray(group, dtype=float)
                if ev is None:
                    per_day[day] = np.nan
                    continue
                ev_vals = np.asarray(ev.reindex(group.index), dtype=float)
                if not np.any(np.isfinite(vals)):
                    per_day[day] = np.nan
                    continue
                per_day[day] = _recovery_day(vals, ev_vals, horizon, residual_fraction)
            out[inst] = pd.Series(per_day, dtype=float)
        if not out:
            return pd.DataFrame(dtype=float)
        return pd.DataFrame(out).sort_index()


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | {"session_event_recovery_score"}
    )


_register_surface()
