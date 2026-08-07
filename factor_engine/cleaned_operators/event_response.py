# -*- coding: utf-8 -*-
"""Historical event-response learning operators (2026-08 V2/V3).

The event-response language answers "given that this kind of event happened
before, what typically happened afterwards?" — the machine learns responses
from history instead of hard-coding what an event means.

* ``event_historical_response_mean``        — mean horizon response after past
  events (P1).
* ``event_historical_response_sign_balance``— mean sign of those responses;
  high magnitude with low sign balance flags a few extreme winners (P1).
* ``event_hawkes_branching_ratio``          — mean number of exponential-decay
  offspring per event: a self-excitation / clustering proxy (P2 research).

``response`` and ``event`` are supplied panels (return / volume change /
fundamental change as the response; any 0/1 indicator as the event).  Only
events with ``s + horizon <= t`` enter the estimate, so the output is
prefix-causal.  Deterministic.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like

_EPS = 1e-12


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="event_response",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "event_response", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:price_volume",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _aligned(rv: np.ndarray, ev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(rv) & np.isfinite(ev)
    return rv[ok], ev[ok]


def _horizon_response(
    response: np.ndarray,
    event: np.ndarray,
    history_window: int,
    horizon: int,
    mode: str,
    min_events: int,
    sign_balance: bool,
) -> np.ndarray:
    n = response.shape[0]
    hw = max(2, int(history_window))
    H = max(1, int(horizon))
    me = max(1, int(min_events))
    out = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - hw)
        last_event = t - H                 # need s + H <= t
        if last_event < lo:
            continue
        rs: list[float] = []
        for s in range(lo, last_event + 1):
            if not np.isfinite(event[s]) or event[s] == 0.0:
                continue
            if not np.isfinite(response[s + 1]):
                continue
            if not np.isfinite(response[s + H]):
                continue
            window_vals = response[s + 1 : s + H + 1]
            finite = window_vals[np.isfinite(window_vals)]
            if finite.size == 0:
                continue
            if mode == "sum":
                rs.append(float(finite.sum()))
            else:
                rs.append(float(finite.mean()))
        if len(rs) < me:
            continue
        if sign_balance:
            out[t] = float(np.mean(np.sign(rs)))
        else:
            out[t] = float(np.mean(rs))
    return out


@register_operator(
    name="event_historical_response_mean",
    category="event_response",
    business_category="event_response",
    canonical="event_historical_response_mean",
    source="event_response",
)
class EventHistoricalResponseMean(SeriesOperator):
    """历史事件的平均 horizon 响应 ``mean_s(R_s)``。

    事件 s 满足 ``s+H ≤ t``；``R_s = sum/mean(response[s+1..s+H])``（mode
    控制），输出这些历史事件的响应均值。事件与响应都由外部输入定义（涨停/
    炸板/成交量冲击/财报 surprise 都可），算子本身只学"过去这种事件发生后通常
    发生什么"。PIT 安全、确定性。
    """

    metadata = _metadata(
        "event_historical_response_mean",
        "历史事件后的平均 horizon 响应（sum/mean 模式）。",
        ["response", "event", "history_window", "horizon", "mode", "min_events"],
        unit="ratio",
        cost=4,
    )

    def _calculate_series(
        self,
        response: pd.DataFrame,
        event: pd.DataFrame,
        history_window: int = 120,
        horizon: int = 5,
        mode: str = "mean",
        min_events: int = 5,
        **_: Any,
    ) -> pd.DataFrame:
        m = str(mode).lower()
        if m not in {"sum", "mean"}:
            raise ValueError("event_historical_response_mean requires mode in {'sum','mean'}")
        rv = response.to_numpy(dtype=float)
        ev = event.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            out[:, c] = _horizon_response(
                rv[:, c], ev[:, c], history_window, horizon, m, min_events, sign_balance=False
            )
        return frame_like(response, out)


@register_operator(
    name="event_historical_response_sign_balance",
    category="event_response",
    business_category="event_response",
    canonical="event_historical_response_sign_balance",
    source="event_response",
)
class EventHistoricalResponseSignBalance(SeriesOperator):
    """历史事件响应的符号平衡 ``mean_s(sign(R_s))`` ∈ [-1,1]。

    mean 响应 +3% 但 sign balance 接近 0 = 平均被少数极端事件拉高；sign
    balance 接近 +0.9 = 绝大多数事件都涨。对自动搜索判断"响应是否稳健/广泛"
    很关键。共享 event-response kernel。PIT 安全、确定性。
    """

    metadata = _metadata(
        "event_historical_response_sign_balance",
        "历史事件响应符号平衡 mean(sign(R_s))（[-1,1]）。",
        ["response", "event", "history_window", "horizon", "min_events"],
        unit="ratio",
        cost=4,
    )

    def _calculate_series(
        self,
        response: pd.DataFrame,
        event: pd.DataFrame,
        history_window: int = 120,
        horizon: int = 5,
        min_events: int = 5,
        **_: Any,
    ) -> pd.DataFrame:
        rv = response.to_numpy(dtype=float)
        ev = event.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            out[:, c] = _horizon_response(
                rv[:, c], ev[:, c], history_window, horizon, "mean", min_events, sign_balance=True
            )
        return frame_like(response, out)


# ---------------------------------------------------------------------------
# event_hawkes_branching_ratio (P2 research)
# ---------------------------------------------------------------------------

def _hawkes_branching_ratio_series(event: np.ndarray, window: int, max_lag: int, min_events: int) -> np.ndarray:
    n = event.shape[0]
    w = max(2, int(window))
    L = max(1, int(max_lag))
    me = max(2, int(min_events))
    # beta chosen so the exponential kernel support is ~ max_lag bars.
    beta = 3.0 / max(L, 1)
    out = np.full(n, np.nan)
    times = np.flatnonzero(np.isfinite(event) & (event != 0.0))
    for t in range(n):
        lo = max(0, t - w)
        events_in = times[(times >= lo) & (times < t)]
        if events_in.size < me:
            continue
        total = 0.0
        for i in range(events_in.shape[0]):
            t_i = events_in[i]
            after = events_in[(events_in > t_i) & (events_in <= t_i + L)]
            if after.size == 0:
                continue
            dt = after.astype(np.float64) - t_i
            total += float(np.sum(np.exp(-beta * dt)))
        out[t] = float(total / events_in.shape[0])
    return out


@register_operator(
    name="event_hawkes_branching_ratio",
    category="event_response",
    business_category="event_response",
    canonical="event_hawkes_branching_ratio",
    source="event_response",
    status="experimental",
)
class EventHawkesBranchingRatio(SeriesOperator):
    """Hawkes 分支比代理：每事件在 ``max_lag`` 内的指数衰减后代均值。

    ``n* = mean_i Σ_{j: t_i<t_j≤t_i+L} exp(-β(t_j-t_i))``，``β=3/max_lag``。
    度量事件自激/聚集强度；regime shift 与模型误设会产生虚假高值，因此仅
    P2 / Research，不作为默认高权重搜索基元。
    """

    metadata = _metadata(
        "event_hawkes_branching_ratio",
        "Hawkes 分支比代理 mean(指数衰减后代数/事件)。",
        ["event", "window", "max_lag", "min_events"],
        unit="ratio",
        cost=6,
    )

    def _calculate_series(
        self, event: pd.DataFrame, window: int = 120, max_lag: int = 10, min_events: int = 5, **_: Any
    ) -> pd.DataFrame:
        ev = event.to_numpy(dtype=float)
        rows, cols = ev.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            out[:, c] = _hawkes_branching_ratio_series(ev[:, c], window, max_lag, min_events)
        return frame_like(event, out)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS)
        | {"event_historical_response_mean", "event_historical_response_sign_balance"}
    )
    _surface.RESEARCH_ONLY_CANONICALS = frozenset(
        set(_surface.RESEARCH_ONLY_CANONICALS) | {"event_hawkes_branching_ratio"}
    )


_register_surface()
