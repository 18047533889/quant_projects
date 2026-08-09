# -*- coding: utf-8 -*-
"""Historical event-response learning operators (2026-08 V2/V3).

The event-response language answers "given that this kind of event happened
before, what typically happened afterwards?" — the machine learns responses
from history instead of hard-coding what an event means.

* ``event_historical_response_mean``        — mean horizon response after past
  events (P1).
* ``event_historical_response_sign_balance``— mean sign of those responses;
  high magnitude with low sign balance flags a few extreme winners (P1).
* ``event_hawkes_branching_ratio_proxy``    — mean number of exponential-decay
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


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int, output_unit: str | None = None) -> OperatorMetadata:
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
        output_unit=output_unit,
    )


def _aligned(rv: np.ndarray, ev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(rv) & np.isfinite(ev)
    return rv[ok], ev[ok]


def _collapse_events(event_times: np.ndarray, refractory: int) -> np.ndarray:
    """First-event-of-episode collapse (P1, round 7).

    Consecutive events separated by fewer than ``refractory`` bars belong to one
    episode — 5 consecutive limit-ups must not count as 5 independent overlapping
    response paths.  Returns only the FIRST event of each episode.
    """
    if refractory < 1 or event_times.size == 0:
        return event_times
    firsts: list[int] = []
    prev = -10**9
    for s in event_times:
        if s - prev >= refractory:
            firsts.append(int(s))
        prev = int(s)
    return np.asarray(firsts, dtype=int)


def _event_episode_metrics_series(
    event: np.ndarray, history_window: int, horizon: int, refractory: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row episode quality metrics ``(effective_events, overlap_ratio)``.

    ``effective_events`` is the number of *first events of episodes* in the
    cohort ``[t-W, t-H]`` (same cohort the mean/dispersion kernels use);
    ``overlap_ratio`` is ``1 - effective/raw`` — how much of the raw event count
    is absorbed by episode collapse.  Both are 0/NaN when there are no events.
    """
    n = event.shape[0]
    hw = max(2, int(history_window))
    H = max(1, int(horizon))
    refr = max(0, int(refractory))
    eff = np.full(n, np.nan)
    ovr = np.full(n, np.nan)
    times = np.flatnonzero(np.isfinite(event) & (event != 0.0))
    for t in range(n):
        lo = max(0, t - hw)
        last_event = t - H
        if last_event < lo:
            continue
        ev = times[(times >= lo) & (times <= last_event)]
        if ev.size < 1:
            continue
        raw = int(ev.size)
        firsts = _collapse_events(ev, refr) if refr > 0 else ev
        eff[t] = int(firsts.size)
        ovr[t] = 1.0 - firsts.size / raw
    return eff, ovr


def _horizon_response(
    response: np.ndarray,
    event: np.ndarray,
    history_window: int,
    horizon: int,
    mode: str,
    min_events: int,
    sign_balance: bool,
    require_full_horizon: bool = True,
    refractory: int = 0,
) -> np.ndarray:
    n = response.shape[0]
    hw = max(2, int(history_window))
    H = max(1, int(horizon))
    me = max(1, int(min_events))
    refr = max(0, int(refractory))
    out = np.full(n, np.nan)
    times = np.flatnonzero(np.isfinite(event) & (event != 0.0))
    for t in range(n):
        lo = max(0, t - hw)
        last_event = t - H                 # need s + H <= t
        if last_event < lo:
            continue
        ev = times[(times >= lo) & (times <= last_event)]
        # P1 (round 7): episode collapse — consecutive events within
        # ``refractory`` bars are one episode; only the first event's response
        # path enters, so 5 consecutive limit-ups do not count 5 times.
        if refr > 0:
            ev = _collapse_events(ev, refr)
        rs: list[float] = []
        for s in ev:
            window_vals = response[s + 1 : s + H + 1]
            # P1-005: unify cohort selection with the shape ops
            # (peak_lag/decay/dispersion/reversal): an event only enters the
            # cohort when its ENTIRE response path s+1..s+H is fully observed.
            # Partial-path averaging (a gap quietly shortened the horizon) would
            # make mean/peak-lag compare different effective horizons.
            if require_full_horizon:
                if not np.all(np.isfinite(window_vals)):
                    continue
                seg = window_vals.astype(float)
            else:
                finite = window_vals[np.isfinite(window_vals)]
                if finite.size == 0:
                    continue
                seg = finite
            if mode == "sum":
                rs.append(float(seg.sum()))
            else:
                rs.append(float(seg.mean()))
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
    控制），输出这些历史事件的响应均值。输出单位继承 ``response``（返回序列的
    均值仍是返回单位），不是无量纲比值；只有 ``sign_balance`` /
    ``peak_lag_normalized`` 这类才是 ratio。事件与响应都由外部输入定义（涨停/
    炸板/成交量冲击/财报 surprise 都可），算子本身只学"过去这种事件发生后通常
    发生什么"。PIT 安全、确定性。
    """

    metadata = _metadata(
        "event_historical_response_mean",
        "历史事件后的平均 horizon 响应（sum/mean 模式，full-horizon cohort，可按 episode 聚合）。",
        ["response", "event", "history_window", "horizon", "mode", "min_events", "require_full_horizon", "refractory"],
        unit="same_as:response",
        output_unit="same_as:response",
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
        require_full_horizon: bool = True,
        refractory: int = 0,
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
                rv[:, c], ev[:, c], history_window, horizon, m, min_events,
                sign_balance=False, require_full_horizon=bool(require_full_horizon),
                refractory=refractory,
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
        "历史事件响应符号平衡 mean(sign(R_s))（[-1,1]，full-horizon cohort）。",
        ["response", "event", "history_window", "horizon", "min_events", "require_full_horizon", "refractory"],
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
        require_full_horizon: bool = True,
        refractory: int = 0,
        **_: Any,
    ) -> pd.DataFrame:
        rv = response.to_numpy(dtype=float)
        ev = event.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            out[:, c] = _horizon_response(
                rv[:, c], ev[:, c], history_window, horizon, "mean", min_events,
                sign_balance=True, require_full_horizon=bool(require_full_horizon),
                refractory=refractory,
            )
        return frame_like(response, out)


# ---------------------------------------------------------------------------
# event_hawkes_branching_ratio_proxy (P2 research)
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
    # P1 (round 7): observed-event-clock / exposure mask — a NaN row is NOT
    # "no event", it is missing coverage.  Gaps advance time but hide events, so
    # without this mask they look like quiet periods and bias the rate down.
    obs = np.isfinite(event)
    for t in range(n):
        lo = max(0, t - w)
        events_in = times[(times >= lo) & (times < t)]
        # P1 (round 7): right-censoring — a parent within ``max_lag`` of ``t``
        # has not yet observed its full offspring window; counting it in the
        # denominator biases the proxy down when events are recent.  Only
        # *matured* parents (``parent_time + max_lag <= t - 1``) contribute.
        matured = events_in[events_in + L <= t - 1]
        if matured.size < me:
            continue
        total = 0.0
        denom = 0.0
        for t_i in matured:
            window_obs = obs[t_i + 1 : t_i + L + 1]
            exposure = float(window_obs.mean()) if window_obs.size else 0.0
            if exposure <= 0.0:
                continue  # no observed coverage for this parent -> no information
            after = events_in[(events_in > t_i) & (events_in <= t_i + L)]
            # Only count offspring reachable through a fully-observed path: a
            # NaN gap may hide events and must not be read as "no event".
            contrib = 0.0
            for t_j in after:
                if obs[t_i + 1 : t_j + 1].all():
                    contrib += float(np.exp(-beta * (t_j - t_i)))
            total += contrib
            # Effective parent count = exposure-weighted, so a partially
            # observed window counts proportionally (gaps look like "no events"
            # otherwise).
            denom += exposure
        if denom <= _EPS:
            continue
        out[t] = float(total / denom)
    return out


@register_operator(
    name="event_hawkes_branching_ratio_proxy",
    category="event_response",
    business_category="event_response",
    canonical="event_hawkes_branching_ratio_proxy",
    source="event_response",
    status="experimental",
)
class EventHawkesBranchingRatio(SeriesOperator):
    """Hawkes 分支比**代理**：每事件在 ``max_lag`` 内的指数衰减后代均值。

    ``n* = mean_i Σ_{j: t_i<t_j≤t_i+L} exp(-β(t_j-t_i))``，``β=3/max_lag``。
    只统计*成熟*父事件（``parent_time + max_lag ≤ t-1``，避免右删失把近期事件
    分母压低）；对每个父事件的 offspring 窗口按观测覆盖（``obs = isfinite``）
    做曝光修正，NaN 缺口不会假装成"没有事件"。注意这是固定指数核的后代聚集
    proxy，**没有**拟合真正的 Hawkes ``λ(t)=μ+Σαe^{-β(t-t_i)}`` 也没有 MLE 求
    ``α/β``；命名用 ``_proxy`` 以免后续被误读为拟合的分支比。度量事件自激/聚集
    强度；regime shift 与模型误设会产生虚假高值，因此仅 P2 / Research。
    """

    metadata = _metadata(
        "event_hawkes_branching_ratio_proxy",
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


# ---------------------------------------------------------------------------
# event_response curve shape (P1 deepening)
# ---------------------------------------------------------------------------

def _response_curve_stats_series(
    response: np.ndarray,
    event: np.ndarray,
    history_window: int,
    horizon: int,
    min_events: int,
    refractory: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-row event-response *curve* statistics (strictly causal, s+H <= t).

    For each past event with a fully observed horizon path ``[s+1, s+H]`` build
    the per-lag response vector ``r_s(ℓ) = response[s+ℓ]``; the mean curve is
    ``m(ℓ) = mean_s r_s(ℓ)``.  Returns ``(peak_lag, decay, dispersion, reversal)``:
      peak_lag   — ``argmax_ℓ |m(ℓ)| / H`` (when does the response typically peak)
      decay      — OLS slope of ``log(|m(ℓ)|+ε)`` vs ``ℓ`` (negative = decaying;
                   near 0 = persistent; positive = late build-up)
      dispersion — ``std(R_s) / mean(|R_s|)`` over per-event total responses
                   R_s = Σ_ℓ r_s(ℓ) (heavy-tailed response distribution > 1)
      reversal   — early/late sign flip of the mean curve, signed magnitude
                   ``(m_late - m_early) / mean(|m|)``, 0 when no flip

    ``refractory`` (P1, round 7) collapses consecutive events within that many
    bars into episodes (first-event-only) so overlapping response paths are not
    double counted when clustering dispersion / significance by episode.
    """
    n = response.shape[0]
    hw = max(2, int(history_window))
    H = max(1, int(horizon))
    me = max(1, int(min_events))
    refr = max(0, int(refractory))
    peak_lag = np.full(n, np.nan)
    decay = np.full(n, np.nan)
    dispersion = np.full(n, np.nan)
    reversal = np.full(n, np.nan)
    times = np.flatnonzero(np.isfinite(event) & (event != 0.0))
    for t in range(n):
        lo = max(0, t - hw)
        last_event = t - H
        if last_event < lo:
            continue
        ev = times[(times >= lo) & (times <= last_event)]
        if refr > 0:
            ev = _collapse_events(ev, refr)
        curves: list[np.ndarray] = []
        totals: list[float] = []
        for s in ev:
            seg = response[s + 1 : s + H + 1]
            if seg.size != H or not np.all(np.isfinite(seg)):
                continue
            curves.append(seg.astype(float))
            totals.append(float(seg.sum()))
        if len(curves) < me:
            continue
        C = np.vstack(curves)                       # (n_ev, H)
        m = C.mean(axis=0)                          # mean response curve
        scale_curve = float(np.mean(np.abs(m))) if H else np.nan
        if np.isfinite(scale_curve) and scale_curve > _EPS:
            peak_lag[t] = float(int(np.argmax(np.abs(m))) + 1) / H
            half = max(1, H // 2)
            early = float(np.mean(m[:half])) if half > 0 else 0.0
            late = float(np.mean(m[half:])) if H - half > 0 else 0.0
            if early * late < 0.0:
                reversal[t] = float((late - early) / scale_curve)
            # decay via OLS of log(|m|+eps) on lag index
            y = np.log(np.abs(m) + _EPS)
            xs = np.arange(1.0, H + 1.0)
            denom = float(np.sum((xs - xs.mean()) ** 2))
            if denom > _EPS:
                decay[t] = float(np.sum((xs - xs.mean()) * (y - y.mean())) / denom)
        tot = np.asarray(totals, dtype=float)
        mean_abs = float(np.mean(np.abs(tot)))
        if tot.size >= 2 and mean_abs > _EPS and np.isfinite(tot).all():
            dispersion[t] = float(np.std(tot, ddof=1) / mean_abs)
    return peak_lag, decay, dispersion, reversal


@register_operator(
    name="event_response_peak_lag",
    category="event_response",
    business_category="event_response",
    canonical="event_response_peak_lag",
    source="event_response",
)
class EventResponsePeakLag(SeriesOperator):
    """历史事件响应的峰值时滞 ``argmax_ℓ |m(ℓ)| / H``。

    m(ℓ) 为过去事件（``s+H ≤ t``、路径全程可观测）的逐 lag 平均响应曲线；输出
    平均响应绝对值最大的 lag 相对 H 的归一位置。0 = 事件后立即见峰值（即时
    冲击）；接近 1 = 响应在窗口后期才见顶（滞后兑现）。PIT 安全、确定性。
    """

    metadata = _metadata(
        "event_response_peak_lag",
        "平均响应曲线峰值时滞 argmax|m(ℓ)|/H（[0,1]）。",
        ["response", "event", "history_window", "horizon", "min_events"],
        unit="ratio",
        cost=5,
    )

    def _calculate_series(
        self,
        response: pd.DataFrame,
        event: pd.DataFrame,
        history_window: int = 120,
        horizon: int = 10,
        min_events: int = 3,
        **_: Any,
    ) -> pd.DataFrame:
        rv = response.to_numpy(dtype=float)
        ev = event.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            pl, _, _, _ = _response_curve_stats_series(rv[:, c], ev[:, c], history_window, horizon, min_events)
            out[:, c] = pl
        return frame_like(response, out)


@register_operator(
    name="event_response_decay_rate",
    category="event_response",
    business_category="event_response",
    canonical="event_response_decay_rate",
    source="event_response",
)
class EventResponseDecayRate(SeriesOperator):
    """历史事件响应曲线的衰减速率（``log|m(ℓ)|`` 对 ℓ 的 OLS 斜率）。

    负 = 响应随 lag 衰减（事件冲击逐步消退）；近 0 = 响应持久；正 = 响应后期
    才积累。与 ``event_response_peak_lag`` 共享同一响应曲线内核。只用已完成
    事件，无前视。
    """

    metadata = _metadata(
        "event_response_decay_rate",
        "平均响应曲线 log|m(ℓ)| 对 ℓ 的斜率（负=衰减）。",
        ["response", "event", "history_window", "horizon", "min_events"],
        unit="ratio",
        cost=5,
    )

    def _calculate_series(
        self,
        response: pd.DataFrame,
        event: pd.DataFrame,
        history_window: int = 120,
        horizon: int = 10,
        min_events: int = 3,
        **_: Any,
    ) -> pd.DataFrame:
        rv = response.to_numpy(dtype=float)
        ev = event.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            _, dec, _, _ = _response_curve_stats_series(rv[:, c], ev[:, c], history_window, horizon, min_events)
            out[:, c] = dec
        return frame_like(response, out)


@register_operator(
    name="event_response_dispersion",
    category="event_response",
    business_category="event_response",
    canonical="event_response_dispersion",
    source="event_response",
)
class EventResponseDispersion(SeriesOperator):
    """历史事件响应分布的离散度 ``std(R_s)/mean|R_s|``（CV 型）。

    R_s = 事件 s 的 horizon 总响应。>1 = 响应被少数大事件主导（胖尾）；≈0 =
    响应高度一致。与 ``event_historical_response_sign_balance`` 互补：sign
    balance 看方向广泛性，dispersion 看规模均匀性。只用已完成事件。
    """

    metadata = _metadata(
        "event_response_dispersion",
        "事件响应分布离散度 std(R)/mean|R|（>1 胖尾，可按 episode 聚合）。",
        ["response", "event", "history_window", "horizon", "min_events", "refractory"],
        unit="ratio",
        cost=5,
    )

    def _calculate_series(
        self,
        response: pd.DataFrame,
        event: pd.DataFrame,
        history_window: int = 120,
        horizon: int = 10,
        min_events: int = 3,
        refractory: int = 0,
        **_: Any,
    ) -> pd.DataFrame:
        rv = response.to_numpy(dtype=float)
        ev = event.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            _, _, disp, _ = _response_curve_stats_series(
                rv[:, c], ev[:, c], history_window, horizon, min_events, refractory=refractory
            )
            out[:, c] = disp
        return frame_like(response, out)


@register_operator(
    name="event_response_reversal_strength",
    category="event_response",
    business_category="event_response",
    canonical="event_response_reversal_strength",
    source="event_response",
)
class EventResponseReversalStrength(SeriesOperator):
    """历史事件响应曲线首尾反转强度（早期↔晚期符号反转时≠0）。

    定义 early=前 half 个 lag 均值、late=后 half 个 lag 均值；当 early·late<0
    时输出 ``(late-early)/mean|m(ℓ)|``（带符号，late 方向），否则 0。高正值 =
    事件后先跌后强势反弹（V 型）；负 = 先涨后回落（冲高回落）。无反转 = 0。
    确定性、PIT 安全。
    """

    metadata = _metadata(
        "event_response_reversal_strength",
        "响应曲线首尾反转强度（V 型>0 / 冲高回落<0 / 无反转=0）。",
        ["response", "event", "history_window", "horizon", "min_events"],
        unit="ratio",
        cost=5,
    )

    def _calculate_series(
        self,
        response: pd.DataFrame,
        event: pd.DataFrame,
        history_window: int = 120,
        horizon: int = 10,
        min_events: int = 3,
        **_: Any,
    ) -> pd.DataFrame:
        rv = response.to_numpy(dtype=float)
        ev = event.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            _, _, _, rev = _response_curve_stats_series(rv[:, c], ev[:, c], history_window, horizon, min_events)
            out[:, c] = rev
        return frame_like(response, out)


@register_operator(
    name="event_response_effective_events",
    category="event_response",
    business_category="event_response",
    canonical="event_response_effective_events",
    source="event_response",
)
class EventResponseEffectiveEvents(SeriesOperator):
    """历史响应 cohort 的有效事件数（episode 首事件数）。

    P1 (round 7) 质量度量：``refractory`` 把间隔 < refractory 的连续事件折叠为
    一个 episode，只计首事件。5 个连续涨停 → 有效事件数 1（而非 5）。与
    ``event_response_overlap_ratio`` 一起用于判断 mean/dispersion 的显著性是否被
    重叠响应路径灌水。refractory=None 时默认等于 horizon（重叠响应路径即同 episode）。
    """

    metadata = _metadata(
        "event_response_effective_events",
        "episode 折叠后的有效事件数（refractory=None 默认=horizon）。",
        ["event", "history_window", "horizon", "refractory"],
        unit="count",
        cost=3,
    )

    def _calculate_series(
        self,
        event: pd.DataFrame,
        history_window: int = 120,
        horizon: int = 5,
        refractory: int | None = None,
        **_: Any,
    ) -> pd.DataFrame:
        ev = event.to_numpy(dtype=float)
        rows, cols = ev.shape
        refr = int(horizon) if refractory is None else int(refractory)
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            eff, _ = _event_episode_metrics_series(ev[:, c], history_window, horizon, refr)
            out[:, c] = eff
        return frame_like(event, out)


@register_operator(
    name="event_response_overlap_ratio",
    category="event_response",
    business_category="event_response",
    canonical="event_response_overlap_ratio",
    source="event_response",
)
class EventResponseOverlapRatio(SeriesOperator):
    """历史响应 cohort 的重叠比 ``1 - effective/raw`` ∈ [0,1]。

    P1 (round 7) 质量度量：raw 事件中多大比例被 episode 折叠吸收（连续涨停 /
    同一事件簇内的重叠响应路径）。0 = 无重叠（事件彼此独立）；接近 1 = 几乎全是
    同一 episode 的连续触发。配合 ``event_response_effective_events`` 判断
    mean/dispersion 显著性是否被独立事件数夸大。
    """

    metadata = _metadata(
        "event_response_overlap_ratio",
        "episode 重叠比 1-effective/raw（[0,1]，质量诊断）。",
        ["event", "history_window", "horizon", "refractory"],
        unit="ratio",
        cost=3,
    )

    def _calculate_series(
        self,
        event: pd.DataFrame,
        history_window: int = 120,
        horizon: int = 5,
        refractory: int | None = None,
        **_: Any,
    ) -> pd.DataFrame:
        ev = event.to_numpy(dtype=float)
        rows, cols = ev.shape
        refr = int(horizon) if refractory is None else int(refractory)
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            _, ovr = _event_episode_metrics_series(ev[:, c], history_window, horizon, refr)
            out[:, c] = ovr
        return frame_like(event, out)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({
            "event_historical_response_mean",
            "event_historical_response_sign_balance",
            "event_response_peak_lag",
            "event_response_decay_rate",
            "event_response_dispersion",
            "event_response_reversal_strength",
            "event_response_effective_events",
            "event_response_overlap_ratio",
        })
    _surface.extend_research_only({"event_hawkes_branching_ratio_proxy"})
    # P1-006: the pre-rename name stays as an alias to the honest _proxy canonical
    # (registered here, AFTER the class decorator registered the target).
    from cleaned_operators.registry import OperatorRegistry

    OperatorRegistry.register_alias(
        "event_hawkes_branching_ratio", "event_hawkes_branching_ratio_proxy"
    )


_register_surface()
