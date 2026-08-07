# -*- coding: utf-8 -*-
"""Local Markov / discrete state-dynamics operators (2026-08 V2/V3).

A single per-column ``DiscreteStateDynamicsKernel`` estimates, over the
strictly-past window ``[t-W, t-1]``, deterministic quantile state edges, the
lagged transition matrix (Jeffreys-smoothed), empirical state frequencies and
per-state Kramers-Moyal D1/D2 coefficients.  The family shares one state
estimate so the operators never roll seven windows independently:

* ``ts_markov_persistence``            — P_kk for the current state k (P1).
* ``ts_markov_state_entropy``          — normalized transition-row entropy (P1).
* ``ts_markov_transition_surprisal``   — ``-log P_ij`` of today's observed
  state jump (P1).
* ``ts_kramers_moyal_local_stability`` — ``-D'(x_k)``: restoring (attractor) vs
  repelling state (P1).
* ``ts_markov_entropy_production``     — rolling entropy-production rate (P2).
* ``ts_active_information_storage``    — ``I(S_t; (S_{t-1},..,S_{t-k}))`` (P2).

The strictly-past window ``[t-W, t-1]`` builds every statistic; the current
value ``x_t`` only selects a state / observed transition, never enters the
historical estimates.  Missing values fail closed (NaN) rather than being read
as a state.  All kernels are deterministic and prefix-causal.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like

_EPS = 1e-12
_LN2 = float(np.log(2.0))


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="state_dynamics",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "state_dynamics", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:price_volume",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Deterministic quantile bin edges over a window (never degenerate)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.array([-np.inf, np.inf], dtype=float)
    if n_bins < 2:
        n_bins = 2
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size == 1:
        return np.array([edges[0] - 1.0, edges[0] + 1.0], dtype=float)
    if edges.size < n_bins + 1:
        lo, hi = float(edges[0]), float(edges[-1])
        if hi - lo <= _EPS:
            lo, hi = lo - 1.0, hi + 1.0
        edges = np.linspace(lo, hi, n_bins + 1)
        edges[0] = -np.inf
        edges[-1] = np.inf
    return edges


def _bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)


def _state_dynamics_series(
    series: np.ndarray,
    window: int,
    bins: int,
    lag: int,
    min_count: int,
) -> dict[str, Any]:
    """DiscreteStateDynamicsKernel over one column.

    Returns per-row arrays:
      state  — current state index (NaN when ``x_t`` is not finite)
      edges  — quantile edges used at each row
      P      — (n, B, B) smoothed transition matrix from ``[t-W, t-1]``
      counts — (n, B) empirical state frequencies in the window
      pi     — (n, B) normalised state frequencies
      D1/D2  — (n, B) per-state Kramers-Moyal coefficients (lagged increments)
      centers— (n, B) bin centers
      total_trans — number of lagged transitions in the window
    """
    n = series.shape[0]
    B = int(bins)
    lg = int(lag)
    mc = max(1, int(min_count))
    state = np.full(n, np.nan)
    edges_out = np.full((n, B + 1), np.nan)
    P = np.full((n, B, B), np.nan)
    counts = np.full((n, B), np.nan)
    pi = np.full((n, B), np.nan)
    D1 = np.full((n, B), np.nan)
    D2 = np.full((n, B), np.nan)
    centers = np.full((n, B), np.nan)
    total_trans = np.full(n, np.nan)

    for t in range(n):
        lo = max(0, t - window)
        past = series[lo:t]                     # strictly past [t-W, t-1]
        cur = series[t]
        if not np.isfinite(cur) or lg < 1:
            continue
        finite = past[np.isfinite(past)]
        if finite.size < 2:
            continue
        edges = _quantile_edges(past, B)
        edges_out[t] = edges
        centers[t] = 0.5 * (edges[:-1] + edges[1:])
        k = int(_bin(np.asarray([cur]), edges)[0])
        state[t] = k
        states_past = _bin(past, edges)
        counts[t] = np.bincount(states_past, minlength=B).astype(float)
        total = float(len(states_past))
        pi[t] = counts[t] / max(total, 1.0)

        d1_row = np.full(B, np.nan)
        d2_row = np.full(B, np.nan)
        if len(past) > lg:
            base = past[:-lg]
            inc = past[lg:] - base
            base_bin = states_past[:-lg]
            N = np.bincount(base_bin * B + states_past[lg:], minlength=B * B).reshape(B, B)
            for bbin in range(B):
                sel = base_bin == bbin
                if int(sel.sum()) < mc:
                    continue
                dx = inc[sel]
                if dx.size == 0:
                    continue
                d1_row[bbin] = float(np.mean(dx))
                d2_row[bbin] = float(np.mean(dx * dx)) / (2.0 * lg)
        D1[t] = d1_row
        D2[t] = d2_row
        total_trans[t] = float(len(past) - lg)
        if len(past) > lg:
            N = np.bincount(base_bin * B + states_past[lg:], minlength=B * B).reshape(B, B)
            P[t] = (N + 0.5) / (N.sum(axis=1, keepdims=True) + 0.5 * B)
    return {
        "state": state,
        "edges": edges_out,
        "P": P,
        "counts": counts,
        "pi": pi,
        "D1": D1,
        "D2": D2,
        "centers": centers,
        "total_trans": total_trans,
    }


def _run_kernel(
    x: pd.DataFrame, window: int, bins: int, lag: int, min_count: int
) -> dict[str, np.ndarray]:
    w = max(2, int(window))
    b = max(2, int(bins))
    lg = max(1, int(lag))
    mc = max(1, int(min_count))
    if w <= lg:
        raise ValueError("window must exceed lag")
    cols = x.shape[1]
    keys = ["state", "P", "counts", "pi", "D1", "D2", "centers", "total_trans", "edges"]
    gathered: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    xv = x.to_numpy(dtype=float)
    for c in range(cols):
        res = _state_dynamics_series(xv[:, c], w, b, lg, mc)
        for k in keys:
            gathered[k].append(res[k])
    out: dict[str, np.ndarray] = {}
    for k in keys:
        arrs = gathered[k]
        # 1-D per column (state, total_trans) -> (n, cols); 2-D/3-D -> stack axis 1.
        out[k] = np.column_stack(arrs) if arrs[0].ndim == 1 else np.stack(arrs, axis=1)
    return out


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return frame_like(template, values)


# ---------------------------------------------------------------------------
# ts_markov_persistence
# ---------------------------------------------------------------------------

def _persistence_series(series: np.ndarray, res: dict[str, np.ndarray], col: int, min_count: int) -> np.ndarray:
    n = series.shape[0]
    out = np.full(n, np.nan)
    for t in range(n):
        k = res["state"][t, col]
        if not np.isfinite(k):
            continue
        k = int(k)
        if res["counts"][t, col, k] < min_count:
            continue
        p = res["P"][t, col, k, k]
        if np.isfinite(p):
            out[t] = float(p)
    return out


@register_operator(
    name="ts_markov_persistence",
    category="state_dynamics",
    business_category="state_dynamics",
    canonical="ts_markov_persistence",
    source="markov_dynamics",
)
class TsMarkovPersistence(SeriesOperator):
    """当前状态的历史保持概率 ``P_kk``。

    状态分箱用严格过去窗口 ``[t-W, t-1]`` 的分位数边缘；转移矩阵 P 也只用该
    窗口内的滞后对。输出当前 ``x_t`` 所在状态 k 的 ``P_kk``（进入后继续留在
    该状态的历史频率）。高 = 该状态历史上容易保持；低 = 通常很快离开。
    """

    metadata = _metadata(
        "ts_markov_persistence",
        "当前状态的历史保持概率 P_kk（分位数状态 + 滞后转移）。",
        ["x", "window", "bins", "lag", "min_count"],
        unit="probability",
        cost=4,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, bins: int = 3, lag: int = 1, min_count: int = 3, **_: Any
    ) -> pd.DataFrame:
        res = _run_kernel(x, window, bins, lag, min_count)
        xv = x.to_numpy(dtype=float)
        cols = x.shape[1]
        mc = max(1, int(min_count))
        out = np.column_stack([_persistence_series(xv[:, c], res, c, mc) for c in range(cols)])
        return _frame_like(x, out)


# ---------------------------------------------------------------------------
# ts_markov_state_entropy
# ---------------------------------------------------------------------------

def _state_entropy_series(series: np.ndarray, res: dict[str, np.ndarray], col: int, bins: int, min_count: int) -> np.ndarray:
    n = series.shape[0]
    out = np.full(n, np.nan)
    log_b = np.log(max(2, int(bins)))
    for t in range(n):
        k = res["state"][t, col]
        if not np.isfinite(k):
            continue
        k = int(k)
        if res["counts"][t, col, k] < min_count:
            continue
        row = res["P"][t, col, k]
        if not np.all(np.isfinite(row)):
            continue
        row = row[row > 0.0]
        h = -float(np.sum(row * np.log(row)))
        out[t] = float(h / log_b)
    return out


@register_operator(
    name="ts_markov_state_entropy",
    category="state_dynamics",
    business_category="state_dynamics",
    canonical="ts_markov_state_entropy",
    source="markov_dynamics",
)
class TsMarkovStateEntropy(SeriesOperator):
    """当前状态后续转移行的归一化熵 ``H_k = -Σ_j P_kj log P_kj / log B``。

    低 = 当前 state 的后续路径高度可预测；高 = 高不确定性的分叉状态。与已有
    ``ts_permutation_transition_entropy``（窗口总体转换混乱度）不同：这里回答
    "我现在所处的这个具体 state 后续有多乱"。
    """

    metadata = _metadata(
        "ts_markov_state_entropy",
        "当前状态转移行熵 H_k/log(B)（[0,1]，低=可预测）。",
        ["x", "window", "bins", "lag", "min_count"],
        unit="entropy",
        cost=4,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, bins: int = 3, lag: int = 1, min_count: int = 3, **_: Any
    ) -> pd.DataFrame:
        res = _run_kernel(x, window, bins, lag, min_count)
        xv = x.to_numpy(dtype=float)
        cols = x.shape[1]
        b = max(2, int(bins))
        mc = max(1, int(min_count))
        out = np.column_stack([_state_entropy_series(xv[:, c], res, c, b, mc) for c in range(cols)])
        return _frame_like(x, out)


# ---------------------------------------------------------------------------
# ts_markov_transition_surprisal
# ---------------------------------------------------------------------------

def _surprisal_series(series: np.ndarray, res: dict[str, np.ndarray], col: int, lag: int, min_count: int) -> np.ndarray:
    n = series.shape[0]
    out = np.full(n, np.nan)
    for t in range(n):
        edges = res["edges"][t, col]
        if not np.isfinite(edges).any():
            continue
        j = res["state"][t, col]
        if not np.isfinite(j):
            continue
        j = int(j)
        # The lagged endpoint ``t - lag`` lies strictly in the past (lag >= 1),
        # so its state under the *same* historical edges is PIT-safe.
        if t - lag < 0:
            continue
        i_val = series[t - lag]
        if not np.isfinite(i_val):
            continue
        i = int(_bin(np.asarray([i_val]), edges)[0])
        if res["counts"][t, col, i] < min_count:
            continue
        p = res["P"][t, col, i, j]
        if np.isfinite(p) and p > 0.0:
            out[t] = float(-np.log(p))
    return out


@register_operator(
    name="ts_markov_transition_surprisal",
    category="state_dynamics",
    business_category="state_dynamics",
    canonical="ts_markov_transition_surprisal",
    source="markov_dynamics",
)
class TsMarkovTransitionSurprisal(SeriesOperator):
    """当前实际发生的状态转换 ``-log P_ij`` 的历史罕见度（nats）。

    转移矩阵只用 ``[t-W, t-1]`` 构建；当前观测转换 ``i=S_{t-lag}, j=S_t`` 的
    概率从历史矩阵中读出。高 = 今天发生了历史上少见的状态跳跃（如低波动→
    极高波动）。PIT 安全：矩阵不包含 t 及以后的信息。
    """

    metadata = _metadata(
        "ts_markov_transition_surprisal",
        "当前状态跳变的历史罕见度 -log P_ij（nats）。",
        ["x", "window", "bins", "lag", "min_count"],
        unit="nats",
        cost=4,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, bins: int = 3, lag: int = 1, min_count: int = 3, **_: Any
    ) -> pd.DataFrame:
        res = _run_kernel(x, window, bins, lag, min_count)
        xv = x.to_numpy(dtype=float)
        cols = x.shape[1]
        lg = max(1, int(lag))
        mc = max(1, int(min_count))
        out = np.column_stack([_surprisal_series(xv[:, c], res, c, lg, mc) for c in range(cols)])
        return _frame_like(x, out)


# ---------------------------------------------------------------------------
# ts_markov_entropy_production (P2 research)
# ---------------------------------------------------------------------------

def _entropy_production_series(res: dict[str, np.ndarray], col: int, min_periods: int) -> np.ndarray:
    n = res["P"].shape[0]
    out = np.full(n, np.nan)
    for t in range(n):
        if res["total_trans"][t, col] < min_periods:
            continue
        P = res["P"][t, col]
        pi = res["pi"][t, col]
        if not np.all(np.isfinite(P)) or not np.all(np.isfinite(pi)):
            continue
        B = P.shape[0]
        sigma = 0.0
        for i in range(B):
            if pi[i] <= 0.0:
                continue
            for j in range(B):
                if i == j:
                    continue
                pij = P[i, j]
                pji = P[j, i]
                if pij <= 0.0 or pji <= 0.0:
                    continue
                sigma += pi[i] * pij * np.log(pij / pji)
        out[t] = float(max(sigma, 0.0))
    return out


@register_operator(
    name="ts_markov_entropy_production",
    category="state_dynamics",
    business_category="state_dynamics",
    canonical="ts_markov_entropy_production",
    source="markov_dynamics",
    status="experimental",
)
class TsMarkovEntropyProduction(SeriesOperator):
    """滚动 Markov 熵产生率 ``σ = Σ π_i P_ij log(P_ij/P_ji)``（nats）。

    窗口内经验状态频率 π 作为不变测度；对称过程 σ≈0，时间不可逆状态演化给出
    正值。这是窗口级统计（不是单状态），用于探测状态动力学的时间不可逆性。
    P2 / Research。
    """

    metadata = _metadata(
        "ts_markov_entropy_production",
        "窗口 Markov 熵产生率 Σ π_i P_ij log(P_ij/P_ji)（nats）。",
        ["x", "window", "bins", "lag", "min_periods"],
        unit="nats",
        cost=5,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, bins: int = 3, lag: int = 1, min_periods: int = 5, **_: Any
    ) -> pd.DataFrame:
        res = _run_kernel(x, window, bins, lag, 1)
        cols = x.shape[1]
        mp = max(2, int(min_periods))
        out = np.column_stack([_entropy_production_series(res, c, mp) for c in range(cols)])
        return _frame_like(x, out)


# ---------------------------------------------------------------------------
# ts_kramers_moyal_local_stability
# ---------------------------------------------------------------------------

def _local_stability_series(res: dict[str, np.ndarray], col: int, min_count: int) -> np.ndarray:
    n = res["P"].shape[0]
    out = np.full(n, np.nan)
    B = res["D1"].shape[2]
    for t in range(n):
        k = res["state"][t, col]
        if not np.isfinite(k):
            continue
        k = int(k)
        if not (0 < k < B - 1):
            continue  # boundary bins have no two-sided drift derivative
        d1 = res["D1"][t, col]
        c = res["centers"][t, col]
        if (
            not np.isfinite(d1[k - 1]) or not np.isfinite(d1[k]) or not np.isfinite(d1[k + 1])
            or not np.isfinite(c[k - 1]) or not np.isfinite(c[k + 1])
        ):
            continue
        if res["counts"][t, col, k] < min_count:
            continue
        denom = c[k + 1] - c[k - 1]
        if abs(denom) <= _EPS:
            continue
        drift_deriv = (d1[k + 1] - d1[k - 1]) / denom
        out[t] = float(-drift_deriv)
    return out


@register_operator(
    name="ts_kramers_moyal_local_stability",
    category="state_dynamics",
    business_category="state_dynamics",
    canonical="ts_kramers_moyal_local_stability",
    source="markov_dynamics",
)
class TsKramersMoyalLocalStability(SeriesOperator):
    """当前状态的局部漂移导数稳定性 ``-D'(x_k)``。

    ``D'(x_k) = (D1_{k+1} - D1_{k-1}) / (c_{k+1} - c_{k-1})``（中心差分），
    输出负值：>0 = 偏离后 drift 会把变量拉回（稳定吸引子）；<0 = drift 继续推远
    （不稳定逃逸）。与 ``ts_kramers_moyal_drift`` 共享同一分位数状态估计。
    """

    metadata = _metadata(
        "ts_kramers_moyal_local_stability",
        "当前状态局部稳定性 -D'(x_k)（>0 吸引子 / <0 排斥）。",
        ["x", "window", "bins", "lag", "min_count"],
        unit="ratio",
        cost=5,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, bins: int = 5, lag: int = 1, min_count: int = 3, **_: Any
    ) -> pd.DataFrame:
        res = _run_kernel(x, window, bins, lag, min_count)
        cols = x.shape[1]
        mc = max(1, int(min_count))
        out = np.column_stack([_local_stability_series(res, c, mc) for c in range(cols)])
        return _frame_like(x, out)


# ---------------------------------------------------------------------------
# ts_active_information_storage (P2 research)
# ---------------------------------------------------------------------------

def _ais_series(series: np.ndarray, window: int, bins: int, history_length: int) -> np.ndarray:
    n = series.shape[0]
    B = max(2, int(bins))
    k = max(1, int(history_length))
    w = max(2, int(window))
    out = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - w)
        past = series[lo:t]
        if not np.any(np.isfinite(past)):
            continue
        edges = _quantile_edges(past, B)
        S = _bin(past, edges)
        if len(S) <= k:
            continue
        joint = np.zeros((B, B ** k), dtype=np.float64)
        n_blocks = 0
        for s in range(k, len(S)):
            hcode = 0
            for m in range(k):
                hcode = hcode * B + int(S[s - k + m])
            joint[int(S[s]), hcode] += 1.0
            n_blocks += 1
        if n_blocks < 3:
            continue
        joint += 0.5
        joint /= joint.sum()
        p_s = joint.sum(axis=1)
        p_h = joint.sum(axis=0)
        ais = 0.0
        for s in range(B):
            for h in range(B ** k):
                p = joint[s, h]
                if p <= _EPS or p_s[s] <= _EPS or p_h[h] <= _EPS:
                    continue
                ais += p * (np.log(p) - np.log(p_s[s]) - np.log(p_h[h]))
        out[t] = float(max(ais, 0.0))
    return out


@register_operator(
    name="ts_active_information_storage",
    category="state_dynamics",
    business_category="state_dynamics",
    canonical="ts_active_information_storage",
    source="markov_dynamics",
    status="experimental",
)
class TsActiveInformationStorage(SeriesOperator):
    """活性信息存储 ``AIS = I(S_t; (S_{t-1},...,S_{t-k}))``（nats）。

    X 自己的 k 步历史块携带多少关于下一状态的信息——非线性多历史可预测性。
    ``bins`` 限制 [2,3]、``history_length`` 限制 [1,2] 以避免状态空间爆炸。
    P2 / Research。
    """

    metadata = _metadata(
        "ts_active_information_storage",
        "活性信息存储 I(S_t; 过去 k 状态块)（nats）。",
        ["x", "window", "bins", "history_length"],
        unit="nats",
        cost=7,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 120, bins: int = 3, history_length: int = 1, **_: Any
    ) -> pd.DataFrame:
        w = max(2, int(window))
        b = int(bins)
        k = int(history_length)
        if not 2 <= b <= 3:
            raise ValueError("ts_active_information_storage requires bins in [2, 3]")
        if not 1 <= k <= 2:
            raise ValueError("ts_active_information_storage requires history_length in [1, 2]")
        xv = x.to_numpy(dtype=float)
        cols = x.shape[1]
        out = np.column_stack([_ais_series(xv[:, c], w, b, k) for c in range(cols)])
        return _frame_like(x, out)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS)
        | {
            "ts_markov_persistence",
            "ts_markov_state_entropy",
            "ts_markov_transition_surprisal",
            "ts_kramers_moyal_local_stability",
        }
    )
    _surface.RESEARCH_ONLY_CANONICALS = frozenset(
        set(_surface.RESEARCH_ONLY_CANONICALS)
        | {
            "ts_markov_entropy_production",
            "ts_active_information_storage",
        }
    )


_register_surface()
