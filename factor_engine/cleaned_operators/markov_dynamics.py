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

R11 round-2 P0 degenerate-state contract (shared with the Polars twin):

* Transition rows are estimated ONLY from ``>= min_count`` observed lagged
  transitions.  A state that appears in the window but has no observed outgoing
  transitions gets an all-NaN row — never a 0/1 row or a uniform-prior row
  presented as a confident estimate.  ``min_count`` is enforced as a relational
  feasibility ``min_count <= window - lag`` at the call boundary and in the
  kernel, so a guaranteed-NaN combination is rejected before rolling.
* The only prior is the Jeffreys (+0.5 per-cell) pseudo-count, applied to the
  observed-support counts.  A destination with zero observed incoming
  transitions therefore keeps at most a small prior-only contribution; consumers
  gate on observed incoming support (``N_obs``) so such a cell is never
  *presented* as a confident estimate (reverse direction of the prior-only rule).
* A window that observed only a single state has no estimated transition
  distribution: persistence, state entropy, transition/stationary surprisal,
  committor, MFPT, spectral gap and entropy production all fail closed to NaN
  instead of reporting a degenerate ~1 / ~0 / trivial value.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, RelationalParamSpec, SeriesOperator, register_operator
from cleaned_operators.closure.strict_scalar import strict_int, strict_float
from cleaned_operators.rolling_pack import frame_like

_EPS = 1e-12
_LN2 = float(np.log(2.0))

# R11 round-2 P0: ``min_count`` / ``min_periods`` are *relational* feasibility
# gates — a window ``[t-W, t-1]`` holds at most ``window - lag`` lagged
# transitions, so a minimum above that is guaranteed-NaN for every window and is
# rejected at the call boundary (binding) before the kernel rolls.  At runtime a
# window with fewer than the minimum *observed* valid transitions emits NaN.
_MIN_COUNT_RELATIONAL = [
    RelationalParamSpec(
        "min_count <= window - lag",
        "min_count must not exceed available transitions (window - lag); "
        "got min_count={min_count}, window={window}, lag={lag}",
    )
]
_MIN_PERIODS_RELATIONAL = [
    RelationalParamSpec(
        "min_periods <= window - lag",
        "min_periods must not exceed available transitions (window - lag); "
        "got min_periods={min_periods}, window={window}, lag={lag}",
    )
]


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    unit: str,
    cost: int,
    relational_specs: list[RelationalParamSpec] | None = None,
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="state_dynamics",
        description=description,
        param_names=params,
        return_type="series",
        relational_specs=list(relational_specs) if relational_specs else [],
        tags=[
            "state_dynamics", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:price_volume",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Deterministic tie-aware quantile bin edges over a window; n_bins+1 edges.

    The raw quantile boundaries are used as-is (never silently switched to
    equal-width linspace between runs — audit P1 tie-aware discretizer).  Under
    heavy ties (A-share limit-up/down, integer states) consecutive quantile
    boundaries coincide; those duplicate edges are kept, which yields zero-width
    cells that stay empty — i.e. fewer *effective* states — the honest reflection
    of the tied data.  ``_bin`` assigns a value to the *leftmost* cell whose
    right edge is ``>= value`` (right-closed intervals, like ``pandas.cut``), so
    a tied-but-distinct value (e.g. a 1-vs-100 bimodal series) still lands in a
    distinct cell from the tied bulk.  Only a genuinely constant / all-NaN
    window gets a fixed finite spread (documented degenerate fallback; edges are
    finite so bin centers can never be ±Inf, audit P0).
    """
    finite = values[np.isfinite(values)]
    B = max(2, int(n_bins))
    if finite.size == 0:
        return np.linspace(-1.0, 1.0, B + 1)
    raw = np.quantile(finite, np.linspace(0.0, 1.0, B + 1))
    if np.ptp(raw) <= _EPS:
        v = float(raw[0])
        return np.linspace(v - 1.0, v + 1.0, B + 1)
    return raw


def _bin_centers(values: np.ndarray, states: np.ndarray, edges: np.ndarray, B: int) -> np.ndarray:
    """Per-bin center for the drift/diffusion derivatives.

    The outer-bin center of the naive ``0.5*(edges[:-1]+edges[1:])`` formula is
    ±Inf whenever the edge is open; downstream derivative operators (local
    stability, diffusion gradient, equilibrium interpolation) then go NaN.
    Instead the center of every bin is the CONDITIONAL empirical median of the
    samples that actually fall in that bin (finite by construction).  Empty bins
    (zero width under ties, or a constant window) fall back to the midpoint of
    the bin's finite edges clipped into the observed data range — always finite
    (audit P0 bin-center ±Inf).
    """
    centers = np.full(B, np.nan)
    valid = np.isfinite(values) & (states >= 0)
    obs_lo = float(np.min(values[valid])) if np.any(valid) else -1.0
    obs_hi = float(np.max(values[valid])) if np.any(valid) else 1.0
    for b in range(B):
        sel = valid & (states == b)
        if np.any(sel):
            centers[b] = float(np.median(values[sel]))
            continue
        lo, hi = edges[b], edges[b + 1]
        if not np.isfinite(lo):
            lo = obs_lo
        if not np.isfinite(hi):
            hi = obs_hi
        mid = lo if hi <= lo else 0.5 * (lo + hi)
        centers[b] = float(np.clip(mid, obs_lo, obs_hi))
    return centers


def _stationary_distribution(P: np.ndarray) -> np.ndarray | None:
    """Left eigenvector of the (Jeffreys-smoothed) transition matrix ``πP = π``.

    Used wherever a *stationary* measure is required (entropy production,
    stationary surprisal) so that ``πP = π`` holds exactly — the empirical window
    frequency does not (audit P0).  Guards against complex / un-normalisable
    solutions and returns ``None`` to fail closed.
    """
    B = P.shape[0]
    try:
        evals, evecs = np.linalg.eig(P.T)
    except np.linalg.LinAlgError:
        return None
    if evals.size == 0 or not np.all(np.isfinite(evals)):
        return None
    idx = int(np.argmin(np.abs(evals - 1.0)))
    pi = np.real_if_close(evecs[:, idx])
    if np.iscomplexobj(pi):
        return None
    pi = pi.astype(float)
    if not np.all(np.isfinite(pi)):
        return None
    s = float(pi.sum())
    if not np.isfinite(s) or abs(s) <= _EPS:
        return None
    pi = pi / s
    if float(pi.min()) < -1e-8:
        return None
    pi = np.clip(pi, 0.0, None)
    s2 = float(pi.sum())
    if not np.isfinite(s2) or s2 <= _EPS:
        return None
    return pi / s2


def _observed_reachable(N_obs: np.ndarray, start: int, target: int) -> bool:
    """True if ``target`` is reachable from ``start`` through *observed*
    transitions only (pre-pseudo-count support), self-reach inclusive.

    Jeffreys smoothing makes every ``P_ij > 0``, so an unreachable state would
    otherwise still produce a finite MFPT / committor.  This transitive closure
    over the observed-support digraph is the reachability gate that makes those
    quantities NaN for genuinely unreachable targets (audit P1).
    """
    B = N_obs.shape[0]
    reach = (N_obs > 0).astype(bool)
    np.fill_diagonal(reach, True)
    for kk in range(B):
        src = reach[:, kk]
        if src.any():
            reach[src] |= reach[kk]
    return bool(reach[start, target])


def _bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # Tie-aware binning (audit P1): right-closed intervals ``(edges[i-1], edges[i]]``,
    # like ``pandas.cut``.  ``side="left"`` maps a value equal to a (possibly
    # duplicated) interior edge to the *leftmost* cell containing that edge, so
    # heavy ties never collapse distinct values into one top cell.  For strictly
    # increasing edges this differs from ``side="right"`` only on exact-boundary
    # hits (measure-zero for continuous data).
    out = np.clip(np.searchsorted(edges, values, side="left") - 1, 0, len(edges) - 2)
    # Fail-closed (audit P0): NaN/±Inf must never become a legal state.
    # ``np.searchsorted`` sorts NaN to the END, so a raw call would silently
    # map missing data into the top bin and pollute state counts, the transition
    # matrix, D1/D2 and the AIS blocks.  Return a -1 sentinel that callers mask.
    out = np.where(np.isfinite(values), out, -1)
    return out


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
      pi     — (n, B) stationary distribution of P (left eigenvector; NaN when
               P is unavailable) — satisfies ``πP = π``
      pi_empirical — (n, B) empirical window state frequency (historical-state
               quantities), kept distinct from the stationary measure
      N_obs  — (n, B, B) observed lagged transition counts *before* pseudo-count
               (empirical-support matrix; used for reachability gates)
      D1/D2  — (n, B) per-state Kramers-Moyal coefficients (lagged *rates*:
               D1 = mean(Δx)/lag, D2 = mean(Δx²)/(2·lag))
      centers— (n, B) bin centers (conditional empirical median per bin)
      total_trans — number of lagged transitions in the window
      n_states_obs — distinct states observed in the window (degenerate single-
               state window -> < 2, used to fail closed on no-estimate outputs)
    """
    n = series.shape[0]
    B = int(bins)
    lg = int(lag)
    mc = max(1, int(min_count))
    w = int(window)
    # R11 round-2 P0 (TASK 3): ``min_count`` is a relational feasibility gate —
    # the window ``[t-W, t-1]`` holds at most ``window - lag`` lagged pairs, so a
    # minimum above that is guaranteed-NaN and rejected at binding even when this
    # kernel is reached directly (e.g. the Polars twin bypasses ``_run_kernel``).
    if mc > w - lg:
        raise ValueError(
            f"min_count must not exceed available transitions (window - lag = {w - lg}); "
            f"got min_count={mc}"
        )
    state = np.full(n, np.nan)
    edges_out = np.full((n, B + 1), np.nan)
    P = np.full((n, B, B), np.nan)
    counts = np.full((n, B), np.nan)
    pi = np.full((n, B), np.nan)
    pi_empirical = np.full((n, B), np.nan)
    N_obs = np.zeros((n, B, B), dtype=float)
    D1 = np.full((n, B), np.nan)
    D2 = np.full((n, B), np.nan)
    centers = np.full((n, B), np.nan)
    total_trans = np.full(n, np.nan)
    n_states_obs = np.full(n, np.nan)

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
        states_past = _bin(past, edges)
        centers[t] = _bin_centers(past, states_past, edges, B)
        k = int(_bin(np.asarray([cur]), edges)[0])
        state[t] = k
        # Fail-closed on missing values: NaN/Inf states are -1 sentinels and are
        # excluded from every statistic (audit P0).
        valid_states = states_past >= 0
        counts[t] = np.bincount(states_past[valid_states], minlength=B).astype(float)
        total = float(int(valid_states.sum()))
        pi_empirical[t] = counts[t] / max(total, 1.0)
        n_states_obs[t] = float(int(np.count_nonzero(counts[t])))

        d1_row = np.full(B, np.nan)
        d2_row = np.full(B, np.nan)
        N = np.zeros((B, B), dtype=float)
        n_trans = 0
        if len(past) > lg:
            base = past[:-lg]
            inc = past[lg:] - base
            base_bin = states_past[:-lg]
            nxt_bin = states_past[lg:]
            trans_ok = (base_bin >= 0) & (nxt_bin >= 0)
            n_trans = int(trans_ok.sum())
            if n_trans > 0:
                N[...] = np.bincount(
                    base_bin[trans_ok] * B + nxt_bin[trans_ok],
                    minlength=B * B,
                ).reshape(B, B)
            for bbin in range(B):
                sel = trans_ok & (base_bin == bbin)
                if int(sel.sum()) < mc:
                    continue
                dx = inc[sel]
                if dx.size == 0:
                    continue
                # audit P0: D1 is a drift *rate* — divide by the lag step.
                d1_row[bbin] = float(np.mean(dx)) / lg
                d2_row[bbin] = float(np.mean(dx * dx)) / (2.0 * lg)
        N_obs[t] = N
        D1[t] = d1_row
        D2[t] = d2_row
        total_trans[t] = float(n_trans)
        # R11 round-2 P0 (TASK 1 + TASK 3): estimate the transition matrix ONLY
        # from at least ``min_count`` observed transitions, and fail closed on
        # prior-only rows:
        #   * a source state that APPEARS in the window (count > 0) but has no
        #     observed outgoing transitions gets an all-NaN row — never a
        #     uniform Jeffreys row presented as a confident estimate.
        #   * a destination with zero observed incoming transitions keeps only
        #     its documented Jeffreys prior (+0.5 pseudo-count); consumers gate
        #     on observed incoming support (``N_obs``) so such a cell is never
        #     *presented* as a confident estimate (reverse direction, TASK 1).
        # Jeffreys (+0.5) smoothing is applied to every observed-support cell so
        # a single observed transition does not read as probability 1.  Rows of
        # bins that never appear (count 0) are never read by any consumer and
        # keep the prior row, which keeps P finite for the linear-solve /
        # spectral consumers (committor, MFPT, spectral gap, entropy production).
        if n_trans >= mc:
            row_sum = N.sum(axis=1)
            P_smooth = (N + 0.5) / (row_sum[:, None] + 0.5 * B)
            P_smooth[(counts[t] > 0) & (row_sum == 0), :] = np.nan
            P[t] = P_smooth
            # audit P0: stationary measure as the left eigenvector of the
            # *smoothed* P (πP = π), not the empirical frequency.  A P with any
            # fail-closed NaN row propagates NaN through eig -> pi stays NaN.
            pi_s = _stationary_distribution(P[t])
            if pi_s is not None:
                pi[t] = pi_s
    return {
        "state": state,
        "edges": edges_out,
        "P": P,
        "counts": counts,
        "pi": pi,
        "pi_empirical": pi_empirical,
        "N_obs": N_obs,
        "D1": D1,
        "D2": D2,
        "centers": centers,
        "total_trans": total_trans,
        "n_states_obs": n_states_obs,
    }


_KM_BINS_GRID = (3, 5, 8)
_KM_LAG_GRID = (1, 2, 3)


def _run_kernel(
    x: pd.DataFrame, window: int, bins: int, lag: int, min_count: int
) -> dict[str, np.ndarray]:
    w = max(2, int(window))
    b = max(2, int(bins))
    lg = max(1, int(lag))
    mc = max(1, int(min_count))
    if w <= lg:
        raise ValueError("window must exceed lag")
    # R11 round-2 P0 (TASK 3): relational feasibility at binding — a window
    # holds at most ``window - lag`` lagged transitions.
    if mc > w - lg:
        raise ValueError(
            f"min_count must not exceed available transitions (window - lag = {w - lg}); "
            f"got min_count={mc}"
        )
    # P1-011: the Markov/KM parameter grid is bounded (anti parameter-explosion in
    # the search grammar): bins ∈ {3,5,8}, lag ∈ {1,2,3}.
    if int(bins) not in _KM_BINS_GRID:
        raise ValueError(f"bins must be in {_KM_BINS_GRID}, got {bins!r}")
    if int(lag) not in _KM_LAG_GRID:
        raise ValueError(f"lag must be in {_KM_LAG_GRID}, got {lag!r}")
    cols = x.shape[1]
    keys = ["state", "P", "counts", "pi", "pi_empirical", "N_obs", "D1", "D2", "centers", "total_trans", "edges", "n_states_obs"]
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
        # R11 round-2 P0 (TASK 2): a window that observed only a single state has
        # no estimated transition distribution — P_kk there is a degenerate ~1,
        # not evidence of persistence.  Fail closed to NaN.
        if res["n_states_obs"][t, col] < 2:
            continue
        if res["counts"][t, col, k] < min_count:
            continue
        # R11 round-2 P0 (TASK 1, reverse): a state that was never *entered*
        # (zero observed incoming transitions) has no persistence to report —
        # P_kk there is only the documented Jeffreys prior.  Fail closed.
        if res["N_obs"][t, col, :, k].sum() <= 0:
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
        relational_specs=_MIN_COUNT_RELATIONAL,
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
        # R11 round-2 P0 (TASK 2): a single-state window has no estimated
        # transition distribution — entropy there is a degenerate ~0, never a
        # confident value.  Fail closed to NaN.
        if res["n_states_obs"][t, col] < 2:
            continue
        if res["counts"][t, col, k] < min_count:
            continue
        row = res["P"][t, col, k]
        # TASK 1 reverse: only destinations with observed incoming transitions
        # are real outcomes; a never-entered destination keeps only the
        # documented Jeffreys prior and must not inflate/deflate the entropy.
        col_support = res["N_obs"][t, col].sum(axis=0) > 0
        row = row[np.isfinite(row) & col_support]
        # A row with fewer than two estimable destinations has no genuine
        # distribution to measure -> NaN (also catches all-NaN fail-closed rows).
        if row.size < 2:
            continue
        s = float(row.sum())
        if not np.isfinite(s) or s <= _EPS:
            continue
        row = row / s
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
        relational_specs=_MIN_COUNT_RELATIONAL,
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
        # R11 round-2 P0 (TASK 2): in a single-state window every observed jump
        # is trivially s->s, so -log P_ss is a degenerate ~0 — not a rare event.
        if res["n_states_obs"][t, col] < 2:
            continue
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
        relational_specs=_MIN_COUNT_RELATIONAL,
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
        # R11 round-2 P0 (TASK 2): a single-state window is time-reversible by
        # construction (only s->s) — σ there would be a degenerate 0, not an
        # estimated entropy-production rate.
        if res["n_states_obs"][t, col] < 2:
            continue
        if res["total_trans"][t, col] < min_periods:
            continue
        P = res["P"][t, col]
        pi = res["pi"][t, col]
        if not np.all(np.isfinite(P)) or not np.all(np.isfinite(pi)):
            continue
        B = P.shape[0]
        # audit P0: standard probability-flux form of the entropy-production rate
        #   σ = Σ_ij (π_i P_ij) · log( (π_i P_ij) / (π_j P_ji) )
        # (the stationary flux π_i P_ij belongs in both the ratio and the
        # prefactor).  No max(σ, 0) clipping — a legitimately negative value is
        # kept (NaN is only returned for genuinely undefined inputs).
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
                flux_ij = pi[i] * pij
                flux_ji = pi[j] * pji
                if flux_ij <= 0.0 or flux_ji <= 0.0:
                    continue
                sigma += flux_ij * np.log(flux_ij / flux_ji)
        out[t] = float(sigma)
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
    """滚动 Markov 熵产生率 ``σ = Σ_ij (π_i P_ij) log((π_i P_ij)/(π_j P_ji))``（nats）。

    π 为转移矩阵 P 的平稳分布（左特征向量，πP=π，而非经验频率）。对称（细致
    平衡）过程 σ=0；时间不可逆状态演化给出非零值。不裁剪负值：若修正后的公式
    出现合法负值则保留。这是窗口级统计（不是单状态），用于探测状态动力学的时间
    不可逆性。P2 / Research。
    """

    metadata = _metadata(
        "ts_markov_entropy_production",
        "窗口 Markov 熵产生率 Σ_ij (π_i P_ij) log((π_i P_ij)/(π_j P_ji))（nats）。",
        ["x", "window", "bins", "lag", "min_periods"],
        unit="nats",
        cost=5,
        relational_specs=_MIN_PERIODS_RELATIONAL,
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
        relational_specs=_MIN_COUNT_RELATIONAL,
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
    # Master Spec A-4/5: bins/history_length/window are user parameters — invalid
    # values raise (never silently clamped into a false search space).
    B = strict_int(bins, "bins", lower=2)
    k = strict_int(history_length, "history_length", lower=1)
    w = strict_int(window, "window", lower=2)
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
            # Fail-closed on missing values: any NaN/Inf in the k-history block
            # or the next state (the -1 sentinel) invalidates the block (audit
            # P0 — missing data must not be re-encoded as the top bin).
            if np.any(S[s - k : s + 1] < 0):
                continue
            hcode = 0
            for m in range(k):
                hcode = hcode * B + int(S[s - k + m])
            joint[int(S[s]), hcode] += 1.0
            n_blocks += 1
        # audit P1 sufficiency gate: the Jeffreys prior (+0.5) dominates a joint
        # histogram with far fewer blocks than cells.  Require at least
        # ``c = 5`` valid blocks per joint cell (B * B^k cells) or fail closed
        # to NaN instead of emitting a prior-dominated value.
        n_cells = B * (B ** k)
        if n_blocks < 5 * n_cells:
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
    充分性门：要求有效块数 ``n_blocks >= 5 · B · B^k``（每联合单元至少 5 个
    样本），否则输出 NaN 而非先验主导值（audit P1）。P2 / Research。
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


# ---------------------------------------------------------------------------
# ts_markov_committor (P1 deepening)
# ---------------------------------------------------------------------------

def _committor_series(res: dict[str, np.ndarray], col: int, min_count: int) -> np.ndarray:
    """Committor q_k: probability that the process, started in state k, reaches
    the upper boundary state B-1 before the lower boundary state 0.

    Solves the harmonic system ``(I - P_int) q_int = P[:, B-1]`` on interior
    states with boundary q_0 = 0, q_{B-1} = 1.  All P entries come from the
    strictly-past window; the current state only *selects* q_k (PIT-safe).
    """
    n = res["P"].shape[0]
    B = res["P"].shape[2]
    out = np.full(n, np.nan)
    for t in range(n):
        k = res["state"][t, col]
        if not np.isfinite(k):
            continue
        k = int(k)
        # R11 round-2 P0 (TASK 2): no transition structure in a single-state
        # window — the committor would be trivially 0/1, not an estimate.
        if res["n_states_obs"][t, col] < 2:
            continue
        if res["counts"][t, col, k] < min_count:
            continue
        P = res["P"][t, col]
        if not np.all(np.isfinite(P)):
            continue
        # audit P1: empirical-support gate.  Jeffreys smoothing makes every
        # P_ij > 0, so an unreachable target would otherwise still return a
        # finite committor.  Require (a) a minimum observed edge count, (b) the
        # upper target actually visited in the window, (c) an observed-support
        # path from the current state to the target.
        N_obs = res["N_obs"][t, col]
        if int(np.count_nonzero(N_obs)) < 2:
            continue
        if int(res["counts"][t, col, B - 1]) < 1:
            continue
        if not _observed_reachable(N_obs, k, B - 1):
            continue
        if B == 2:
            out[t] = float(k)  # q_0 = 0, q_1 = 1 exactly
            continue
        interior = list(range(1, B - 1))
        if not interior:
            continue
        A = np.eye(len(interior)) - P[np.ix_(interior, interior)]
        rhs = P[interior, B - 1]
        try:
            q_int = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(q_int)):
            continue
        if k == 0:
            out[t] = 0.0
        elif k == B - 1:
            out[t] = 1.0
        else:
            out[t] = float(np.clip(q_int[k - 1], 0.0, 1.0))
    return out


@register_operator(
    name="ts_markov_committor",
    category="state_dynamics",
    business_category="state_dynamics",
    canonical="ts_markov_committor",
    source="markov_dynamics",
)
class TsMarkovCommittor(SeriesOperator):
    """当前状态的 committor 概率 ``q_k = P(先到上边界 B-1, 而非下边界 0)``。

    从严格过去窗口 ``[t-W, t-1]`` 估计转移矩阵 P，在内部状态上解调和方程
    ``(I-P_int) q = P[:, B-1]``（q_0=0, q_{B-1}=1）。输出 ``q_k ∈ [0,1]``：
    ≈1 = 历史动力学显示当前状态更容易最终进入上侧极端；≈0 = 更易先触下侧；
    ≈0.5 = 两侧相当。与 ``ts_markov_persistence``（留下概率）正交。
    """

    metadata = _metadata(
        "ts_markov_committor",
        "当前状态先达上边界而非下边界的概率 q_k（[0,1]）。",
        ["x", "window", "bins", "lag", "min_count"],
        unit="probability",
        cost=5,
        relational_specs=_MIN_COUNT_RELATIONAL,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, bins: int = 3, lag: int = 1, min_count: int = 3, **_: Any
    ) -> pd.DataFrame:
        res = _run_kernel(x, window, bins, lag, min_count)
        cols = x.shape[1]
        mc = max(1, int(min_count))
        out = np.column_stack([_committor_series(res, c, mc) for c in range(cols)])
        return _frame_like(x, out)


# ---------------------------------------------------------------------------
# ts_markov_mean_first_passage_time (P1 deepening)
# ---------------------------------------------------------------------------

def _mfpt_series(res: dict[str, np.ndarray], col: int, min_count: int, target: str, lag: int) -> np.ndarray:
    """Mean first passage time to a target state set A, for the current state.

    Solves ``(I - P_notA) m = 1`` on non-target states; m_i is the expected
    number of steps to first enter A.  Converted to calendar days via ``*lag``.
    Only the strictly-past transition matrix is used (PIT-safe).
    """
    n = res["P"].shape[0]
    B = res["P"].shape[2]
    out = np.full(n, np.nan)
    for t in range(n):
        k = res["state"][t, col]
        if not np.isfinite(k):
            continue
        k = int(k)
        # R11 round-2 P0 (TASK 2): no transition structure in a single-state
        # window — MFPT would be a degenerate 0/1, not an estimate.
        if res["n_states_obs"][t, col] < 2:
            continue
        if res["counts"][t, col, k] < min_count:
            continue
        P = res["P"][t, col]
        if not np.all(np.isfinite(P)):
            continue
        if target == "upper":
            A = {B - 1}
        elif target == "lower":
            A = {0}
        elif target == "extreme":
            A = {0, B - 1}
        else:
            raise ValueError(f"unknown target {target!r}; expected upper/lower/extreme")
        if k in A:
            out[t] = 0.0
            continue
        # audit P1: empirical-support gate.  Jeffreys smoothing makes every
        # P_ij > 0, so an unreachable target would otherwise still return a
        # finite MFPT.  Require (a) a minimum observed edge count, (b) at least
        # one target state actually visited in the window, (c) an observed-
        # support path from the current state to some target.
        N_obs = res["N_obs"][t, col]
        if int(np.count_nonzero(N_obs)) < 2:
            continue
        if not any(int(res["counts"][t, col, a]) >= 1 for a in A):
            continue
        if not any(_observed_reachable(N_obs, k, a) for a in A):
            continue
        nonA = [i for i in range(B) if i not in A]
        if not nonA:
            continue
        M = np.eye(len(nonA)) - P[np.ix_(nonA, nonA)]
        try:
            m = np.linalg.solve(M, np.ones(len(nonA)))
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(m)):
            continue
        steps = float(m[nonA.index(k)])
        out[t] = float(max(steps, 0.0)) * lag
    return out


@register_operator(
    name="ts_markov_mean_first_passage_time",
    category="state_dynamics",
    business_category="state_dynamics",
    canonical="ts_markov_mean_first_passage_time",
    source="markov_dynamics",
)
class TsMarkovMeanFirstPassageTime(SeriesOperator):
    """当前状态到目标状态集的平均首达时间（交易日，MFPT）。

    解 ``(I - P_notA) m = 1``（非目标状态），``m_k × lag`` 换算为天数。目标
    集合为 ``target`` 枚举：upper={B-1} / lower={0} / extreme={0,B-1}。与
    committor 互补：committor 回答"去哪边"，MFPT 回答"大概多久到"。只用严格
    过去窗口的转移矩阵，无前视。
    """

    metadata = _metadata(
        "ts_markov_mean_first_passage_time",
        "当前状态到 target 状态集的平均首达时间（交易日）。",
        ["x", "window", "bins", "lag", "min_count", "target"],
        unit="days",
        cost=5,
        relational_specs=_MIN_COUNT_RELATIONAL,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, bins: int = 3, lag: int = 1,
        min_count: int = 3, target: str = "upper", **_: Any
    ) -> pd.DataFrame:
        res = _run_kernel(x, window, bins, lag, min_count)
        cols = x.shape[1]
        mc = max(1, int(min_count))
        lg = max(1, int(lag))
        out = np.column_stack([_mfpt_series(res, c, mc, target, lg) for c in range(cols)])
        return _frame_like(x, out)


# ---------------------------------------------------------------------------
# ts_markov_spectral_gap (P1 deepening)
# ---------------------------------------------------------------------------

def _spectral_gap_series(res: dict[str, np.ndarray], col: int, min_periods: int) -> np.ndarray:
    """``Gap = 1 - |λ2|`` where λ2 is the second-largest-magnitude eigenvalue
    of the transition matrix.  High = fast mixing (state forgets its initial
    condition); low = metastability / long memory.  Clamped to [0, 1]."""
    n = res["P"].shape[0]
    out = np.full(n, np.nan)
    for t in range(n):
        # R11 round-2 P0 (TASK 2): a single-state window has a trivial 1x1
        # "chain" — the spectral gap is a degenerate 0, not a mixing estimate.
        if res["n_states_obs"][t, col] < 2:
            continue
        if res["total_trans"][t, col] < min_periods:
            continue
        P = res["P"][t, col]
        if not np.all(np.isfinite(P)):
            continue
        try:
            ev = np.linalg.eigvals(P)
        except np.linalg.LinAlgError:
            continue
        if ev.size == 0 or not np.all(np.isfinite(ev)):
            continue
        mag = np.abs(ev)
        order = np.argsort(mag)[::-1]
        second = float(mag[order[1]]) if ev.size >= 2 else 0.0
        out[t] = float(np.clip(1.0 - second, 0.0, 1.0))
    return out


@register_operator(
    name="ts_markov_spectral_gap",
    category="state_dynamics",
    business_category="state_dynamics",
    canonical="ts_markov_spectral_gap",
    source="markov_dynamics",
)
class TsMarkovSpectralGap(SeriesOperator):
    """窗口转移矩阵谱隙 ``Gap = 1 - |λ2|``（市场状态记忆强度）。

    λ1=1 为平凡特征值；|λ2| 越接近 1，状态越久不遗忘初始条件（metastable /
    强记忆），Gap 越低。高 Gap = 快速混合。与自相关不同：这是状态转移动力学的
    整体混合速率。窗口级统计，只用 ``[t-W, t-1]``。
    """

    metadata = _metadata(
        "ts_markov_spectral_gap",
        "转移矩阵谱隙 1-|λ2|（低=metastability，高=快速混合）。",
        ["x", "window", "bins", "lag", "min_periods"],
        unit="ratio",
        cost=5,
        relational_specs=_MIN_PERIODS_RELATIONAL,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 120, bins: int = 3, lag: int = 1, min_periods: int = 5, **_: Any
    ) -> pd.DataFrame:
        res = _run_kernel(x, window, bins, lag, min_periods)
        cols = x.shape[1]
        mp = max(2, int(min_periods))
        out = np.column_stack([_spectral_gap_series(res, c, mp) for c in range(cols)])
        return _frame_like(x, out)


# ---------------------------------------------------------------------------
# ts_markov_stationary_surprisal (P1 deepening)
# ---------------------------------------------------------------------------

def _stationary_surprisal_series(res: dict[str, np.ndarray], col: int, min_count: int) -> np.ndarray:
    n = res["pi"].shape[0]
    out = np.full(n, np.nan)
    for t in range(n):
        k = res["state"][t, col]
        if not np.isfinite(k):
            continue
        k = int(k)
        # R11 round-2 P0 (TASK 2): a single-state window has stationary mass 1 on
        # the one observed state — -log π_k is a degenerate 0, not "long-run
        # rarity".
        if res["n_states_obs"][t, col] < 2:
            continue
        if res["counts"][t, col, k] < min_count:
            continue
        # TASK 1 reverse: a state that was never entered has no stationary
        # occupancy — -log π_k would be the Jeffreys prior, not a rare-state
        # signal.  Fail closed.
        if res["N_obs"][t, col, :, k].sum() <= 0:
            continue
        pi = res["pi"][t, col]
        if not np.all(np.isfinite(pi)):
            continue
        p = float(pi[k])
        out[t] = float(-np.log(p + _EPS))
    return out


@register_operator(
    name="ts_markov_stationary_surprisal",
    category="state_dynamics",
    business_category="state_dynamics",
    canonical="ts_markov_stationary_surprisal",
    source="markov_dynamics",
)
class TsMarkovStationarySurprisal(SeriesOperator):
    """当前状态在长期动力学中的罕见度 ``-log(π_k)``（nats）。

    π 为转移矩阵 P 的**平稳分布**（左特征向量，πP=π，由 P 计算而非经验频率，
    audit P1 诚实命名）；输出当前状态 k 的负对数平稳概率。与
    ``ts_markov_transition_surprisal`` 正交：那个回答"这次跳转怪不怪"，这个
    回答"当前所处位置本身在长期动力学里有多稀有"。只用 ``[t-W,t-1]``。
    """

    metadata = _metadata(
        "ts_markov_stationary_surprisal",
        "当前状态在 P 的平稳分布下的长期稀有度 -log π_k（nats）。",
        ["x", "window", "bins", "lag", "min_count"],
        unit="nats",
        cost=4,
        relational_specs=_MIN_COUNT_RELATIONAL,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, bins: int = 3, lag: int = 1, min_count: int = 3, **_: Any
    ) -> pd.DataFrame:
        res = _run_kernel(x, window, bins, lag, min_count)
        cols = x.shape[1]
        mc = max(1, int(min_count))
        out = np.column_stack([_stationary_surprisal_series(res, c, mc) for c in range(cols)])
        return _frame_like(x, out)


# ---------------------------------------------------------------------------
# ts_km_equilibrium_distance (P1 deepening)
# ---------------------------------------------------------------------------

def _equilibrium_distance_series(
    series: np.ndarray, res: dict[str, np.ndarray], col: int, window: int, min_count: int
) -> np.ndarray:
    """``(x_t - x*) / MAD`` where x* is the stable fixed point of the drift
    D1 (zero crossing with a downward slope, i.e. an attractor).  The center
    of the scaled rolling z-score is therefore the *empirically-estimated
    attractor* rather than the window mean.  MAD fallback to window span/1.0
    when the window is degenerate."""
    n = series.shape[0]
    w = max(2, int(window))
    out = np.full(n, np.nan)
    for t in range(n):
        k = res["state"][t, col]
        if not np.isfinite(k):
            continue
        k = int(k)
        if res["counts"][t, col, k] < min_count:
            continue
        d1 = res["D1"][t, col]
        c = res["centers"][t, col]
        if not np.all(np.isfinite(d1)) or not np.all(np.isfinite(c)):
            continue
        B = d1.shape[0]
        xstar = None
        for m in range(B - 1):
            a, b = d1[m], d1[m + 1]
            if np.isnan(a) or np.isnan(b):
                continue
            if a > 0.0 and b < 0.0:
                denom = a - b
                xstar = c[m] + (c[m + 1] - c[m]) * (a / denom) if abs(denom) > _EPS else c[m]
                break
        if xstar is None:
            continue  # no stable fixed point in window
        lo = max(0, t - w)
        past = series[lo:t]
        finite = past[np.isfinite(past)]
        mad = 1.0
        if finite.size >= 4:
            med = np.median(finite)
            dev = np.abs(finite - med)
            madv = np.median(dev)
            if np.isfinite(madv) and madv > _EPS:
                mad = madv
            else:
                span = float(np.nanmax(finite) - np.nanmin(finite))
                if np.isfinite(span) and span > _EPS:
                    mad = span
        xt = series[t]
        if not np.isfinite(xt):
            continue
        out[t] = float((xt - xstar) / mad)
    return out


@register_operator(
    name="ts_km_equilibrium_distance",
    category="state_dynamics",
    business_category="state_dynamics",
    canonical="ts_km_equilibrium_distance",
    source="markov_dynamics",
)
class TsKmEquilibriumDistance(SeriesOperator):
    """当前值相对历史动力学吸引子的距离 ``(x_t - x*) / MAD``。

    从窗口 D1(x) 找稳定不动点 x*（D1 由正变负的过零点）。比 rolling z-score
    高一层的中心：不是均值，而是**从历史动力学估计出来的吸引子**。非常适合
    valuation / turnover / volatility / price deviation。无 x* 时 fail-closed。
    """

    metadata = _metadata(
        "ts_km_equilibrium_distance",
        "当前值相对 D1 吸引子 x* 的距离 (x_t-x*)/MAD。",
        ["x", "window", "bins", "lag", "min_count"],
        unit="zscore",
        cost=5,
        relational_specs=_MIN_COUNT_RELATIONAL,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, bins: int = 5, lag: int = 1, min_count: int = 3, **_: Any
    ) -> pd.DataFrame:
        res = _run_kernel(x, window, bins, lag, min_count)
        xv = x.to_numpy(dtype=float)
        cols = x.shape[1]
        w = max(2, int(window))
        mc = max(1, int(min_count))
        out = np.column_stack([_equilibrium_distance_series(xv[:, c], res, c, w, mc) for c in range(cols)])
        return _frame_like(x, out)


# ---------------------------------------------------------------------------
# ts_km_diffusion_gradient (P1/P2 deepening)
# ---------------------------------------------------------------------------

def _diffusion_gradient_series(res: dict[str, np.ndarray], col: int, min_count: int) -> np.ndarray:
    """``dD2/dx`` at the current bin (central difference over bin centers).
    High positive = stochastic dispersion widens rapidly when the state moves
    up = state-dependent heteroskedasticity / multiplicative noise."""
    n = res["D2"].shape[0]
    B = res["D2"].shape[2]
    out = np.full(n, np.nan)
    for t in range(n):
        k = res["state"][t, col]
        if not np.isfinite(k):
            continue
        k = int(k)
        if not (0 < k < B - 1):
            continue
        d2 = res["D2"][t, col]
        c = res["centers"][t, col]
        if (
            not np.isfinite(d2[k - 1]) or not np.isfinite(d2[k + 1])
            or not np.isfinite(c[k - 1]) or not np.isfinite(c[k + 1])
        ):
            continue
        if res["counts"][t, col, k] < min_count:
            continue
        denom = c[k + 1] - c[k - 1]
        if abs(denom) <= _EPS:
            continue
        out[t] = float((d2[k + 1] - d2[k - 1]) / denom)
    return out


@register_operator(
    name="ts_km_diffusion_gradient",
    category="state_dynamics",
    business_category="state_dynamics",
    canonical="ts_km_diffusion_gradient",
    source="markov_dynamics",
)
class TsKmDiffusionGradient(SeriesOperator):
    """当前状态的扩散梯度 ``dD2/dx``（state-dependent heteroskedasticity）。

    中心差分 ``(D2_{k+1} - D2_{k-1}) / (c_{k+1} - c_{k-1})``。高正值 = 状态向
    上移动时随机离散度迅速扩大（乘性噪声）。与 ``ts_kramers_moyal_local_stability``
    （漂移导数）互补。
    """

    metadata = _metadata(
        "ts_km_diffusion_gradient",
        "当前状态扩散梯度 dD2/dx（状态依赖异方差）。",
        ["x", "window", "bins", "lag", "min_count"],
        unit="diffusion",
        cost=5,
        relational_specs=_MIN_COUNT_RELATIONAL,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, bins: int = 5, lag: int = 1, min_count: int = 3, **_: Any
    ) -> pd.DataFrame:
        res = _run_kernel(x, window, bins, lag, min_count)
        cols = x.shape[1]
        mc = max(1, int(min_count))
        out = np.column_stack([_diffusion_gradient_series(res, c, mc) for c in range(cols)])
        return _frame_like(x, out)


# ---------------------------------------------------------------------------
# ts_km_quasipotential_depth (P2 deepening)
# ---------------------------------------------------------------------------

def _quasipotential_depth_series(res: dict[str, np.ndarray], col: int, min_count: int) -> np.ndarray:
    """Depth of the potential well containing the current state.

    Discrete quasi-potential ``U[k] = -Σ_{m<k} D1[m]/(D2[m]+ε) · Δc``; well =
    nearest local minimum of U to the current bin, barrier = lower of the two
    flanking ridge maxima.  Depth = U_barrier - U_well.  High = historically
    hard to escape the current basin; low = easily pushed out."""
    n = res["D1"].shape[0]
    B = res["D1"].shape[2]
    out = np.full(n, np.nan)
    for t in range(n):
        k = res["state"][t, col]
        if not np.isfinite(k):
            continue
        k = int(k)
        if res["counts"][t, col, k] < min_count:
            continue
        d1 = res["D1"][t, col]
        d2 = res["D2"][t, col]
        c = res["centers"][t, col]
        if not np.all(np.isfinite(d1)) or not np.all(np.isfinite(d2)) or not np.all(np.isfinite(c)):
            continue
        if B < 3:
            continue
        dx = np.diff(c)
        U = np.zeros(B)
        for m in range(1, B):
            U[m] = U[m - 1] - (d1[m - 1] / (d2[m - 1] + _EPS)) * dx[m - 1]
        mins = [
            m for m in range(B)
            if (m == 0 or U[m] <= U[m - 1]) and (m == B - 1 or U[m] <= U[m + 1])
        ]
        maxs = [
            m for m in range(B)
            if (m == 0 or U[m] >= U[m - 1]) and (m == B - 1 or U[m] >= U[m + 1])
        ]
        if not mins:
            continue
        well = min(mins, key=lambda m: abs(m - k))
        left = [m for m in maxs if m < well]
        right = [m for m in maxs if m > well]
        levels: list[float] = []
        if left:
            levels.append(float(max(U[m] for m in left)))
        if right:
            levels.append(float(max(U[m] for m in right)))
        if not levels:
            continue
        depth = float(min(levels)) - float(U[well])
        if np.isfinite(depth) and depth > _EPS:
            out[t] = depth
    return out


@register_operator(
    name="ts_km_quasipotential_depth",
    category="state_dynamics",
    business_category="state_dynamics",
    canonical="ts_km_quasipotential_depth",
    source="markov_dynamics",
)
class TsKmQuasipotentialDepth(SeriesOperator):
    """当前状态所在势阱的深度（准势 U 的井底-鞍点差）。

    ``U(x) = -∫ D1/(D2+ε) dx``（离散累计）；井 = 距当前 bin 最近的 U 局部极小，
    势垒 = 两侧脊极大中的较低者；depth = U_barrier - U_well。高 = 当前 basin
    历史上难以逃出（比 state persistence 更深）；低 = 看似稳定但易被冲击推出。
    数值保守 fail-closed（无井/无鞍 → NaN）。
    """

    metadata = _metadata(
        "ts_km_quasipotential_depth",
        "当前状态势阱深度（井底-鞍点差）。",
        ["x", "window", "bins", "lag", "min_count"],
        unit="potential",
        cost=6,
        relational_specs=_MIN_COUNT_RELATIONAL,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 120, bins: int = 5, lag: int = 1, min_count: int = 3, **_: Any
    ) -> pd.DataFrame:
        res = _run_kernel(x, window, bins, lag, min_count)
        cols = x.shape[1]
        mc = max(1, int(min_count))
        out = np.column_stack([_quasipotential_depth_series(res, c, mc) for c in range(cols)])
        return _frame_like(x, out)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({
            "ts_markov_persistence",
            "ts_markov_state_entropy",
            "ts_markov_transition_surprisal",
            "ts_kramers_moyal_local_stability",
            "ts_markov_committor",
            "ts_markov_mean_first_passage_time",
            "ts_markov_spectral_gap",
            "ts_markov_stationary_surprisal",
            "ts_km_equilibrium_distance",
            "ts_km_diffusion_gradient",
            "ts_km_quasipotential_depth",
        })
    _surface.extend_research_only(
        {
            "ts_markov_entropy_production",
            "ts_active_information_storage",
        }
    )


_register_surface()
