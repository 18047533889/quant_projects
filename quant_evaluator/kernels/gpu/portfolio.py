"""Batched GPU cohort portfolio PnL (spec §16, §17, §19, Wave 3).

Replicates the CPU ``compute_cohort_pnl`` in
``metrics/probe_portfolio/_core.py`` exactly, but batches over the factor
axis F on the GPU-friendly (T, F, N) layout.

Semantics preserved from the CPU reference (spec §16):
  - signal day s: factor(s) quantile -> next trading day (s+1) VWAP entry,
    hold H days, exit at (s+H) (exit day included in PnL).
  - Long = top bucket (D10), each stock +long_weight/N_long;
    Short = D1, each stock short_weight/N_short.
  - Each cohort uses 1/H of capital.
  - Rolling active weight W_{t,n} = sum over active cohorts of entry weights
    (computed via cumsum + shifted-diff, avoiding a T x H x N x F loop).
  - PnL_t = sum_n W_{t,n} * r_{t,n} (daily return, NaN -> 0).
  - cost per-side: entry/exit each charged one per-side cost on notional.

The rolling active weight is derived from the per-signal-day entry weight
matrix W_entry[s, n] (the position weight of the cohort created at signal
day s).  A cohort s is active on day t iff s+1 <= t <= min(s+H, T-1), i.e.
s in [t-H, t-1].  Hence

    W_t = (1/H) * sum_{s=max(0,t-H)}^{t-1} W_entry[s]
        = (1/H) * (C[t-1] - C[t-H-1])          (C = cumsum of W_entry)

which is a single cumsum plus a shifted difference, fully vectorized over
(T, F, N).
"""

from __future__ import annotations

from typing import Optional, Dict

import numpy as np

from quant_evaluator.kernels.gpu.rank import batched_quantile_assignment

EPS = 1e-12


def _import_cp():
    import cupy as cp
    return cp


def compute_cohort_pnl_batch_gpu(
    factor_values,
    daily_returns,
    tradability=None,
    n_quantiles: int = 10,
    holding: int = 20,
    long_weight: float = 0.5,
    short_weight: float = -0.5,
    per_side_cost: float = 0.0,
    min_bucket_size: int = 1,
    require_tradable: bool = True,
) -> Dict[str, "cp.ndarray"]:
    """Batched cohort portfolio PnL over (T, F, N) factors.

    Args:
        factor_values : (T, F, N) or (T, N) factor panel (row = trading day).
        daily_returns : (T, N) daily holding return r_t = P_t / P_{t-1} - 1
            (vwap->vwap).  NaN treated as 0 (flat) exactly like the CPU.
        tradability   : (T, N) bool; True where the stock is tradable on that
            day (VWAP present / not suspended).  None => all tradable.
        n_quantiles   : number of quantile buckets (default 10 -> D1..D10).
        holding       : holding period H in trading days (default 20).
        long_weight   : D10 per-stock target weight = long_weight / N_long.
        short_weight  : D1 per-stock target weight = short_weight / N_short.
        per_side_cost : per-side proportional cost (default 0 = gross).
        min_bucket_size: min members per bucket; degenerate days skipped.
        require_tradable: if True, entry-day untradable stocks are dropped.

    Returns:
        dict of device arrays, each (T, F) unless noted:
            pnl_net        : daily net PnL (1/H scaled, net of costs).
            gross_exposure : daily gross |w|/H (long+short).
            cohort_weight  : per-signal-day new-cohort count / H.
            entry_cost     : daily entry cost (net-of-return basis).
            exit_cost      : daily exit cost (net-of-return basis).
            n_long         : (T, F) D10 member count per signal day.
            n_short        : (T, F) D1 member count per signal day.
    """
    cp = _import_cp()
    fv = cp.asarray(factor_values, dtype=cp.float64)
    if fv.ndim == 2:
        fv = fv[:, None, :]  # (T, 1, N)
    if fv.ndim != 3:
        raise ValueError(f"factor_values must be (T, F, N) or (T, N), got ndim={fv.ndim}")
    T, F, N = fv.shape

    ret = cp.asarray(daily_returns, dtype=cp.float64)  # (T, N)
    if ret.ndim != 2 or ret.shape != (T, N):
        raise ValueError(f"daily_returns must be (T, N) == {(T, N)}, got {ret.shape}")

    if tradability is None:
        trad = cp.ones((T, N), dtype=cp.bool_)
    else:
        trad = cp.asarray(tradability, dtype=cp.bool_)
        if trad.ndim == 1:
            trad = trad[:, None]
        if trad.shape != (T, N):
            raise ValueError(f"tradability must be (T, N) == {(T, N)}, got {trad.shape}")

    if holding < 1:
        raise ValueError(f"holding must be >= 1, got {holding}")
    if not (0.0 <= per_side_cost < 1.0):
        raise ValueError(f"per_side_cost must be in [0, 1), got {per_side_cost}")

    # ---- quantile assignment over (T*F, N) ----
    q = batched_quantile_assignment(
        fv.reshape(T * F, N), n_quantiles=n_quantiles, method="max"
    )  # (T*F, N) int32, NaN -> -1
    q = q.reshape(T, F, N)
    top = n_quantiles - 1
    long_mask = q == top  # (T, F, N)
    short_mask = q == 0

    if min_bucket_size > 1:
        long_cnt = cp.sum(long_mask, axis=2)
        short_cnt = cp.sum(short_mask, axis=2)
        degenerate = (long_cnt < min_bucket_size) | (short_cnt < min_bucket_size)
        long_mask = long_mask & (~degenerate[:, :, None])
        short_mask = short_mask & (~degenerate[:, :, None])

    if require_tradable:
        # cohort s enters at s+1 -> use tradability at entry day s+1.
        trad_entry = cp.concatenate(
            [trad[1:, None, :], cp.zeros((1, 1, N), dtype=cp.bool_)], axis=0
        )  # (T, 1, N); last row 0 (no entry day for s = T-1)
        long_mask = long_mask & trad_entry
        short_mask = short_mask & trad_entry

    # ---- per-cohort entry weights ----
    n_long = cp.sum(long_mask, axis=2)  # (T, F)
    n_short = cp.sum(short_mask, axis=2)
    w_long = cp.where(long_mask, long_weight / cp.maximum(n_long[:, :, None], 1.0), 0.0)
    w_short = cp.where(short_mask, short_weight / cp.maximum(n_short[:, :, None], 1.0), 0.0)
    W_entry = w_long + w_short  # (T, F, N)

    side_notional = cp.sum(cp.abs(w_long), axis=2) + cp.sum(cp.abs(w_short), axis=2)
    valid = (n_long + n_short) > 0
    valid = valid & (side_notional > EPS)
    W_entry = cp.where(valid[:, :, None], W_entry, 0.0)
    side_notional = cp.where(valid, side_notional, 0.0)

    # ---- rolling active weight via cumsum + shifted diff ----
    C = cp.cumsum(W_entry, axis=0)  # (T, F, N)
    C_pad = cp.concatenate([cp.zeros((1, F, N), dtype=cp.float64), C], axis=0)  # (T+1, F, N)
    idx_t = cp.arange(T)
    lo = cp.maximum(0, idx_t - holding)
    W_t = (C_pad[idx_t] - C_pad[lo]) / holding  # (T, F, N)

    # ---- PnL from daily returns (NaN -> 0) ----
    ret0 = cp.where(cp.isfinite(ret), ret, 0.0)  # (T, N)
    pnl_gross = cp.sum(W_t * ret0[:, None, :], axis=2)  # (T, F)

    # ---- costs (per-side on notional, entry + exit) ----
    cost_scale = per_side_cost / holding
    entry_cost = cp.zeros((T, F), dtype=cp.float64)
    if T >= 2:
        entry_cost[1:] = cost_scale * side_notional[:-1]
    exit_cost = cp.zeros((T, F), dtype=cp.float64)
    if holding > 1:
        if T > holding:
            # s in [0, T-1-holding] -> exit at s+holding (t in [holding, T-1])
            src = side_notional[: T - holding]  # (T-holding, F)
            exit_cost[holding:T] += cost_scale * src
        # s in [T-holding, T-3] -> exit clamped to T-1 (not same-day)
        if T - 3 >= T - holding:
            exit_cost[T - 1] += cost_scale * cp.sum(side_notional[T - holding : T - 2], axis=0)

    pnl_net = pnl_gross - entry_cost - exit_cost

    # ---- gross exposure (rolling sum of side_notional / H) ----
    csn = cp.cumsum(side_notional, axis=0)  # (T, F)
    csn_pad = cp.concatenate([cp.zeros((1, F), dtype=cp.float64), csn], axis=0)
    gross = (csn_pad[idx_t] - csn_pad[lo]) / holding  # (T, F)

    # ---- per-signal-day cohort weight (1/H if a cohort was created) ----
    cohort_weight = cp.where(valid, 1.0 / holding, 0.0)  # (T, F)

    return {
        "pnl_net": pnl_net,
        "gross_exposure": gross,
        "cohort_weight": cohort_weight,
        "entry_cost": entry_cost,
        "exit_cost": exit_cost,
        "n_long": n_long,
        "n_short": n_short,
    }
