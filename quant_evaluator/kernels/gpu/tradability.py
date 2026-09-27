"""GPU tradability / capacity kernels (spec §26).

Batch-vectorized over the factor axis F on the GPU-friendly (T, F, N)
layout, matching the CPU reference in :mod:`quant_evaluator.metrics.turnover`
and :mod:`quant_evaluator.metrics.temporal` exactly:

  - ``batched_factor_turnover_rate`` -> ``compute_factor_turnover_rate``
  - ``batched_weighted_turnover``    -> ``compute_weighted_turnover``
  - ``batched_turnover_contribution`` -> ``compute_turnover_contribution``

Semantics preserved:
  - factor turnover rate: fraction of assets that change top/bottom quantile
    membership between adjacent days (min 10 jointly-finite assets).
  - weighted turnover: 0.5 * sum(|delta w| * avg_size) / sum(avg_size) over
    jointly-finite neighbours, NaN where no valid observations.
  - turnover contribution: per-asset 0.5 * |delta w| over jointly-finite
    neighbours, first row NaN.

CuPy is imported lazily so the module loads on a CPU-only environment.
"""

from __future__ import annotations

import numpy as np


# Bound temporary arrays for time-vectorized membership turnover.
_FACTOR_TURNOVER_WORKSPACE_BYTES = 128 << 20
_FACTOR_TURNOVER_BYTES_PER_CELL = 128


def _import_cp():
    import cupy as cp
    return cp


def _as_tfn(x):
    cp = _import_cp()
    a = cp.asarray(x, dtype=cp.float64)
    if a.ndim == 2:
        a = a[:, None, :]
    if a.ndim != 3:
        raise ValueError(f"expected (T, F, N) or (T, N), got ndim={a.ndim}")
    return a


def _nanquantile_rows(safe, n_fin, q):
    """Linear-interpolation quantile of the finite subset per row.

    ``safe`` is (R, N) with +inf marking invalid; the finite values sort to the
    front.  ``n_fin`` is (R,) the finite count per row.  Matches
    ``np.nanquantile(..., q)`` default linear method.
    """
    cp = _import_cp()
    order = cp.argsort(safe, axis=1, kind="stable")
    sv = cp.take_along_axis(safe, order, axis=1)  # (R, N) sorted, inf at end
    R, N = sv.shape
    pos = q * (n_fin - 1.0)  # (R,)
    lo = cp.floor(pos).astype(cp.int64)
    frac = pos - lo
    lo = cp.clip(lo, 0, N - 2)
    hi = lo + 1
    v_lo = cp.take_along_axis(sv, lo[:, None], axis=1)[:, 0]
    v_hi = cp.take_along_axis(sv, hi[:, None], axis=1)[:, 0]
    return v_lo + frac * (v_hi - v_lo)  # (R,)


def batched_factor_turnover_rate(factor_values, quantile: float = 0.9):
    """Top/bottom quantile membership turnover, (T-1, F).

    Matches ``compute_factor_turnover_rate``: for each (t, f) the threshold is
    the ``quantile`` quantile of the jointly-finite values; membership is
    ``>= threshold`` (quantile > 0.5) or ``<= threshold`` (quantile < 0.5);
    turnover = mean of membership changes between adjacent days.  NaN where
    fewer than 10 jointly-finite assets.
    """
    cp = _import_cp()
    x = _as_tfn(factor_values)
    T, F, N = x.shape
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    if T < 2:
        return cp.full((0, F), cp.nan, dtype=cp.float64).get()
    out = cp.full((T - 1, F), cp.nan, dtype=cp.float64)

    # Process adjacent dates in bounded batches. The old per-date loop
    # synchronized the host and device once per date via bool(cp.any(ok));
    # these row-wise operations are independent across (date, factor), so
    # batching removes those synchronizations without changing semantics.
    cells_per_pair = max(F * N, 1)
    pairs_per_chunk = max(
        1,
        int(_FACTOR_TURNOVER_WORKSPACE_BYTES // (
            _FACTOR_TURNOVER_BYTES_PER_CELL * cells_per_pair
        )),
    )
    for start in range(0, T - 1, pairs_per_chunk):
        stop = min(start + pairs_per_chunk, T - 1)
        a = x[start:stop].reshape(-1, N)          # (pairs*F, N)
        b = x[start + 1:stop + 1].reshape(-1, N)  # (pairs*F, N)
        va, vb = cp.isfinite(a), cp.isfinite(b)
        na, nb = va.sum(axis=1), vb.sum(axis=1)
        n = cp.sum(va | vb, axis=1)
        ok = (na >= 10) & (nb >= 10)
        a_safe = cp.where(va, a, cp.inf)
        b_safe = cp.where(vb, b, cp.inf)
        thr_a = _nanquantile_rows(a_safe, na, quantile)
        thr_b = _nanquantile_rows(b_safe, nb, quantile)
        if quantile > 0.5:
            in_a = a_safe >= thr_a[:, None]
            in_b = b_safe >= thr_b[:, None]
        else:
            in_a = a_safe <= thr_a[:, None]
            in_b = b_safe <= thr_b[:, None]
        in_a &= va
        in_b &= vb
        # Unknown next-day values for a selected asset do not imply an exit.
        ok &= ~cp.any(in_a & ~vb, axis=1)
        changed = in_a != in_b
        rate = cp.sum(changed, axis=1) / cp.maximum(n, 1.0)
        out[start:stop] = cp.where(ok, rate, cp.nan).reshape(stop - start, F)
    return out.get()


def batched_weighted_turnover(weights, position_sizes):
    """Position-size-weighted turnover, (T,).

    Matches ``compute_weighted_turnover`` on a single (T, N) weight matrix.
    Returns a (T,) series with the first element NaN.
    """
    cp = _import_cp()
    w = cp.asarray(weights, dtype=cp.float64)
    ps = cp.asarray(position_sizes, dtype=cp.float64)
    T, N = w.shape
    if T < 2:
        return cp.full(T, cp.nan, dtype=cp.float64).get()
    w_t0 = w[:-1, :]
    w_t1 = w[1:, :]
    ps_t0 = ps[:-1, :]
    ps_t1 = ps[1:, :]
    avg_size = 0.5 * (ps_t0 + ps_t1)  # (T-1, N)
    finite = (
        cp.isfinite(w_t0) & cp.isfinite(w_t1)
        & cp.isfinite(ps_t0) & cp.isfinite(ps_t1)
    )
    delta = cp.abs(w_t1 - w_t0)
    weighted_delta = cp.where(finite, delta * avg_size, 0.0)
    numerator = cp.sum(weighted_delta, axis=1)
    denominator = cp.sum(cp.where(finite, avg_size, 0.0), axis=1)
    wt = 0.5 * numerator / cp.maximum(denominator, 1e-300)
    n_valid = cp.sum(finite, axis=1)
    wt = cp.where((n_valid == N) & (N > 0) & (denominator != 0), wt, cp.nan)
    return cp.concatenate([cp.array([cp.nan]), wt]).get()


def batched_turnover_contribution(weights):
    """Per-asset turnover contribution, (T, N).

    Matches ``compute_turnover_contribution``: 0.5 * |delta w| over
    jointly-finite neighbours; first row NaN.
    """
    cp = _import_cp()
    w = cp.asarray(weights, dtype=cp.float64)
    T, N = w.shape
    contribution = cp.full((T, N), cp.nan, dtype=cp.float64)
    if T < 2:
        return contribution.get()
    w_t0 = w[:-1, :]
    w_t1 = w[1:, :]
    finite = cp.isfinite(w_t0) & cp.isfinite(w_t1)
    delta = w_t1 - w_t0
    contribution[1:, :] = 0.5 * cp.where(finite, cp.abs(delta), 0.0)
    contribution[1:, :] = cp.where(cp.all(finite, axis=1)[:, None], contribution[1:, :], cp.nan)
    return contribution.get()
