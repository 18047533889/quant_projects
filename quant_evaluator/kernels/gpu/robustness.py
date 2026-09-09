"""GPU robustness kernels (spec §20, Wave 4).

HAC is batch-vectorized over F; bootstrap streams factors while sharing time
draws. All inputs are daily IC series of shape (T, F).

Semantics preserved from the CPU oracle in
:mod:`quant_evaluator.metrics.robustness`:

  * HAC variance / t-stat   — endpoint trimming only; internal gaps and
    infinities give insufficient (NaN) evidence, never compressed calendar lags.
  * Block bootstrap CI      — deterministic PCG64 RNG, shared original-time
    draws across all factors; incomplete columns give NaN, not compressed blocks.
  * Subsample IC/std        — deterministic PCG64 RNG; per-bootstrap
    without-replacement subset; mean / std (ddof=1) over columns.

CuPy is imported lazily so the module (and any importers) load on a
CPU-only environment.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from quant_evaluator.kernels.gpu.temporal import (
    ic_autocorrelation_series as _ic_acf,
    half_life as _half_life,
)

__all__ = [
    "hac_variance",
    "hac_tstat",
    "ic_autocorrelation_series",
    "half_life",
    "block_bootstrap_ci",
    "subsample_ic",
    "subsample_ic_std",
    "rolling_ic_stats",
]


def _import_cp():
    import cupy as cp
    return cp


def _front_packed(ic):
    """Pack contiguous finite segments, marking internal gaps insufficient.

    Returns (packed (T,F), n_finite (F,)).  Slot k of factor f holds the k-th
    finite value of column f, preserving original order.
    """
    cp = _import_cp()
    finite = cp.isfinite(ic)
    T, F = ic.shape
    rank = cp.cumsum(finite, axis=0) - 1
    n_f = cp.sum(finite, axis=0).astype(cp.int64)
    time_rows = cp.arange(T)[:, None]
    if T:
        first = cp.min(cp.where(finite, time_rows, T), axis=0)
        last = cp.max(cp.where(finite, time_rows, -1), axis=0)
        contiguous = ((last - first + 1) == n_f) & ~cp.isinf(ic).any(axis=0)
    else:
        contiguous = cp.zeros(F, dtype=bool)
    fgrid = cp.broadcast_to(cp.arange(F)[None, :], (T, F))
    rows = rank[finite]
    cols = fgrid[finite]
    vals = ic[finite].astype(cp.float64)
    A = cp.zeros((T, F), dtype=cp.float64)
    A[rows, cols] = vals
    return A, cp.where(contiguous, n_f, 0)


def _hac_variance_impl(series, max_lag=5, kernel="bartlett"):
    """Contiguous-sample HAC variance; internal, over a GPU array (T,F)."""
    cp = _import_cp()
    from quant_evaluator.metrics.robustness import _validate_hac_policy
    _validate_hac_policy(max_lag, kernel)
    ic = cp.asarray(series)
    if ic.ndim == 1:
        ic = ic[:, None]
    if ic.ndim != 2:
        raise ValueError("HAC requires a time series or T x F matrix")
    T, F = ic.shape
    A, n_f = _front_packed(ic)
    mean_f = cp.sum(A, axis=0) / cp.maximum(n_f, 1)
    Ssq = cp.sum(A * A, axis=0)
    S = cp.sum(A, axis=0)
    cov0 = (Ssq - n_f * mean_f * mean_f) / cp.maximum(n_f, 1)
    hac = cov0 * 1.0

    tvec = cp.arange(T, dtype=cp.float64)[:, None]
    count = n_f.astype(cp.float64) - 1  # max valid slot index per factor (n_f-1)
    for lag in range(1, min(max_lag + 1, T)):
        # pairs (slot k, slot k+lag) available where k <= n_f-lag-1
        valid_slots = n_f - lag                       # (F,)
        slot_mask = tvec < valid_slots[None, :].astype(cp.float64)  # (T,F)
        A_lag = cp.zeros_like(A)
        A_lag[:T - lag, :] = A[lag:, :]
        num = cp.sum(slot_mask * A * A_lag, axis=0)
        s_prev = cp.sum(slot_mask * A, axis=0)
        s_curr = cp.sum(slot_mask * A_lag, axis=0)
        cnt = valid_slots.astype(cp.float64)
        cov = (num - mean_f * (s_prev + s_curr) + cnt * mean_f * mean_f) / cp.maximum(n_f, 1)
        if kernel == "bartlett":
            w = 1.0 - lag / (max_lag + 1)
        else:
            w = 1.0
        hac = hac + 2.0 * w * cov

    hac_var = hac / cp.maximum(n_f, 1)
    ok = n_f >= (max_lag + 10)
    hac_var = cp.where(ok, hac_var, cp.nan)
    return hac_var


def hac_variance(ic_series, max_lag=5, kernel="bartlett"):
    """Newey-West HAC variance, parity with ``compute_hac_variance``.

    Returns shape (F,).  ``max_lag`` and ``kernel`` mirror the CPU oracle
    (``kernel`` in {"bartlett", "uniform"}).
    """
    if kernel not in ("bartlett", "uniform"):
        raise ValueError(f"Unknown kernel: {kernel}, must be 'bartlett' or 'uniform'")
    return _hac_variance_impl(ic_series, max_lag=max_lag, kernel=kernel).get()


def hac_tstat(ic_series, max_lag=5, kernel="bartlett"):
    """HAC-robust t-stat and standard error, parity with ``compute_hac_tstat``.

    Returns (t_stat (F,), se (F,)), NaN where insufficient data or non-positive
    HAC variance.
    """
    cp = _import_cp()
    ic = cp.asarray(ic_series)
    if ic.ndim == 1:
        ic = ic[:, None]
    T, F = ic.shape
    A, n_f = _front_packed(ic)
    mean_f = cp.sum(A, axis=0) / cp.maximum(n_f, 1)
    # NOTE: pass the ORIGINAL ic to _hac_variance_impl (it re-derives its own
    # front-packed A and finite counts); passing the already-packed A would be
    # double-compressed and corrupt n_f.
    hac_var = _hac_variance_impl(ic, max_lag=max_lag, kernel=kernel)
    se_hac = cp.sqrt(hac_var)
    t_stat = mean_f / se_hac
    bad = (hac_var <= 0) | cp.isnan(hac_var) | (n_f < max_lag + 10)
    t_stat = cp.where(bad, cp.nan, t_stat)
    se_hac = cp.where(bad, cp.nan, se_hac)
    return t_stat.get(), se_hac.get()


def _rng(seed):
    """Deterministic numpy Generator (PCG64) used to draw the sampled indices.

    The CPU oracle (:mod:`quant_evaluator.metrics.robustness`) draws its
    sampled indices from ``np.random.default_rng(random_seed)``.  To achieve
    exact (1e-8) parity we generate the *same* index sequences with the same
    PRNG on the host, then upload them to the device for the batched
    gather/reduction.  The expensive statistics are still computed GPU-side and
    batch-vectorized over F; only the small O(B*sqrt) index draw happens on
    host.  A fresh Generator per draw keeps the state fully deterministic under
    a fixed seed.
    """
    import numpy as np
    return np.random.default_rng(seed)


def block_bootstrap_ci(
    ic_series,
    block_length=10,
    num_bootstrap=1000,
    confidence_level=0.95,
    random_seed=None,
):
    """Block-bootstrap CI for mean IC, parity with ``compute_block_bootstrap_ci``.

    Deterministic (fixed seed) and uses block-index tiling so the B x T x F
    bootstrap matrix is never materialized.  Returns (ci_lower (F,),
    ci_upper (F,)).
    """
    cp = _import_cp()
    ic = cp.asarray(ic_series)
    if ic.ndim == 1:
        ic = ic[:, None]
    if ic.ndim != 2:
        raise ValueError("bootstrap requires a time series or T x F matrix")
    T, F = ic.shape
    from quant_evaluator.metrics.robustness import _validate_bootstrap_policy
    _validate_bootstrap_policy(T, block_length, num_bootstrap, confidence_level, random_seed)

    alpha = 1.0 - confidence_level
    lower_percentile = 100.0 * (alpha / 2)
    upper_percentile = 100.0 * (1.0 - alpha / 2)

    finite = cp.isfinite(ic)
    n_f = cp.sum(finite, axis=0)

    ci_lower = cp.full(F, cp.nan, dtype=cp.float64)
    ci_upper = cp.full(F, cp.nan, dtype=cp.float64)

    f_slots = cp.where((n_f == T) & (n_f >= block_length * 2))[0]

    # One shared original-calendar draw matrix, independent of factor ordering.
    rng = _rng(random_seed)
    shared_starts = rng.integers(0, T - block_length + 1,
                                size=(num_bootstrap, (T + block_length - 1) // block_length))

    for f in f_slots.get().tolist():
        col = ic[:, f]
        valid_ic = col[finite[:, f]].astype(cp.float64)
        n = int(n_f[f])
        num_blocks = (n + block_length - 1) // block_length
        m = n - block_length + 1  # number of possible block starts

        # Deterministic host index draw (PCG64 parity with the CPU oracle).
        starts = shared_starts  # Identical draws for every candidate and factor order.

        # Build the block-concatenation index tensor (B, num_blocks*block_length),
        # streamed in factor tiles of bootstrap count — never B x T x F.
        st = cp.asarray(starts)                      # (B, nbk)
        offsets = cp.arange(block_length, dtype=cp.int64)[None, None, :]  # (1,1,BL)
        idx3 = st[:, :, None] + offsets             # (B, nbk, BL)
        idx3 = idx3.reshape(num_bootstrap, -1)[:, :n]  # trim to original length
        sample = valid_ic[idx3]                      # (B, n)
        boot_means = cp.mean(sample, axis=1)         # (B,)

        lo = cp.percentile(boot_means, lower_percentile)
        hi = cp.percentile(boot_means, upper_percentile)
        ci_lower[f] = lo
        ci_upper[f] = hi

    return ci_lower.get(), ci_upper.get()


def subsample_ic(
    ic_series,
    num_subsamples=100,
    subsample_fraction=0.8,
    random_seed=None,
):
    """Bootstrap-subsampled IC means / stds, parity with ``compute_subsample_ic``.

    Returns (subsample_means (num_subsamples, F), subsample_stds
    (num_subsamples, F)).  Deterministic under a fixed seed.
    """
    cp = _import_cp()
    ic = cp.asarray(ic_series)
    if ic.ndim == 1:
        ic = ic[:, None]
    T, F = ic.shape

    if subsample_fraction <= 0 or subsample_fraction >= 1:
        raise ValueError(f"subsample_fraction must be in (0, 1), got {subsample_fraction}")

    subsample_size = max(1, int(T * subsample_fraction))
    means = cp.full((num_subsamples, F), cp.nan, dtype=cp.float64)
    stds = cp.full((num_subsamples, F), cp.nan, dtype=cp.float64)

    finite = cp.isfinite(ic)

    # Deterministic host index draw (PCG64 parity with the CPU oracle):
    # per bootstrap b, ``int(T*frac)`` indices sampled WITHOUT replacement.
    rng = _rng(random_seed)
    idxs_host = np.empty((num_subsamples, subsample_size), dtype=np.int64)
    for b in range(num_subsamples):
        # A single choice call reproduces the oracle (default_rng.choice).
        idxs_host[b] = rng.choice(T, size=subsample_size, replace=False)
    idxs = cp.asarray(idxs_host)  # (B, K)

    # Vectorized, no per-factor loop: gather + column mean/std over all factors.
    def _mean_nan(sub):
        fm = cp.isfinite(sub).astype(cp.float64)
        cnt = cp.sum(fm, axis=1)
        s = cp.where(cp.isfinite(sub), sub, 0.0)
        return cp.sum(s, axis=1) / cp.maximum(cnt, 1.0), cnt

    # loop over bootstrap tile to bound device memory (small B x K x F)
    TILE = max(1, (1 << 22) // max(subsample_size * F, 1))
    for start_b in range(0, num_subsamples, TILE):
        nb = min(TILE, num_subsamples - start_b)
        sub = ic[idxs[start_b:start_b + nb]]           # (nb, K, F)
        m, cnt = _mean_nan(sub)                        # (nb, F),(nb,F)
        s = cp.where(cp.isfinite(sub), sub, 0.0)
        var = cp.sum(cp.where(cp.isfinite(sub), (s - m[:, None, :]) ** 2, 0.0), axis=1)
        std = cp.sqrt(var / cp.maximum(cnt - 1, 1.0))
        std = cp.where(cnt >= 2, std, np.nan)
        means[start_b:start_b + nb] = m
        stds[start_b:start_b + nb] = std

    return means.get(), stds.get()


def subsample_ic_std(
    ic_series,
    num_subsamples=100,
    subsample_fraction=0.8,
    random_seed=None,
):
    """Std of subsample mean IC, parity with ``compute_subsample_ic_std``."""
    cp = _import_cp()
    means, _ = subsample_ic(
        ic_series, num_subsamples, subsample_fraction, random_seed
    )
    m = cp.asarray(means)
    return cp.nanstd(m, axis=0, ddof=1).get()


def rolling_ic_stats(ic_series, window=60, min_periods=20):
    """Rolling IC mean / IR, parity with ``compute_rolling_ic_stats``.

    Batch-vectorized over F and T (no factor loop).  Returns (rolling_mean
    (T,F), rolling_ir (T,F)).  Both arrays carry NaN until ``min_periods``
    finite observations accumulate in a trailing window; IR additionally
    requires >= max(2, min_periods) finite values and a nonzero window std
    (ddof=1).
    """
    cp = _import_cp()
    ic = cp.asarray(ic_series)
    if ic.ndim == 1:
        ic = ic[:, None]
    T, F = ic.shape

    finite = cp.isfinite(ic)
    x = cp.where(finite, ic, 0.0)

    # trailing sums via prefix sums
    pref = cp.concatenate([cp.zeros((1, F)), cp.cumsum(x, axis=0)], axis=0)      # (T+1,F)
    fpref = cp.concatenate([cp.zeros((1, F)), cp.cumsum(finite, axis=0)], axis=0)
    pref_sq = cp.concatenate([cp.zeros((1, F)), cp.cumsum(x * x, axis=0)], axis=0)

    ends = cp.arange(1, T + 1)                                   # (T,)
    start = cp.maximum(ends - window, 0)                        # (T,)
    # 1-D advanced index so fpref[ends] is (T,F), not (T,1,F)
    cnt = fpref[ends] - fpref[start]                            # (T,F)
    s = pref[ends] - pref[start]
    s_sq = pref_sq[ends] - pref_sq[start]

    mean = s / cp.maximum(cnt, 1.0)
    mean_ok = cnt >= min_periods

    var = s_sq - cnt * mean * mean
    std = cp.sqrt(cp.maximum(var, 0.0) / cp.maximum(cnt - 1, 1.0))
    # NaN when std not finite or <= 1e-12
    ir = mean / std
    ir = cp.where((std <= 1e-12) | (~cp.isfinite(std)), cp.nan, ir)
    ir = cp.where(cnt >= max(2, min_periods), ir, cp.nan)

    rmean = cp.where(mean_ok, mean, cp.nan)
    return rmean.get(), ir.get()
