"""GPU data-quality kernels (spec §35).

Batch-vectorized over the factor axis F on the GPU-friendly (T, F, N)
layout, matching the CPU reference in :mod:`quant_evaluator.metrics.data_quality`
exactly:

  - ``batched_missing_ratio``
  - ``batched_missing_timeline``
  - ``batched_staleness``
  - ``batched_effective_n``
  - ``batched_distinct_level_ratio``
  - ``batched_tie_ratio``
  - ``batched_cross_section_cardinality``
  - ``batched_tradable_coverage``
  - ``batched_universe_churn``

All return per-factor arrays of shape (F,).  CuPy is imported lazily so the
module loads on a CPU-only environment.
"""

from __future__ import annotations

import numpy as np


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


def batched_missing_ratio(factor_values):
    """Fraction of (T, N) cells with non-finite values, (F,)."""
    cp = _import_cp()
    x = _as_tfn(factor_values)
    T, F, N = x.shape
    missing = cp.sum(~cp.isfinite(x), axis=(0, 2))
    return (missing / (T * N)).get()


def batched_missing_timeline(factor_values):
    """Fraction of time periods with any missing value, (F,)."""
    cp = _import_cp()
    x = _as_tfn(factor_values)
    T, F, N = x.shape
    any_missing = cp.any(~cp.isfinite(x), axis=2)  # (T, F)
    return (cp.sum(any_missing, axis=0) / T).get()


def batched_staleness(factor_values):
    """Mean fraction of assets unchanged from prior day, (F,)."""
    cp = _import_cp()
    x = _as_tfn(factor_values)
    T, F, N = x.shape
    if T < 2:
        return cp.full(F, cp.nan, dtype=cp.float64).get()
    prev = x[:-1]
    curr = x[1:]
    both = cp.isfinite(prev) & cp.isfinite(curr)
    unchanged = (prev == curr) & both
    n = cp.sum(both, axis=(0, 2))
    same = cp.sum(unchanged, axis=(0, 2))
    return (same / cp.maximum(n, 1)).get()


def batched_effective_n(factor_values):
    """Mean number of finite values per day, (F,)."""
    cp = _import_cp()
    x = _as_tfn(factor_values)
    T, F, N = x.shape
    n = cp.sum(cp.isfinite(x), axis=2)  # (T, F)
    return cp.mean(n, axis=0).get()


def batched_distinct_level_ratio(factor_values):
    """Mean fraction of distinct values among finite values per day, (F,)."""
    cp = _import_cp()
    x = _as_tfn(factor_values)
    T, F, N = x.shape
    out = cp.full(F, cp.nan, dtype=cp.float64)
    ssum = cp.zeros(F, dtype=cp.float64)
    cnt = cp.zeros(F, dtype=cp.float64)
    for t in range(T):
        row = x[t]
        finite = cp.isfinite(row)
        n_fin = cp.sum(finite, axis=1)
        x_safe = cp.where(finite, row, cp.inf)
        order = cp.argsort(x_safe, axis=1, kind="stable")
        sv = cp.take_along_axis(x_safe, order, axis=1)
        new_fin = cp.zeros_like(sv, dtype=cp.bool_)
        new_fin[:, 0] = sv[:, 0] != cp.inf
        new_fin[:, 1:] = (sv[:, 1:] != sv[:, :-1]) & (sv[:, 1:] != cp.inf)
        n_distinct = cp.sum(new_fin, axis=1)
        ratio = n_distinct / cp.maximum(n_fin, 1)
        ok = n_fin > 0
        ssum = ssum + cp.where(ok, ratio, 0.0)
        cnt = cnt + ok.astype(cp.float64)
    out = cp.where(cnt > 0, ssum / cp.maximum(cnt, 1), cp.nan)
    return out.get()


def batched_tie_ratio(factor_values):
    """Mean fraction of finite values tied with another value, (F,)."""
    cp = _import_cp()
    return 1.0 - batched_distinct_level_ratio(factor_values)


def batched_cross_section_cardinality(factor_values):
    """Mean number of distinct values per day, (F,)."""
    cp = _import_cp()
    x = _as_tfn(factor_values)
    T, F, N = x.shape
    ssum = cp.zeros(F, dtype=cp.float64)
    cnt = cp.zeros(F, dtype=cp.float64)
    for t in range(T):
        row = x[t]
        finite = cp.isfinite(row)
        n_fin = cp.sum(finite, axis=1)
        x_safe = cp.where(finite, row, cp.inf)
        order = cp.argsort(x_safe, axis=1, kind="stable")
        sv = cp.take_along_axis(x_safe, order, axis=1)
        new_fin = cp.zeros_like(sv, dtype=cp.bool_)
        new_fin[:, 0] = sv[:, 0] != cp.inf
        new_fin[:, 1:] = (sv[:, 1:] != sv[:, :-1]) & (sv[:, 1:] != cp.inf)
        n_distinct = cp.sum(new_fin, axis=1)
        ok = n_fin > 0
        ssum = ssum + cp.where(ok, n_distinct, 0.0)
        cnt = cnt + ok.astype(cp.float64)
    out = cp.where(cnt > 0, ssum / cp.maximum(cnt, 1), cp.nan)
    return out.get()


def batched_tradable_coverage(factor_values, labels, min_assets=10):
    """Fraction of days with >= min_assets jointly valid cells, (F,)."""
    cp = _import_cp()
    x = _as_tfn(factor_values)
    y = cp.asarray(labels, dtype=cp.float64)
    T, F, N = x.shape
    label_finite = cp.isfinite(y)  # (T, N)
    factor_finite = cp.isfinite(x)  # (T, F, N)
    valid = factor_finite & label_finite[:, None, :]  # (T, F, N)
    per_day = cp.sum(valid, axis=2)  # (T, F)
    return (cp.sum(per_day >= min_assets, axis=0) / T).get()


def batched_universe_churn(factor_values, labels):
    """Mean fraction of tradable universe changing membership per day, (F,)."""
    cp = _import_cp()
    x = _as_tfn(factor_values)
    y = cp.asarray(labels, dtype=cp.float64)
    T, F, N = x.shape
    if T < 2:
        return cp.full(F, cp.nan, dtype=cp.float64).get()
    label_finite = cp.isfinite(y)
    factor_finite = cp.isfinite(x)
    tradable = factor_finite & label_finite[:, None, :]  # (T, F, N)
    prev = tradable[:-1]
    curr = tradable[1:]
    churn = cp.sum(prev != curr, axis=2)  # (T-1, F)
    return (cp.mean(churn / N, axis=0)).get()
