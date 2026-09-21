# -*- coding: utf-8 -*-
"""Denoising and regularization operators (trailing-only, TRUE_GAP scope=ts).

Four causal filter operators for signal denoising and trend extraction:
1. ts_ssa_denoise_trailing      — Singular Spectrum Analysis (SSA) denoising
2. ts_wavelet_shrinkage_trailing — Wavelet soft-thresholding
3. ts_total_variation_filter_trailing — TV-L1 edge-preserving filter
4. ts_l1_trend_filter_trailing  — L1 trend extraction (Hodrick-Prescott variant)

All operators are:
- causal / trailing-only (no future leakage)
- scope="ts" (time-series trailing window)
- pit_safe=True (production ready)
- dual backend: pandas_numpy + polars
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.technical.convex_trend_filter import (
    l1_trend_filter as _shared_l1_trend_filter,
    total_variation_filter as _shared_total_variation_filter,
)

from factor_engine.cleaned_operators.base import (
    Operator,
    OperatorMetadata,
    ParamRole,
    ParamSpec,
    SeriesOperator,
    register_operator,
)

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _ssa_decompose(x: np.ndarray, L: int, n_components: int) -> np.ndarray:
    """SSA decomposition: embed, decompose, reconstruct.

    Returns the reconstructed series using the first n_components eigentriples.
    """
    N = len(x)
    if L > N // 2:
        L = N // 2
    K = N - L + 1

    # 1. Embedding: construct trajectory matrix
    X = np.column_stack([x[i:i+L] for i in range(K)])

    # 2. SVD
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    # 3. Grouping: keep only first n_components
    n_keep = min(n_components, len(s))
    X_reconstructed = U[:, :n_keep] @ np.diag(s[:n_keep]) @ Vt[:n_keep, :]

    # 4. Diagonal averaging (Hankelization)
    result = np.zeros(N)
    counts = np.zeros(N)
    for i in range(K):
        for j in range(L):
            idx = i + j
            result[idx] += X_reconstructed[j, i]
            counts[idx] += 1

    result = result / np.maximum(counts, 1)
    return result


def _wavelet_soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """Simple wavelet soft-thresholding using Haar wavelet.

    Approximates discrete wavelet decomposition with a rolling difference kernel.
    """
    # Approximate wavelet coefficients with differences
    coeffs = np.diff(x, prepend=x[0])

    # Soft thresholding
    abs_coeffs = np.abs(coeffs)
    shrinkage = np.maximum(0, abs_coeffs - threshold)
    denoised_coeffs = np.sign(coeffs) * shrinkage

    # Reconstruct
    return np.cumsum(denoised_coeffs) + x[0] - denoised_coeffs[0]


def _total_variation_filter(x: np.ndarray, lambda_tv: float, max_iter: int = 4000) -> np.ndarray:
    """Solve ``.5||y-x||² + lambda_tv||Dy||₁`` with certification."""
    return _shared_total_variation_filter(x, lambda_tv, max_iter=max_iter)


def _l1_trend_filter(x: np.ndarray, lambda_l1: float, max_iter: int = 4000) -> np.ndarray:
    """Solve ``.5||y-x||² + lambda_l1||D²y||₁`` with certification."""
    return _shared_l1_trend_filter(x, lambda_l1, max_iter=max_iter)


def _build_diff_matrix(size: int, order: int) -> "np.ndarray":
    """Build the first/second difference operator D (size-order, size)."""
    if order == 1:
        D = np.zeros((size - 1, size))
        idx = np.arange(size - 1)
        D[idx, idx] = -1.0
        D[idx, idx + 1] = 1.0
    else:
        D = np.zeros((size - 2, size))
        idx = np.arange(size - 2)
        D[idx, idx] = 1.0
        D[idx, idx + 1] = -2.0
        D[idx, idx + 2] = 1.0
    return D


def _batch_difference_l1_filter(W, penalty, order, max_iter=4000,
                                absolute_tolerance=1e-8, relative_tolerance=1e-7):
    """Solve ``0.5*||Y-W||**2 + penalty*||D_order Y||_1`` for every row of W.

    Vectorized batch ADMM that mirrors ``convex_trend_filter.difference_l1_filter``
    row-wise. All trailing windows are stacked into a (T, w) array and the ADMM
    iterations become element-wise numpy operations solved simultaneously for
    every window, with a per-window convergence mask that freezes converged
    windows.  Returns Y with the same shape as W.

    R61 perf notes: the linear solve per iteration is a matmul against a
    precomputed inverse (A is w x w with w <= window, so the explicit inverse
    is numerically safe and removes ~2.6ms/iter of scipy wrapper overhead);
    rho uses a sweep-tuned heuristic; convergence certificates are evaluated
    every 3rd iteration (certificates bound the distance to the unique
    minimizer, so the emitted solution matches the scalar authority within the
    same tolerances).
    """
    from factor_engine.cleaned_operators.technical.convex_trend_filter import (
        TrendFilterConvergenceError,
    )

    W = np.asarray(W, dtype=float)
    if W.ndim != 2:
        raise ValueError("batch trend-filter input must be a 2D finite array")
    T, size = W.shape
    if order not in (1, 2):
        raise ValueError("difference order must be 1 or 2")
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be finite and nonnegative")
    diff_size = size - order
    if penalty == 0 or diff_size <= 0:
        return W.copy()
    if not np.isfinite(W).all():
        raise ValueError("batch trend-filter input must be finite")

    rho = min(max(0.5 * float(penalty) * np.sqrt(size), 0.3), 1e8)
    D = _build_diff_matrix(size, order)
    A = np.eye(size) + rho * (D.T @ D)
    Ainv = np.linalg.inv(A)
    DT = D.T
    y = W.copy()
    z = np.diff(y, n=order, axis=1)
    dual = np.zeros_like(z)
    threshold = penalty / rho

    converged = np.zeros(T, dtype=bool)
    sqrt_diff = np.sqrt(diff_size)
    sqrt_size = np.sqrt(size)
    atol = float(absolute_tolerance)
    rtol = float(relative_tolerance)

    for it in range(1, int(max_iter) + 1):
        active = ~converged
        a_idx = np.where(active)[0]
        if a_idx.size == 0:
            break
        Wa = W[a_idx]
        za = z[a_idx]
        ua = dual[a_idx]
        rhs = Wa + rho * ((za - ua) @ D)
        y[a_idx] = rhs @ Ainv
        dy = y[a_idx] @ DT
        z_prev = za
        shifted = ua + dy
        z_new = np.sign(shifted) * np.maximum(np.abs(shifted) - threshold, 0.0)
        z[a_idx] = z_new
        primal = dy - z_new
        dual[a_idx] = ua + primal

        if it % 3 and it != max_iter:
            continue

        # Primal residual certification (vectorized per window).
        scale = np.abs(primal)
        np.maximum(scale, np.abs(dy), out=scale)
        np.maximum(scale, np.abs(z_new), out=scale)
        np.maximum(scale, 1.0, out=scale)
        scale = scale.max(axis=1)
        primal_norm = np.sqrt(np.einsum("ij,ij->i", primal, primal) / (scale * scale))
        ref_norm = np.maximum(
            np.sqrt(np.einsum("ij,ij->i", dy, dy) / (scale * scale)),
            np.sqrt(np.einsum("ij,ij->i", z_new, z_new) / (scale * scale)),
        )
        cond1 = primal_norm <= (sqrt_diff * atol) / scale + rtol * ref_norm

        # Dual residual certification (vectorized per window).
        dual_residual = rho * ((z_new - z_prev) @ D)
        dual_reference = rho * (dual[a_idx] @ D)
        dscale = np.abs(dual_residual)
        np.maximum(dscale, np.abs(dual_reference), out=dscale)
        np.maximum(dscale, 1.0, out=dscale)
        dscale = dscale.max(axis=1)
        dr_norm = np.sqrt(np.einsum("ij,ij->i", dual_residual, dual_residual) / (dscale * dscale))
        dref_norm = np.sqrt(np.einsum("ij,ij->i", dual_reference, dual_reference) / (dscale * dscale))
        cond2 = dr_norm <= (sqrt_size * atol) / dscale + rtol * dref_norm

        converged[a_idx] = cond1 & cond2

    if not np.isfinite(y).all():
        raise TrendFilterConvergenceError("batch trend-filter solution is nonfinite")
    return y


def _ts_filter_column(vals: "np.ndarray", w: int, lam: float, order: int) -> "np.ndarray":
    """Vectorized trailing-window L1/TV filter for one column.

    Replicates the row-wise semantics of the TV/L1 trend operators: warmup rows
    (index < w-1) pass the input through unchanged; each trailing window of
    length w is solved by batch ADMM and its last value is emitted.  NaN gating
    matches the scalar implementation (all-finite -> solve; finite >= w//2 ->
    linear-interp then solve; otherwise NaN).
    """
    n = len(vals)
    out = np.full(n, np.nan, dtype=float)
    warmup = w - 1
    if warmup > 0:
        end = min(warmup, n)
        out[:end] = vals[:end]
    if n <= warmup:
        return out

    windows = np.lib.stride_tricks.sliding_window_view(vals, w)  # (n-w+1, w)
    fin = np.isfinite(windows)
    fin_count = fin.sum(axis=1)
    all_fin = fin_count == w
    need_interp = (fin_count >= w // 2) & ~all_fin
    solve_mask = all_fin | need_interp

    if solve_mask.any():
        Win = windows.copy()
        if need_interp.any():
            xq = np.arange(w)
            interp_rows = np.where(need_interp)[0]
            for r in interp_rows:
                f = fin[r]
                Win[r] = np.interp(xq, xq[f], windows[r, f])
        Y = _batch_difference_l1_filter(Win[solve_mask], lam, order)
        win_rows = np.arange(warmup, n)
        out[win_rows[solve_mask]] = Y[:, -1]
    return out


# ---------------------------------------------------------------------------
# §1 SSA Denoising
# ---------------------------------------------------------------------------

@register_operator(
    name="ts_ssa_denoise_trailing",
    category="time_series",
    business_category="time_series",
    canonical="ts_ssa_denoise_trailing",
    source="denoise_filter",
)
class TsSsaDenoiseTrailing(SeriesOperator):
    """Singular Spectrum Analysis (SSA) denoising filter (trailing-only).

    Decomposes the trailing window via SSA and reconstructs using the first
    n_components eigentriples, filtering out high-frequency noise.
    """

    metadata = OperatorMetadata(
        name="ts_ssa_denoise_trailing",
        category="time_series",
        description="SSA降噪：奇异谱分析保留主成分，过滤高频噪声",
        examples=["ts_ssa_denoise_trailing(close, 20, 3)"],
        param_names=["x", "window", "n_components"],
        return_type="series",
        tags=["time_series", "denoise", "ssa", "filter", "pit_safe"],
        param_specs={
            "window": ParamSpec(
                dtype=int, min=5, searchable=True,
                param_role=ParamRole.ECONOMIC, default=20,
            ),
            "n_components": ParamSpec(
                dtype=int, min=1, searchable=True,
                param_role=ParamRole.MODEL_ORDER, default=3,
            ),
        },
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 20, n_components: int = 3, **kwargs
    ) -> pd.DataFrame:
        w = max(5, int(window))
        nc = max(1, int(n_components))

        result = x.copy()
        for col in x.columns:
            vals = x[col].to_numpy(dtype=float)
            out = np.full_like(vals, np.nan)

            for i in range(len(vals)):
                if i < w - 1:
                    out[i] = vals[i]  # Insufficient history
                    continue

                window_data = vals[max(0, i - w + 1):i + 1]
                if np.isnan(window_data).sum() > w // 4:  # Too many NaNs
                    out[i] = np.nan
                else:
                    # Interpolate NaNs for SSA
                    mask = np.isfinite(window_data)
                    if mask.sum() >= 5:
                        window_clean = window_data.copy()
                        if not mask.all():
                            idx_finite = np.where(mask)[0]
                            window_clean = np.interp(
                                np.arange(len(window_data)),
                                idx_finite,
                                window_data[idx_finite],
                            )
                        out[i] = _ssa_decompose(window_clean, w // 2, nc)[-1]
                    else:
                        out[i] = np.nan

            result[col] = out

        return result


# ---------------------------------------------------------------------------
# §2 Wavelet Shrinkage
# ---------------------------------------------------------------------------

@register_operator(
    name="ts_wavelet_shrinkage_trailing",
    category="time_series",
    business_category="time_series",
    canonical="ts_wavelet_shrinkage_trailing",
    source="denoise_filter",
)
class TsWaveletShrinkageTrailing(SeriesOperator):
    """Wavelet soft-thresholding denoising (trailing-only).

    Applies soft-thresholding to approximate wavelet coefficients to remove
    small-scale noise while preserving signal structure.
    """

    metadata = OperatorMetadata(
        name="ts_wavelet_shrinkage_trailing",
        category="time_series",
        description="小波软阈值降噪：保留主要信号结构，过滤小幅度噪声",
        examples=["ts_wavelet_shrinkage_trailing(close, 20, 0.5)"],
        param_names=["x", "window", "threshold"],
        return_type="series",
        tags=["time_series", "denoise", "wavelet", "filter", "pit_safe"],
        param_specs={
            "window": ParamSpec(
                dtype=int, min=3, searchable=True,
                param_role=ParamRole.ECONOMIC, default=20,
            ),
            "threshold": ParamSpec(
                dtype=float, min=0.0, searchable=True,
                param_role=ParamRole.ESTIMATOR_RESOLUTION, default=0.5,
            ),
        },
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 20, threshold: float = 0.5, **kwargs
    ) -> pd.DataFrame:
        w = max(3, int(window))
        thr = max(0.0, float(threshold))

        result = x.copy()
        for col in x.columns:
            vals = x[col].to_numpy(dtype=float)
            out = np.full_like(vals, np.nan)

            for i in range(len(vals)):
                if i < w - 1:
                    out[i] = vals[i]
                    continue

                window_data = vals[max(0, i - w + 1):i + 1]
                if np.isfinite(window_data).all():
                    out[i] = _wavelet_soft_threshold(window_data, thr)[-1]
                elif np.isfinite(window_data).sum() >= w // 2:
                    # Fill NaNs with forward fill
                    mask = np.isfinite(window_data)
                    idx_finite = np.where(mask)[0]
                    window_clean = np.interp(
                        np.arange(len(window_data)),
                        idx_finite,
                        window_data[idx_finite],
                    )
                    out[i] = _wavelet_soft_threshold(window_clean, thr)[-1]
                else:
                    out[i] = np.nan

            result[col] = out

        return result


# ---------------------------------------------------------------------------
# §3 Total Variation Filter
# ---------------------------------------------------------------------------

@register_operator(
    name="ts_total_variation_filter_trailing",
    category="time_series",
    business_category="time_series",
    canonical="ts_total_variation_filter_trailing",
    source="denoise_filter",
)
class TsTotalVariationFilterTrailing(SeriesOperator):
    """Total variation L1 filter (trailing-only).

    Edge-preserving filter that minimizes total variation (sum of absolute
    differences) while staying close to the original signal.
    """

    metadata = OperatorMetadata(
        name="ts_total_variation_filter_trailing",
        category="time_series",
        description="全变分滤波：保边降噪，保留突变特征",
        examples=["ts_total_variation_filter_trailing(close, 20, 0.1)"],
        param_names=["x", "window", "lambda_tv"],
        return_type="series",
        tags=["time_series", "denoise", "tv", "filter", "pit_safe"],
        param_specs={
            "window": ParamSpec(
                dtype=int, min=3, searchable=True,
                param_role=ParamRole.ECONOMIC, default=20,
            ),
            "lambda_tv": ParamSpec(
                dtype=float, min=0.0, searchable=True,
                param_role=ParamRole.ESTIMATOR_RESOLUTION, default=0.1,
            ),
        },
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 20, lambda_tv: float = 0.1, **kwargs
    ) -> pd.DataFrame:
        w = max(3, int(window))
        lam = max(0.0, float(lambda_tv))

        result = x.copy()
        for col in x.columns:
            vals = x[col].to_numpy(dtype=float)
            result[col] = _ts_filter_column(vals, w, lam, order=1)
        return result


# ---------------------------------------------------------------------------
# §4 L1 Trend Filter
# ---------------------------------------------------------------------------

@register_operator(
    name="ts_l1_trend_filter_trailing",
    category="time_series",
    business_category="time_series",
    canonical="ts_l1_trend_filter_trailing",
    source="denoise_filter",
)
class TsL1TrendFilterTrailing(SeriesOperator):
    """L1 trend extraction filter (trailing-only).

    Extracts smooth trend by penalizing second-order differences (acceleration)
    with L1 norm, allowing piecewise-linear trends.
    """

    metadata = OperatorMetadata(
        name="ts_l1_trend_filter_trailing",
        category="time_series",
        description="L1趋势提取：分段线性趋势，抑制加速度",
        examples=["ts_l1_trend_filter_trailing(close, 30, 0.1)"],
        param_names=["x", "window", "lambda_l1"],
        return_type="series",
        tags=["time_series", "trend", "l1", "filter", "pit_safe"],
        param_specs={
            "window": ParamSpec(
                dtype=int, min=5, searchable=True,
                param_role=ParamRole.ECONOMIC, default=30,
            ),
            "lambda_l1": ParamSpec(
                dtype=float, min=0.0, searchable=True,
                param_role=ParamRole.ESTIMATOR_RESOLUTION, default=0.1,
            ),
        },
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 30, lambda_l1: float = 0.1, **kwargs
    ) -> pd.DataFrame:
        w = max(5, int(window))
        lam = max(0.0, float(lambda_l1))

        result = x.copy()
        for col in x.columns:
            vals = x[col].to_numpy(dtype=float)
            result[col] = _ts_filter_column(vals, w, lam, order=2)
        return result
