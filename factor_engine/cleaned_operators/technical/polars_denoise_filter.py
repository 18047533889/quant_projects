# -*- coding: utf-8 -*-
"""Polars implementations for denoising and regularization operators.

Dual backend for ts_ssa_denoise_trailing, ts_wavelet_shrinkage_trailing,
ts_total_variation_filter_trailing, ts_l1_trend_filter_trailing.
"""
from __future__ import annotations

import numpy as np

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

from factor_engine.cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator

_SKIP = frozenset({"date", "stock_code"})


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _ssa_decompose(x: np.ndarray, L: int, n_components: int) -> np.ndarray:
    """SSA decomposition (same as pandas version)."""
    N = len(x)
    if L > N // 2:
        L = N // 2
    K = N - L + 1
    X = np.column_stack([x[i:i+L] for i in range(K)])
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    n_keep = min(n_components, len(s))
    X_reconstructed = U[:, :n_keep] @ np.diag(s[:n_keep]) @ Vt[:n_keep, :]
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
    """Wavelet soft-thresholding (same as pandas version)."""
    coeffs = np.diff(x, prepend=x[0])
    abs_coeffs = np.abs(coeffs)
    shrinkage = np.maximum(0, abs_coeffs - threshold)
    denoised_coeffs = np.sign(coeffs) * shrinkage
    return np.cumsum(denoised_coeffs) + x[0] - denoised_coeffs[0]


def _total_variation_filter(x: np.ndarray, lambda_tv: float, max_iter: int = 50) -> np.ndarray:
    """Total variation L1 filter (same as pandas version)."""
    y = x.copy()
    step = 0.5
    for _ in range(max_iter):
        grad = y - x
        diff = np.diff(y, prepend=y[0])
        soft_diff = np.sign(diff) * np.maximum(0, np.abs(diff) - lambda_tv * step)
        y_new = y - step * grad
        y_new = x + np.cumsum(soft_diff)
        if np.allclose(y, y_new, atol=1e-6):
            break
        y = y_new
    return y


def _l1_trend_filter(x: np.ndarray, lambda_l1: float, max_iter: int = 50) -> np.ndarray:
    """L1 trend filter (same as pandas version)."""
    y = x.copy()
    step = 0.1
    for _ in range(max_iter):
        d2y = np.diff(y, n=2, prepend=[y[0], y[1]])
        soft_d2y = np.sign(d2y) * np.maximum(0, np.abs(d2y) - lambda_l1 * step)
        y_new = x + np.cumsum(np.cumsum(soft_d2y - d2y))
        if np.allclose(y, y_new, atol=1e-6):
            break
        y = y_new
    return y


def _rolling_apply_ssa(arr: np.ndarray, window: int, n_components: int) -> float:
    """Rolling SSA for single point."""
    if len(arr) < window:
        return float(arr[-1]) if len(arr) > 0 else np.nan
    window_data = arr[-window:]
    if np.isnan(window_data).sum() > window // 4:
        return np.nan
    mask = np.isfinite(window_data)
    if mask.sum() < 5:
        return np.nan
    window_clean = window_data.copy()
    if not mask.all():
        idx_finite = np.where(mask)[0]
        window_clean = np.interp(np.arange(len(window_data)), idx_finite, window_data[idx_finite])
    return float(_ssa_decompose(window_clean, window // 2, n_components)[-1])


def _rolling_apply_wavelet(arr: np.ndarray, threshold: float) -> float:
    """Rolling wavelet for single point."""
    if len(arr) < 3:
        return float(arr[-1]) if len(arr) > 0 else np.nan
    window_data = arr
    if np.isfinite(window_data).all():
        return float(_wavelet_soft_threshold(window_data, threshold)[-1])
    elif np.isfinite(window_data).sum() >= len(window_data) // 2:
        mask = np.isfinite(window_data)
        idx_finite = np.where(mask)[0]
        window_clean = np.interp(np.arange(len(window_data)), idx_finite, window_data[idx_finite])
        return float(_wavelet_soft_threshold(window_clean, threshold)[-1])
    return np.nan


def _rolling_apply_tv(arr: np.ndarray, lambda_tv: float) -> float:
    """Rolling TV for single point."""
    if len(arr) < 3:
        return float(arr[-1]) if len(arr) > 0 else np.nan
    window_data = arr
    if np.isfinite(window_data).all():
        return float(_total_variation_filter(window_data, lambda_tv)[-1])
    elif np.isfinite(window_data).sum() >= len(window_data) // 2:
        mask = np.isfinite(window_data)
        idx_finite = np.where(mask)[0]
        window_clean = np.interp(np.arange(len(window_data)), idx_finite, window_data[idx_finite])
        return float(_total_variation_filter(window_clean, lambda_tv)[-1])
    return np.nan


def _rolling_apply_l1(arr: np.ndarray, lambda_l1: float) -> float:
    """Rolling L1 trend for single point."""
    if len(arr) < 5:
        return float(arr[-1]) if len(arr) > 0 else np.nan
    window_data = arr
    if np.isfinite(window_data).all():
        return float(_l1_trend_filter(window_data, lambda_l1)[-1])
    elif np.isfinite(window_data).sum() >= len(window_data) // 2:
        mask = np.isfinite(window_data)
        idx_finite = np.where(mask)[0]
        window_clean = np.interp(np.arange(len(window_data)), idx_finite, window_data[idx_finite])
        return float(_l1_trend_filter(window_clean, lambda_l1)[-1])
    return np.nan


@register_operator(
    name="ts_ssa_denoise_trailing",
    category="time_series",
    business_category="time_series",
    canonical="ts_ssa_denoise_trailing",
    source="denoise_filter_polars",
)
class TsSsaDenoiseTrailingPolars(SeriesOperator):
    """Polars SSA denoising filter."""

    metadata = OperatorMetadata(
        name="ts_ssa_denoise_trailing",
        category="time_series",
        description="SSA降噪（Polars）",
        param_names=["x", "window", "n_components"],
        return_type="series",
        tags=["time_series", "polars", "denoise", "pit_safe"],
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, n_components: int = 3, **kwargs
    ) -> pl.DataFrame:
        w = max(5, int(window))
        nc = max(1, int(n_components))
        cols = _numeric_cols(x)

        def _apply(arr: np.ndarray) -> float:
            return _rolling_apply_ssa(arr, w, nc)

        return x.with_columns([
            pl.col(c).rolling_map(_apply, window_size=w, min_samples=w).alias(c)
            for c in cols
        ])


@register_operator(
    name="ts_wavelet_shrinkage_trailing",
    category="time_series",
    business_category="time_series",
    canonical="ts_wavelet_shrinkage_trailing",
    source="denoise_filter_polars",
)
class TsWaveletShrinkageTrailingPolars(SeriesOperator):
    """Polars wavelet shrinkage filter."""

    metadata = OperatorMetadata(
        name="ts_wavelet_shrinkage_trailing",
        category="time_series",
        description="小波软阈值降噪（Polars）",
        param_names=["x", "window", "threshold"],
        return_type="series",
        tags=["time_series", "polars", "denoise", "pit_safe"],
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, threshold: float = 0.5, **kwargs
    ) -> pl.DataFrame:
        w = max(3, int(window))
        thr = max(0.0, float(threshold))
        cols = _numeric_cols(x)

        def _apply(arr: np.ndarray) -> float:
            return _rolling_apply_wavelet(arr, thr)

        return x.with_columns([
            pl.col(c).rolling_map(_apply, window_size=w, min_samples=w).alias(c)
            for c in cols
        ])


@register_operator(
    name="ts_total_variation_filter_trailing",
    category="time_series",
    business_category="time_series",
    canonical="ts_total_variation_filter_trailing",
    source="denoise_filter_polars",
)
class TsTotalVariationFilterTrailingPolars(SeriesOperator):
    """Polars total variation filter."""

    metadata = OperatorMetadata(
        name="ts_total_variation_filter_trailing",
        category="time_series",
        description="全变分滤波（Polars）",
        param_names=["x", "window", "lambda_tv"],
        return_type="series",
        tags=["time_series", "polars", "denoise", "pit_safe"],
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, lambda_tv: float = 0.1, **kwargs
    ) -> pl.DataFrame:
        w = max(3, int(window))
        lam = max(0.0, float(lambda_tv))
        cols = _numeric_cols(x)

        def _apply(arr: np.ndarray) -> float:
            return _rolling_apply_tv(arr, lam)

        return x.with_columns([
            pl.col(c).rolling_map(_apply, window_size=w, min_samples=w).alias(c)
            for c in cols
        ])


@register_operator(
    name="ts_l1_trend_filter_trailing",
    category="time_series",
    business_category="time_series",
    canonical="ts_l1_trend_filter_trailing",
    source="denoise_filter_polars",
)
class TsL1TrendFilterTrailingPolars(SeriesOperator):
    """Polars L1 trend filter."""

    metadata = OperatorMetadata(
        name="ts_l1_trend_filter_trailing",
        category="time_series",
        description="L1趋势提取（Polars）",
        param_names=["x", "window", "lambda_l1"],
        return_type="series",
        tags=["time_series", "polars", "trend", "pit_safe"],
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 30, lambda_l1: float = 0.1, **kwargs
    ) -> pl.DataFrame:
        w = max(5, int(window))
        lam = max(0.0, float(lambda_l1))
        cols = _numeric_cols(x)

        def _apply(arr: np.ndarray) -> float:
            return _rolling_apply_l1(arr, lam)

        return x.with_columns([
            pl.col(c).rolling_map(_apply, window_size=w, min_samples=w).alias(c)
            for c in cols
        ])
