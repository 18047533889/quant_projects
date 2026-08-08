# -*- coding: utf-8 -*-
"""Haar wavelet energy and spectral (FFT) operators (P2, experimental).

Fixed wavelet family (Haar, orthogonal), fixed boundary handling (power-of-two
truncation) and a minimum window, so semantics are backend-independent.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import SeriesOperator, register_operator
from cleaned_operators.ts_model._rolling_core import frame_like, metadata

_CANONICALS: list[str] = []


def _register(name: str, description: str, params: list[str], unit: str, fn, cost: int = 8):
    @register_operator(
        name=name,
        category="time_series_regression",
        business_category="time_series_regression",
        canonical=name,
        source="ts_model.wavelet_spectral",
        backend="pandas_numpy",
        status="experimental",
    )
    class _WaveOp(SeriesOperator):
        metadata = metadata(name, description, params, unit=unit, cost=cost)

        def _calculate_series(self, *args, **kwargs):
            return fn(*args, **kwargs)

    _CANONICALS.append(name)
    import cleaned_operators.operator_surface as _surface

    _surface.RESEARCH_ONLY_CANONICALS = frozenset(
        set(_surface.RESEARCH_ONLY_CANONICALS) | {name}
    )
    return _WaveOp


def _apply(x: pd.DataFrame, fn) -> pd.DataFrame:
    xv = x.to_numpy(dtype=float)
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            out[row, col] = fn(xv[: row + 1, col])
    return frame_like(x, out)


def _trailing_contiguous_finite(vals: np.ndarray) -> np.ndarray:
    """Trailing contiguous finite suffix of ``vals`` (no gap bridging).

    A wavelet / spectral transform needs a real time axis; deleting NaN and
    splicing the remaining points shifts scale / frequency / phase.  We keep the
    original axis and transform only the *trailing* contiguous finite block so a
    recent gap never reaches back into stale pre-gap data (review R4-64 / P1-15).
    """
    n = len(vals)
    j = n
    while j > 0 and not np.isfinite(vals[j - 1]):
        j -= 1
    i = j
    while i > 0 and np.isfinite(vals[i - 1]):
        i -= 1
    return vals[i:j].astype(float)


def _haar_energy(vals: np.ndarray, window: int) -> list[float]:
    """Return per-level detail energy for a power-of-two Haar DWT."""
    seg = vals[-int(window):]
    finite = _trailing_contiguous_finite(seg)
    n = len(finite)
    if n < 8:
        return []
    m = 2 ** int(np.floor(np.log2(n)))
    # Keep the NEWEST m observations within the contiguous run: a trailing
    # window that truncates to a power of two must retain the data closest to
    # ``t`` — the oldest prefix carries stale information for a trailing factor
    # (review P0-12).
    x = finite[-m:].copy()
    levels: list[float] = []
    while len(x) >= 2:
        approx = (x[::2] + x[1::2]) / np.sqrt(2.0)
        detail = (x[::2] - x[1::2]) / np.sqrt(2.0)
        levels.append(float(np.sum(detail * detail)))
        x = approx
    levels.reverse()  # low -> high
    return levels


def _wavelet_stats(vals: np.ndarray, window: int, stat: str) -> float:
    levels = _haar_energy(vals, int(window))
    if len(levels) < 2:
        return np.nan
    total = float(sum(levels))
    if total <= 1e-12:
        return np.nan
    if stat == "low":
        return float(levels[0] / total)
    if stat == "high":
        return float(levels[-1] / total)
    if stat == "entropy":
        w = np.array([e / total for e in levels])
        # A zero-energy level contributes 0*log(0) = NaN; drop it before the
        # sum (highly regular series frequently have one empty band — review
        # P0-13).  The normalization uses the number of *present* bands.
        w = w[w > 0.0]
        if w.size < 2:
            return np.nan
        h = -float(np.sum(w * np.log(w)))
        return float(h / np.log(w.size))
    if stat == "slope":
        # Haar level j has dyadic scale 2^j, so the regressor is log(2^j) =
        # j*log(2), not log(j) — review P1-16.  Population cov/var (same ddof)
        # avoid the n/(n-1) inflation of np.cov/np.var — review P1-17.
        scales = np.log(2.0) * np.arange(1, len(levels) + 1, dtype=float)
        log_e = np.log(np.maximum(levels, 1e-15))
        sb = float(np.mean(scales))
        lb = float(np.mean(log_e))
        cov = float(np.mean((scales - sb) * (log_e - lb)))
        var = float(np.mean((scales - sb) ** 2))
        return cov / var if var > 0.0 else np.nan
    return np.nan


_register("ts_wavelet_low_frequency_ratio", "低频小波能量占比。", ["x", "window"], "ratio",
           lambda x, window=128: _apply(x, lambda v: _wavelet_stats(v, int(window), "low")))
_register("ts_wavelet_high_frequency_ratio", "高频小波能量占比。", ["x", "window"], "ratio",
           lambda x, window=128: _apply(x, lambda v: _wavelet_stats(v, int(window), "high")))
_register("ts_wavelet_entropy", "小波各尺度能量熵。", ["x", "window"], "level",
           lambda x, window=128: _apply(x, lambda v: _wavelet_stats(v, int(window), "entropy")))
_register("ts_wavelet_energy_slope", "小波能量随尺度变化的斜率。", ["x", "window"], "level",
           lambda x, window=128: _apply(x, lambda v: _wavelet_stats(v, int(window), "slope")))


def _spectral_low_ratio(vals: np.ndarray, window: int) -> float:
    seg = vals[-int(window):]
    # P1-15: same no-time-axis-compression rule as the Haar kernel — use the
    # trailing contiguous finite suffix (frequency/phase semantics need a real
    # time axis; review R4-64).
    finite = _trailing_contiguous_finite(seg)
    if len(finite) < 16:
        return np.nan
    m = len(finite)
    demean = finite - np.mean(finite)
    spec = np.abs(np.fft.rfft(demean)) ** 2
    total = float(np.sum(spec))
    if total <= 1e-12:
        return np.nan
    half = max(1, len(spec) // 2)
    return float(np.sum(spec[:half]) / total)


_register("ts_spectral_low_frequency_ratio", "傅里叶低频能量占比。", ["x", "window"], "ratio",
           lambda x, window=128: _apply(x, lambda v: _spectral_low_ratio(v, int(window))))
