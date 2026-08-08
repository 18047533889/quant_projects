# -*- coding: utf-8 -*-
"""Cross-spectral primitives between two series (2026-08-08 Gemini V2 round).

Both operators use one kernel: the trailing window is linearly detrended and
Hann-windowed for *both* series, an FFT produces the cross-spectrum
``S_xy(f) = X(f)·conj(Y(f))`` and the auto-spectra, and two summaries are
returned:

* ``ts_cross_spectral_coherence`` — the mean coherence
  ``|S_xy|² / (S_xx·S_yy)`` over the positive-frequency band (0 = unrelated,
  1 = perfectly coherent phase-locked dynamics at that frequency).
* ``ts_cross_spectral_phase`` — the coherence-amplitude-weighted circular mean
  phase ``arg(Σ_f |S_xy|·e^{iφ_f})`` in radians.  This is a genuine circular
  mean (never a plain arithmetic mean of angles, which wraps); the weighted
  form equals ``arg(Σ_f S_xy)``.

Both are strict-PIT, deterministic, and NaN fail-closed: a NaN in either
series invalidates the window (the time axis is never compressed).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.gemini_v2_common import (
    frame_like,
    register_dual,
    trailing_contiguous_multi,
    union_extended,
)

_EPS = 1e-12


def _cross_spectrum_window(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """Welch-averaged cross-coherence and weighted mean phase.

    The per-frequency squared coherence ``|S_xy|²/(S_xx·S_yy)`` of a single
    unsmoothed periodogram is identically 1 (``|a·conj(b)|² = |a|²·|b|²`` for
    complex ``a, b``), so the spectra must be ensemble-averaged before the
    ratio is taken.  The window is split into 50%-overlap Hann segments, the
    cross-spectrum and auto-spectra are averaged across segments, and only
    then is the coherence formed.
    """
    n = x.shape[0]
    if n < 12:
        return None
    seg_len = max(16, n // 2)
    hop = seg_len // 2
    n_seg = 1 + (n - seg_len) // hop
    if n_seg < 2:
        return None
    i_max = seg_len // 2
    Sxx = np.zeros(i_max, dtype=float)
    Syy = np.zeros(i_max, dtype=float)
    Sxy = np.zeros(i_max, dtype=complex)
    t = np.arange(seg_len, dtype=float)
    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * t / (seg_len - 1.0))) if seg_len > 1 else np.ones(seg_len)
    for s in range(n_seg):
        i0 = s * hop
        xs = x[i0 : i0 + seg_len]
        ys = y[i0 : i0 + seg_len]
        if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
            return None
        xd = xs - np.polyval(np.polyfit(t, xs, 1), t)
        yd = ys - np.polyval(np.polyfit(t, ys, 1), t)
        X = np.fft.rfft(xd * hann)
        Y = np.fft.rfft(yd * hann)
        Xf = X[1 : i_max + 1]
        Yf = Y[1 : i_max + 1]
        Sxx += np.abs(Xf) ** 2
        Syy += np.abs(Yf) ** 2
        Sxy += Xf * np.conj(Yf)
    denom = Sxx * Syy
    denom = np.where(denom > _EPS, denom, np.nan)
    coh = np.abs(Sxy) ** 2 / denom
    mean_coh = float(np.nanmean(coh))
    if not np.isfinite(mean_coh):
        mean_coh = np.nan
    total_phase = float(np.angle(np.sum(Sxy)))
    return mean_coh, total_phase


def _cross_series(x2d: np.ndarray, y2d: np.ndarray, window: int, which: str) -> np.ndarray:
    rows, cols = x2d.shape
    w = max(8, int(window))
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            lo = max(0, r - w + 1)
            run = trailing_contiguous_multi(x2d[lo : r + 1, c], y2d[lo : r + 1, c])
            if run is None:
                continue
            if run[0].shape[0] < 8:
                continue
            res = _cross_spectrum_window(run[0], run[1])
            if res is None:
                continue
            val = res[0] if which == "coherence" else res[1]
            if np.isfinite(val):
                out[r, c] = val
    return out


def _ts_cross_spectral_coherence(x: pd.DataFrame, y: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    if int(window) < 8:
        raise ValueError("ts_cross_spectral_coherence requires window >= 8")
    out = _cross_series(x.to_numpy(dtype=float), y.to_numpy(dtype=float), int(window), "coherence")
    return frame_like(x, out)


def _ts_cross_spectral_phase(x: pd.DataFrame, y: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    if int(window) < 8:
        raise ValueError("ts_cross_spectral_phase requires window >= 8")
    out = _cross_series(x.to_numpy(dtype=float), y.to_numpy(dtype=float), int(window), "phase")
    return frame_like(x, out)


_SPECS: dict[str, dict[str, Any]] = {
    "ts_cross_spectral_coherence": {
        "fn": _ts_cross_spectral_coherence,
        "params": ["x", "y", "window"],
        "category": "time_series_spectral",
        "domain": "cross_spectral",
        "unit": "ratio",
        "cost": 5,
        "tags_extra": [],
        "output_unit": "ratio",
    },
    "ts_cross_spectral_phase": {
        "fn": _ts_cross_spectral_phase,
        "params": ["x", "y", "window"],
        "category": "time_series_spectral",
        "domain": "cross_spectral",
        "unit": "radians",
        "cost": 5,
        "tags_extra": [],
        "output_unit": "radians",
    },
}


def _register() -> None:
    for canonical, spec in _SPECS.items():
        register_dual(
            canonical,
            spec["fn"],
            spec["params"],
            category=spec["category"],
            domain=spec["domain"],
            unit=spec["unit"],
            cost=spec["cost"],
            source="cross_spectrum",
            tags_extra=spec["tags_extra"],
            output_unit=spec.get("output_unit"),
        )
    union_extended(*_SPECS.keys())


_register()
