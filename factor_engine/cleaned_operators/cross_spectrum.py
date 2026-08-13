# -*- coding: utf-8 -*-
"""Cross-spectral primitives between two series (2026-08-08 Gemini V2 round).

Both operators use one kernel: the trailing window is linearly detrended and
Hann-windowed for *both* series, an FFT produces the cross-spectrum
``S_xy(f) = X(f)·conj(Y(f))`` and the auto-spectra, and two summaries are
returned:

* ``ts_cross_spectral_coherence`` — the mean coherence
  ``|S_xy|² / (S_xx·S_yy)`` over a fixed positive-frequency ``band``
  (``all``/``long``/``medium``/``short``; 0 = unrelated, 1 = perfectly
  coherent phase-locked dynamics at that frequency).
* ``ts_cross_spectral_phase`` — the coherence-amplitude-weighted circular mean
  phase ``arg(Σ_f |S_xy|·e^{iφ_f})`` in radians.  This is a genuine circular
  mean (never a plain arithmetic mean of angles, which wraps); the weighted
  form equals ``arg(Σ_f S_xy)``.  With ``S_xy = X·conj(Y)`` a POSITIVE phase
  means ``x`` LEADS ``y``.  R5 P1-08: when the band coherence drops below
  ``min_coherence`` the phase is NaN (a random angle carries no signal).
  P1-14: the DEFAULT gate is a non-trivial ``min_coherence=0.2`` (was ``0.0``)
  — below it the coherence is statistically indistinguishable from noise and
  the phase is a random angle, so gating to NaN is the honest default; callers
  who explicitly want every finite spectrum's angle can pass ``0.0``.

Both are strict-PIT, deterministic, and NaN fail-closed: a NaN in either
series invalidates the window (the time axis is never compressed), and the
window contract is ``window >= 24`` so at least two Welch segments always
resolve (R5 P1-07).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import ParamSpec, ParamRole
from cleaned_operators.gemini_v2_common import (
    frame_like,
    register_dual,
    trailing_contiguous_multi,
    union_extended,
)

_EPS = 1e-12

# R5 P1-01/P1-09: ``band`` is a discrete reviewed enum (no floating frequency
# exposed to the search grammar), ``min_coherence`` is a bounded float.
_CROSS_SPEC = {
    "window": ParamSpec(dtype=int, min=24),
    "band": ParamSpec(dtype=str, choices=("all", "long", "medium", "short"), searchable=False),
}
_PHASE_SPEC = {
    "window": ParamSpec(dtype=int, min=24),
    "band": ParamSpec(dtype=str, choices=("all", "long", "medium", "short"), searchable=False),
    "min_coherence": ParamSpec(dtype=float, min=0.0, max=1.0, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
}

# Discrete reviewed frequency bands (R5 P1-09): a *fixed* partition of the
# positive-frequency bins — no arbitrary floating frequency is exposed to the
# search grammar.  "long" = low frequencies (business-cycle/weekly style),
# "medium" = intermediate, "short" = high-frequency.
_BANDS = {
    "all": lambda lo, hi, i_max: (lo, hi),
    "long": lambda lo, hi, i_max: (lo, max(lo, hi // 3)),
    "medium": lambda lo, hi, i_max: (max(lo, hi // 3), max(lo, (2 * hi) // 3)),
    "short": lambda lo, hi, i_max: (max(lo, (2 * hi) // 3), hi),
}


def _band_range(band: str, i_max: int) -> tuple[int, int]:
    fn = _BANDS.get(str(band).lower(), _BANDS["all"])
    return fn(0, i_max, i_max)


def _cross_spectrum_window(
    x: np.ndarray, y: np.ndarray, band: str = "all", min_coherence: float = 0.0
) -> tuple[float, float] | None:
    """Welch-averaged cross-coherence and weighted mean phase over ``band``.

    The per-frequency squared coherence ``|S_xy|²/(S_xx·S_yy)`` of a single
    unsmoothed periodogram is identically 1 (``|a·conj(b)|² = |a|²·|b|²`` for
    complex ``a, b``), so the spectra must be ensemble-averaged before the
    ratio is taken.  The window is split into 50%-overlap Hann segments, the
    cross-spectrum and auto-spectra are averaged across segments, and only
    then is the coherence formed.

    Phase sign convention (R5 P1-08): with ``S_xy = X(f)·conj(Y(f))``, a
    POSITIVE phase means ``x`` LEADS ``y`` (``y`` is a delayed copy of ``x``:
    ``y(t) = x(t−δ) ⇒ Y = X·e^{−2πifδ} ⇒ S_xy = |X|²·e^{+2πifδ}``).
    """
    n = x.shape[0]
    # R5 P1-07: the public window contract guarantees >= 2 Welch segments
    # (seg_len = max(16, n/2); a window below 24 yields a single segment and an
    # identically-1 coherence, so it is rejected up front).
    if n < 24:
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
    hann = np.where((seg_len - 1.0))) if seg_len > 1 else np.ones(seg_len) != 0, 0.5 * (1.0 - np.cos(2.0 * np.pi * t / (seg_len - 1.0))) if seg_len > 1 else np.ones(seg_len), np.nan)
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
    lo, hi = _band_range(band, i_max)
    if hi - lo < 1:
        return None
    Sxx_b = Sxx[lo:hi]
    Syy_b = Syy[lo:hi]
    Sxy_b = Sxy[lo:hi]
    denom = Sxx_b * Syy_b
    ok = denom > _EPS
    # P1-33: energy-weighted band coherence ``Σ|S_xy|² / Σ(S_xx·S_yy)`` over the
    # resolvable band frequencies — not a plain arithmetic mean of the ratios
    # (which weights every bin equally regardless of the energy actually at that
    # bin).  By Cauchy-Schwarz this stays in [0, 1].
    num_coh = float(np.sum(np.abs(Sxy_b)[ok] ** 2))
    den_coh = float(np.sum(denom[ok]))
    mean_coh = num_coh / den_coh if den_coh > _EPS else np.nan
    if not np.isfinite(mean_coh):
        mean_coh = np.nan
    # R5 P1-08 / P1-14: at (near-)zero coherence the phase is a random angle —
    # gate it to NaN below ``min_coherence`` instead of emitting noise.  The
    # public phase operator now defaults to ``min_coherence=0.2`` so the gate
    # is on by default (a coherence too close to noise yields NaN, not a
    # spurious angle).
    if mean_coh < float(min_coherence):
        total_phase = np.nan
    else:
        total_phase = float(np.angle(np.sum(Sxy_b)))
    return mean_coh, total_phase


def _cross_series(
    x2d: np.ndarray, y2d: np.ndarray, window: int, which: str,
    band: str = "all", min_coherence: float = 0.0,
) -> np.ndarray:
    rows, cols = x2d.shape
    w = max(24, int(window))
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            lo = max(0, r - w + 1)
            run = trailing_contiguous_multi(x2d[lo : r + 1, c], y2d[lo : r + 1, c])
            if run is None:
                continue
            if run[0].shape[0] < 24:
                continue
            res = _cross_spectrum_window(run[0], run[1], band, min_coherence)
            if res is None:
                continue
            val = res[0] if which == "coherence" else res[1]
            if np.isfinite(val):
                out[r, c] = val
    return out


def _ts_cross_spectral_coherence(
    x: pd.DataFrame, y: pd.DataFrame, window: int = 60, band: str = "all"
) -> pd.DataFrame:
    if int(window) < 24:
        raise ValueError("ts_cross_spectral_coherence requires window >= 24")
    out = _cross_series(
        x.to_numpy(dtype=float), y.to_numpy(dtype=float), int(window), "coherence", band
    )
    return frame_like(x, out)


def _ts_cross_spectral_phase(
    x: pd.DataFrame, y: pd.DataFrame, window: int = 60, band: str = "all",
    min_coherence: float = 0.2,
) -> pd.DataFrame:
    if int(window) < 24:
        raise ValueError("ts_cross_spectral_phase requires window >= 24")
    if not 0.0 <= float(min_coherence) <= 1.0:
        raise ValueError("ts_cross_spectral_phase requires 0 <= min_coherence <= 1")
    out = _cross_series(
        x.to_numpy(dtype=float), y.to_numpy(dtype=float), int(window), "phase",
        band, float(min_coherence),
    )
    return frame_like(x, out)


_SPECS: dict[str, dict[str, Any]] = {
    "ts_cross_spectral_coherence": {
        "fn": _ts_cross_spectral_coherence,
        "params": ["x", "y", "window", "band"],
        "category": "time_series_spectral",
        "domain": "cross_spectral",
        "unit": "ratio",
        "cost": 5,
        "tags_extra": [],
        "output_unit": "ratio",
        "param_specs": _CROSS_SPEC,
    },
    "ts_cross_spectral_phase": {
        "fn": _ts_cross_spectral_phase,
        "params": ["x", "y", "window", "band", "min_coherence"],
        "category": "time_series_spectral",
        "domain": "cross_spectral",
        "unit": "radians",
        "cost": 5,
        "tags_extra": [],
        "output_unit": "radians",
        "param_specs": _PHASE_SPEC,
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
            param_specs=spec.get("param_specs"),
        )
    union_extended(*_SPECS.keys())


_register()
