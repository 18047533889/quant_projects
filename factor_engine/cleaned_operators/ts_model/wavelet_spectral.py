# -*- coding: utf-8 -*-
"""Haar wavelet energy and spectral (FFT) operators (P2, experimental).

Fixed wavelet family (Haar, orthogonal), fixed boundary handling and a minimum
window, so semantics are backend-independent.

Round-3 audit #56: the DWT boundary no longer truncates to the LARGEST power of
two (127 obs -> 64, 128 obs -> 128: the factor definition doubled at the
boundary). The public ``window`` domain is the fixed anchor set
{32, 64, 128, 256}; unsupported requests fail instead of silently expanding
the physical lookback. Round-3 audit #58: the FFT spectral factor requires
the FULL trailing window (no 16-bar -> 60-bar partial warmup).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import (
    ParamRole,
    ParamSpec,
    SeriesOperator,
    register_operator,
)
from factor_engine.cleaned_operators.ts_model._rolling_core import frame_like, metadata
from factor_engine.cleaned_operators.common.strict_params import (
    strict_int,
    strict_positive_int,
)

_CANONICALS: list[str] = []


def _register(
    name: str,
    description: str,
    params: list[str],
    unit: str,
    fn,
    cost: int = 8,
    *,
    fixed_window: bool = False,
    window_spec: ParamSpec | None = None,
):
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
        metadata = metadata(
            name,
            description,
            params,
            unit=unit,
            cost=cost,
            param_specs={
                "window": ParamSpec(
                    dtype=int,
                    choices=_FIXED_WINDOW_ANCHORS,
                    default=128,
                    param_role=ParamRole.HORIZON,
                )
            } if fixed_window else ({"window": window_spec} if window_spec else None),
        )

        def _calculate_series(self, *args, **kwargs):
            return fn(*args, **kwargs)

    _CANONICALS.append(name)
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_research_only({name})
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

    Audit #55: a NaN at the CURRENT (last) row yields an empty block — a missing
    current value must produce NaN at the current output, never fall back to
    yesterday's trailing run (this matches the shared
    ``_rolling_core.trailing_contiguous_finite`` / ``gemini_v2_common``
    fail-closed-current-row contract).
    """
    n = len(vals)
    if n == 0 or not np.isfinite(vals[-1]):
        return np.empty(0, dtype=float)
    i = n
    while i > 0 and np.isfinite(vals[i - 1]):
        i -= 1
    return vals[i:].astype(float)


_FIXED_WINDOW_ANCHORS = (32, 64, 128, 256)


def _fixed_window_anchor(window: int) -> int:
    """Validate the public Haar window against the supported anchors.

    Hidden rounding can make the physical lookback exceed the declared window
    (for example, 100 -> 128), which breaks warmup and cache identity.  Haar
    operators expose the discrete supported domain and reject every other value.
    """
    requested = strict_int(window, "window")
    if requested not in _FIXED_WINDOW_ANCHORS:
        raise ValueError(
            f"window must be one of {_FIXED_WINDOW_ANCHORS}; got {requested}"
        )
    return requested


def _haar_energy(vals: np.ndarray, window: int) -> list[tuple[int, float]]:
    """Return ``(dyadic scale, detail energy)`` pairs, coarse to fine."""
    anchor = _fixed_window_anchor(window)
    seg = vals[-anchor:]
    finite = _trailing_contiguous_finite(seg)
    # Require the FULL fixed window (audit #56): a 127-observation window must
    # NOT silently drop down to a 64-observation Haar transform.  If the fixed
    # anchor is not yet available (or the current value is missing, audit #55)
    # fail closed with NaN.
    if finite.size < anchor:
        return []
    # Keep the NEWEST ``anchor`` observations within the contiguous run: the
    # oldest prefix carries stale information for a trailing factor (review
    # P0-12).
    x = finite[-anchor:].copy()
    levels: list[tuple[int, float]] = []
    scale = 2
    while len(x) >= 2:
        approx = (x[::2] + x[1::2]) / np.sqrt(2.0)
        detail = (x[::2] - x[1::2]) / np.sqrt(2.0)
        levels.append((scale, float(np.sum(detail * detail))))
        x = approx
        scale *= 2
    levels.reverse()  # coarse/low-frequency -> fine/high-frequency
    return levels


def _wavelet_stats(vals: np.ndarray, window: int, stat: str) -> float:
    levels = _haar_energy(vals, window)
    if len(levels) < 2:
        return np.nan
    energies = [energy for _scale, energy in levels]
    total = float(sum(energies))
    if total <= 1e-12:
        return np.nan
    if stat == "low":
        return float(energies[0] / total)
    if stat == "high":
        return float(energies[-1] / total)
    if stat == "entropy":
        w = np.array([e / total for e in energies])
        # A zero-energy level contributes 0*log(0) = NaN; drop it before the
        # sum (highly regular series frequently have one empty band — review
        # P0-13).  The normalization uses the number of *present* bands.
        w = w[w > 0.0]
        if w.size == 0:
            return np.nan  # no positive energy at all (degenerate)
        if w.size == 1:
            # Audit #57: p = (1, 0, 0, ...) has Shannon entropy H = 0 *theoretically*
            # (a single band carries all the energy).  The old code returned NaN
            # because the normalization denominator log(1) = 0; a single positive
            # band is a perfectly valid, maximally-concentrated distribution.
            return 0.0
        h = -float(np.sum(w * np.log(w)))
        return float(h / np.log(w.size))
    if stat == "slope":
        # Haar level j has dyadic scale 2^j, so the regressor is log(2^j) =
        # j*log(2), not log(j) — review P1-16.  Population cov/var (same ddof)
        # avoid the n/(n-1) inflation of np.cov/np.var — review P1-17.
        scales = np.log(np.array([scale for scale, _energy in levels], dtype=float))
        log_e = np.log(np.maximum(energies, 1e-15))
        sb = float(np.mean(scales))
        lb = float(np.mean(log_e))
        cov = float(np.mean((scales - sb) * (log_e - lb)))
        var = float(np.mean((scales - sb) ** 2))
        return cov / var if var > 0.0 else np.nan
    return np.nan


_register("ts_wavelet_low_frequency_ratio", "低频小波能量占比。", ["x", "window"], "ratio",
           lambda x, window=128: _apply(x, lambda v: _wavelet_stats(v, window, "low")), fixed_window=True)
_register("ts_wavelet_high_frequency_ratio", "高频小波能量占比。", ["x", "window"], "ratio",
           lambda x, window=128: _apply(x, lambda v: _wavelet_stats(v, window, "high")), fixed_window=True)
_register("ts_wavelet_entropy", "小波各尺度能量熵。", ["x", "window"], "level",
           lambda x, window=128: _apply(x, lambda v: _wavelet_stats(v, window, "entropy")), fixed_window=True)
_register("ts_wavelet_energy_slope", "小波能量随尺度变化的斜率。", ["x", "window"], "level",
           lambda x, window=128: _apply(x, lambda v: _wavelet_stats(v, window, "slope")), fixed_window=True)


def _spectral_low_ratio(vals: np.ndarray, window: int) -> float:
    w = strict_positive_int(window, "window")
    seg = vals[-w:]
    # P1-15: same no-time-axis-compression rule as the Haar kernel — use the
    # trailing contiguous finite suffix (frequency/phase semantics need a real
    # time axis; review R4-64).
    finite = _trailing_contiguous_finite(seg)
    # Audit #58: no partial warmup — a trailing spectral factor must not emit a
    # 16-bar spectrum, then 17, ... up to the full window (the front would be a
    # DIFFERENT factor than the 60-bar tail).  Require the FULL trailing window
    # (contiguous finite) or NaN.  (A missing current value already yields an
    # empty block via ``_trailing_contiguous_finite``, audit #55.)
    if finite.size < w:
        return np.nan
    demean = finite - np.mean(finite)
    spec = np.abs(np.fft.rfft(demean)) ** 2
    weights = np.full(spec.size, 2.0, dtype=float)
    weights[0] = 1.0
    if w % 2 == 0:
        weights[-1] = 1.0
    energy = weights * spec
    total = float(np.sum(energy))
    if total <= 1e-12:
        return np.nan
    half = max(1, len(spec) // 2)
    return float(np.sum(energy[:half]) / total)


_register("ts_spectral_low_frequency_ratio", "傅里叶低频能量占比。", ["x", "window"], "ratio",
           lambda x, window=128: _apply(x, lambda v: _spectral_low_ratio(v, window)),
           window_spec=ParamSpec(dtype=int, min=1, default=128, param_role=ParamRole.HORIZON))
