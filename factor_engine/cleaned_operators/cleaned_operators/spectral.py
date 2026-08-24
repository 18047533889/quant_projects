# -*- coding: utf-8 -*-
"""Spectral-shape operators (2026-08 geometry/math expansion).

Every operator shares one kernel: the trailing window is *linearly detrended*,
multiplied by a *Hann window*, an FFT produces the periodogram ``P(f)`` over the
positive frequencies ``f_i = i/N`` for ``i = 1..floor(N/2)`` (DC is excluded,
Nyquist is ``floor(N/2)/N``):

* ``ts_spectral_centroid``          — Σ f·P / Σ P, normalized by Nyquist.
* ``ts_spectral_flatness``          — geometric mean / arithmetic mean of P ∈ [0,1].
* ``ts_spectral_peak_concentration``— max(P) / Σ P.
* ``ts_spectral_quality_factor``    — peak frequency / half-power bandwidth.

All operators are trailing-window, prefix-causal and deterministic.  Missing
policy (P1-009): the trailing window must be fully contiguous-finite — a NaN is
never zero-padded (a missing value must not inject spectral energy).  A window
with fewer than 16 rows, or any NaN row, emits NaN; a degenerate (all-zero)
spectrum emits NaN.  Cross-day price inputs must be continuous price or return
(not raw close).  Invalid parameters raise ``ValueError``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata, ParamSpec, SeriesOperator, register_operator
from factor_engine.cleaned_operators.rolling_pack import frame_like, register_polars_bridge

_EPS = 1e-12
_MIN_FINITE = 16

# Audit #80: the shared periodogram kernel requires >= 16 fully-finite rows; a
# window of 4..15 would only ever emit NaN, so it must not enter the search
# surface.  Declared here so binder/search reject those windows up front.
_SPECTRAL_PARAM_SPECS: dict[str, ParamSpec] = {
    "window": ParamSpec(dtype=int, min=_MIN_FINITE),
}


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    meta = OperatorMetadata(
        name=name,
        category="spectral",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "spectral", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:spectral",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )
    meta.param_specs = dict(_SPECTRAL_PARAM_SPECS)
    return meta


def _periodogram(chunk: np.ndarray) -> tuple[np.ndarray, int] | None:
    """Linear-detrend + Hann + FFT periodogram, positive freqs (P1-009).

    Returns ``(P, i_max)`` where ``P[i]`` corresponds to ``f = (i+1)/N`` for
    ``i = 0..i_max-1`` and ``i_max = floor(N/2)``.  ``None`` when the window is
    not FULLY finite (trailing contiguous window): a missing value is never
    zero-padded — zero-padding injects spurious low-frequency energy from the
    missing pattern into the spectrum.  Cross-day price inputs must be continuous
    price or return (not raw close) so the detrend + spectrum describes the
    return-generating process, not the un-adjusted level path.
    """
    n = chunk.size
    if n < _MIN_FINITE:
        return None
    v = chunk.astype(float)
    if not np.all(np.isfinite(v)):
        return None
    t = np.arange(n, dtype=float)
    # linear detrend on the full window (all finite by the check above)
    slope, intercept = np.polyfit(t, v, 1)
    resid = v - (slope * t + intercept)
    # Hann window
    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * t / (n - 1.0))) if n > 1 else np.ones(n)
    resid = resid * hann
    spectrum = np.fft.rfft(resid)
    power = (np.abs(spectrum) ** 2) / float(n)
    i_max = n // 2
    p_pos = power[1 : i_max + 1]  # drop DC, keep positive frequencies
    if p_pos.size == 0:
        return None
    return p_pos, i_max


def _spectral_centroid_series(x2d: np.ndarray, window: int) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            pg = _periodogram(col[i0 : r + 1])
            if pg is None:
                continue
            p, i_max = pg
            n = col[i0 : r + 1].size
            total = float(p.sum())
            if total <= 0 or not np.isfinite(total):
                continue
            freqs = np.arange(1, i_max + 1, dtype=float) / float(n)
            sc = float(np.sum(freqs * p) / total)
            nyq = i_max / float(n)
            if nyq > 0:
                out[r, c] = sc / nyq
    return out


def _spectral_flatness_series(x2d: np.ndarray, window: int) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            pg = _periodogram(col[i0 : r + 1])
            if pg is None:
                continue
            p, _ = pg
            mean_p = float(p.mean())
            if mean_p <= 0 or not np.isfinite(mean_p):
                continue
            with np.errstate(divide="ignore"):
                geo = float(np.exp(np.mean(np.log(p))))
            out[r, c] = geo / (mean_p + _EPS)
    return out


def _spectral_peak_concentration_series(x2d: np.ndarray, window: int) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            pg = _periodogram(col[i0 : r + 1])
            if pg is None:
                continue
            p, _ = pg
            total = float(p.sum())
            if total <= 0 or not np.isfinite(total):
                continue
            out[r, c] = float(p.max() / total)
    return out


def _spectral_quality_factor_series(x2d: np.ndarray, window: int) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            pg = _periodogram(col[i0 : r + 1])
            if pg is None:
                continue
            p, i_max = pg
            n = col[i0 : r + 1].size
            total = float(p.sum())
            if total <= 0 or not np.isfinite(total):
                continue
            k_star = int(np.argmax(p))
            half = float(p.max()) / 2.0
            k_lo = k_star
            while k_lo - 1 >= 0 and p[k_lo - 1] >= half:
                k_lo -= 1
            k_hi = k_star
            while k_hi + 1 < p.size and p[k_hi + 1] >= half:
                k_hi += 1
            i_star = k_star + 1
            i_lo = k_lo + 1
            i_hi = k_hi + 1
            f_star = i_star / float(n)
            df = (i_hi - i_lo) / float(n)
            # Audit #81: a single-bin peak has half-power width df = 0 — the
            # old ``f_star / (df + EPS)`` emitted Q ~ f_star/EPS (huge,
            # meaningless).  Use AT LEAST one-bin resolution: the FFT cannot
            # resolve a width narrower than 1/N, so Q is bounded by the peak
            # bin index (<= n/2) and never explodes.
            df = max(df, 1.0 / float(n))
            out[r, c] = f_star / df
    return out


def _check_spectral_params(window: int) -> int:
    w = int(window)
    if w < _MIN_FINITE:
        raise ValueError(f"window must be >= {_MIN_FINITE}")
    return w


@register_operator(
    name="ts_spectral_centroid",
    category="spectral",
    business_category="spectral",
    canonical="ts_spectral_centroid",
    source="spectral",
)
class TsSpectralCentroid(SeriesOperator):
    """谱质心：``SC = Σ f·P(f)/Σ P(f)`` 再除以 Nyquist（``i_max/N``），∈[0,1]。

    高 → 能量集中在高频（短周期/噪声主导）；低 → 低频（趋势/慢周期）主导。P2。
    """

    metadata = _metadata(
        "ts_spectral_centroid",
        "周期图正频加权质心，按 Nyquist 归一（频率重心）。",
        ["x", "window"],
        unit="ratio",
        cost=3,
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, **_: Any) -> pd.DataFrame:
        w = _check_spectral_params(window)
        return frame_like(x, _spectral_centroid_series(x.to_numpy(dtype=float), w))


@register_operator(
    name="ts_spectral_flatness",
    category="spectral",
    business_category="spectral",
    canonical="ts_spectral_flatness",
    source="spectral",
)
class TsSpectralFlatness(SeriesOperator):
    """谱平坦度：``exp(mean(log P))/(mean P + eps)`` ∈ [0,1]。

    白噪声 → 接近 1；强周期/窄带 → 接近 0。任一频点能量为 0 → 0。P2。
    """

    metadata = _metadata(
        "ts_spectral_flatness",
        "周期图几何均值 / 算术均值（频谱平坦 vs 尖峰）。",
        ["x", "window"],
        unit="ratio",
        cost=3,
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, **_: Any) -> pd.DataFrame:
        w = _check_spectral_params(window)
        return frame_like(x, _spectral_flatness_series(x.to_numpy(dtype=float), w))


@register_operator(
    name="ts_spectral_peak_concentration",
    category="spectral",
    business_category="spectral",
    canonical="ts_spectral_peak_concentration",
    source="spectral",
)
class TsSpectralPeakConcentration(SeriesOperator):
    """谱峰集中度：``max(P)/Σ P`` ∈ (0,1]。

    高 → 单一强主导周期；低 → 能量铺开在多个频率。P2。
    """

    metadata = _metadata(
        "ts_spectral_peak_concentration",
        "周期图最大峰值占比（单一主导周期 vs 频谱铺开）。",
        ["x", "window"],
        unit="ratio",
        cost=3,
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, **_: Any) -> pd.DataFrame:
        w = _check_spectral_params(window)
        return frame_like(x, _spectral_peak_concentration_series(x.to_numpy(dtype=float), w))


@register_operator(
    name="ts_spectral_quality_factor",
    category="spectral",
    business_category="spectral",
    canonical="ts_spectral_quality_factor",
    source="spectral",
)
class TsSpectralQualityFactor(SeriesOperator):
    """谱品质因子：``Q = f*/(Δf + eps)``，f*=峰值频率，Δf=半功率带宽。

    Q 大 → 周期峰尖锐（稳定周期）；Q 小 → 峰宽（周期漂移 / 噪声）。P2。
    """

    metadata = _metadata(
        "ts_spectral_quality_factor",
        "峰值频率 / 半功率带宽（周期峰尖锐度）。",
        ["x", "window"],
        unit="ratio",
        cost=4,
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, **_: Any) -> pd.DataFrame:
        w = _check_spectral_params(window)
        return frame_like(x, _spectral_quality_factor_series(x.to_numpy(dtype=float), w))


_NEW_CANONICALS = (
    "ts_spectral_centroid",
    "ts_spectral_flatness",
    "ts_spectral_peak_concentration",
    "ts_spectral_quality_factor",
)


def _register_surface() -> None:
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
