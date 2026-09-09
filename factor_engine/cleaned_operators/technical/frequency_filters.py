# -*- coding: utf-8 -*-
"""Frequency-domain / causal filtering operators (R47 TRUE_GAP batch).

Four causal trailing-only frequency/smoothing filters for daily wide panels.
All operators are strictly causal (no future leakage), produce NaN until warmup,
and support dual backend (pandas_numpy + polars).

  10. ts_bessel_lowpass_causal      -- IIR Bessel low-pass (maximally flat group delay)
  11. ts_fir_lowpass_causal          -- FIR low-pass (windowed sinc, linear phase)
  12. ts_spectral_lowpass_trailing   -- FFT-based trailing spectral filter
  13. ts_causal_savgol_endpoint      -- Savitzky-Golay smoothing (causal endpoint)

Registration follows R47 conventions: scope="ts", pit_safe=True, dual backend,
registered to EXTENDED_ONLY_CANONICALS, explicit policies in r47_policy_pack.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import (
    OperatorMetadata,
    ParamSpec,
    ParamRole,
    RelationalParamSpec,
    SeriesOperator,
    register_operator,
)
from factor_engine.backend.operator_errors import OperatorParameterError
from factor_engine.cleaned_operators.common.strict_params import strict_int, strict_float, strict_enum

_EPS = 1e-12


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _meta(
    name: str,
    description: str,
    params: list[str],
    *,
    unit: str,
    param_specs: dict[str, ParamSpec] | None = None,
    relational_specs: list[RelationalParamSpec] | None = None,
) -> OperatorMetadata:
    """Metadata factory for frequency filter operators."""
    return OperatorMetadata(
        name=name,
        category="technical_signal",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "technical_signal",
            "daily",
            "pit_safe",
            "causal",
            "typed_v2",
            f"signature:{','.join(params)}->series",
            "domain:technical_signal",
            f"unit:{unit}",
            "cost:2",
            "frequency_filter",
        ],
        param_specs=param_specs or {},
        relational_specs=relational_specs or [],
    )


def _bessel_sos(order: int, cutoff: float) -> np.ndarray:
    """Phase-normalized digital Bessel sections, cutoff in cycles per bar.

    Digital design prewarps the requested frequency and retains ZPK until SOS
    formation.  In particular, no ill-conditioned high-order BA polynomial is
    formed.  Phase normalization preserves the existing prototype convention;
    for order > 1 its cutoff must not be interpreted as a -3 dB frequency.
    """
    from scipy.signal import bessel
    return bessel(order, cutoff, fs=1.0, norm="phase", output="sos")


def _apply_iir_filter(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Apply IIR filter causally (forward pass only, zero initial conditions)."""
    from scipy.signal import sosfilt
    return sosfilt(sos, x, axis=0)


def _fir_lowpass_window(ntaps: int, cutoff: float, window: str = "hamming") -> np.ndarray:
    """Design FIR lowpass filter using windowed sinc method.

    ntaps: number of taps (must be odd for symmetric kernel)
    cutoff: normalized cutoff frequency (0, 0.5)
    window: 'hamming', 'hann', 'blackman'
    """
    from scipy.signal import firwin
    return firwin(ntaps, cutoff, window=window, fs=1.0)


def _savgol_coeffs_causal(window: int, polyorder: int) -> np.ndarray:
    """Savitzky-Golay coefficients for causal (right-aligned) window.

    Returns coefficients for smoothing the LAST point in a trailing window.
    """
    from scipy.signal import savgol_coeffs
    # Standard savgol_coeffs gives symmetric centered window;
    # for causal we use the last half of the symmetric window
    # or compute directly for the endpoint
    # Here we use mode 'nearest' equivalent: full window, last point
    return savgol_coeffs(window, polyorder, deriv=0, delta=1.0, pos=window - 1, use='dot')


# ---------------------------------------------------------------------------
# 10. ts_bessel_lowpass_causal
# ---------------------------------------------------------------------------
@register_operator(
    name="ts_bessel_lowpass_causal",
    category="technical_signal",
    business_category="technical_signal",
    canonical="ts_bessel_lowpass_causal",
    source="technical.frequency_filters",
)
class BesselLowpassCausal(SeriesOperator):
    """Causal IIR Bessel low-pass filter.

    Maximally flat group delay in passband (minimizes phase distortion).
    Strictly causal: uses only trailing data, no future leakage.

    Parameters
    ----------
    x : pd.DataFrame
        Input daily panel (wide: columns = instruments, index = dates).
    order : int
        Filter order (1-8 recommended; higher = steeper rolloff).
    cutoff : float
        Normalized cutoff frequency (0, 0.5) where 0.5 = Nyquist.
        Phase-normalized digital critical frequency (not generally -3 dB).
        Example: 0.05 = roughly 20-day period for daily data.

    Returns
    -------
    pd.DataFrame
        Filtered panel, same shape as input. The first ``order`` finite
        observations are masked. Missing/nonfinite observations freeze the SOS
        state on an event clock and remain NaN; an observed zero still advances
        state. Zero initial conditions are retained. Full-history replay is
        required; a finite warmup is not a checkpoint substitute.
    """

    metadata = _meta(
        "ts_bessel_lowpass_causal",
        "因果 Bessel 低通滤波器：最大平坦群延迟，严格因果性。",
        ["x", "order", "cutoff"],
        unit="price",
        param_specs={
            "order": ParamSpec(dtype=int, min=1, max=8, param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "cutoff": ParamSpec(dtype=float, min=1e-4, max=0.499, param_role=ParamRole.ECONOMIC),
        },
    )

    def _calculate_series(
        self, x: pd.DataFrame, order: int = 4, cutoff: float = 0.05, **_: Any
    ) -> pd.DataFrame:
        order = strict_int(order, name="order", minimum=1, maximum=8)
        cutoff = strict_float(cutoff, name="cutoff", minimum=1e-4, maximum=0.499)

        sos = _bessel_sos(order, cutoff)

        out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
        for col in x.columns:
            series = x[col].values.copy()
            valid = np.isfinite(series)
            if not valid.any():
                continue
            # Event-clock freeze: a missing price is not a zero-price event.
            # Filtering the finite subsequence is exactly the same recurrence
            # as retaining every SOS state across each missing physical row.
            filtered = _apply_iir_filter(series[valid], sos)
            # Warmup counts actual state updates, not missing physical rows.
            filtered[:order] = np.nan
            out.loc[valid, col] = filtered
        return out


# ---------------------------------------------------------------------------
# 11. ts_fir_lowpass_causal
# ---------------------------------------------------------------------------
@register_operator(
    name="ts_fir_lowpass_causal",
    category="technical_signal",
    business_category="technical_signal",
    canonical="ts_fir_lowpass_causal",
    source="technical.frequency_filters",
)
class FIRLowpassCausal(SeriesOperator):
    """Causal FIR low-pass filter using windowed sinc design.

    Linear phase (symmetric group delay), no feedback (stable by design).
    Strictly causal: applies filter causally with trailing window.

    Parameters
    ----------
    x : pd.DataFrame
        Input daily panel (wide: columns = instruments, index = dates).
    ntaps : int
        Number of filter taps (must be odd; higher = sharper transition).
    cutoff : float
        Normalized cutoff frequency (0, 0.5) where 0.5 = Nyquist.
    window : str
        Window type: 'hamming' (default), 'hann', 'blackman'.

    Returns
    -------
    pd.DataFrame
        Filtered panel, same shape as input. The first valid row is ntaps-1.
        Any nonfinite input invalidates the full trailing tap support.
    """

    metadata = _meta(
        "ts_fir_lowpass_causal",
        "因果 FIR 低通滤波器：线性相位，窗函数法，严格因果性。",
        ["x", "ntaps", "cutoff", "window"],
        unit="price",
        param_specs={
            "ntaps": ParamSpec(dtype=int, min=3, max=201, choices=tuple(range(3, 202, 2)),
                               default=21, history_semantics="max_rows",
                               param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "cutoff": ParamSpec(dtype=float, min=1e-4, max=0.499, param_role=ParamRole.ECONOMIC),
            "window": ParamSpec(dtype=str, choices=("hamming", "hann", "blackman"), param_role=ParamRole.POLICY),
        },
    )

    def _calculate_series(
        self, x: pd.DataFrame, ntaps: int = 21, cutoff: float = 0.1, window: str = "hamming", **_: Any
    ) -> pd.DataFrame:
        ntaps = strict_int(ntaps, "ntaps", minimum=3, maximum=201)
        if ntaps % 2 == 0:
            raise OperatorParameterError("ntaps must be odd; no implicit tap expansion")
        cutoff = strict_float(cutoff, "cutoff", minimum=1e-4, maximum=0.499)
        window = strict_enum(window, "window", ("hamming", "hann", "blackman"))

        h = _fir_lowpass_window(ntaps, cutoff, window=window)

        out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
        for col in x.columns:
            series = x[col].values.copy()
            valid = np.isfinite(series)
            if not valid.any():
                continue
            if len(series) < ntaps:
                continue
            # Full convolution's first N entries are the one-sided FIR: no
            # centered alignment and no dependence on a future prefix length.
            filtered = np.convolve(np.where(valid, series, 0.0), h, mode="full")[:len(series)]
            support = np.convolve(valid.astype(np.int64), np.ones(ntaps, dtype=np.int64),
                                  mode="full")[:len(series)]
            filtered[support != ntaps] = np.nan
            out[col] = filtered
        return out


# ---------------------------------------------------------------------------
# 12. ts_spectral_lowpass_trailing
# ---------------------------------------------------------------------------
@register_operator(
    name="ts_spectral_lowpass_trailing",
    category="technical_signal",
    business_category="technical_signal",
    canonical="ts_spectral_lowpass_trailing",
    source="technical.frequency_filters",
)
class SpectralLowpassTrailing(SeriesOperator):
    """FFT-based trailing spectral low-pass filter.

    Applies FFT to a trailing window, zeros high-frequency components,
    and inverse-transforms. Strictly causal: uses only trailing data.

    Parameters
    ----------
    x : pd.DataFrame
        Input daily panel (wide: columns = instruments, index = dates).
    window : int
        Trailing window length for FFT (power of 2 recommended).
    cutoff_freq : int
        Cutoff frequency index: zero all frequencies > cutoff_freq.

    Returns
    -------
    pd.DataFrame
        Filtered panel, same shape as input. NaN until window bars.
    """

    metadata = _meta(
        "ts_spectral_lowpass_trailing",
        "频域低通滤波器：FFT 滑动窗口，零高频分量，严格因果性。",
        ["x", "window", "cutoff_freq"],
        unit="price",
        param_specs={
            "window": ParamSpec(dtype=int, min=8, max=512, history_semantics="max_rows",
                                param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "cutoff_freq": ParamSpec(dtype=int, min=1, max=256, param_role=ParamRole.ECONOMIC),
        },
        relational_specs=[RelationalParamSpec("cutoff_freq <= window // 2",
                                              "cutoff_freq exceeds the trailing window's Nyquist bin")],
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 64, cutoff_freq: int = 5, **_: Any
    ) -> pd.DataFrame:
        window = strict_int(window, "window", minimum=8, maximum=512)
        cutoff_freq = strict_int(cutoff_freq, "cutoff_freq", minimum=1, maximum=256)
        if cutoff_freq > window // 2:
            raise OperatorParameterError("cutoff_freq exceeds the trailing window's Nyquist bin")

        out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
        n = len(x)

        for col in x.columns:
            series = x[col].values.copy()
            valid = np.isfinite(series)
            if not valid.any():
                continue

            result = np.full(n, np.nan, dtype=float)
            for i in range(window, n + 1):
                segment = series[i - window:i]
                seg_valid = valid[i - window:i]
                if not seg_valid.all():
                    continue  # Skip windows with missing data

                # FFT
                fft_vals = np.fft.rfft(segment)
                # Zero high frequencies
                fft_vals[cutoff_freq + 1:] = 0.0
                # IFFT
                filtered_seg = np.fft.irfft(fft_vals, n=window)
                # Take the last (current) value as the causal output
                result[i - 1] = filtered_seg[-1]

            out[col] = result
        return out


# ---------------------------------------------------------------------------
# 13. ts_causal_savgol_endpoint
# ---------------------------------------------------------------------------
@register_operator(
    name="ts_causal_savgol_endpoint",
    category="technical_signal",
    business_category="technical_signal",
    canonical="ts_causal_savgol_endpoint",
    source="technical.frequency_filters",
)
class CausalSavGolEndpoint(SeriesOperator):
    """Causal Savitzky-Golay smoothing filter (endpoint mode).

    Fits a polynomial to a trailing window and evaluates at the endpoint.
    Preserves higher moments (slope, curvature) better than simple smoothing.
    Strictly causal: uses only trailing data.

    Parameters
    ----------
    x : pd.DataFrame
        Input daily panel (wide: columns = instruments, index = dates).
    window : int
        Trailing window length (must be > polyorder).
    polyorder : int
        Polynomial order for local fit (1 = linear, 2 = quadratic, etc.).

    Returns
    -------
    pd.DataFrame
        Smoothed panel, same shape as input. NaN until window bars.
    """

    metadata = _meta(
        "ts_causal_savgol_endpoint",
        "因果 Savitzky-Golay 平滑：多项式局部拟合端点，保持高阶矩。",
        ["x", "window", "polyorder"],
        unit="price",
        param_specs={
            "window": ParamSpec(dtype=int, min=3, max=101, history_semantics="max_rows",
                                param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "polyorder": ParamSpec(dtype=int, min=1, max=5, param_role=ParamRole.POLICY),
        },
        relational_specs=[RelationalParamSpec("polyorder < window",
                                              "polyorder must be smaller than the trailing window")],
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 21, polyorder: int = 2, **_: Any
    ) -> pd.DataFrame:
        window = strict_int(window, "window", minimum=3, maximum=101)
        polyorder = strict_int(polyorder, "polyorder", minimum=1, maximum=5)
        if polyorder >= window:
            raise OperatorParameterError("polyorder must be smaller than the trailing window")

        coeffs = _savgol_coeffs_causal(window, polyorder)

        out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
        for col in x.columns:
            series = x[col].values.copy()
            valid = np.isfinite(series)
            if not valid.any():
                continue

            # Apply causal convolution
            result = np.full(len(series), np.nan, dtype=float)
            for i in range(window - 1, len(series)):
                seg = series[i - window + 1:i + 1]
                seg_valid = valid[i - window + 1:i + 1]
                if seg_valid.all():
                    result[i] = np.dot(coeffs, seg)

            out[col] = result
        return out


# ---------------------------------------------------------------------------
# Recursive history contract
# ---------------------------------------------------------------------------
# The SOS state has unbounded history.  Until the runtime checkpoint adapter
# owns its per-section state, a finite warmup must never stand in for replay.
from factor_engine.runtime.execution_contract import declare_stateful

declare_stateful(
    "ts_bessel_lowpass_causal",
    state_model="recursive",
    chunking="required_full_history",
)
