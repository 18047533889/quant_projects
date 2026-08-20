# -*- coding: utf-8 -*-
"""R21 §32 P18: Numba CPU kernel implementations for EWM pairwise correlation/covariance.

Registers ts_ewm_corr and ts_ewm_cov with the
:class:`~backend.numba_kernel_registry.NumbaKernelRegistry`.

Replaces the POLARS_PANDAS_DELEGATE path with stateful Numba JIT kernels
that implement the same recurrence:

    state_t = alpha * x_t * y_t + (1-alpha) * state_{t-1}

Contract (verified against pandas 2.3.3):
- alpha = 2/(span+1) via ``span=window`` (span mapping).
- adjust=False, ignore_na=False, min_periods=2 (pandas ewmcov default).
- pairwise-finite: a row is valid only when BOTH operands are finite;
  Inf/-Inf is INVALID (pre-masked to NaN).
- bias=False (unbiased) for cov: cov = state / (sum_wt^2 - sum_wt2).
- corr = cov(x,y) / sqrt(var(x) * var(y)).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from backend.numba_kernel_registry import (  # noqa: E402
    NUMBA_AVAILABLE,
    NumbaKernelRegistry,
)

if NUMBA_AVAILABLE:
    from numba import njit  # noqa: E402
    _HAS_NUMBA = True
else:  # pragma: no cover
    _HAS_NUMBA = False


# --------------------------------------------------------------------------
# Reference kernels (canonical NumPy implementations)
# --------------------------------------------------------------------------

def _ewm_pairwise_reference(
    x: np.ndarray, y: np.ndarray, alpha: float, corr: bool
) -> np.ndarray:
    """Reference EWM pairwise corr/cov matching pandas ewm(span, adjust=False).

    State machine tracks:
    - sum_wt: cumulative weight sum (for bias correction denominator)
    - sum_wt2: cumulative squared weight sum (for bias correction denominator)
    - state_xy: EWM state for x*y
    - state_xx: EWM state for x*x
    - state_yy: EWM state for y*y

    cov(x,y) = state_xy / (sum_wt - sum_wt2/sum_wt)  [unbiased]
    var(x) = state_xx / (sum_wt - sum_wt2/sum_wt)
    corr(x,y) = state_xy / sqrt(state_xx * state_yy)
    """
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)

    sum_wt = 0.0
    sum_wt2 = 0.0
    state_xy = 0.0
    state_xx = 0.0
    state_yy = 0.0
    valid_count = 0

    for i in range(n):
        xi = x[i]
        yi = y[i]

        # Pairwise-finite: both must be finite
        if np.isfinite(xi) and np.isfinite(yi):
            valid_count += 1
            # Update weight accumulation
            wt = (1.0 - alpha) ** (valid_count - 1)
            sum_wt += wt
            sum_wt2 += wt * wt

            # Update EWM states
            state_xy = alpha * xi * yi + (1.0 - alpha) * state_xy
            state_xx = alpha * xi * xi + (1.0 - alpha) * state_xx
            state_yy = alpha * yi * yi + (1.0 - alpha) * state_yy

            # Need at least 2 valid pairs for corr/cov (bias correction)
            if valid_count >= 2:
                # Bias correction denominator
                denom = sum_wt - sum_wt2 / sum_wt
                if denom > 0:
                    if corr:
                        # corr = cov(x,y) / sqrt(var(x) * var(y))
                        # cov = state_xy / denom, var(x) = state_xx / denom
                        # corr = state_xy / sqrt(state_xx * state_yy)
                        denom_corr = state_xx * state_yy
                        if denom_corr > 0:
                            out[i] = state_xy / np.sqrt(denom_corr)
                        # else: out[i] stays NaN (zero variance)
                    else:
                        # cov (unbiased)
                        out[i] = state_xy / denom
                # else: denom <= 0, out[i] stays NaN
        # else: invalid pair, carry previous value (state unchanged)
        # Note: out[i] already NaN from initialization; we don't update state

    return out


def _ewm_pairwise_reference_with_state(
    x: np.ndarray, y: np.ndarray, alpha: float, corr: bool
) -> tuple[np.ndarray, dict]:
    """Same as reference but returns final state for testing."""
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)

    sum_wt = 0.0
    sum_wt2 = 0.0
    state_xy = 0.0
    state_xx = 0.0
    state_yy = 0.0
    valid_count = 0

    for i in range(n):
        xi = x[i]
        yi = y[i]

        if np.isfinite(xi) and np.isfinite(yi):
            valid_count += 1
            wt = (1.0 - alpha) ** (valid_count - 1)
            sum_wt += wt
            sum_wt2 += wt * wt

            state_xy = alpha * xi * yi + (1.0 - alpha) * state_xy
            state_xx = alpha * xi * xi + (1.0 - alpha) * state_xx
            state_yy = alpha * yi * yi + (1.0 - alpha) * state_yy

            if valid_count >= 2:
                denom = sum_wt - sum_wt2 / sum_wt
                if denom > 0:
                    if corr:
                        denom_corr = state_xx * state_yy
                        if denom_corr > 0:
                            out[i] = state_xy / np.sqrt(denom_corr)
                    else:
                        out[i] = state_xy / denom

    state = {
        "sum_wt": sum_wt,
        "sum_wt2": sum_wt2,
        "state_xy": state_xy,
        "state_xx": state_xx,
        "state_yy": state_yy,
        "valid_count": valid_count,
    }
    return out, state


# --------------------------------------------------------------------------
# Numba kernels (compiled, fastmath=False, nogil=True)
# --------------------------------------------------------------------------

if _HAS_NUMBA:

    @njit(cache=True, nogil=True, fastmath=False)
    def _ewm_pairwise_numba(
        x: np.ndarray, y: np.ndarray, alpha: float, corr: bool
    ) -> np.ndarray:
        """Numba-compiled EWM pairwise corr/cov kernel.

        Same recurrence as reference, but JIT-compiled for performance.
        """
        n = x.shape[0]
        out = np.full(n, np.nan)

        sum_wt = 0.0
        sum_wt2 = 0.0
        state_xy = 0.0
        state_xx = 0.0
        state_yy = 0.0
        valid_count = 0

        for i in range(n):
            xi = x[i]
            yi = y[i]

            # Pairwise-finite: both must be finite
            if np.isfinite(xi) and np.isfinite(yi):
                valid_count += 1
                # Update weight accumulation
                wt = (1.0 - alpha) ** (valid_count - 1)
                sum_wt += wt
                sum_wt2 += wt * wt

                # Update EWM states
                state_xy = alpha * xi * yi + (1.0 - alpha) * state_xy
                state_xx = alpha * xi * xi + (1.0 - alpha) * state_xx
                state_yy = alpha * yi * yi + (1.0 - alpha) * state_yy

                # Need at least 2 valid pairs for corr/cov (bias correction)
                if valid_count >= 2:
                    # Bias correction denominator
                    denom = sum_wt - sum_wt2 / sum_wt
                    if denom > 0:
                        if corr:
                            # corr = state_xy / sqrt(state_xx * state_yy)
                            denom_corr = state_xx * state_yy
                            if denom_corr > 0:
                                out[i] = state_xy / np.sqrt(denom_corr)
                        else:
                            # cov (unbiased)
                            out[i] = state_xy / denom
            # else: invalid pair, state unchanged, output carries previous NaN

        return out

    _EWM_PAIRWISE_NUMBA = _ewm_pairwise_numba
else:  # pragma: no cover
    _EWM_PAIRWISE_NUMBA = None


def _register() -> None:
    NumbaKernelRegistry.register(
        "ts_ewm", "ts_ewm_corr", _ewm_pairwise_reference, _EWM_PAIRWISE_NUMBA,
        semantic_version="1.0", supported_dtypes=("float64",),
        supported_param_domain="alpha in (0, 1], min_valid_pairs >= 2",
        nogil=True, parallel=False,
    )
    NumbaKernelRegistry.register(
        "ts_ewm", "ts_ewm_cov", _ewm_pairwise_reference, _EWM_PAIRWISE_NUMBA,
        semantic_version="1.0", supported_dtypes=("float64",),
        supported_param_domain="alpha in (0, 1], min_valid_pairs >= 2",
        nogil=True, parallel=False,
    )


_register()
