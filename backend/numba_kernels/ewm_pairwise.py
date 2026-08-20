# -*- coding: utf-8 -*-
"""R21 §32 P18: Numba CPU kernel implementations for EWM pairwise correlation/covariance.

Registers ts_ewm_corr and ts_ewm_cov with the
:class:`~backend.numba_kernel_registry.NumbaKernelRegistry`.

Replaces the POLARS_PANDAS_DELEGATE path with stateful Numba JIT kernels
that reproduce pandas ``ewm(span=window, adjust=False, ignore_na=False,
min_periods=2).corr/cov``.

Contract (verified against pandas 2.3.3):
- alpha = 2/(span+1) via ``span=window`` (span mapping).
- adjust=False, ignore_na=False, min_periods=2 (pandas ewmcov default).
- pairwise-finite: a row is valid only when BOTH operands are finite;
  Inf/-Inf is INVALID (pre-masked to NaN).
- normalized EWM states (weights renormalized so sum_wt == 1 each step).

Key formulas (weights renormalized after every observation, so sum_wt == 1):
- On a valid row with ``gap`` invalid rows since the last valid row:
  D = (1-alpha)**(gap+1), w = alpha/(D+alpha),
  state = w*x + (1-w)*state  (for x, y, xy, xx, yy)
  sum_wt2 = (D^2*sum_wt2 + alpha^2) / (D+alpha)^2
- cov(bias=True)  = state_xy - state_x*state_y
- cov(bias=False) = (state_xy - state_x*state_y) / (1 - sum_wt2)
- corr = (state_xy - state_x*state_y)
         / sqrt((state_xx - state_x^2) * (state_yy - state_y^2))
- Invalid rows freeze the state; the output carries the previous value
  forward (pandas ewm.corr/cov convention with ignore_na=False).
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
    x: np.ndarray, y: np.ndarray, alpha: float, corr: bool, bias: bool = False
) -> np.ndarray:
    """Reference EWM pairwise corr/cov matching pandas ewm(span, adjust=False).

    State machine (normalized weights, sum_wt == 1):
    - state_x, state_y, state_xy, state_xx, state_yy: normalized EWM states
    - sum_wt2: sum of squared normalized weights (for bias correction)
    - gap: number of invalid rows since the last valid row

    corr = (state_xy - state_x*state_y) / sqrt(var_x * var_y)
    cov(bias=True)  = state_xy - state_x*state_y
    cov(bias=False) = (state_xy - state_x*state_y) / (1 - sum_wt2)
    """
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)

    state_x = 0.0
    state_y = 0.0
    state_xy = 0.0
    state_xx = 0.0
    state_yy = 0.0
    sum_wt2 = 0.0
    valid_count = 0
    gap = 0
    first_valid = True

    for i in range(n):
        xi = x[i]
        yi = y[i]

        # Pairwise-finite: both must be finite
        if np.isfinite(xi) and np.isfinite(yi):
            valid_count += 1

            if first_valid:
                # Initialize normalized EWM states with first valid value
                state_x = xi
                state_y = yi
                state_xy = xi * yi
                state_xx = xi * xi
                state_yy = yi * yi
                sum_wt2 = 1.0
                first_valid = False
            else:
                # Weight accounts for the gap of invalid rows
                # (pandas ignore_na=False: distances count raw periods).
                d = (1.0 - alpha) ** (gap + 1)
                w = alpha / (d + alpha)
                state_x = w * xi + (1.0 - w) * state_x
                state_y = w * yi + (1.0 - w) * state_y
                state_xy = w * xi * yi + (1.0 - w) * state_xy
                state_xx = w * xi * xi + (1.0 - w) * state_xx
                state_yy = w * yi * yi + (1.0 - w) * state_yy
                # Renormalized sum of squared weights
                sum_wt2 = (d * d * sum_wt2 + alpha * alpha) / ((d + alpha) ** 2)
            gap = 0
        else:
            gap += 1

        # Need at least 2 valid pairs (min_periods=2). On invalid rows the
        # state is frozen and the previous statistic carries forward.
        if valid_count >= 2:
            cov_biased = state_xy - state_x * state_y
            var_xx = state_xx - state_x * state_x
            var_yy = state_yy - state_y * state_y

            if corr:
                denom_corr = var_xx * var_yy
                if denom_corr > 0:
                    out[i] = cov_biased / np.sqrt(denom_corr)
                # else: zero variance -> stays NaN
            else:
                if bias:
                    out[i] = cov_biased
                else:
                    denom = 1.0 - sum_wt2
                    if denom > 0:
                        out[i] = cov_biased / denom
                    # else: stays NaN

    return out


def _ewm_pairwise_reference_with_state(
    x: np.ndarray, y: np.ndarray, alpha: float, corr: bool, bias: bool = False
) -> tuple[np.ndarray, dict]:
    """Same as reference but returns final state for testing."""
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)

    state_x = 0.0
    state_y = 0.0
    state_xy = 0.0
    state_xx = 0.0
    state_yy = 0.0
    sum_wt2 = 0.0
    valid_count = 0
    gap = 0
    first_valid = True

    for i in range(n):
        xi = x[i]
        yi = y[i]

        if np.isfinite(xi) and np.isfinite(yi):
            valid_count += 1

            if first_valid:
                state_x = xi
                state_y = yi
                state_xy = xi * yi
                state_xx = xi * xi
                state_yy = yi * yi
                sum_wt2 = 1.0
                first_valid = False
            else:
                d = (1.0 - alpha) ** (gap + 1)
                w = alpha / (d + alpha)
                state_x = w * xi + (1.0 - w) * state_x
                state_y = w * yi + (1.0 - w) * state_y
                state_xy = w * xi * yi + (1.0 - w) * state_xy
                state_xx = w * xi * xi + (1.0 - w) * state_xx
                state_yy = w * yi * yi + (1.0 - w) * state_yy
                sum_wt2 = (d * d * sum_wt2 + alpha * alpha) / ((d + alpha) ** 2)
            gap = 0
        else:
            gap += 1

        if valid_count >= 2:
            cov_biased = state_xy - state_x * state_y
            var_xx = state_xx - state_x * state_x
            var_yy = state_yy - state_y * state_y

            if corr:
                denom_corr = var_xx * var_yy
                if denom_corr > 0:
                    out[i] = cov_biased / np.sqrt(denom_corr)
            else:
                if bias:
                    out[i] = cov_biased
                else:
                    denom = 1.0 - sum_wt2
                    if denom > 0:
                        out[i] = cov_biased / denom

    state = {
        "state_x": state_x,
        "state_y": state_y,
        "state_xy": state_xy,
        "state_xx": state_xx,
        "state_yy": state_yy,
        "sum_wt2": sum_wt2,
        "valid_count": valid_count,
    }
    return out, state


# --------------------------------------------------------------------------
# Numba kernels (compiled, fastmath=False, nogil=True)
# --------------------------------------------------------------------------

if _HAS_NUMBA:

    @njit(cache=True, nogil=True, fastmath=False)
    def _ewm_pairwise_numba(
        x: np.ndarray, y: np.ndarray, alpha: float, corr: bool, bias: bool = False
    ) -> np.ndarray:
        """Numba-compiled EWM pairwise corr/cov kernel.

        Same recurrence as reference, but JIT-compiled for performance.
        """
        n = x.shape[0]
        out = np.full(n, np.nan)

        state_x = 0.0
        state_y = 0.0
        state_xy = 0.0
        state_xx = 0.0
        state_yy = 0.0
        sum_wt2 = 0.0
        valid_count = 0
        gap = 0
        first_valid = True

        for i in range(n):
            xi = x[i]
            yi = y[i]

            # Pairwise-finite: both must be finite
            if np.isfinite(xi) and np.isfinite(yi):
                valid_count += 1

                if first_valid:
                    # Initialize normalized EWM states with first valid value
                    state_x = xi
                    state_y = yi
                    state_xy = xi * yi
                    state_xx = xi * xi
                    state_yy = yi * yi
                    sum_wt2 = 1.0
                    first_valid = False
                else:
                    # Weight accounts for the gap of invalid rows
                    d = (1.0 - alpha) ** (gap + 1)
                    w = alpha / (d + alpha)
                    state_x = w * xi + (1.0 - w) * state_x
                    state_y = w * yi + (1.0 - w) * state_y
                    state_xy = w * xi * yi + (1.0 - w) * state_xy
                    state_xx = w * xi * xi + (1.0 - w) * state_xx
                    state_yy = w * yi * yi + (1.0 - w) * state_yy
                    sum_wt2 = (d * d * sum_wt2 + alpha * alpha) / ((d + alpha) ** 2)
                gap = 0
            else:
                gap += 1

            # Need at least 2 valid pairs (min_periods=2). On invalid rows the
            # state is frozen and the previous statistic carries forward.
            if valid_count >= 2:
                cov_biased = state_xy - state_x * state_y
                var_xx = state_xx - state_x * state_x
                var_yy = state_yy - state_y * state_y

                if corr:
                    denom_corr = var_xx * var_yy
                    if denom_corr > 0:
                        out[i] = cov_biased / np.sqrt(denom_corr)
                else:
                    if bias:
                        out[i] = cov_biased
                    else:
                        denom = 1.0 - sum_wt2
                        if denom > 0:
                            out[i] = cov_biased / denom

        return out

    _EWM_PAIRWISE_NUMBA = _ewm_pairwise_numba
else:  # pragma: no cover
    _EWM_PAIRWISE_NUMBA = None


def _register() -> None:
    NumbaKernelRegistry.register(
        "ts_ewm", "ts_ewm_corr", _ewm_pairwise_reference, _EWM_PAIRWISE_NUMBA,
        semantic_version="4.0", supported_dtypes=("float64",),
        supported_param_domain="alpha in (0, 1], min_valid_pairs >= 2",
        nogil=True, parallel=False,
    )
    NumbaKernelRegistry.register(
        "ts_ewm", "ts_ewm_cov", _ewm_pairwise_reference, _EWM_PAIRWISE_NUMBA,
        semantic_version="4.0", supported_dtypes=("float64",),
        supported_param_domain="alpha in (0, 1], min_valid_pairs >= 2",
        nogil=True, parallel=False,
    )


_register()
