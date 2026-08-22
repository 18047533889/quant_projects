# -*- coding: utf-8 -*-
"""R21 §32 P18: Numba CPU kernel implementations for EWM pairwise correlation/covariance.

Registers ts_ewm_corr and ts_ewm_cov with the
:class:`~backend.numba_kernel_registry.NumbaKernelRegistry`.

Replaces the POLARS_PANDAS_DELEGATE path with stateful Numba JIT kernels
that implement the SAME recurrence as pandas ``ewm(span, adjust=False,
ignore_na=False, min_periods=2)`` ``cov``/``corr`` (the Cython ``ewmcov``
online algorithm, verified against pandas 2.3.3 on hostile fixtures).

Contract (verified against pandas 2.3.3):
- alpha = 2/(span+1) via ``span=window`` (span mapping).
- adjust=False, ignore_na=False, min_periods=2 (pandas ewmcov default).
- pairwise-finite: a row is valid only when BOTH operands are finite;
  Inf/-Inf is INVALID (pre-masked to NaN).
- bias=False (unbiased) for cov.  The online recurrence keeps an
  exponentially-weighted running mean of x and y, an exponentially-weighted
  covariance, and the pandas weight accumulators (``sum_wt``/``sum_wt2`` with
  the ``old_wt`` reset that defines ``adjust=False``).  A row where the pair is
  invalid decays the weights by ``(1-alpha)`` but does NOT update the moments,
  exactly matching pandas ``ignore_na=False`` on the pair-masked series
  (the pandas ``ewmcov`` pairwise path masks non-finite pairs to NaN and then
  runs the single-series recurrence).
- corr = cov(x,y) / sqrt(var(x) * var(y)); zero-variance streams yield NaN.
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

def _ewm_pairwise_legacy_reference(
    x: np.ndarray, y: np.ndarray, alpha: float, corr: bool
) -> np.ndarray:
    """LEGACY (superseded) pairwise recurrence, kept only for documentation
    and reference comparison in tests.

    This is the ORIGINAL kernel semantics (EWMA of x*y / x*x / y*y with
    ``sum_wt``/``sum_wt2`` and no mean subtraction).  It is NOT pandas
    ``ewm.cov``/``ewm.corr`` equivalent: with holes or a non-zero mean it
    diverges from pandas.  The active canonical semantics live in the second
    ``_ewm_pairwise_reference`` (pandas ``ewmcov`` online algorithm).

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


def _ewm_pairwise_legacy_state(
    x: np.ndarray, y: np.ndarray, alpha: float, corr: bool
) -> tuple[np.ndarray, dict]:
    """Legacy state-returning variant (superseded — kept for old tests)."""
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

        Same recurrence as reference (pandas ``ewmcov`` online algorithm,
        ``adjust=False, ignore_na=False, bias=False``), JIT-compiled.
        """
        n = x.shape[0]
        out = np.full(n, np.nan)

        mean_x = x[0]
        mean_y = y[0]
        valid0 = np.isfinite(mean_x) and np.isfinite(mean_y)
        nobs = 1 if valid0 else 0
        if not valid0:
            mean_x = np.nan
            mean_y = np.nan

        cov = 0.0
        cov_xx = 0.0
        cov_yy = 0.0
        sum_wt = 1.0
        sum_wt2 = 1.0
        old_wt = 1.0
        old_wt_factor = 1.0 - alpha
        new_wt = alpha  # adjust=False

        for i in range(1, n):
            cur_x = x[i]
            cur_y = y[i]
            is_obs = np.isfinite(cur_x) and np.isfinite(cur_y)
            if is_obs:
                nobs += 1

            if mean_x == mean_x:  # mean_x is not NaN
                # ignore_na=False: a NaN pair still decays the weights
                sum_wt *= old_wt_factor
                sum_wt2 *= old_wt_factor * old_wt_factor
                old_wt *= old_wt_factor
                if is_obs:
                    old_mean_x = mean_x
                    old_mean_y = mean_y

                    # avoid numerical errors on constant series (pandas parity)
                    if mean_x != cur_x:
                        mean_x = (old_wt * old_mean_x + new_wt * cur_x) / (old_wt + new_wt)
                    if mean_y != cur_y:
                        mean_y = (old_wt * old_mean_y + new_wt * cur_y) / (old_wt + new_wt)
                    cov = (
                        old_wt * (cov + (old_mean_x - mean_x) * (old_mean_y - mean_y))
                        + new_wt * (cur_x - mean_x) * (cur_y - mean_y)
                    ) / (old_wt + new_wt)
                    cov_xx = (
                        old_wt * (cov_xx + (old_mean_x - mean_x) * (old_mean_x - mean_x))
                        + new_wt * (cur_x - mean_x) * (cur_x - mean_x)
                    ) / (old_wt + new_wt)
                    cov_yy = (
                        old_wt * (cov_yy + (old_mean_y - mean_y) * (old_mean_y - mean_y))
                        + new_wt * (cur_y - mean_y) * (cur_y - mean_y)
                    ) / (old_wt + new_wt)
                    sum_wt += new_wt
                    sum_wt2 += new_wt * new_wt
                    old_wt += new_wt
                    # adjust=False reset
                    sum_wt /= old_wt
                    sum_wt2 /= old_wt * old_wt
                    old_wt = 1.0
            elif is_obs:
                mean_x = cur_x
                mean_y = cur_y

            if nobs >= 2:
                numerator = sum_wt * sum_wt
                denominator = numerator - sum_wt2
                if denominator > 0:
                    correction = numerator / denominator
                    if corr:
                        var_x = correction * cov_xx
                        var_y = correction * cov_yy
                        if var_x > 0.0 and var_y > 0.0:
                            out[i] = (correction * cov) / np.sqrt(var_x * var_y)
                    else:
                        out[i] = correction * cov
                # else: denominator <= 0 → NaN (constant stream)

        return out


    # Rebind: when numba is present this is the compiled function; the plain
    # Python reference (below) is kept for parity checks.
    _EWM_PAIRWISE_NUMBA = _ewm_pairwise_numba


def _ewm_pairwise_reference(
    x: np.ndarray, y: np.ndarray, alpha: float, corr: bool
) -> np.ndarray:
    """Reference EWM pairwise corr/cov — pandas ``ewmcov`` online algorithm.

    This is a direct port of the pandas 2.3.3 Cython ``ewmcov`` recurrence
    (``pandas/_libs/window/aggregations.pyx``) specialised to
    ``adjust=False, ignore_na=False, bias=False``.  It is the CANONICAL
    semantics that the compiled kernel must reproduce.

    State machine:
    - ``mean_x`` / ``mean_y``: exponentially-weighted running means
      (updated with the pandas ``old_wt``/``new_wt`` weight schedule);
    - ``cov``: exponentially-weighted running covariance (Welford-style);
    - ``cov_xx`` / ``cov_yy``: the same recurrence run on x-with-x and
      y-with-y over the SAME pairwise-valid observation set (a row is valid
      only when BOTH operands are finite — pandas pairwise masking);
    - ``sum_wt`` / ``sum_wt2``: pandas weight accumulators, with the
      ``adjust=False`` reset (``sum_wt /= old_wt; sum_wt2 /= old_wt^2`` after
      each valid pair) that makes the final correction
      ``numerator/denominator = sum_wt^2/(sum_wt^2 - sum_wt2)``.

    Pair-invalid rows (either operand NaN/Inf) DECAY the weights by
    ``(1-alpha)`` (``ignore_na=False``) but leave the moments untouched, which
    is exactly pandas ``ignore_na=False`` on the pair-masked series.
    """
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)

    mean_x = float(x[0])
    mean_y = float(y[0])
    valid0 = np.isfinite(mean_x) and np.isfinite(mean_y)
    nobs = int(valid0)
    if not valid0:
        mean_x = np.nan
        mean_y = np.nan

    cov = 0.0
    cov_xx = 0.0
    cov_yy = 0.0
    sum_wt = 1.0
    sum_wt2 = 1.0
    old_wt = 1.0
    old_wt_factor = 1.0 - alpha
    new_wt = alpha  # adjust=False

    for i in range(1, n):
        cur_x = float(x[i])
        cur_y = float(y[i])
        is_obs = bool(np.isfinite(cur_x) and np.isfinite(cur_y))
        nobs += is_obs

        if mean_x == mean_x:  # mean_x is not NaN
            # ignore_na=False: a NaN pair still decays the weights
            sum_wt *= old_wt_factor
            sum_wt2 *= old_wt_factor * old_wt_factor
            old_wt *= old_wt_factor
            if is_obs:
                old_mean_x = mean_x
                old_mean_y = mean_y

                # avoid numerical errors on constant series (pandas parity)
                if mean_x != cur_x:
                    mean_x = (old_wt * old_mean_x + new_wt * cur_x) / (old_wt + new_wt)
                if mean_y != cur_y:
                    mean_y = (old_wt * old_mean_y + new_wt * cur_y) / (old_wt + new_wt)
                cov = (
                    old_wt * (cov + (old_mean_x - mean_x) * (old_mean_y - mean_y))
                    + new_wt * (cur_x - mean_x) * (cur_y - mean_y)
                ) / (old_wt + new_wt)
                cov_xx = (
                    old_wt * (cov_xx + (old_mean_x - mean_x) ** 2)
                    + new_wt * (cur_x - mean_x) ** 2
                ) / (old_wt + new_wt)
                cov_yy = (
                    old_wt * (cov_yy + (old_mean_y - mean_y) ** 2)
                    + new_wt * (cur_y - mean_y) ** 2
                ) / (old_wt + new_wt)
                sum_wt += new_wt
                sum_wt2 += new_wt * new_wt
                old_wt += new_wt
                # adjust=False reset
                sum_wt /= old_wt
                sum_wt2 /= old_wt * old_wt
                old_wt = 1.0
        elif is_obs:
            mean_x = cur_x
            mean_y = cur_y

        if nobs >= 2:
            numerator = sum_wt * sum_wt
            denominator = numerator - sum_wt2
            if denominator > 0:
                correction = numerator / denominator
                if corr:
                    var_x = correction * cov_xx
                    var_y = correction * cov_yy
                    if var_x > 0.0 and var_y > 0.0:
                        out[i] = (correction * cov) / np.sqrt(var_x * var_y)
                else:
                    out[i] = correction * cov
        # else: out[i] stays NaN (warmup)

    return out


def _ewm_self_var(vals: np.ndarray, alpha: float, upto: int, partner: np.ndarray) -> float:
    """Running ``ewm`` variance (bias=False, adjust=False, ignore_na=False)
    of ``vals[:upto]`` where a row is valid only when both ``vals[i]`` and
    ``partner[i]`` are finite (pandas pairwise masking)."""
    if upto < 2:
        return np.nan
    mean_x = float(vals[0])
    nobs = int(np.isfinite(mean_x) and np.isfinite(partner[0]))
    if not nobs:
        mean_x = np.nan
    cov = 0.0
    sum_wt = 1.0
    sum_wt2 = 1.0
    old_wt = 1.0
    old_wt_factor = 1.0 - alpha
    new_wt = alpha
    out = np.nan
    for i in range(1, upto):
        cur_x = float(vals[i])
        is_obs = bool(np.isfinite(cur_x) and np.isfinite(partner[i]))
        nobs += is_obs
        if mean_x == mean_x:
            if is_obs or True:
                sum_wt *= old_wt_factor
                sum_wt2 *= old_wt_factor * old_wt_factor
                old_wt *= old_wt_factor
                if is_obs:
                    old_mean_x = mean_x
                    if mean_x != cur_x:
                        mean_x = (old_wt * old_mean_x + new_wt * cur_x) / (old_wt + new_wt)
                    cov = (
                        old_wt * (cov + (old_mean_x - mean_x) ** 2)
                        + new_wt * (cur_x - mean_x) ** 2
                    ) / (old_wt + new_wt)
                    sum_wt += new_wt
                    sum_wt2 += new_wt * new_wt
                    old_wt += new_wt
                    sum_wt /= old_wt
                    sum_wt2 /= old_wt * old_wt
                    old_wt = 1.0
        elif is_obs:
            mean_x = cur_x
        if nobs >= 2:
            numerator = sum_wt * sum_wt
            denominator = numerator - sum_wt2
            if denominator > 0:
                out = (numerator / denominator) * cov
    return out


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
