# -*- coding: utf-8 -*-
"""Change-point detection primitives (2026-08-08 Gemini V2 round).

Three deterministic, trailing-window change-point scores.  Each scans a window
for the best breakpoint ``τ`` (``min_segment ≤ τ ≤ n - min_segment``) using a
Gaussian profile-likelihood or rank statistic, and returns a *signed* score
(never a p-value — financial series are serially correlated and the
independence assumptions behind change-point p-values do not hold).
The GLR scores are ``sign · sqrt(calibrated_LLR)`` — bounded by a documented
cap (P0-L-69: the strongest possible change saturates at ``sqrt(40) ≈ 6.32``,
never a program-constant-dominated value), and null-calibrated against the
number of scanned breakpoints (P0-L-70).  Only the rank-based Pettitt score is
normalised into [-1, 1].  ``window >= 2·min_segment + 2`` is required for any
breakpoint to exist — smaller windows are rejected up front instead of silently
returning all-NaN.

* ``ts_glr_mean_shift_score``      — generalised likelihood ratio for a mean
  shift: ``LLR_τ = (n/2)·ln(σ̂² / σ̂_w²)``, output ``sign(μ_post - μ_pre) ·
  sqrt(calibrated_LLR)``.  Prefix sums make the whole scan ``O(n)`` per window.
* ``ts_glr_variance_shift_score``  — same scan against H1 "two variances":
  ``LLR_τ = (n/2)ln(σ̂²) - (n₁/2)ln(σ̂₁²) - (n₂/2)ln(σ̂₂²)``, output
  ``sign(ln(var_post/var_pre)) · sqrt(calibrated_LLR)``.
* ``ts_pettitt_change_score``      — rank-based Pettitt statistic
  ``U_t = 2·Σ_{i≤t} r_i - t(n+1)`` (average-rank ties), output
  ``sign(U_τ*) · |U_τ*| / (n²/4)`` (bounded in [-1, 1]; 0 ≈ no change).

All are strict-PIT (only the trailing contiguous finite window), deterministic
and NaN fail-closed.

P0-L round-3 calibration (both GLR scores):
* *Perfect-split cap.*  A perfect split (residual within-segment SS ≈ 0) has an
  unbounded true LLR.  The old ``EPS * full_ss`` denominator made the max score
  a data-independent function of the program constant ``1e-12``.  The LLR is
  now computed from the genuine floating-point cancellation floor of the
  prefix-sum SS (relative error ~ ``n·eps``) purely to *classify* a split as
  perfect, and every LLR is clamped at the documented cap ``_GLR_MAX_LLR = 40``
  (output score saturates at ``sqrt(40) ≈ 6.32``).
* *Scan-selection null calibration.*  A window of ``n`` bars scans
  ``m = n - 2·min_segment + 1`` candidate breakpoints, so the raw max LLR's null
  rises with the window (multiple-testing bias).  A BIC-style ``ln(m)`` penalty
  is subtracted so the null is centred near 0 regardless of window.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import ParamSpec, RelationalParamSpec
from cleaned_operators.gemini_v2_common import (
    frame_like,
    register_dual,
    trailing_contiguous_finite,
    union_extended,
)

_EPS = 1e-12  # used ONLY for near-zero classification in the rank Pettitt score

# P0-L-69: GLR perfect-split handling.  A change-point GLR divides by the pooled
# within-segment residual sum of squares; a *perfect* split (both segments
# constant) drives that residual SS to floating-point zero and the true LLR is
# unbounded.  The old ``max(pooled, _EPS * full_ss)`` denominator made the max
# score a data-independent function of an arbitrary program epsilon
# (``sqrt((n/2)·ln 1e12)``).  Instead:
#   * ``_GLR_MEASURE_NOISE_MULT`` scales the genuine floating-point cancellation
#     floor of the prefix-sum SS computation (relative error ~ n·eps) and is used
#     ONLY to classify a split as "perfect" — NEVER as a denominator;
#   * ``_GLR_MAX_LLR`` is the documented cap on any change-point LLR; the output
#     score saturates at ``sqrt(_GLR_MAX_LLR)`` so it is never dominated by a
#     program constant.
_GLR_MEASURE_NOISE_MULT = 8.0
_GLR_MAX_LLR = 40.0

# P0-L-70: max-over-breakpoint scan-selection bias.  A window of n bars scans
# m = n - 2·min_segment + 1 candidate breakpoints, so the raw max LLR's null
# rises with the window (multiple-testing bias).  Null-calibrate by subtracting
# a BIC-style ``ln(m)`` penalty: under the null the sup of ~chi-square_1 GLR
# stats grows ~ ln m, so subtracting ``ln m`` centres the null near 0
# independent of the window.  Output is ``sign · sqrt(max(calibrated_LLR, 0))``.
_GLR_SCAN_PENALTY_COEF = 1.0


def _ss_noise_floor(full_ss: float, n: int) -> float:
    """Floating-point measurement-noise floor for a prefix-sum SS.

    Prefix-sum SS cancellation carries relative error ~ ``n·eps_machine``; any
    pooled within-segment SS at or below this floor is indistinguishable from a
    PERFECT split (residual SS = 0).  Used only as a classifier, never as a
    denominator."""
    return full_ss * n * _GLR_MEASURE_NOISE_MULT * np.finfo(float).eps


def _null_calibrate_llr(best: float, n: int, min_segment: int) -> float:
    """Apply the documented LLR cap and the scan-selection null calibration.

    ``best`` is the max raw LLR over all candidate breakpoints (``inf`` when a
    perfect split was found).  Returns the calibrated LLR in ``[0, _GLR_MAX_LLR]``:
    subtract the ``ln(#candidates)`` penalty, then clamp at the documented cap.
    A perfect split therefore scores ``sqrt(_GLR_MAX_LLR)`` — a documented cap,
    never a function of a program epsilon."""
    if best <= 0.0:
        return 0.0
    m = max(n - 2 * min_segment + 1, 2)
    adj = best - _GLR_SCAN_PENALTY_COEF * np.log(float(m))
    if not np.isfinite(adj):
        # perfect split: unbounded raw LLR -> the documented cap
        adj = _GLR_MAX_LLR
    return min(adj, _GLR_MAX_LLR)


# R5 P1-01: ``min_segment`` is an int (5.9 -> 5 is rejected).
_GLR_PARAM_SPECS = {
    "window": ParamSpec(dtype=int, min=4),
    "min_segment": ParamSpec(dtype=int, min=2),
}
# R6-24: window >= 2*min_segment + 2 is required for any breakpoint to exist —
# declared as a relational constraint so search never emits a guaranteed-NaN
# combination (the runtime also raises, but search should prune first).
_GLR_RELATIONAL_SPECS = [
    RelationalParamSpec(
        "window >= 2 * min_segment + 2",
        "window must be >= 2*min_segment + 2 for a breakpoint to exist "
        "(window={window}, min_segment={min_segment})",
    )
]


def _mean_shift_score(v: np.ndarray, min_segment: int) -> float:
    n = v.shape[0]
    if n < 2 * min_segment + 2:
        return np.nan
    ps = np.concatenate(([0.0], np.cumsum(v)))
    pss = np.concatenate(([0.0], np.cumsum(v * v)))
    full_ss = pss[n] - ps[n] * ps[n] / n
    if not np.isfinite(full_ss) or full_ss <= _EPS:
        return np.nan
    floor = _ss_noise_floor(full_ss, n)
    best = 0.0
    best_sign = 0.0
    for tau in range(min_segment, n - min_segment + 1):
        n1 = tau
        n2 = n - tau
        mu1 = ps[tau] / n1
        mu2 = (ps[n] - ps[tau]) / n2
        ss1 = pss[tau] - ps[tau] * ps[tau] / n1
        ss2 = (pss[n] - pss[tau]) - (ps[n] - ps[tau]) ** 2 / n2
        pooled = ss1 + ss2
        # P0-L-69: a PERFECT change point (constant A then constant B) has
        # pooled residual SS at floating-point zero — the strongest evidence in
        # the sample.  The old ``max(pooled, _EPS*full_ss)`` denominator made
        # the max score a data-independent function of 1e-12.  A perfect split
        # has an unbounded true LLR, so record ``inf`` and let
        # ``_null_calibrate_llr`` clamp it at the documented cap.
        if pooled <= floor:
            llr = np.inf
        else:
            llr = (n / 2.0) * np.log(full_ss / pooled)
        if llr > best:
            best = llr
            best_sign = 1.0 if mu2 > mu1 else (-1.0 if mu2 < mu1 else 0.0)
    best = _null_calibrate_llr(best, n, min_segment)
    if best <= 0.0:
        return 0.0
    return float(best_sign * np.sqrt(best))


def _variance_shift_score(v: np.ndarray, min_segment: int) -> float:
    n = v.shape[0]
    if n < 2 * min_segment + 2:
        return np.nan
    ps = np.concatenate(([0.0], np.cumsum(v)))
    pss = np.concatenate(([0.0], np.cumsum(v * v)))
    full_ss = pss[n] - ps[n] * ps[n] / n
    if not np.isfinite(full_ss) or full_ss <= _EPS:
        return np.nan
    floor = _ss_noise_floor(full_ss, n)
    best = 0.0
    best_sign = 0.0
    for tau in range(min_segment, n - min_segment + 1):
        n1 = tau
        n2 = n - tau
        ss1 = pss[tau] - ps[tau] * ps[tau] / n1
        ss2 = (pss[n] - pss[tau]) - (ps[n] - ps[tau]) ** 2 / n2
        # P0-L-69: a segment with zero variance (perfectly constant) is the
        # strongest variance evidence — but the old ``max(ss_i, _EPS*full_ss)``
        # denominator made the LLR a data-independent function of 1e-12.  A
        # one-side-constant split has an unbounded variance-ratio LLR -> record
        # ``inf`` (clamped at the documented cap); a both-sides-constant split
        # carries no variance DIFFERENCE evidence -> skip.
        if ss1 <= floor and ss2 <= floor:
            continue
        if ss1 <= floor or ss2 <= floor:
            llr = np.inf
        else:
            var1 = ss1 / n1
            var2 = ss2 / n2
            # R6-160: the null model for a VARIANCE shift must use a CONSISTENT
            # location model — the pooled WITHIN-SEGMENT variance, not the global
            # variance around the grand mean.  The old ``full_ss / n`` mixed a pure
            # mean shift (which inflates the grand-mean variance) into the H0 term,
            # so an equal-variance mean shift still fired the "variance shift"
            # detector.  Using the pooled within-segment variance keeps the LLR
            # measuring variance change only, with the location terms cancelling.
            pooled_var = (ss1 + ss2) / n
            llr = (n / 2.0) * np.log(pooled_var) - (n1 / 2.0) * np.log(var1) - (n2 / 2.0) * np.log(var2)
        if llr > best:
            best = llr
            if ss1 <= floor:
                best_sign = 1.0  # var2 > var1 = 0 -> post-variance higher
            elif ss2 <= floor:
                best_sign = -1.0  # var1 > var2 = 0 -> post-variance lower
            else:
                best_sign = 1.0 if np.log(var2 / var1) > 0.0 else -1.0
    best = _null_calibrate_llr(best, n, min_segment)
    if best <= 0.0:
        return 0.0
    return float(best_sign * np.sqrt(best))


def _average_ranks(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="stable")
    n = v.shape[0]
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and v[order[j]] == v[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg
        i = j
    return ranks


def _pettitt_score(v: np.ndarray, min_segment: int) -> float:
    n = v.shape[0]
    if n < 2 * min_segment + 2:
        return np.nan
    r = _average_ranks(v)
    psr = np.concatenate(([0.0], np.cumsum(r)))
    best_k = 0.0
    best_t = min_segment
    best_sign = 0.0
    for t in range(min_segment, n - min_segment + 1):
        u = 2.0 * psr[t] - t * (n + 1.0)
        if abs(u) > best_k:
            best_k = abs(u)
            best_t = t
            # R5 P1-06: unify the change-point family sign — "positive = the post
            # regime is higher".  A post-higher regime gives the pre-half LOW
            # ranks, hence a negative Pettitt U*, so the sign is flipped.
            best_sign = -1.0 if u > 0.0 else (1.0 if u < 0.0 else 0.0)
    if best_k <= _EPS:
        return 0.0
    normalized = best_k / (n * n / 4.0)
    return float(best_sign * min(normalized, 1.0))


def _glr_series(x2d: np.ndarray, window: int, min_segment: int, which: str) -> np.ndarray:
    rows, cols = x2d.shape
    w = max(4, int(window))
    ms = max(2, int(min_segment))
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            lo = max(0, r - w + 1)
            v = trailing_contiguous_finite(col[lo : r + 1])
            if v.size < 2 * ms + 2:
                continue
            if which == "mean":
                val = _mean_shift_score(v, ms)
            elif which == "variance":
                val = _variance_shift_score(v, ms)
            else:
                val = _pettitt_score(v, ms)
            if np.isfinite(val):
                out[r, c] = float(val)
    return out


def _glr_feasibility(window: int, min_segment: int, canonical: str) -> tuple[int, int]:
    w = int(window)
    ms = max(2, int(min_segment))
    # P1-29: no breakpoint exists unless the window can hold two segments — a
    # window below ``2*min_segment + 2`` would silently return all-NaN for the
    # whole parameter combination.
    if w < 2 * ms + 2:
        raise ValueError(f"{canonical} requires window >= 2*min_segment + 2 (got window={w}, min_segment={ms})")
    return w, ms


def _ts_glr_mean_shift_score(x: pd.DataFrame, window: int = 60, min_segment: int = 10) -> pd.DataFrame:
    w, ms = _glr_feasibility(window, min_segment, "ts_glr_mean_shift_score")
    out = _glr_series(x.to_numpy(dtype=float), w, ms, "mean")
    return frame_like(x, out)


def _ts_glr_variance_shift_score(x: pd.DataFrame, window: int = 60, min_segment: int = 10) -> pd.DataFrame:
    w, ms = _glr_feasibility(window, min_segment, "ts_glr_variance_shift_score")
    out = _glr_series(x.to_numpy(dtype=float), w, ms, "variance")
    return frame_like(x, out)


def _ts_pettitt_change_score(x: pd.DataFrame, window: int = 120, min_segment: int = 10) -> pd.DataFrame:
    w, ms = _glr_feasibility(window, min_segment, "ts_pettitt_change_score")
    out = _glr_series(x.to_numpy(dtype=float), w, ms, "pettitt")
    return frame_like(x, out)


_SPECS: dict[str, dict[str, Any]] = {
    "ts_glr_mean_shift_score": {
        "fn": _ts_glr_mean_shift_score,
        "params": ["x", "window", "min_segment"],
        "category": "time_series_change",
        "domain": "change_point",
        # R6-162: sqrt(max_LLR) is a dimensionless score, NOT a level.
        "unit": "dimensionless_score",
        "cost": 3,
        "tags_extra": ["condition"],
        "param_specs": _GLR_PARAM_SPECS,
    },
    "ts_glr_variance_shift_score": {
        "fn": _ts_glr_variance_shift_score,
        "params": ["x", "window", "min_segment"],
        "category": "time_series_change",
        "domain": "change_point",
        "unit": "dimensionless_score",
        "cost": 3,
        "tags_extra": ["condition"],
        "param_specs": _GLR_PARAM_SPECS,
    },
    "ts_pettitt_change_score": {
        "fn": _ts_pettitt_change_score,
        "params": ["x", "window", "min_segment"],
        "category": "time_series_change",
        "domain": "change_point",
        "unit": "ratio",
        "cost": 4,
        "tags_extra": ["state"],
        "param_specs": _GLR_PARAM_SPECS,
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
            source="glr_change",
            tags_extra=spec["tags_extra"],
            output_unit=spec["unit"],
            param_specs=spec.get("param_specs"),
            relational_specs=_GLR_RELATIONAL_SPECS,
        )
    union_extended(*_SPECS.keys())


_register()
