# -*- coding: utf-8 -*-
"""Change-point detection primitives (2026-08-08 Gemini V2 round).

Three deterministic, trailing-window change-point scores.  Each scans a window
for the best breakpoint ``τ`` (``min_segment ≤ τ ≤ n - min_segment``) using a
Gaussian profile-likelihood or rank statistic, and returns a *signed* score
(never a p-value — financial series are serially correlated and the
independence assumptions behind change-point p-values do not hold).
The GLR scores are ``sign · sqrt(max_LLR)`` — NOT bounded (P1-29: a change of
arbitrary size yields an arbitrarily large score); only the rank-based Pettitt
score is normalised into [-1, 1].  ``window >= 2·min_segment + 2`` is required
for any breakpoint to exist — smaller windows are rejected up front instead of
silently returning all-NaN.

* ``ts_glr_mean_shift_score``      — generalised likelihood ratio for a mean
  shift: ``LLR_τ = (n/2)·ln(σ̂² / σ̂_w²)``, output ``sign(μ_post - μ_pre) ·
  sqrt(max_LLR)``.  Prefix sums make the whole scan ``O(n)`` per window.
* ``ts_glr_variance_shift_score``  — same scan against H1 "two variances":
  ``LLR_τ = (n/2)ln(σ̂²) - (n₁/2)ln(σ̂₁²) - (n₂/2)ln(σ̂₂²)``, output
  ``sign(ln(var_post/var_pre)) · sqrt(max_LLR)``.
* ``ts_pettitt_change_score``      — rank-based Pettitt statistic
  ``U_t = 2·Σ_{i≤t} r_i - t(n+1)`` (average-rank ties), output
  ``sign(U_τ*) · |U_τ*| / (n²/4)`` (bounded in [-1, 1]; 0 ≈ no change).

All are strict-PIT (only the trailing contiguous finite window), deterministic
and NaN fail-closed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import ParamSpec
from cleaned_operators.gemini_v2_common import (
    frame_like,
    register_dual,
    trailing_contiguous_finite,
    union_extended,
)

_EPS = 1e-12

# R5 P1-01: ``min_segment`` is an int (5.9 -> 5 is rejected).
_GLR_PARAM_SPECS = {
    "window": ParamSpec(dtype=int, min=4),
    "min_segment": ParamSpec(dtype=int, min=2),
}


def _mean_shift_score(v: np.ndarray, min_segment: int) -> float:
    n = v.shape[0]
    if n < 2 * min_segment + 2:
        return np.nan
    ps = np.concatenate(([0.0], np.cumsum(v)))
    pss = np.concatenate(([0.0], np.cumsum(v * v)))
    full_ss = pss[n] - ps[n] * ps[n] / n
    if full_ss <= _EPS:
        return np.nan
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
        if pooled <= _EPS:
            continue
        llr = (n / 2.0) * np.log(full_ss / pooled)
        if llr > best:
            best = llr
            best_sign = 1.0 if mu2 > mu1 else (-1.0 if mu2 < mu1 else 0.0)
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
    if full_ss <= _EPS:
        return np.nan
    best = 0.0
    best_sign = 0.0
    for tau in range(min_segment, n - min_segment + 1):
        n1 = tau
        n2 = n - tau
        ss1 = pss[tau] - ps[tau] * ps[tau] / n1
        ss2 = (pss[n] - pss[tau]) - (ps[n] - ps[tau]) ** 2 / n2
        if ss1 <= _EPS or ss2 <= _EPS:
            continue
        var1 = ss1 / n1
        var2 = ss2 / n2
        llr = (n / 2.0) * np.log(full_ss / n) - (n1 / 2.0) * np.log(var1) - (n2 / 2.0) * np.log(var2)
        if llr > best:
            best = llr
            best_sign = 1.0 if np.log(var2 / var1) > 0.0 else -1.0
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
        "unit": "level",
        "cost": 3,
        "tags_extra": ["condition"],
        "param_specs": _GLR_PARAM_SPECS,
    },
    "ts_glr_variance_shift_score": {
        "fn": _ts_glr_variance_shift_score,
        "params": ["x", "window", "min_segment"],
        "category": "time_series_change",
        "domain": "change_point",
        "unit": "level",
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
        )
    union_extended(*_SPECS.keys())


_register()
