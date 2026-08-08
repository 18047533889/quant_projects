# -*- coding: utf-8 -*-
"""Extreme-value / event-count primitives (2026-08-08 Gemini V2 round).

* ``ts_pickands_tail_index`` — the Pickands (1975) tail-index estimator
  ``ξ̂ = (1/ln 2)·ln( (X_(n-k) - X_(n-2k)) / (X_(n-2k) - X_(n-4k)) )``.  This is
  the third estimator of the tail shape beside the existing Hill and GPD-PWM
  operators — estimator diversity, not a new tail concept, hence P1/extended.
  ``k`` is a fixed scan offset (not a search parameter surface).  For the
  lower tail the estimator is applied to the negated series.
* ``ts_evt_threshold_stability`` — the standard deviation of the Hill
  estimate ``ξ̂_Hill(k)`` as the threshold moves over ``k ∈ [k_min, k_max]``.
  A low value means the tail index is stable across thresholds (a reliable
  tail); a high value flags a fragile / regime-mixed tail.  Outputs the
  dispersion (never a p-value).
* ``event_allan_factor`` — the Allan factor of a daily 0/1 event series:
  for each scale ``m ∈ {1, 2, 4, …, max_scale}`` count events in non-
  overlapping blocks of length ``m`` and take ``AF(m) = Var(count) / Mean``,
  then return the mean of ``log2(AF(m))`` over the scales (≈ 0 for a Poisson
  process, > 0 for clustered events, < 0 for anti-clustering).

All operators are strict-PIT, deterministic and NaN fail-closed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.gemini_v2_common import (
    frame_like,
    register_dual,
    trailing_contiguous_finite,
    union_extended,
)

_EPS = 1e-12
_LN2 = np.log(2.0)


# ---------------------------------------------------------------------------
# Pickands tail index
# ---------------------------------------------------------------------------
def _pickands_tail(v: np.ndarray, k: int, upper: bool) -> float:
    n = v.shape[0]
    kk = max(1, int(k))
    if n < 4 * kk + 1:
        return np.nan
    z = v if upper else -v
    s = np.sort(z)
    # order stats: X_(n), X_(n-1), ...
    a = s[n - 4 * kk]
    b = s[n - 2 * kk]
    d = s[n - kk]
    denom = b - a
    numer = d - b
    if denom <= _EPS or numer <= _EPS:
        return np.nan
    return float(np.log(numer / denom) / _LN2)


# ---------------------------------------------------------------------------
# EVT threshold stability (std of Hill over k)
# ---------------------------------------------------------------------------
def _hill_xi(s: np.ndarray, k: int) -> float:
    n = s.shape[0]
    if k < 1 or n - 1 - k < 0:
        return np.nan
    threshold = s[n - 1 - k]
    if threshold <= _EPS:
        return np.nan
    top = s[n - 1 - k + 1 : n]
    if top.size != k or np.any(top <= _EPS):
        return np.nan
    return float(np.mean(np.log(top / threshold)))


def _threshold_stability(v: np.ndarray, k_min: int, k_max: int, upper: bool) -> float:
    n = v.shape[0]
    z = v if upper else -v
    if np.any(z <= 0.0):
        return np.nan
    s = np.sort(z)
    kmax = min(int(k_max), n // 2)
    kmin = max(2, int(k_min))
    if kmax <= kmin:
        return np.nan
    hills = np.asarray([_hill_xi(s, k) for k in range(kmin, kmax + 1)], dtype=float)
    ok = np.isfinite(hills)
    if int(ok.sum()) < 3:
        return np.nan
    return float(np.std(hills[ok]))


def _ts_pickands_tail_index(x: pd.DataFrame, window: int = 120, k: int = 10, side: str = "upper") -> pd.DataFrame:
    if int(window) < 6:
        raise ValueError("ts_pickands_tail_index requires window >= 6")
    side_s = str(side).lower()
    if side_s not in {"upper", "lower"}:
        raise ValueError("ts_pickands_tail_index requires side in {'upper','lower'}")
    rows, cols = x.shape
    w = int(window)
    arr = x.to_numpy(dtype=float)
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        col = arr[:, c]
        for r in range(rows):
            lo = max(0, r - w + 1)
            v = trailing_contiguous_finite(col[lo : r + 1])
            if v.size < 4 * int(k) + 1:
                continue
            val = _pickands_tail(v, int(k), side_s == "upper")
            if np.isfinite(val):
                out[r, c] = val
    return frame_like(x, out)


def _ts_evt_threshold_stability(x: pd.DataFrame, window: int = 120, k_min: int = 5, k_max: int = 20, side: str = "upper") -> pd.DataFrame:
    if int(window) < 10:
        raise ValueError("ts_evt_threshold_stability requires window >= 10")
    side_s = str(side).lower()
    if side_s not in {"upper", "lower"}:
        raise ValueError("ts_evt_threshold_stability requires side in {'upper','lower'}")
    rows, cols = x.shape
    w = int(window)
    arr = x.to_numpy(dtype=float)
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        col = arr[:, c]
        for r in range(rows):
            lo = max(0, r - w + 1)
            v = trailing_contiguous_finite(col[lo : r + 1])
            if v.size < int(k_max) * 2 + 2:
                continue
            val = _threshold_stability(v, int(k_min), int(k_max), side_s == "upper")
            if np.isfinite(val):
                out[r, c] = val
    return frame_like(x, out)


# ---------------------------------------------------------------------------
# Allan factor
# ---------------------------------------------------------------------------
def _allan_factor(v: np.ndarray, max_scale: int) -> float:
    n = v.shape[0]
    if n < 8:
        return np.nan
    ms = max(2, int(max_scale))
    scales = [1]
    while scales[-1] * 2 <= ms:
        scales.append(scales[-1] * 2)
    logs: list[float] = []
    for m in scales:
        if m > n:
            continue
        n_blocks = n // m
        if n_blocks < 2:
            continue
        blocks = v[n - n_blocks * m :].reshape(n_blocks, m)
        counts = blocks.sum(axis=1)
        mean_c = float(np.mean(counts))
        if mean_c <= _EPS:
            continue
        var_c = float(np.var(counts))
        af = var_c / mean_c
        if af <= _EPS:
            af = _EPS
        logs.append(float(np.log2(af)))
    if len(logs) < 2:
        return np.nan
    return float(np.mean(logs))


def _ts_event_allan_factor(x: pd.DataFrame, window: int = 120, max_scale: int = 16) -> pd.DataFrame:
    if int(window) < 8:
        raise ValueError("event_allan_factor requires window >= 8")
    rows, cols = x.shape
    w = int(window)
    arr = x.to_numpy(dtype=float)
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        col = arr[:, c]
        for r in range(rows):
            lo = max(0, r - w + 1)
            v = trailing_contiguous_finite(col[lo : r + 1])
            if v.size < 8:
                continue
            val = _allan_factor(v, int(max_scale))
            if np.isfinite(val):
                out[r, c] = val
    return frame_like(x, out)


_SPECS: dict[str, dict[str, Any]] = {
    "ts_pickands_tail_index": {
        "fn": _ts_pickands_tail_index,
        "params": ["x", "window", "k", "side"],
        "category": "extreme_value",
        "domain": "tail",
        "unit": "level",
        "cost": 4,
        "tags_extra": ["state"],
        "output_unit": "level",
    },
    "ts_evt_threshold_stability": {
        "fn": _ts_evt_threshold_stability,
        "params": ["x", "window", "k_min", "k_max", "side"],
        "category": "extreme_value",
        "domain": "tail",
        "unit": "level",
        "cost": 4,
        "tags_extra": ["state"],
        "output_unit": "level",
    },
    "event_allan_factor": {
        "fn": _ts_event_allan_factor,
        "params": ["event", "window", "max_scale"],
        "category": "event_counting",
        "domain": "event",
        "unit": "level",
        "cost": 3,
        "tags_extra": ["state"],
        "output_unit": "level",
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
            source="evt_allan",
            tags_extra=spec["tags_extra"],
            output_unit=spec.get("output_unit"),
        )
    union_extended(*_SPECS.keys())


_register()
