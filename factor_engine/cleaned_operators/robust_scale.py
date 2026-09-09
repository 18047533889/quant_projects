# -*- coding: utf-8 -*-
"""Robust scale / location primitives (2026-08-08 Gemini V2 round).

* ``ts_qn_scale`` — the Rousseeuw & Croux (1993) Qn scale estimator:
  ``Qn = D∞ · d_n · {|x_i - x_j| : i < j}_{(k)}`` with ``k = C(h, 2)``,
  ``h = ⌊n/2⌋ + 1``.  Uses the corrected consistency constant
  ``D∞ = 1/(√2·Φ⁻¹(5/8)) ≈ 2.21914446598508`` (the 2.2219 in early papers was
  a typo) and the robustbase finite-sample corrections: exact ``d_n`` for
  ``n ≤ 12`` and the published odd/even polynomials beyond.  The pairwise
  differences are collected in a compact upper-triangle array (never a dense
  ``W×W`` matrix) and the k-th order statistic is selected in linear time.
* ``ts_hodges_lehmann_location`` — the Hodges-Lehmann location estimator
  ``median{(x_i + x_j)/2 : i ≤ j}`` (robust to heavy tails, more efficient
  than the median at normality).  P1 / extended: it is not uniformly better
  than the median on asymmetric distributions, so it stays out of the default
  grammar.

Both are strict-PIT, deterministic, and NaN fail-closed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import ParamSpec
from factor_engine.cleaned_operators.gemini_v2_common import (
    frame_like,
    register_dual,
    trailing_contiguous_finite,
    union_extended,
)

_EPS = 1e-12

# R6-165: Qn and Hodges-Lehmann are O(n²) per rolling row (Qn builds all
# pairwise absolute differences; HL sorts all pair means).  A window cap keeps
# automatic search from generating an exploding-cost parameter; search should
# default to the reviewed grid {20, 60, 120} which is well below the cap.
_ROBUST_SCALE_SPECS = {
    "window": ParamSpec(dtype=int, min=2, max=512),
    "min_periods": ParamSpec(dtype=int, min=2),
}

# Corrected consistency constant (robustbase / Akinshin 2022).
_QN_D_INF = 2.21914446598508
# robustbase 0.95-0 Table 2 finite-sample corrections for n <= 12.
_QN_C_TABLE = {
    2: 0.399356,
    3: 0.99365,
    4: 0.51321,
    5: 0.84401,
    6: 0.61220,
    7: 0.85877,
    8: 0.66993,
    9: 0.87344,
    10: 0.72014,
    11: 0.88906,
    12: 0.75743,
}


def _qn_dn(n: int) -> float:
    if n in _QN_C_TABLE:
        return _QN_C_TABLE[n]
    if n % 2 == 1:
        return 1.0 / (1.0 + 1.60188 / n - 2.1284 / (n * n) - 5.172 / (n**3))
    return 1.0 / (1.0 + 3.67561 / n + 1.9654 / (n * n) + 6.987 / (n**3) - 77.0 / (n**4))


def _pairwise_abs_diffs(v: np.ndarray) -> np.ndarray:
    n = v.shape[0]
    # compact upper triangle — O(n^2) memory in one flat array, no W x W matrix
    out = np.empty(n * (n - 1) // 2, dtype=float)
    idx = 0
    for i in range(n - 1):
        vi = v[i]
        seg = v[i + 1 :]
        out[idx : idx + seg.shape[0]] = np.abs(seg - vi)
        idx += seg.shape[0]
    return out


def _qn_scale(v: np.ndarray) -> float:
    n = v.shape[0]
    if n < 2:
        return np.nan
    h = n // 2 + 1
    k = h * (h - 1) // 2
    diffs = _pairwise_abs_diffs(v)
    if k < 1 or k > diffs.shape[0]:
        return np.nan
    order_stat = float(np.partition(diffs, k - 1)[k - 1])
    return float(_QN_D_INF * _qn_dn(n) * order_stat)


def _finite_midpoint(a, b):
    """Finite binary64 midpoint, avoiding both overflow and tiny-value halving."""
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    direct = ((np.signbit(a) != np.signbit(b)) |
              ((np.abs(a) <= np.finfo(float).max / 2) &
               (np.abs(b) <= np.finfo(float).max / 2)))
    result = np.empty(a.shape, dtype=float)
    result[direct] = (a[direct] + b[direct]) / 2
    result[~direct] = a[~direct] / 2 + b[~direct] / 2
    return result


def _hodges_lehmann(v: np.ndarray) -> float:
    n = v.shape[0]
    if n < 2:
        return np.nan
    # all pairwise midpoints (i <= j) — compact array
    mids = np.empty(n * (n + 1) // 2, dtype=float)
    idx = 0
    for i in range(n):
        seg = v[i:]
        mids[idx : idx + seg.shape[0]] = _finite_midpoint(seg, v[i])
        idx += seg.shape[0]
    middle = mids.size // 2
    if mids.size % 2:
        return float(np.partition(mids, middle)[middle])
    central = np.partition(mids, (middle - 1, middle))[middle - 1:middle + 1]
    return float(_finite_midpoint(central[0], central[1]))


def _ts_qn_scale(x: pd.DataFrame, window: int = 60, min_periods: int = 8) -> pd.DataFrame:
    if int(window) < 2:
        raise ValueError("ts_qn_scale requires window >= 2")
    rows, cols = x.shape
    w = int(window)
    mp = max(2, int(min_periods))
    arr = x.to_numpy(dtype=float)
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        col = arr[:, c]
        for r in range(rows):
            lo = max(0, r - w + 1)
            v = trailing_contiguous_finite(col[lo : r + 1])
            if v.size < mp:
                continue
            val = _qn_scale(v)
            if np.isfinite(val):
                out[r, c] = float(val)
    return frame_like(x, out)


def _ts_hodges_lehmann_location(x: pd.DataFrame, window: int = 60, min_periods: int = 8) -> pd.DataFrame:
    if int(window) < 2:
        raise ValueError("ts_hodges_lehmann_location requires window >= 2")
    rows, cols = x.shape
    w = int(window)
    mp = max(2, int(min_periods))
    arr = x.to_numpy(dtype=float)
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        col = arr[:, c]
        for r in range(rows):
            lo = max(0, r - w + 1)
            v = trailing_contiguous_finite(col[lo : r + 1])
            if v.size < mp:
                continue
            val = _hodges_lehmann(v)
            if np.isfinite(val):
                out[r, c] = float(val)
    return frame_like(x, out)


_SPECS: dict[str, dict[str, Any]] = {
    "ts_qn_scale": {
        "fn": _ts_qn_scale,
        "params": ["x", "window", "min_periods"],
        "category": "robust_statistics",
        "domain": "statistics",
        # Round-11 §37-D: same_as:x — 'target' is not a declared param; a
        # scale/location statistic of the input carries the input's unit.
        "unit": "same_as:x",
        "cost": 5,
        "tags_extra": [],
        "output_unit": "same_as:x",
        "param_specs": _ROBUST_SCALE_SPECS,
    },
    "ts_hodges_lehmann_location": {
        "fn": _ts_hodges_lehmann_location,
        "params": ["x", "window", "min_periods"],
        "category": "robust_statistics",
        "domain": "statistics",
        # Round-11 §37-D: same_as:x (see ts_qn_scale).
        "unit": "same_as:x",
        "cost": 5,
        "tags_extra": [],
        "output_unit": "same_as:x",
        "param_specs": _ROBUST_SCALE_SPECS,
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
            source="robust_scale",
            tags_extra=spec["tags_extra"],
            output_unit=spec.get("output_unit"),
            param_specs=spec.get("param_specs"),
        )
    union_extended(*_SPECS.keys())


_register()
