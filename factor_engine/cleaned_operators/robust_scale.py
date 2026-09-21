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


def _hl_run_bounds(col: np.ndarray, w: int):
    """Per-row trailing contiguous finite run [s, lf] (must end at the current
    row) plus run length k, mirroring trailing_contiguous_finite semantics."""
    n = col.size
    finite = np.isfinite(col)
    idx = np.where(finite, np.arange(n), -1)
    last_fin = np.maximum.accumulate(idx)
    nan_idx = np.where(~finite, np.arange(n), -1)
    last_nan = np.maximum.accumulate(nan_idx)
    t = np.arange(n)
    lo = np.maximum(t - w + 1, 0)
    lf = np.maximum(last_fin, -1)
    s = np.maximum(last_nan[np.maximum(lf, 0)] + 1, lo)
    s = np.where(lf >= 0, s, lo)
    # trailing_contiguous_finite returns EMPTY when the last row is NaN
    k = np.where((lf >= s) & (lf == t), lf - s + 1, 0)
    return s, k, lf


def _vec_hodges_lehmann(col: np.ndarray, w: int, mp: int) -> np.ndarray:
    """Column-wise Hodges-Lehmann location over trailing windows (vectorized).

    The authority takes the median of the compact upper-triangular (i <= j)
    pairwise midpoints of the trailing contiguous finite run.  A power-of-two
    column scaling (loss-free) keeps the arithmetic identical on 1e+-300 data.
    """
    n = col.size
    out = np.full(n, np.nan)
    if w > 400:
        # defensive: fall back to the scalar kernel for absurd windows
        for r in range(n):
            lo = max(0, r - w + 1)
            v = trailing_contiguous_finite(col[lo : r + 1])
            if v.size >= mp:
                out[r] = _hodges_lehmann(v)
        return out

    fin = col[np.isfinite(col)]
    scale = 1.0
    if fin.size:
        m = float(np.max(np.abs(fin)))
        if np.isfinite(m) and m > 0.0:
            scale = 2.0 ** (-int(np.frexp(m)[1]))
    vals = col * scale

    s, k, lf = _hl_run_bounds(vals, w)
    t = np.arange(n)
    ok = k >= max(mp, 2)
    if not ok.any():
        return out

    # padded window matrix: S[t, j] = vals[t-w+1+j] (NaN outside [s_t, t])
    src_idx = t[:, None] - (w - 1) + np.arange(w)[None, :]
    valid = (src_idx >= 0) & (src_idx >= s[:, None]) & np.isfinite(vals)[np.clip(src_idx, 0, n - 1)]
    S = vals[np.clip(src_idx, 0, n - 1)]
    S = np.where(valid, S, np.nan)

    ii, jj = np.triu_indices(w)          # pairs i <= j (order is irrelevant to the median)
    a = S[:, ii]
    b = S[:, jj]
    with np.errstate(invalid="ignore", over="ignore"):
        same_sign = np.signbit(a) != np.signbit(b)
        half_safe = (np.abs(a) <= np.finfo(float).max / 2) & (np.abs(b) <= np.finfo(float).max / 2)
        mids = np.where(same_sign | half_safe, (a + b) / 2, a / 2 + b / 2)
    # run-length aware valid counts: pairs within the run = k(k+1)/2
    c = (k * (k + 1) // 2).astype(np.int64)

    full = ok & (k == w)
    part = ok & (k < w)
    if full.any():
        Mc = mids[full]
        kc = w
        cc = kc * (kc + 1) // 2
        mid_r = cc // 2
        if cc % 2:
            out[np.where(full)[0]] = np.partition(Mc, mid_r, axis=1)[:, mid_r]
        else:
            P = np.partition(Mc, (mid_r - 1, mid_r), axis=1)
            lo_v = P[:, mid_r - 1]
            hi_v = P[:, mid_r]
            same = np.signbit(lo_v) != np.signbit(hi_v)
            half = (np.abs(lo_v) <= np.finfo(float).max / 2) & (np.abs(hi_v) <= np.finfo(float).max / 2)
            out[np.where(full)[0]] = np.where(same | half, (lo_v + hi_v) / 2, lo_v / 2 + hi_v / 2)
    if part.any():
        Mv = mids[part]
        kv = k[part]
        rows_p = np.where(part)[0]
        for r_i in range(Mv.shape[0]):
            cc = int(kv[r_i]) * (int(kv[r_i]) + 1) // 2
            if cc < 1:
                continue
            flat = Mv[r_i]
            flat = flat[np.isfinite(flat)]
            if flat.size < cc:       # NaN midpoints inside the run cannot happen
                continue
            mid_r = cc // 2
            if cc % 2:
                out[rows_p[r_i]] = np.partition(flat, mid_r)[mid_r]
            else:
                P = np.partition(flat, (mid_r - 1, mid_r))
                lo_v, hi_v = P[mid_r - 1], P[mid_r]
                same = np.signbit(lo_v) != np.signbit(hi_v)
                half = (np.abs(lo_v) <= np.finfo(float).max / 2) & (np.abs(hi_v) <= np.finfo(float).max / 2)
                out[rows_p[r_i]] = (lo_v + hi_v) / 2 if (same or half) else lo_v / 2 + hi_v / 2
    # power-of-two scaling is a pure exponent shift: the location estimate is
    # scale-equivariant, so unscale value-dimension output.
    if scale != 1.0:
        inv = 1.0 / scale
        e = int(np.frexp(inv)[1])
        out = out * (2.0 ** (e // 2)) * (inv / 2.0 ** (e // 2))
    return out


def _ts_hodges_lehmann_location(x: pd.DataFrame, window: int = 60, min_periods: int = 8) -> pd.DataFrame:
    if int(window) < 2:
        raise ValueError("ts_hodges_lehmann_location requires window >= 2")
    rows, cols = x.shape
    w = int(window)
    mp = max(2, int(min_periods))
    arr = x.to_numpy(dtype=float)
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        out[:, c] = _vec_hodges_lehmann(arr[:, c], w, mp)
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
