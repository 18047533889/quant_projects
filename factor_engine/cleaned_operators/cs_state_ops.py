# -*- coding: utf-8 -*-
"""Cross-sectional / global-state primitives (2026-08-08 Gemini V2 round).

* ``cs_hartigan_dip`` — the Hartigan dip statistic of the day's cross-section.
  A faithful port of the reference algorithm (Hartigan 1985, Algorithm AS 217;
  as implemented in the ``diptest`` R/Python packages via GCM/LCM change-point
  cycling).  Output is ``sqrt(N)·dip`` so it is comparable across days with
  different listing counts.  This is a GLOBAL state (tags ``global_state``):
  every instrument on the same day receives the same value and it must only be
  used as a regime/condition input, never as a standalone ranking factor.
  ``N >= min_cross`` is required; no Monte-Carlo p-value is produced.
* ``group_wasserstein_barycenter_distance`` — the 1-D Wasserstein (earth
  mover) distance between an instrument's trailing window distribution and
  its group's pooled distribution (the empirical group "barycenter"): how far
  a name's recent return distribution sits from its peers' average
  distribution.  Computed on a fixed quantile grid (deterministic, exact for
  equal grid).  P1 / extended.
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


# ---------------------------------------------------------------------------
# Hartigan dip — faithful port of the reference GCM/LCM algorithm
# ---------------------------------------------------------------------------
def _dip_statistic(sorted_x: np.ndarray) -> float:
    n = int(sorted_x.shape[0])
    if n < 2 or sorted_x[0] == sorted_x[-1]:
        return 0.0
    X = np.zeros(n + 1, dtype=float)
    X[1:] = np.asarray(sorted_x, dtype=float)
    mn = np.zeros(n + 1, dtype=np.int64)
    mj = np.zeros(n + 1, dtype=np.int64)
    gcm = np.zeros(n + 2, dtype=np.int64)
    lcm = np.zeros(n + 2, dtype=np.int64)

    # GCM change-point indices (minorant): offset +1, start 1
    mn[1] = 1
    for i in range(2, n + 1):
        mn[i] = i - 1
        while True:
            ind = mn[i]
            ind_iter = mn[ind]
            if ind == 1:
                break
            if (X[i] - X[ind]) * (ind - ind_iter) < (X[ind] - X[ind_iter]) * (i - ind):
                break
            mn[i] = ind_iter
    # LCM change-point indices (majorant): offset -1, start n
    mj[n] = n
    for i in range(n - 1, 0, -1):
        mj[i] = i + 1
        while True:
            ind = mj[i]
            ind_iter = mj[ind]
            if ind == n:
                break
            if (X[i] - X[ind]) * (ind - ind_iter) < (X[ind] - X[ind_iter]) * (i - ind):
                break
            mj[i] = ind_iter

    low = 1
    high = n
    dip = 0.0
    while True:
        # collect GCM change points from high down to low
        gcm[1] = high
        i = 1
        while gcm[i] > low:
            gcm[i + 1] = mn[gcm[i]]
            i += 1
        l_gcm = i
        gcm_x = l_gcm
        gcm_y = l_gcm - 1
        # collect LCM change points from low up to high
        lcm[1] = low
        i = 1
        while lcm[i] < high:
            lcm[i + 1] = mj[lcm[i]]
            i += 1
        l_lcm = i
        lcm_x = l_lcm
        lcm_y = 2

        # max distance between GCM and LCM
        if l_gcm != 2 or l_lcm != 2:
            d = 0.0
            gy = gcm_y
            ly = lcm_y
            while True:
                gcm_val = gcm[gy]
                lcm_val = lcm[ly]
                is_maj = 1 if gcm_val > lcm_val else 0
                ii = gcm_val if is_maj else lcm_val
                jj = lcm_val if is_maj else gcm_val
                i1 = gcm[gy + 1] if is_maj else lcm[ly - 1]
                sign = 2 * is_maj - 1
                denom = X[ii] - X[i1]
                if denom != 0.0:
                    dx = sign * ((jj - i1 + sign) - (X[jj] - X[i1]) * (ii - i1) / denom)
                else:
                    dx = -1.0
                gy -= 1 - is_maj
                ly += is_maj
                if dx >= d:
                    d = dx
                    gcm_x = gy + 1
                    lcm_x = ly - is_maj
                if gy < 1:
                    gy = 1
                if ly > l_lcm:
                    ly = l_lcm
                if gcm[gy] == lcm[ly]:
                    break
        else:
            d = 0.0

        if d < dip:
            break

        # dip of the convex minorant
        dip_l = 0.0
        for j in range(gcm_x, l_gcm):
            tmp = 1.0
            j_start = gcm[j + 1]
            j_end = gcm[j]
            if j_end - j_start > 1 and X[j_end] != X[j_start]:
                c_ = (j_end - j_start) / (X[j_end] - X[j_start])
                x_js = X[j_start]
                for jj in range(j_start, j_end + 1):
                    dd = (jj - j_start + 1) - (X[jj] - x_js) * c_
                    if dd > tmp:
                        tmp = dd
            if tmp > dip_l:
                dip_l = tmp
        # dip of the concave majorant
        dip_u = 0.0
        for j in range(lcm_x, l_lcm):
            tmp = 1.0
            j_start = lcm[j]
            j_end = lcm[j + 1]
            if j_end - j_start > 1 and X[j_end] != X[j_start]:
                c_ = (j_end - j_start) / (X[j_end] - X[j_start])
                x_js = X[j_start]
                for jj in range(j_start, j_end + 1):
                    dd = -((jj - j_start - 1) - (X[jj] - x_js) * c_)
                    if dd > tmp:
                        tmp = dd
            if tmp > dip_u:
                dip_u = tmp

        tmp_dip = dip_l if dip_l >= dip_u else dip_u
        if dip < tmp_dip:
            dip = tmp_dip
        flag = (low == gcm[gcm_x] and high == lcm[lcm_x])
        low = gcm[gcm_x]
        high = lcm[lcm_x]
        if flag:
            break
    return float(dip / (2.0 * n))


def _ts_cs_hartigan_dip(x: pd.DataFrame, min_cross: int = 100) -> pd.DataFrame:
    mc = max(2, int(min_cross))
    arr = x.to_numpy(dtype=float)
    rows, cols = arr.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for r in range(rows):
        row = arr[r]
        finite = row[np.isfinite(row)]
        if finite.size < mc:
            continue
        dip = _dip_statistic(np.sort(finite))
        out[r, :] = float(np.sqrt(finite.size) * dip)
    return frame_like(x, out)


# ---------------------------------------------------------------------------
# group Wasserstein barycenter distance
# ---------------------------------------------------------------------------
_GRID = 200


def _quantiles(v: np.ndarray) -> np.ndarray:
    return np.quantile(v, np.linspace(0.5 / _GRID, 1.0 - 0.5 / _GRID, _GRID))


def _wasserstein_pool(me_q: np.ndarray, pool_q: np.ndarray) -> float:
    return float(np.mean(np.abs(me_q - pool_q)))


def _group_wasserstein_barycenter_distance(
    x: pd.DataFrame,
    group: pd.DataFrame,
    window: int = 60,
    min_group_size: int = 3,
) -> pd.DataFrame:
    if int(window) < 4:
        raise ValueError("group_wasserstein_barycenter_distance requires window >= 4")
    w = int(window)
    mgs = max(2, int(min_group_size))
    arr = x.to_numpy(dtype=float)
    garr = group.to_numpy()
    rows, cols = arr.shape
    out = np.full((rows, cols), np.nan, dtype=float)

    for r in range(rows):
        lo = max(0, r - w + 1)
        # group members at row r
        groups: dict[Any, list[int]] = {}
        for c in range(cols):
            lab = garr[r, c]
            if lab is None or (isinstance(lab, float) and np.isnan(lab)):
                continue
            groups.setdefault(lab, []).append(c)
        # per-group pooled quantiles (computed once, reused for every member)
        for lab, members in groups.items():
            if len(members) < mgs:
                continue
            pool_parts: list[np.ndarray] = []
            for c in members:
                v = trailing_contiguous_finite(arr[lo : r + 1, c])
                if v.size >= 4:
                    pool_parts.append(v)
            if len(pool_parts) < mgs:
                continue
            pool = np.concatenate(pool_parts)
            if pool.size < 12:
                continue
            pool_q = _quantiles(pool)
            for c in members:
                v = trailing_contiguous_finite(arr[lo : r + 1, c])
                if v.size < 4:
                    continue
                me_q = _quantiles(v)
                out[r, c] = _wasserstein_pool(me_q, pool_q)
    return frame_like(x, out)


_SPECS: dict[str, dict[str, Any]] = {
    "cs_hartigan_dip": {
        "fn": _ts_cs_hartigan_dip,
        "params": ["x", "min_cross"],
        "category": "cross_sectional_state",
        "domain": "distribution_shape",
        "unit": "level",
        "cost": 4,
        "tags_extra": ["global_state"],
        "output_unit": "level",
    },
    "group_wasserstein_barycenter_distance": {
        "fn": _group_wasserstein_barycenter_distance,
        "params": ["x", "group", "window", "min_group_size"],
        "category": "cross_sectional_state",
        "domain": "transport",
        "unit": "same_as_target",
        "cost": 5,
        "tags_extra": [],
        "output_unit": "same_as_target",
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
            source="cs_state_ops",
            tags_extra=spec["tags_extra"],
            output_unit=spec.get("output_unit"),
        )
    union_extended(*_SPECS.keys())


_register()
