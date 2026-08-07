# -*- coding: utf-8 -*-
"""L-moments and unimodality operators (2026-08 geometry/math expansion).

* ``ts_l_skewness`` / ``ts_l_kurtosis`` — L-moment ratios computed with
  Hosking's probability-weighted-moment estimators on the sorted window:

      b_r = (1/n) sum_{i=r}^{n-1} [C(i, r) / C(n-1, r)] * x_{i:n}

  followed by ``lambda_1 = b_0``, ``lambda_2 = 2b_1 - b_0``,
  ``lambda_3 = 6b_2 - 6b_1 + b_0``, ``lambda_4 = 20b_3 - 30b_2 + 12b_1 - b_0``
  and the ratios ``tau_3 = lambda_3/lambda_2``, ``tau_4 = lambda_4/lambda_2``.
  L-moment ratios are bounded (``|tau_3| < 1``, ``tau_4 < 1``), robust and
  defined for any distribution with a finite mean — unlike conventional
  skewness/kurtosis.

* ``ts_hartigan_dip`` — the Hartigan dip statistic: the maximum vertical
  distance between the empirical CDF and the closest unimodal distribution
  (AS 217, S-version).  The implementation follows the classic iterative
  algorithm that alternates between the greatest convex minorant (GCM) on the
  left of the modal interval and the least concave majorant (LCM) on the
  right, skipping through the change points of those envelopes until the modal
  interval stops improving.  Output is the raw dip in ``[0, 0.25]``: near 0
  for unimodal data, large for multimodal data.

All operators are trailing-window, prefix-causal and deterministic.  Degenerate
windows (too few points, constant values) emit NaN.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like, register_polars_bridge


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="moments",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "moments", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:distribution_shape",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _check_window(window: int) -> int:
    w = int(window)
    if w < 2:
        raise ValueError("window must be >= 2")
    return w


# ---------------------------------------------------------------------------
# L-moments (Hosking PWM estimators)
# ---------------------------------------------------------------------------
def _l_moments(vals: np.ndarray) -> tuple[float, float, float, float]:
    x = np.sort(vals[np.isfinite(vals)])
    n = int(x.size)
    if n < 4:
        return np.nan, np.nan, np.nan, np.nan
    idx = np.arange(n, dtype=float)
    b0 = float(np.mean(x))
    b1 = float(np.sum(idx[1:] * x[1:]) / (n - 1)) / n
    b2 = float(np.sum(idx[2:] * (idx[2:] - 1.0) * x[2:]) / ((n - 1) * (n - 2))) / n
    b3 = float(np.sum(idx[3:] * (idx[3:] - 1.0) * (idx[3:] - 2.0) * x[3:]) / ((n - 1) * (n - 2) * (n - 3))) / n
    l1 = b0
    l2 = 2.0 * b1 - b0
    l3 = 6.0 * b2 - 6.0 * b1 + b0
    l4 = 20.0 * b3 - 30.0 * b2 + 12.0 * b1 - b0
    return l1, l2, l3, l4


def _l_ratio_series(x2d: np.ndarray, window: int, ratio: str) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            _l1, l2, l3, l4 = _l_moments(col[i0 : r + 1])
            if not np.isfinite(l2) or abs(l2) <= 1e-12:
                continue
            if ratio == "skew":
                if np.isfinite(l3):
                    out[r, c] = l3 / l2
            else:
                if np.isfinite(l4):
                    out[r, c] = l4 / l2
    return out


# ---------------------------------------------------------------------------
# Hartigan dip statistic (Algorithm AS 217, S-version)
# ---------------------------------------------------------------------------
def _dip_max_distance(
    arr: np.ndarray,
    gcm: np.ndarray,
    lcm: np.ndarray,
    gcm_x: int,
    gcm_y: int,
    lcm_x: int,
    lcm_y: int,
    gcm_rel: int,
    lcm_rel: int,
) -> tuple[float, int, int, int, int]:
    """Largest distance between the GCM and LCM envelopes (AS 217 helper)."""
    ret_d = 0.0
    while True:
        gy = int(gcm[gcm_y])
        ly = int(lcm[lcm_y])
        is_maj = 1 if gy > ly else 0
        i_ = gy if is_maj else ly          # max of the two change points
        j_ = ly if is_maj else gy          # min
        i1 = int(gcm[gcm_y + 1]) if is_maj else int(lcm[lcm_y - 1])
        sign = 2 * is_maj - 1
        denom = arr[i_] - arr[i1]
        if denom == 0.0:
            dx = 0.0
        else:
            dx = sign * ((j_ - i1 + sign) - (arr[j_] - arr[i1]) * (i_ - i1) / denom)
        gcm_y -= 1 - is_maj
        lcm_y += is_maj
        if dx >= ret_d:
            ret_d = dx
            gcm_x = gcm_y + 1
            lcm_x = lcm_y - is_maj
        if gcm_y < 1:
            gcm_y = 1
        if lcm_y > lcm_rel:
            lcm_y = lcm_rel
        if int(gcm[gcm_y]) == int(lcm[lcm_y]):
            break
    return ret_d, gcm_x, gcm_y, lcm_x, lcm_y


def _dip_compute_dip(arr: np.ndarray, optimum: np.ndarray, rel_length: int, x: int, offset: int) -> float:
    """Max distance between one envelope and the empirical CDF points."""
    sign = 1 + -2 * offset
    ret_dip = 0.0
    tmp_val = 1.0  # AS 217 works in "2n * dip" units, so the floor is 1.
    for j in range(x, rel_length):
        j_start = int(optimum[j + 1 - offset])
        j_end = int(optimum[j + offset])
        if j_end - j_start > 1 and arr[j_end] != arr[j_start]:
            c_slope = (j_end - j_start) / (arr[j_end] - arr[j_start])
            arr_js = arr[j_start]
            for jj in range(j_start, j_end + 1):
                d = sign * ((jj - j_start + sign) - (arr[jj] - arr_js) * c_slope)
                if d > tmp_val:
                    tmp_val = d
        if tmp_val > ret_dip:
            ret_dip = tmp_val
        tmp_val = 1.0
    return ret_dip


def _hartigan_dip(x_sorted: np.ndarray) -> float:
    """Hartigan dip statistic (AS 217, S-version) in ``[0, 0.25]``.

    Deterministic translation of the reference ``diptest`` kernel; validated
    bit-exactly against the reference implementation.
    """
    n = int(x_sorted.size)
    if n < 2:
        return 0.0
    if x_sorted[-1] == x_sorted[0]:
        return 0.0
    arr = np.empty(n + 1, dtype=float)
    arr[0] = 0.0
    arr[1:] = x_sorted

    # Change-point chains of the GCM (mn) and LCM (mj) for every start index.
    mn = np.zeros(n + 1, dtype=np.int64)
    mj = np.zeros(n + 1, dtype=np.int64)
    mn[1] = 1
    for i in range(2, n + 1):
        mn[i] = i - 1
        while True:
            a = int(mn[i])
            b = int(mn[a])
            if a == 1:
                break
            if (arr[i] - arr[a]) * (a - b) < (arr[a] - arr[b]) * (i - a):
                break
            mn[i] = b
    mj[n] = n
    for i in range(n - 1, 0, -1):
        mj[i] = i + 1
        while True:
            a = int(mj[i])
            b = int(mj[a])
            if a == n:
                break
            if (arr[i] - arr[a]) * (a - b) < (arr[a] - arr[b]) * (i - a):
                break
            mj[i] = b

    low, high = 1, n
    dip = 0.0
    gcm = np.zeros(n + 1, dtype=np.int64)
    lcm = np.zeros(n + 1, dtype=np.int64)
    # Safety cap: the modal interval strictly shrinks each iteration, so n+10
    # iterations is a generous deterministic bound (never reached in practice).
    for _ in range(n + 10):
        gcm[1] = high
        i = 1
        while int(gcm[i]) > low:
            i += 1
            gcm[i] = mn[int(gcm[i - 1])]
        gcm_rel = i
        gcm_x = gcm_rel
        gcm_y = gcm_rel - 1

        lcm[1] = low
        i = 1
        while int(lcm[i]) < high:
            i += 1
            lcm[i] = mj[int(lcm[i - 1])]
        lcm_rel = i
        lcm_x = lcm_rel
        lcm_y = 2

        if gcm_rel != 2 or lcm_rel != 2:
            d, gcm_x, gcm_y, lcm_x, lcm_y = _dip_max_distance(
                arr, gcm, lcm, gcm_x, gcm_y, lcm_x, lcm_y, gcm_rel, lcm_rel
            )
        else:
            d = 0.0

        if d < dip:
            break

        dip_l = _dip_compute_dip(arr, gcm, gcm_rel, gcm_x, 0)
        dip_u = _dip_compute_dip(arr, lcm, lcm_rel, lcm_x, 1)
        tmp_val = dip_u if dip_l < dip_u else dip_l
        if dip < tmp_val:
            dip = tmp_val

        if low == int(gcm[gcm_x]) and high == int(lcm[lcm_x]):
            break
        low = int(gcm[gcm_x])
        high = int(lcm[lcm_x])
    return dip / (2.0 * n)


def _dip_series(x2d: np.ndarray, window: int) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            chunk = col[i0 : r + 1]
            valid = chunk[np.isfinite(chunk)]
            if valid.size < 4:
                continue
            if float(valid.min()) == float(valid.max()):
                continue  # degenerate constant window -> NaN (never fabricate 0)
            out[r, c] = _hartigan_dip(np.sort(valid))
    return out


@register_operator(
    name="ts_l_skewness",
    category="moments",
    business_category="moments",
    canonical="ts_l_skewness",
    source="moments_ext",
)
class TsLSkewness(SeriesOperator):
    """L 偏度 tau_3 = lambda_3 / lambda_2（Hosking PWM 估计量）。

    有界、对极端值稳健，|tau_3| < 1。0 -> 对称，正 -> 右尾重。P1。
    """

    metadata = _metadata(
        "ts_l_skewness",
        "L 偏度 tau_3 = lambda_3/lambda_2（有界稳健）。",
        ["x", "window"],
        unit="ratio",
        cost=3,
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, **_: Any) -> pd.DataFrame:
        w = _check_window(window)
        return frame_like(x, _l_ratio_series(x.to_numpy(dtype=float), w, "skew"))


@register_operator(
    name="ts_l_kurtosis",
    category="moments",
    business_category="moments",
    canonical="ts_l_kurtosis",
    source="moments_ext",
)
class TsLKurtosis(SeriesOperator):
    """L 峰度 tau_4 = lambda_4 / lambda_2（Hosking PWM 估计量）。

    有界 tau_4 < 1；均匀分布约 0，正态约 0.123，尖峰厚尾更大。P1。
    """

    metadata = _metadata(
        "ts_l_kurtosis",
        "L 峰度 tau_4 = lambda_4/lambda_2（有界稳健）。",
        ["x", "window"],
        unit="ratio",
        cost=3,
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, **_: Any) -> pd.DataFrame:
        w = _check_window(window)
        return frame_like(x, _l_ratio_series(x.to_numpy(dtype=float), w, "kurt"))


@register_operator(
    name="ts_hartigan_dip",
    category="moments",
    business_category="moments",
    canonical="ts_hartigan_dip",
    source="moments_ext",
)
class TsHartiganDip(SeriesOperator):
    """Hartigan dip 统计量：经验 CDF 与最近单峰分布的竖直最大距离。

    AS 217 S-version（GCM/LCM 迭代）。小 -> 单峰；大 -> 多峰。范围 [0, 0.25]。
    与普通偏度/峰度互补，直接度量模态结构。P1。
    """

    metadata = _metadata(
        "ts_hartigan_dip",
        "Hartigan dip 统计量（最近单峰拟合的最大距离，S-version）。",
        ["x", "window"],
        unit="ratio",
        cost=7,
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 120, **_: Any) -> pd.DataFrame:
        w = _check_window(window)
        return frame_like(x, _dip_series(x.to_numpy(dtype=float), w))


_NEW_CANONICALS = (
    "ts_l_skewness",
    "ts_l_kurtosis",
    "ts_hartigan_dip",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | set(_NEW_CANONICALS)
    )
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
