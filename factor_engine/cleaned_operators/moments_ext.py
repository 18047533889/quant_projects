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

R14-P1 statistical-usability gates ("不要把数学上可算当成统计上可用"): dropping
NaN and computing on whatever remains turns a ``window=120`` with only 4-10 real
observations into a number that is not comparable across stocks/time.  Each
operator therefore exposes ``min_periods`` (minimum number of finite values
actually used after dropping NaN) and ``min_coverage_fraction`` (that effective
count divided by the nominal length of the *current* trailing window, i.e.
``min(r+1, window)``).  A window below either gate emits NaN.  L-moments default
to ``min_periods=20`` / ``min_coverage_fraction=0.5``; the Hartigan dip to
``min_periods=20`` / ``min_coverage_fraction=0.8``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata, ParamRole, ParamSpec, RelationalParamSpec, SeriesOperator, register_operator, strict_int_param
from factor_engine.cleaned_operators.rolling_pack import frame_like, register_polars_bridge


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int, window_default: int, coverage_default: float, minimum: int) -> OperatorMetadata:
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
        output_unit="dimensionless",
        panel_params=("x",),
        panel_arity=1,
        scalar_params=("window", "min_periods", "min_coverage_fraction"),
        param_specs={
            "window": ParamSpec(dtype=int, min=minimum, default=window_default, history_semantics="max_rows", param_role=ParamRole.HORIZON),
            "min_periods": ParamSpec(dtype=int, min=minimum, default=20, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
            "min_coverage_fraction": ParamSpec(dtype=float, min=np.nextafter(0.0, 1.0), max=1.0, default=coverage_default, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        },
        relational_specs=[RelationalParamSpec(expression="min_periods <= window")],
    )


def _check_window(window: int, *, minimum: int) -> int:
    return strict_int_param(window, "window", lower=minimum)


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


def _l_ratio_series(
    x2d: np.ndarray,
    window: int,
    ratio: str,
    min_periods: int = 20,
    min_coverage_fraction: float = 0.5,
) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = strict_int_param(window, "window", lower=4)
    mp = strict_int_param(min_periods, "min_periods", lower=4)
    mcf = float(min_coverage_fraction)
    if not np.isfinite(mcf) or not 0.0 < mcf <= 1.0:
        raise ValueError("min_coverage_fraction must be finite and in (0, 1]")
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            chunk = col[i0 : r + 1]
            valid = chunk[np.isfinite(chunk)]
            # R14-P1: statistical-usability gate — effective n after dropping
            # NaN, plus the fraction the finite values cover of the NOMINAL
            # length of this trailing window (``min(r+1, window)``).  A
            # ``window=120`` with only a handful of real observations must emit
            # NaN, not a number that is not comparable across stocks/time.
            if valid.size < mp:
                continue
            if valid.size / float(chunk.size) < mcf:
                continue
            magnitude = float(np.max(np.abs(valid)))
            if magnitude == 0.0:
                continue
            _l1, l2, l3, l4 = _l_moments(valid / magnitude)
            if not np.isfinite(l2) or l2 == 0.0:
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


# ---------------------------------------------------------------------------
# R62: lockstep-batched AS 217 dip (no per-row Python loop)
# ---------------------------------------------------------------------------

def _trailing_windows(x: np.ndarray, w: int) -> np.ndarray:
    """``(T, w)`` trailing-aligned window matrix (NaN before the series start)."""
    n = int(x.shape[0])
    if n == 0:
        return np.empty((0, w), dtype=float)
    src = np.arange(n)[:, None] - (w - 1) + np.arange(w)[None, :]
    return np.where(src >= 0, x[np.clip(src, 0, n - 1)], np.nan)


def _dip_scan(_A: np.ndarray, _optimum: np.ndarray, _rel: np.ndarray, _x: np.ndarray,
              _offset: int, _live: np.ndarray) -> np.ndarray:
    """Batched ``_dip_compute_dip``: max envelope-to-ECDF distance per row.

    The per-row ``for j`` / ``for jj`` pair is evaluated as one ``(rows, L)``
    block per ``j`` step (``L`` = the longest link of that step), with the
    authority's ``tmp_val = 1.0`` floor preserved.
    """
    T, W = _A.shape[0], _A.shape[1] - 1
    rows = np.arange(T)
    sign = 1 + -2 * _offset
    ret = np.zeros(T)
    k = 0
    while True:
        j = _x + k
        act = _live & (j < _rel)
        if not act.any():
            break
        link_lo = _optimum[rows, np.clip(j + 1 - _offset, 0, W)]
        link_hi = _optimum[rows, np.clip(j + _offset, 0, W)]
        usable = act & ((link_hi - link_lo) > 1) & (_A[rows, link_hi] != _A[rows, link_lo])
        length = np.where(usable, link_hi - link_lo + 1, 0)
        lmax = int(length.max())
        if lmax > 0:
            off = np.arange(lmax)[None, :]
            pos = link_lo[:, None] + off
            inside = off < length[:, None]
            dev = _A[rows[:, None], np.clip(pos, 0, W)] - _A[rows, link_lo][:, None]
            with np.errstate(invalid="ignore", divide="ignore"):
                slope = np.where(
                    usable, (link_hi - link_lo) / (_A[rows, link_hi] - _A[rows, link_lo]), 0.0
                )
            vals = sign * ((pos - link_lo[:, None] + sign) - dev * slope[:, None])
            step = np.where(inside, vals, -np.inf).max(axis=1)
            cand = np.where(usable, np.maximum(1.0, step), 1.0)
            ret = np.where(act, np.maximum(ret, cand), ret)
        else:
            ret = np.where(act, np.maximum(ret, 1.0), ret)
        k += 1
    return ret


def _dip_lockstep(_A: np.ndarray, _m: np.ndarray, _active: np.ndarray, w: int) -> np.ndarray:
    """AS 217 (S-version) for every window at once; returns the raw dip*2n."""
    T = _A.shape[0]
    W = w
    rows = np.arange(T)

    # --- GCM change-point chains mn[i] (prefix, walking down) ---
    top = int(_m.max()) if T else 0
    mn = np.zeros((T, W + 1), dtype=np.int64)
    mn[:, 1] = 1
    for i in range(2, top + 1):
        act = _active & (i <= _m)
        if not act.any():
            break
        cur = np.where(act, i - 1, 0)
        ai = _A[:, i]
        while True:
            b = mn[rows, cur]
            aa = _A[rows, cur]
            ab = _A[rows, b]
            done = ~act | (cur <= 1) | ((ai - aa) * (cur - b) < (aa - ab) * (i - cur))
            if done.all():
                break
            cur = np.where(done, cur, b)
        mn[:, i] = np.where(act, cur, 0)

    # --- LCM change-point chains mj[i] (suffix, walking up) ---
    mj = np.zeros((T, W + 1), dtype=np.int64)
    mj[rows, _m] = _m
    for i in range(top - 1, 0, -1):
        act = _active & (i < _m)
        if not act.any():
            continue
        cur = np.where(act, i + 1, 0)
        ai = _A[:, i]
        while True:
            b = mj[rows, cur]
            aa = _A[rows, cur]
            ab = _A[rows, b]
            done = ~act | (cur >= _m) | ((ai - aa) * (cur - b) < (aa - ab) * (i - cur))
            if done.all():
                break
            cur = np.where(done, cur, b)
        mj[:, i] = np.where(act, cur, 0)

    low = np.ones(T, dtype=np.int64)
    high = _m.copy()
    dip = np.zeros(T)
    finished = ~_active
    gcm = np.zeros((T, W + 1), dtype=np.int64)
    lcm = np.zeros((T, W + 1), dtype=np.int64)

    for _ in range(top + 10):
        if not (_active & ~finished).any():
            break
        live = _active & ~finished

        # --- gcm chain: from high down to low through mn ---
        gcm[:, 1] = high
        ptr = np.ones(T, dtype=np.int64)
        while True:
            more = live & (gcm[rows, ptr] > low)
            if not more.any():
                break
            nxt = np.where(more, ptr + 1, ptr)
            src = mn[rows, gcm[rows, nxt - 1]]
            sel = np.where(more)[0]
            gcm[sel, nxt[sel]] = src[sel]
            ptr = nxt
        gcm_rel = ptr.copy()

        # --- lcm chain: from low up to high through mj ---
        lcm[:, 1] = low
        ptr = np.ones(T, dtype=np.int64)
        while True:
            more = live & (lcm[rows, ptr] < high)
            if not more.any():
                break
            nxt = np.where(more, ptr + 1, ptr)
            src = mj[rows, lcm[rows, nxt - 1]]
            sel = np.where(more)[0]
            lcm[sel, nxt[sel]] = src[sel]
            ptr = nxt
        lcm_rel = ptr.copy()

        gcm_x = gcm_rel.copy()
        gcm_y = gcm_rel - 1
        lcm_x = lcm_rel.copy()
        lcm_y = np.full(T, 2, dtype=np.int64)

        # --- _dip_max_distance (only when either chain is not a single link) ---
        run = live & ((gcm_rel != 2) | (lcm_rel != 2))
        d = np.zeros(T)
        if run.any():
            ret_d = np.zeros(T)
            act2 = run.copy()
            while act2.any():
                gy = gcm[rows, gcm_y]
                ly = lcm[rows, lcm_y]
                is_maj = gy > ly
                imaj = is_maj.astype(np.int64)
                i_ = np.where(is_maj, gy, ly)
                j_ = np.where(is_maj, ly, gy)
                i1 = np.where(
                    is_maj,
                    gcm[rows, np.clip(gcm_y + 1, 0, W)],
                    lcm[rows, np.clip(lcm_y - 1, 0, W)],
                )
                sign = 2 * imaj - 1
                with np.errstate(invalid="ignore", divide="ignore"):
                    denom = _A[rows, i_] - _A[rows, i1]
                    dx = np.where(
                        denom == 0.0,
                        0.0,
                        sign * ((j_ - i1 + sign)
                                - (_A[rows, j_] - _A[rows, i1]) * (i_ - i1) / denom),
                    )
                gcm_y = gcm_y - (1 - imaj)
                lcm_y = lcm_y + imaj
                take = act2 & (dx >= ret_d)
                ret_d = np.where(take, dx, ret_d)
                gcm_x = np.where(take, gcm_y + 1, gcm_x)
                lcm_x = np.where(take, lcm_y - imaj, lcm_x)
                gcm_y = np.maximum(gcm_y, 1)
                lcm_y = np.minimum(lcm_y, lcm_rel)
                act2 = act2 & (gcm[rows, gcm_y] != lcm[rows, lcm_y])
            d = ret_d

        dip_l = _dip_scan(_A, gcm, gcm_rel, gcm_x, 0, live)
        dip_u = _dip_scan(_A, lcm, lcm_rel, lcm_x, 1, live)
        # authority: ``tmp_val = dip_u if dip_l < dip_u else dip_l`` -> MAX
        tmp_val = np.maximum(dip_l, dip_u)
        brk = live & (d < dip)
        cont = live & ~brk
        dip = np.where(cont & (dip < tmp_val), tmp_val, dip)
        stop = cont & (low == gcm[rows, np.clip(gcm_x, 0, W)]) & (
            high == lcm[rows, np.clip(lcm_x, 0, W)]
        )
        finished = finished | brk | stop
        upd = cont & ~stop
        low = np.where(upd, gcm[rows, np.clip(gcm_x, 0, W)], low)
        high = np.where(upd, lcm[rows, np.clip(lcm_x, 0, W)], high)
    return dip


def _dip_series(
    x2d: np.ndarray,
    window: int,
    min_periods: int = 20,
    min_coverage_fraction: float = 0.8,
) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = strict_int_param(window, "window", lower=2)
    mp = strict_int_param(min_periods, "min_periods", lower=2)
    mcf = float(min_coverage_fraction)
    if not np.isfinite(mcf) or not 0.0 < mcf <= 1.0:
        raise ValueError("min_coverage_fraction must be finite and in (0, 1]")
    for c in range(cols):
        out[:, c] = _dip_column(x2d[:, c], w, mp, mcf)
    return out


def _dip_column(col: np.ndarray, w: int, mp: int, mcf: float) -> np.ndarray:
    """One column of ``_dip_series`` (the panel row loop is gone; every window
    is a row of the ``(T, w)`` batch)."""
    n = int(col.shape[0])
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return out
    W = w
    win = _trailing_windows(col, W)
    fin = np.isfinite(win)
    m = fin.sum(axis=1)
    nominal = np.minimum(np.arange(n) + 1, W)
    # R14-P1: statistical-usability gate — effective n after dropping NaN, plus
    # the fraction the finite values cover of the NOMINAL length of this
    # trailing window (``min(r+1, window)``).
    active = (m >= mp) & (mp >= 2) & (m / np.maximum(nominal, 1) >= mcf)
    order = np.sort(np.where(fin, win, np.nan), axis=1)
    top_idx = np.clip(m - 1, 0, W - 1)
    vmin = np.where(active, order[:, 0], np.nan)
    vmax = np.where(active, order[np.arange(n), top_idx], np.nan)
    # degenerate constant window -> NaN (never fabricate 0)
    active = active & (vmin != vmax)
    if not active.any():
        return out
    magnitude = np.maximum(np.abs(vmin), np.abs(vmax))
    A = np.zeros((n, W + 1), dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        A[:, 1:] = order / magnitude[:, None]
    raw = _dip_lockstep(A, m, active, W)
    with np.errstate(invalid="ignore", divide="ignore"):
        val = raw / (2.0 * m)
    return np.where(active, val, np.nan)


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
    R14-P1: ``min_periods``（有效有限样本数）与 ``min_coverage_fraction``
    （有效样本/当前滚动窗名义长度）双重门控，样本不足或覆盖率过低 -> NaN。
    """

    metadata = _metadata(
        "ts_l_skewness",
        "L 偏度 tau_3 = lambda_3/lambda_2（有界稳健）。",
        ["x", "window", "min_periods", "min_coverage_fraction"],
        unit="ratio",
        cost=3,
        window_default=60,
        coverage_default=0.5,
        minimum=4,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 60,
        min_periods: int = 20,
        min_coverage_fraction: float = 0.5,
        **_: Any,
    ) -> pd.DataFrame:
        w = _check_window(window, minimum=4)
        return frame_like(
            x,
            _l_ratio_series(
                x.to_numpy(dtype=float), w, "skew",
                min_periods=min_periods, min_coverage_fraction=min_coverage_fraction,
            ),
        )


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
    R14-P1: ``min_periods`` / ``min_coverage_fraction`` 门控同上。
    """

    metadata = _metadata(
        "ts_l_kurtosis",
        "L 峰度 tau_4 = lambda_4/lambda_2（有界稳健）。",
        ["x", "window", "min_periods", "min_coverage_fraction"],
        unit="ratio",
        cost=3,
        window_default=60,
        coverage_default=0.5,
        minimum=4,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 60,
        min_periods: int = 20,
        min_coverage_fraction: float = 0.5,
        **_: Any,
    ) -> pd.DataFrame:
        w = _check_window(window, minimum=4)
        return frame_like(
            x,
            _l_ratio_series(
                x.to_numpy(dtype=float), w, "kurt",
                min_periods=min_periods, min_coverage_fraction=min_coverage_fraction,
            ),
        )


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
    R14-P1: ``min_periods``（默认 20）与 ``min_coverage_fraction``（默认 0.8）
    门控；经验 CDF 几何对稀疏窗敏感，覆盖率要求高于 L 矩。
    """

    metadata = _metadata(
        "ts_hartigan_dip",
        "Hartigan dip 统计量（最近单峰拟合的最大距离，S-version）。",
        ["x", "window", "min_periods", "min_coverage_fraction"],
        unit="ratio",
        cost=7,
        window_default=120,
        coverage_default=0.8,
        minimum=2,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 120,
        min_periods: int = 20,
        min_coverage_fraction: float = 0.8,
        **_: Any,
    ) -> pd.DataFrame:
        w = _check_window(window, minimum=2)
        return frame_like(
            x,
            _dip_series(
                x.to_numpy(dtype=float), w,
                min_periods=min_periods, min_coverage_fraction=min_coverage_fraction,
            ),
        )


_NEW_CANONICALS = (
    "ts_l_skewness",
    "ts_l_kurtosis",
    "ts_hartigan_dip",
)


def _register_surface() -> None:
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
