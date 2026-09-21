# -*- coding: utf-8 -*-
"""Alpha-language distribution-shape and distribution-shift operators (2026-08).

Tail / robust-skew family:

* tail imbalance: whether the tail mass skews to the upside or downside
  (count-based, MAD-thresholded so upper/lower tail counts are not forced equal);
* expected shortfall asymmetry: upside vs downside tail *depth* asymmetry.

Two-window shift family (recent ``[t-Ws+1, t]`` vs older ``[t-Ws-Wl+1, t-Ws]``,
non-overlapping):

* Wasserstein shift: how far the distribution moved (1-D earth mover);
* KS shift: the largest CDF gap (distributional shape change);
* location shift / scale shift: robust median / MAD movement.

All operators are causal trailing transforms (only rows ``<= t``).  Degenerate
windows (too few samples / zero scale) return NaN, never Inf or an invented 0.
"""
from __future__ import annotations

import math

from typing import Any, Callable

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from factor_engine.cleaned_operators.base import OperatorMetadata, ParamRole, ParamSpec, SeriesOperator, register_operator
from factor_engine.cleaned_operators.rolling_pack import (
    check_window,
    frame_like,
    register_polars_bridge,
    valid_values,
)

_EPS = 1e-12

_DISTRIBUTION_DEFAULTS = {
    "ts_tail_imbalance": {"window": 60, "k": 1.0, "min_periods": 8},
    "ts_expected_shortfall_asymmetry": {"window": 60, "q": 0.05, "min_tail_count": 3},
    **{name: {"recent_window": 20, "old_window": 40, "min_periods": mp} for name, mp in {
        "ts_wasserstein_shift": 5, "ts_ks_shift": 5, "ts_location_shift": 5,
        "ts_scale_shift": 5, "ts_quantile_transport_slope": 3,
        "ts_quantile_transport_curvature": 3, "ts_mmd_rbf_shift": 5,
    }.items()},
}

def _distribution_specs(name: str) -> dict[str, ParamSpec]:
    out: dict[str, ParamSpec] = {}
    for key, default in _DISTRIBUTION_DEFAULTS[name].items():
        if key in ("window", "recent_window", "old_window", "min_periods", "min_tail_count"):
            out[key] = ParamSpec(dtype=int, min=1, default=default, param_role=ParamRole.HORIZON)
        elif key == "k":
            out[key] = ParamSpec(dtype=float, min=0.0, default=default, param_role=ParamRole.STATE_THRESHOLD)
        elif key == "q":
            out[key] = ParamSpec(dtype=float, min=0.0, max=0.5, default=default, param_role=ParamRole.ESTIMATOR_RESOLUTION)
    return out


def _metadata(name: str, description: str, params: list[str], *, unit: str) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="time_series_distribution",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "time_series_distribution", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", "domain:distribution",
            f"unit:{unit}", "cost:1",
        ],
        panel_params=tuple(p for p in params if p not in _DISTRIBUTION_DEFAULTS[name]),
        scalar_params=tuple(_DISTRIBUTION_DEFAULTS[name]),
        param_specs=_distribution_specs(name),
    )


def map_two_window(
    values: np.ndarray,
    recent_window: int,
    old_window: int,
    fn: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray:
    """Trailing two-window map (recent vs older, non-overlapping), per column.

    ``fn(recent_chunk, old_chunk)``; recent is the trailing ``recent_window``
    rows ending at the current row, old is the ``old_window`` rows immediately
    before it.  When the old window falls before the panel start the output is
    NaN (no older reference).
    """
    rows, cols = values.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    ws = int(recent_window)
    wl = int(old_window)
    for col in range(cols):
        for row in range(rows):
            recent_lo = max(0, row - ws + 1)
            recent = values[recent_lo : row + 1, col]
            old_end = row - ws
            if old_end < 0:
                continue
            old_lo = max(0, old_end - wl + 1)
            old = values[old_lo : old_end + 1, col]
            if old.size == 0:
                continue
            out[row, col] = fn(recent, old)
    return out


def _mad(vals: np.ndarray) -> float:
    return float(np.median(np.abs(vals - np.median(vals))))


# ---------------------------------------------------------------------------
# R62 vectorised two-window / rolling-window machinery
# ---------------------------------------------------------------------------
_Q_GRID9 = np.linspace(0.1, 0.9, 9)
_A_TRANSPORT9 = np.column_stack(
    [np.ones_like(_Q_GRID9), _Q_GRID9 - 0.5, (_Q_GRID9 - 0.5) ** 2]
)
_TRIU_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _triu_idx(m: int):
    if m not in _TRIU_CACHE:
        _TRIU_CACHE[m] = np.triu_indices(m, k=1)
    return _TRIU_CACHE[m]


def _win_pad_rows(col: np.ndarray, w: int) -> np.ndarray:
    """Trailing windows ``(n, w)``; row r = col[r-w+1 .. r], NaN where negative."""
    pre = np.concatenate([np.full(w - 1, np.nan), col])
    return sliding_window_view(pre, w)


def _quantile_interp(sorted_win: np.ndarray, counts: np.ndarray, levels) -> np.ndarray:
    """``np.quantile(finite_row, levels, method="linear")`` for every row.

    ``sorted_win`` (n, W) holds the finite values first (ascending, NaN in the
    pad tail) and ``counts`` (n,) the finite count.  Reproduces numpy's linear
    method: index ``q*(p-1)`` bracketed by floor/ceil with a fractional lerp.
    """
    lv = np.asarray(levels, dtype=float)
    if lv.ndim == 1:
        lv = lv[None, :]
    pm1 = np.maximum(counts - 1, 0)
    pos = lv * pm1[:, None].astype(np.float64)
    pm1i = np.maximum(pm1.astype(np.int64)[:, None], 0)
    lo = np.clip(np.floor(pos).astype(np.int64), 0, pm1i)
    hi = np.minimum(lo + 1, pm1i)
    frac = pos - lo
    a = np.take_along_axis(sorted_win, lo, axis=1)
    b = np.take_along_axis(sorted_win, hi, axis=1)
    return a + (b - a) * frac


def _rolling_quantile(Wm: np.ndarray, levels) -> np.ndarray:
    """``np.nanquantile`` along axis 1 without numpy's hidden per-row loop."""
    Ws = np.sort(Wm, axis=1)
    cnt = np.isfinite(Wm).sum(axis=1)
    return _quantile_interp(Ws, cnt, levels)


def _two_window_pads(col: np.ndarray, ws: int, wl: int):
    """Recent ``[r-ws+1, r]`` / older ``[r-ws-wl+1, r-ws]`` trailing windows.

    The older window is NaN wherever it falls before the panel start, which is
    exactly ``map_two_window``'s ``old_end < 0`` -> NaN rule.
    """
    n = col.size
    pre = np.concatenate([np.full(ws + wl - 1, np.nan), col])
    return sliding_window_view(pre, ws)[wl : wl + n], sliding_window_view(pre, wl)[0:n]


def _sorted_prefix_median(sorted_rows: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Median of the finite prefix of each ascending-sorted row.

    ``sorted_rows`` (n, W) is finite-first ascending with a NaN pad tail and
    ``count`` (n,) the finite count, so the median rank depends only on
    ``count`` -- a pure gather.  ``np.nanmedian(..., axis=1)`` would be a
    hidden per-row Python loop (numpy's ``apply_along_axis`` fallback).
    """
    last = sorted_rows.shape[1] - 1
    c = np.maximum(count, 1)
    k1 = np.minimum((c - 1) // 2, last)
    k2 = np.minimum(c // 2, last)
    a = np.take_along_axis(sorted_rows, k1[:, None], axis=1)[:, 0]
    b = np.take_along_axis(sorted_rows, k2[:, None], axis=1)[:, 0]
    return np.where(count > 0, 0.5 * (a + b), np.nan)


def _median_mad_sorted(Osorted: np.ndarray):
    """Median and MAD of each sorted row (finite prefix, NaN pad tail)."""
    count = np.isfinite(Osorted).sum(axis=1)
    med = _sorted_prefix_median(Osorted, count)
    dev = np.sort(np.abs(Osorted - med[:, None]), axis=1)
    mad = _sorted_prefix_median(dev, count)
    return med, mad


def _pairwise_median(Comb: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Median of the pairwise absolute differences of each row's finite sample.

    ``Comb`` (n, W) is sorted finite-first with a NaN tail and ``m`` is the
    finite count.  The finite values' *positions* depend only on ``m``, so rows
    are grouped by sample size and each group is selected with a fixed ``kth``
    partition -- no per-row Python work, no ``np.nanmedian`` row loop.
    """
    out = np.full(Comb.shape[0], np.nan)
    ok = m >= 2
    if not ok.any():
        return out
    for mv in np.unique(m[ok]):
        idx = np.flatnonzero(ok & (m == mv))
        ii, jj = _triu_idx(int(mv))
        sub = Comb[idx]
        d = sub[:, jj] - sub[:, ii]          # sorted asc -> jj>ii gives |diff|
        c = d.shape[1]
        k1 = (c - 1) // 2
        k2 = c // 2
        if k1 == k2:
            out[idx] = np.partition(d, k1, axis=1)[:, k1]
        else:
            p = np.partition(d, [k1, k2], axis=1)
            out[idx] = 0.5 * (p[:, k1] + p[:, k2])
    return out


def _es_asymmetry_series(x2d: np.ndarray, window: int, q: float, min_tail: int) -> np.ndarray:
    """``(u - l) / (u + l)`` with ``u`` / ``l`` the mean upper / lower tail depth."""
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        Wm = _win_pad_rows(x2d[:, c], w)
        fin = np.isfinite(Wm)
        cnt = fin.sum(axis=1)
        with np.errstate(all="ignore"):
            quants = _rolling_quantile(Wm, np.array([q, 1.0 - q]))
            q_lo = quants[:, 0]
            q_up = quants[:, 1]
            up = fin & (Wm > q_up[:, None])
            lo = fin & (Wm < q_lo[:, None])
            cup = up.sum(axis=1)
            clo = lo.sum(axis=1)
            u = np.where(up, Wm, 0.0).sum(axis=1) / cup - q_up
            l = q_lo - np.where(lo, Wm, 0.0).sum(axis=1) / clo
            denom = u + l
            val = (u - l) / denom
        ok = ((cnt >= 2 * min_tail) & (cup >= min_tail) & (clo >= min_tail)
              & np.isfinite(denom) & (denom >= _EPS))
        out[:, c] = np.where(ok, val, np.nan)
    return out


def _location_shift_series(x2d, ws, wl, mp):
    """``(median(A) - median(B)) / (MAD(B) + eps)`` on the two windows."""
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        rec, old = _two_window_pads(x2d[:, c], ws, wl)
        Rs = np.sort(rec, axis=1)
        Os = np.sort(old, axis=1)
        cnt_r = np.isfinite(Rs).sum(axis=1)
        cnt_o = np.isfinite(Os).sum(axis=1)
        med_r = _sorted_prefix_median(Rs, cnt_r)
        med_o, mad_o = _median_mad_sorted(Os)
        with np.errstate(all="ignore"):
            val = (med_r - med_o) / mad_o
        ok = (cnt_r >= mp) & (cnt_o >= mp) & np.isfinite(mad_o) & (mad_o >= _EPS)
        out[:, c] = np.where(ok, val, np.nan)
    return out


def _wasserstein_shift_series(x2d, ws, wl, mp):
    """Mean ``|Q_A(u) - Q_B(u)|`` over an ``n``-point grid, scaled by MAD(B)."""
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        rec, old = _two_window_pads(x2d[:, c], ws, wl)
        Rs = np.sort(rec, axis=1)
        Os = np.sort(old, axis=1)
        cnt_r = np.isfinite(Rs).sum(axis=1)
        cnt_o = np.isfinite(Os).sum(axis=1)
        ng = np.maximum(cnt_r, cnt_o)
        ncol = int(ng.max()) if ng.size else 1
        i = np.arange(max(ncol, 1))[None, :]
        # np.linspace(0, 1, n): u_i = i/(n-1) step, last point pinned to 1.
        lv = i / np.maximum(ng - 1, 1)[:, None].astype(np.float64)
        lv = np.where(np.arange(max(ncol, 1))[None, :] < ng[:, None], lv, 0.0)
        qa = _quantile_interp(Rs, cnt_r, lv)
        qb = _quantile_interp(Os, cnt_o, lv)
        gmask = i < ng[:, None]
        with np.errstate(all="ignore"):
            w1 = np.where(gmask, np.abs(qa - qb), 0.0).sum(axis=1) / ng
        _med_o, mad_o = _median_mad_sorted(Os)
        with np.errstate(all="ignore"):
            val = w1 / mad_o
        ok = (cnt_r >= mp) & (cnt_o >= mp) & np.isfinite(mad_o) & (mad_o >= _EPS)
        out[:, c] = np.where(ok, val, np.nan)
    return out


def _transport_series(x2d, ws, wl, mp, which):
    """OLS ``dQ(q) = a + b*z + c*z^2`` on the fixed grid; ``which`` 1 -> b, 2 -> c."""
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        rec, old = _two_window_pads(x2d[:, c], ws, wl)
        Rs = np.sort(rec, axis=1)
        Os = np.sort(old, axis=1)
        cnt_r = np.isfinite(Rs).sum(axis=1)
        cnt_o = np.isfinite(Os).sum(axis=1)
        dq = _quantile_interp(Rs, cnt_r, _Q_GRID9) - _quantile_interp(Os, cnt_o, _Q_GRID9)
        fit_mask = (cnt_r >= 3) & (cnt_o >= 3)
        dqm = np.where(np.isfinite(dq), dq, 0.0)
        dqm = np.where(fit_mask[:, None], dqm, 0.0)
        with np.errstate(all="ignore"):
            coef, *_ = np.linalg.lstsq(_A_TRANSPORT9, dqm.T, rcond=None)
            val = coef[which]
        ok = (cnt_r >= mp) & (cnt_o >= mp) & np.all(np.isfinite(coef), axis=0)
        out[:, c] = np.where(ok, val, np.nan)
    return out


def _mmd_rbf_shift_series(x2d, ws, wl, mp):
    """RBF-kernel MMD^2 with the median-heuristic bandwidth (deterministic).

    The kernel is exactly invariant under a common rescaling of the samples and
    the bandwidth, so the samples are put on an ``np.frexp`` power-of-two column
    scale (lossless) before the pairwise kernels; that keeps ``exp`` finite for
    1e+-300 data.  ``tau`` is still ALSO evaluated on the authority scale so the
    overflow-to-NaN behaviour of the historical kernel is reproduced verbatim.
    """
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    sent = 1e150
    Ri, Rj = _triu_idx(ws)
    Oi, Oj = _triu_idx(wl)
    for c in range(cols):
        col = x2d[:, c]
        rec, old = _two_window_pads(col, ws, wl)
        Rp = np.isfinite(rec)
        Op = np.isfinite(old)
        cnt_r = Rp.sum(axis=1)
        cnt_o = Op.sum(axis=1)
        fin = np.isfinite(col)
        if fin.any():
            ma = float(np.max(np.abs(col[fin])))
            e = math.frexp(ma)[1] if (ma > 0.0 and np.isfinite(ma)) else 0
        else:
            e = 0
        sc = math.ldexp(1.0, -e)
        Rs = np.sort(np.where(Rp, rec * sc, np.nan), axis=1)
        Os = np.sort(np.where(Op, old * sc, np.nan), axis=1)
        Comb = np.sort(np.concatenate([Rs, Os], axis=1), axis=1)
        m = cnt_r + cnt_o
        with np.errstate(all="ignore"):
            sigma_s = _pairwise_median(Comb, m)
            sigma_o = np.ldexp(sigma_s, e)
        # the NaN pad lives in the sorted TAIL, so mask on the sorted array
        Rf = np.where(np.isfinite(Rs), Rs, sent)
        Of = np.where(np.isfinite(Os), Os, sent)
        with np.errstate(all="ignore"):
            tau_o = 2.0 * sigma_o * sigma_o
            tau = 2.0 * sigma_s * sigma_s
            t2 = tau[:, None]
            t3 = tau[:, None, None]
            cntr = cnt_r.astype(np.float64)
            cnto = cnt_o.astype(np.float64)
            nri = (ws - cnt_r).astype(np.float64)
            noi = (wl - cnt_o).astype(np.float64)
            # symmetric blocks: diagonal ones + twice the strict upper triangle
            dr = Rf[:, Ri] - Rf[:, Rj]
            kxx = cntr + 2.0 * np.exp(-(dr * dr) / t2).sum(axis=1) - nri * (nri - 1.0)
            do = Of[:, Oi] - Of[:, Oj]
            kyy = cnto + 2.0 * np.exp(-(do * do) / t2).sum(axis=1) - noi * (noi - 1.0)
            dxy = Rf[:, :, None] - Of[:, None, :]
            kxy = np.exp(-(dxy * dxy) / t3).sum(axis=(1, 2)) - nri * noi
            mmd2 = (kxx / (cntr * cntr) + kyy / (cnto * cnto) - 2.0 * kxy / (cntr * cnto))
            val = np.maximum(mmd2, 0.0)
        ok = ((cnt_r >= mp) & (cnt_o >= mp) & (m >= 4)
              & np.isfinite(sigma_o) & (sigma_o >= _EPS) & np.isfinite(tau_o))
        out[:, c] = np.where(ok, val, np.nan)
    return out


@register_operator(
    name="ts_tail_imbalance",
    category="time_series_distribution",
    business_category="time_series_distribution",
    canonical="ts_tail_imbalance",
    source="alpha_language_distribution")
class TsTailImbalance(SeriesOperator):
    """尾部失衡: (Upper - Lower) / n, 范围 [-1,1]。

    Upper = #{x > m + k*s}, Lower = #{x < m - k*s}; m = 中位数,
    s = 1.4826*MAD。用 MAD 阈值而非分位数, 避免上下尾计数被机械拉平。
    """

    metadata = _metadata(
        "ts_tail_imbalance",
        "MAD 阈值下的上下尾计数失衡。",
        ["x", "window", "k", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, k: float = 1.0, min_periods: int = 8, **_: Any
    ) -> pd.DataFrame:
        w = check_window(window)
        kk = float(k)
        if kk <= 0.0:
            raise ValueError("k must be > 0")
        mp = max(4, int(min_periods))
        xv = x.to_numpy(dtype=float)

        def _fn(chunk: np.ndarray) -> float:
            v = valid_values(chunk)
            n = v.size
            if n < mp:
                return np.nan
            m = float(np.median(v))
            s = 1.4826 * _mad(v)
            if not np.isfinite(s) or s < _EPS:
                return np.nan
            upper = float(np.sum(v > m + kk * s))
            lower = float(np.sum(v < m - kk * s))
            return float((upper - lower) / n)

        from factor_engine.cleaned_operators.rolling_pack import map_rolling

        return frame_like(x, map_rolling(xv, w, _fn))


@register_operator(
    name="ts_expected_shortfall_asymmetry",
    category="time_series_distribution",
    business_category="time_series_distribution",
    canonical="ts_expected_shortfall_asymmetry",
    source="alpha_language_distribution")
class TsExpectedShortfallAsymmetry(SeriesOperator):
    """期望短尾不对称: (U - L) / (U + L + eps), 范围 [-1,1]。

    U = mean(x - Q_{1-q} | x > Q_{1-q}), L = mean(Q_q - x | x < Q_q)。
    正值 = 极端上涨幅度强于极端下跌; 等价 ts_expected_shortfall 上下尾的组合 fused。
    """

    metadata = _metadata(
        "ts_expected_shortfall_asymmetry",
        "上尾深度 vs 下尾深度不对称。",
        ["x", "window", "q", "min_tail_count"],
        unit="ratio",
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, q: float = 0.05, min_tail_count: int = 3, **_: Any
    ) -> pd.DataFrame:
        w = check_window(window)
        quantile = float(q)
        if not 0.0 < quantile < 0.5:
            raise ValueError("q must be in (0, 0.5)")
        min_tail = max(2, int(min_tail_count))
        return frame_like(
            x,
            _es_asymmetry_series(x.to_numpy(dtype=float), w, quantile, min_tail),
        )
@register_operator(
    name="ts_wasserstein_shift",
    category="time_series_distribution",
    business_category="time_series_distribution",
    canonical="ts_wasserstein_shift",
    source="alpha_language_distribution")
class TsWassersteinShift(SeriesOperator):
    """一维 Wasserstein 位移: mean_u |F_A^{-1}(u) - F_B^{-1}(u)| / (MAD(B) + eps)。

    A = 最近窗口, B = 更早窗口(不重叠)。衡量分布移动了多远。
    """

    metadata = _metadata(
        "ts_wasserstein_shift",
        "近窗口 vs 旧窗口一维 Wasserstein 距离(按旧窗口 MAD 归一)。",
        ["x", "recent_window", "old_window", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(
        self, x: pd.DataFrame, recent_window: int = 20, old_window: int = 40, min_periods: int = 5, **_: Any
    ) -> pd.DataFrame:
        ws = check_window(recent_window, name="recent_window")
        wl = check_window(old_window, name="old_window")
        mp = max(3, int(min_periods))
        return frame_like(
            x,
            _wasserstein_shift_series(x.to_numpy(dtype=float), ws, wl, mp),
        )
@register_operator(
    name="ts_ks_shift",
    category="time_series_distribution",
    business_category="time_series_distribution",
    canonical="ts_ks_shift",
    source="alpha_language_distribution")
class TsKsShift(SeriesOperator):
    """KS 位移: sup_z |F_A(z) - F_B(z)|, 范围 [0,1]。

    衡量 CDF 最大变化(分布形状改变); 与 Wasserstein(移动多远)互补。
    """

    metadata = _metadata(
        "ts_ks_shift",
        "近窗口 vs 旧窗口 KS 统计量。",
        ["x", "recent_window", "old_window", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(
        self, x: pd.DataFrame, recent_window: int = 20, old_window: int = 40, min_periods: int = 5, **_: Any
    ) -> pd.DataFrame:
        ws = check_window(recent_window, name="recent_window")
        wl = check_window(old_window, name="old_window")
        mp = max(3, int(min_periods))
        xv = x.to_numpy(dtype=float)

        def _fn(recent: np.ndarray, old: np.ndarray) -> float:
            ra = np.sort(valid_values(recent))
            oa = np.sort(valid_values(old))
            if ra.size < mp or oa.size < mp:
                return np.nan
            combined = np.unique(np.concatenate([ra, oa]))
            ecdf_a = np.searchsorted(ra, combined, side="right") / ra.size
            ecdf_b = np.searchsorted(oa, combined, side="right") / oa.size
            return float(np.max(np.abs(ecdf_a - ecdf_b)))

        return frame_like(x, map_two_window(xv, ws, wl, _fn))


@register_operator(
    name="ts_location_shift",
    category="time_series_distribution",
    business_category="time_series_distribution",
    canonical="ts_location_shift",
    source="alpha_language_distribution")
class TsLocationShift(SeriesOperator):
    """位置位移: (median(A) - median(B)) / (MAD(B) + eps)。稳健的水平移动。"""

    metadata = _metadata(
        "ts_location_shift",
        "近窗口 vs 旧窗口的中位数位移(按旧窗口 MAD 归一)。",
        ["x", "recent_window", "old_window", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(
        self, x: pd.DataFrame, recent_window: int = 20, old_window: int = 40, min_periods: int = 5, **_: Any
    ) -> pd.DataFrame:
        ws = check_window(recent_window, name="recent_window")
        wl = check_window(old_window, name="old_window")
        mp = max(3, int(min_periods))
        return frame_like(
            x,
            _location_shift_series(x.to_numpy(dtype=float), ws, wl, mp),
        )
@register_operator(
    name="ts_scale_shift",
    category="time_series_distribution",
    business_category="time_series_distribution",
    canonical="ts_scale_shift",
    source="alpha_language_distribution")
class TsScaleShift(SeriesOperator):
    """尺度位移: log((MAD(A) + eps) / (MAD(B) + eps))。波动/离散度的换挡。"""

    metadata = _metadata(
        "ts_scale_shift",
        "近窗口 vs 旧窗口 MAD 之比的对数。",
        ["x", "recent_window", "old_window", "min_periods"],
        unit="log",
    )

    def _calculate_series(
        self, x: pd.DataFrame, recent_window: int = 20, old_window: int = 40, min_periods: int = 5, **_: Any
    ) -> pd.DataFrame:
        ws = check_window(recent_window, name="recent_window")
        wl = check_window(old_window, name="old_window")
        mp = max(3, int(min_periods))
        xv = x.to_numpy(dtype=float)

        def _fn(recent: np.ndarray, old: np.ndarray) -> float:
            ra = valid_values(recent)
            oa = valid_values(old)
            if ra.size < mp or oa.size < mp:
                return np.nan
            mad_r = _mad(ra)
            mad_o = _mad(oa)
            if (not np.isfinite(mad_r) or mad_r < _EPS) or (not np.isfinite(mad_o) or mad_o < _EPS):
                return np.nan
            return float(np.log(mad_r / mad_o))

        return frame_like(x, map_two_window(xv, ws, wl, _fn))


_Q_GRID = np.linspace(0.1, 0.9, 9)


def _quantile_transport_fit(recent: np.ndarray, old: np.ndarray) -> tuple[float, float, float] | None:
    """ΔQ(q) = Q_recent(q) - Q_old(q) on a fixed grid, fit a + b·z + c·z² (z=q-0.5).

    Returns ``(a, b, c)`` (location shift, transport slope, transport curvature)
    or None when either window is too small / degenerate."""
    ra = valid_values(recent)
    oa = valid_values(old)
    if ra.size < 3 or oa.size < 3:
        return None
    qr = np.quantile(ra, _Q_GRID)
    qo = np.quantile(oa, _Q_GRID)
    dq = qr - qo
    z = _Q_GRID - 0.5
    if float(np.sum(z * z)) <= _EPS:
        return None
    # OLS on [1, z, z^2] (well conditioned, 9 points).
    A = np.column_stack([np.ones_like(z), z, z * z])
    try:
        coef, *_ = np.linalg.lstsq(A, dq, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(coef)):
        return None
    return float(coef[0]), float(coef[1]), float(coef[2])


@register_operator(
    name="ts_quantile_transport_slope",
    category="time_series_distribution",
    business_category="time_series_distribution",
    canonical="ts_quantile_transport_slope",
    source="alpha_language_distribution")
class TsQuantileTransportSlope(SeriesOperator):
    """分位数输运斜率 b：ΔQ(q) = a + b·(q-0.5) + c·(q-0.5)²。

    b>0 = 上尾向右扩、下尾收缩（distribution spreading）；b<0 = 尾部向中心收
    缩。a（整体位置移动）已由 ``ts_location_shift`` 覆盖，这里只注册斜率。
    固定 q 网格 [0.1..0.9]，确定性、PIT 安全。
    """

    metadata = _metadata(
        "ts_quantile_transport_slope",
        "ΔQ 对 q-0.5 的 OLS 斜率（>0 尾部扩散 / <0 收缩）。",
        ["x", "recent_window", "old_window", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(
        self, x: pd.DataFrame, recent_window: int = 20, old_window: int = 40, min_periods: int = 3, **_: Any
    ) -> pd.DataFrame:
        ws = check_window(recent_window, name="recent_window")
        wl = check_window(old_window, name="old_window")
        mp = max(3, int(min_periods))
        return frame_like(
            x,
            _transport_series(x.to_numpy(dtype=float), ws, wl, mp, 1),
        )
@register_operator(
    name="ts_quantile_transport_curvature",
    category="time_series_distribution",
    business_category="time_series_distribution",
    canonical="ts_quantile_transport_curvature",
    source="alpha_language_distribution")
class TsQuantileTransportCurvature(SeriesOperator):
    """分位数输运曲率 c：ΔQ(q) = a + b·(q-0.5) + c·(q-0.5)²。

    c 捕捉输运位移是否集中在双尾还是中部：c>0 = 位移在尾部更大（双尾外扩），
    c<0 = 位移在中部分布移动。真正意义上的 distribution-shape migration。
    与 ``ts_quantile_transport_slope`` 共享同一拟合。
    """

    metadata = _metadata(
        "ts_quantile_transport_curvature",
        "ΔQ 对 (q-0.5)² 的 OLS 曲率（尾部集中位移）。",
        ["x", "recent_window", "old_window", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(
        self, x: pd.DataFrame, recent_window: int = 20, old_window: int = 40, min_periods: int = 3, **_: Any
    ) -> pd.DataFrame:
        ws = check_window(recent_window, name="recent_window")
        wl = check_window(old_window, name="old_window")
        mp = max(3, int(min_periods))
        return frame_like(
            x,
            _transport_series(x.to_numpy(dtype=float), ws, wl, mp, 2),
        )
def _mmd_rbf(recent: np.ndarray, old: np.ndarray, min_periods: int) -> float:
    """MMD² with an RBF kernel; σ = median heuristic over the combined sample.

    Deterministic (no GP search of σ), scale-aware, and fails closed on
    degenerate/constant windows where every kernel value is 1."""
    ra = valid_values(recent)
    oa = valid_values(old)
    if ra.size < min_periods or oa.size < min_periods:
        return np.nan
    comb = np.concatenate([ra, oa])
    m = comb.size
    if m < 4:
        return np.nan
    d = comb[:, None] - comb[None, :]
    triu = np.abs(d[np.triu_indices(m, k=1)])
    if triu.size == 0:
        return np.nan
    sigma = float(np.median(triu))
    if not np.isfinite(sigma) or sigma < _EPS:
        return np.nan
    tau = 2.0 * sigma * sigma

    def _k(a: np.ndarray, b: np.ndarray) -> float:
        diff = a[:, None] - b[None, :]
        return float(np.exp(-(diff * diff) / tau).mean())

    kxx = _k(ra, ra)
    kyy = _k(oa, oa)
    kxy = _k(ra, oa)
    mmd2 = kxx + kyy - 2.0 * kxy
    return float(max(mmd2, 0.0))


@register_operator(
    name="ts_mmd_rbf_shift",
    category="time_series_distribution",
    business_category="time_series_distribution",
    canonical="ts_mmd_rbf_shift",
    source="alpha_language_distribution")
class TsMmdRbfShift(SeriesOperator):
    """近 vs 旧窗口的 RBF-kernel MMD²（最大均值差异）。

    ``MMD² = E[k(X,X')] + E[k(Y,Y')] - 2E[k(X,Y)]``；σ 用合并样本的 pairwise
    距离中位数（median heuristic），**不做 GP 搜索**（确定性）。能捕捉任意
    光滑分布差异（不只是均值/方差），是 Energy/Wasserstein 的补充。常量窗口
    （σ≈0）fail-closed → NaN。
    """

    metadata = _metadata(
        "ts_mmd_rbf_shift",
        "RBF-kernel MMD²（σ=median heuristic，确定性）。",
        ["x", "recent_window", "old_window", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(
        self, x: pd.DataFrame, recent_window: int = 20, old_window: int = 40, min_periods: int = 5, **_: Any
    ) -> pd.DataFrame:
        ws = check_window(recent_window, name="recent_window")
        wl = check_window(old_window, name="old_window")
        mp = max(4, int(min_periods))
        return frame_like(
            x,
            _mmd_rbf_shift_series(x.to_numpy(dtype=float), ws, wl, mp),
        )
_NEW_CANONICALS = (
    "ts_tail_imbalance",
    "ts_expected_shortfall_asymmetry",
    "ts_wasserstein_shift",
    "ts_ks_shift",
    "ts_location_shift",
    "ts_scale_shift",
    "ts_quantile_transport_slope",
    "ts_quantile_transport_curvature",
    "ts_mmd_rbf_shift",
)


def _register_surface() -> None:
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
