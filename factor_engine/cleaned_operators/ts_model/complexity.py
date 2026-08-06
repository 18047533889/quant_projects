# -*- coding: utf-8 -*-
"""Information complexity, structural-break and regime operators (P2).

Entropy / complexity / Hurst kernels and causal online change-point / regime
probabilities.  All are experimental high-cost operators.
"""
from __future__ import annotations

from itertools import permutations
from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import SeriesOperator, register_operator
from cleaned_operators.ts_model._rolling_core import frame_like, metadata

_CANONICALS: list[str] = []


def _register(name: str, description: str, params: list[str], unit: str, fn, cost: int = 8):
    @register_operator(
        name=name,
        category="time_series_regression",
        business_category="time_series_regression",
        canonical=name,
        source="ts_model.complexity",
        backend="pandas_numpy",
        status="experimental",
    )
    class _ComplexityOp(SeriesOperator):
        metadata = metadata(name, description, params, unit=unit, cost=cost)

        def _calculate_series(self, *args, **kwargs):
            return fn(*args, **kwargs)

    _CANONICALS.append(name)
    import cleaned_operators.operator_surface as _surface

    _surface.RESEARCH_ONLY_CANONICALS = frozenset(
        set(_surface.RESEARCH_ONLY_CANONICALS) | {name}
    )
    return _ComplexityOp


def _apply(x: pd.DataFrame, fn) -> pd.DataFrame:
    xv = x.to_numpy(dtype=float)
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            out[row, col] = fn(xv[: row + 1, col])
    return frame_like(x, out)


def _permutation_entropy(vals: np.ndarray, order: int, window: int) -> float:
    seg = vals[-int(window):]
    finite = seg[np.isfinite(seg)]
    if len(finite) < order + 1:
        return np.nan
    n = len(finite)
    counts: dict[tuple[int, ...], int] = {}
    for i in range(n - order + 1):
        block = finite[i : i + order]
        rank = tuple(np.argsort(np.argsort(block)))
        counts[rank] = counts.get(rank, 0) + 1
    total = sum(counts.values())
    if total <= 1:
        return np.nan
    h = -sum(c / total * np.log(c / total) for c in counts.values())
    return float(h / np.log(len(list(permutations(range(order)))))) if order > 1 else 0.0


_register("ts_permutation_entropy", "序模式排列熵（归一化）。", ["x", "order", "window"], "level",
           lambda x, order=4, window=120: _apply(x, lambda v: _permutation_entropy(v, int(order), int(window))))


def _sample_entropy(vals: np.ndarray, m: int, r_scale: float, window: int) -> float:
    seg = vals[-int(window):]
    finite = seg[np.isfinite(seg)]
    if len(finite) < m + 2:
        return np.nan
    r = float(r_scale) * float(np.std(finite))
    if r <= 0:
        return np.nan
    n = len(finite)

    def _count(pattern_len: int) -> int:
        count = 0
        for i in range(n - pattern_len):
            for j in range(i + 1, n - pattern_len + 1):
                d = np.max(np.abs(finite[i : i + pattern_len] - finite[j : j + pattern_len]))
                if d < r:
                    count += 1
        return count

    b = _count(m)
    a = _count(m + 1)
    if b == 0:
        return np.nan
    return float(-np.log(a / b)) if a > 0 else float(np.log(b))


_register("ts_sample_entropy", "样本熵（相似子序列继续保持相似的概率）。", ["x", "m", "r", "window"], "level",
           lambda x, m=2, r=0.2, window=200: _apply(x, lambda v: _sample_entropy(v, int(m), float(r), int(window))), cost=9)


def _lz_complexity(vals: np.ndarray, window: int, bins: int) -> float:
    seg = vals[-int(window):]
    finite = seg[np.isfinite(seg)]
    if len(finite) < 16:
        return np.nan
    # discretize to bins
    q = np.quantile(finite, np.linspace(0, 1, int(bins) + 1)[1:-1])
    symbols = np.digitize(finite, q).astype(int)
    s = "".join(chr(65 + v) for v in symbols)
    c = 1
    l = len(s)
    i = 0
    k = 1
    k_max = 1
    while True:
        if i + k > l:
            break
        if s[i : i + k] in s[: i + k - 1]:
            k += 1
            if k > k_max:
                k_max = k
        else:
            c += 1
            i += k_max
            k = 1
            k_max = 1
    return float(c * np.log2(l) / l) if l > 1 else np.nan


_register("ts_lz_complexity", "Lempel-Ziv 复杂度（符号化）。", ["x", "window", "bins"], "level",
           lambda x, window=200, bins=8: _apply(x, lambda v: _lz_complexity(v, int(window), int(bins))), cost=8)


def _multiscale_entropy_slope(vals: np.ndarray, max_scale: int, window: int) -> float:
    seg = vals[-int(window):]
    finite = seg[np.isfinite(seg)]
    if len(finite) < 16:
        return np.nan
    scales: list[int] = []
    ents: list[float] = []
    for s in range(1, max(2, int(max_scale)) + 1):
        if len(finite) < s * 3:
            continue
        coarse = finite[: len(finite) // s * s].reshape(-1, s).mean(axis=1)
        e = _permutation_entropy(coarse, 3, len(coarse))
        if np.isfinite(e):
            scales.append(float(s))
            ents.append(e)
    if len(scales) < 2 or np.var(scales) <= 0:
        return np.nan
    return float(np.cov(scales, ents)[0, 1] / np.var(scales))


_register("ts_multiscale_entropy_slope", "多尺度熵相对尺度的斜率。", ["x", "max_scale", "window"], "level",
           lambda x, max_scale=5, window=300: _apply(x, lambda v: _multiscale_entropy_slope(v, int(max_scale), int(window))), cost=9)


def _turning_point_ratio(vals: np.ndarray, window: int) -> float:
    seg = vals[-int(window):]
    finite = seg[np.isfinite(seg)]
    if len(finite) < 4:
        return np.nan
    d = np.diff(finite)
    turns = 0
    for i in range(1, len(d)):
        if d[i] * d[i - 1] < 0:
            turns += 1
    return float(turns / (len(d) - 1))


_register("ts_turning_point_ratio", "局部方向反转次数比例。", ["x", "window"], "ratio",
           lambda x, window=60: _apply(x, lambda v: _turning_point_ratio(v, int(window))), cost=3)


def _dfa_hurst(vals: np.ndarray, window: int, min_scale: int, max_scale: int) -> float:
    seg = vals[-int(window):]
    finite = seg[np.isfinite(seg)]
    if len(finite) < max(32, max_scale * 4):
        return np.nan
    y = np.cumsum(finite - np.mean(finite))
    n = len(y)
    scales: list[int] = []
    f: list[float] = []
    for s in range(max(4, int(min_scale)), min(int(max_scale), n // 4) + 1):
        n_seg = n // s
        if n_seg < 2:
            continue
        fluct = 0.0
        for v in range(n_seg):
            idx = slice(v * s, (v + 1) * s)
            seg_y = y[idx]
            coef = np.polyfit(np.arange(s), seg_y, 1)
            trend = np.polyval(coef, np.arange(s))
            fluct += np.mean((seg_y - trend) ** 2)
        fluct = np.sqrt(fluct / n_seg)
        f.append(fluct)
        scales.append(np.log(s))
    if len(scales) < 3 or np.var(scales) <= 0:
        return np.nan
    return float(np.cov(scales, np.log(f))[0, 1] / np.var(scales))


_register("ts_dfa_hurst", "DFA 估计 Hurst 指数。", ["x", "window", "min_scale", "max_scale"], "level",
           lambda x, window=250, min_scale=4, max_scale=32: _apply(x, lambda v: _dfa_hurst(v, int(window), int(min_scale), int(max_scale))), cost=9)


def _cusum_vol_break(vals: np.ndarray, window: int, min_periods: int) -> float:
    seg = vals[-int(window):]
    sq = seg ** 2
    valid = np.isfinite(sq)
    if valid.sum() < max(min_periods, 10):
        return np.nan
    v = sq[valid]
    mean = float(np.mean(v))
    sd = float(np.std(v))
    if sd <= 1e-12:
        return np.nan
    running = np.cumsum(v - mean) / (sd * np.sqrt(np.arange(1, len(v) + 1)))
    # The last cumulative sum is mechanically ~0 because the window's total
    # deviation from its own mean is zero; the informative statistic is the
    # largest magnitude reached along the path.
    return float(np.max(np.abs(running)))


_register("ts_cusum_vol_break_score", "平方收益 CUSUM 波动突变得分。", ["x", "window", "min_periods"], "level",
           lambda x, window=60, min_periods=10: _apply(x, lambda v: _cusum_vol_break(v, int(window), int(min_periods))), cost=3)


def _regime_filter(vals: np.ndarray, window: int, stat: str) -> float:
    seg = vals[-int(window):]
    finite = seg[np.isfinite(seg)]
    if len(finite) < 10:
        return np.nan
    # Two-state Gaussian filter on volatility (high/low), deterministic grid.
    r = np.array(finite, dtype=float)
    var_all = float(np.var(r))
    sigma_hi = np.sqrt(max(var_all * 2.0, 1e-12))
    sigma_lo = np.sqrt(max(var_all * 0.5, 1e-12))
    p_hi = 0.5
    p_hi_hist: list[float] = []
    for x in r:
        like_hi = np.exp(-0.5 * (x / sigma_hi) ** 2) / sigma_hi
        like_lo = np.exp(-0.5 * (x / sigma_lo) ** 2) / sigma_lo
        p_hi = (like_hi * p_hi) / max(like_hi * p_hi + like_lo * (1 - p_hi), 1e-300)
        p_hi = min(max(p_hi, 1e-6), 1.0 - 1e-6)
        p_hi_hist.append(p_hi)
    if stat == "prob":
        return float(p_hi_hist[-1])
    if stat == "duration":
        # current run length above 0.5
        d = 0
        for p in reversed(p_hi_hist):
            if p > 0.5:
                d += 1
            else:
                break
        return float(d)
    # change point probability: recent jump in filtered prob
    if len(p_hi_hist) < 5:
        return np.nan
    recent = p_hi_hist[-3:]
    older = p_hi_hist[-10:-3]
    return float(abs(np.mean(recent) - np.mean(older)))


_register("ts_two_state_regime_probability", "两状态高波动过滤概率（仅滤波）。", ["x", "window"], "level",
           lambda x, window=120: _apply(x, lambda v: _regime_filter(v, int(window), "prob")), cost=5)
_register("ts_regime_duration", "当前状态持续期（过滤概率>0.5 的连续长度）。", ["x", "window"], "count",
           lambda x, window=120: _apply(x, lambda v: _regime_filter(v, int(window), "duration")), cost=5)
_register("ts_change_point_probability", "在线状态切换概率（滤波概率近突变度）。", ["x", "window"], "level",
           lambda x, window=120: _apply(x, lambda v: _regime_filter(v, int(window), "changepoint")), cost=5)
