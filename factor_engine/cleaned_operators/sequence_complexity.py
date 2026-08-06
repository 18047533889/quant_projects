# -*- coding: utf-8 -*-
"""Sequence complexity, long-memory and roughness operators (2026-08 pack, group 2).

Adds permutation entropy (plain / weighted / transition), sample entropy, DFA
Hurst exponent, Higuchi fractal dimension, variogram slope and autocorrelation
decay half-life.  All are causal daily-panel transforms.

Computational guards (fail-closed)
----------------------------------
* Windows are capped (sample entropy <= 120, Hurst <= 512) so auto-mining cannot
  generate uncontrollable O(n^2) kernels.
* Missing values break order-dependent estimators: embeddings/vectors containing
  NaN are dropped, or the longest contiguous finite run is used — values are
  never re-connected across a gap.
* Constant windows, no matches and degenerate fits return NaN (never Inf).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import (
    check_window,
    frame_like,
    map_rolling,
    register_polars_bridge,
    valid_values,
)


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="complexity",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "complexity", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", "domain:complexity",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _longest_finite_run(chunk: np.ndarray) -> np.ndarray:
    """Longest contiguous finite run (never re-connects across a gap)."""
    best: list[float] = []
    current: list[float] = []
    for value in chunk:
        if np.isfinite(value):
            current.append(float(value))
        else:
            if len(current) > len(best):
                best = current
            current = []
    if len(current) > len(best):
        best = current
    return np.asarray(best, dtype=float)


def _permutation_pattern(values: np.ndarray) -> int:
    """Stable ordinal permutation pattern of a finite embedding."""
    order = np.argsort(values, kind="stable")
    pattern = np.argsort(order, kind="stable")
    code = 0
    for rank in pattern:
        code = code * (len(values) + 1) + int(rank)
    return code


def _permutation_codes(chunk: np.ndarray, order: int, delay: int) -> list[int]:
    """Valid permutation codes over the raw window (NaN embeddings dropped)."""
    codes: list[int] = []
    embed_len = (order - 1) * delay + 1
    n = len(chunk)
    for i in range(n - embed_len + 1):
        idx = [i + d * delay for d in range(order)]
        vals = chunk[idx]
        if np.isfinite(vals).all():
            codes.append(_permutation_pattern(vals))
    return codes


def _entropy_from_counts(counts: list[int], total: int, normalize: int | None) -> float:
    if total < 2:
        return np.nan
    value = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        value -= p * math.log(p)
    if normalize is not None and normalize > 1:
        value /= math.log(normalize)
    return float(value)


@register_operator(
    name="ts_permutation_entropy",
    category="complexity",
    business_category="sequence_complexity",
    canonical="ts_permutation_entropy",
    source="sequence_complexity",
)
class TsPermutationEntropy(SeriesOperator):
    """排列熵：窗口内连续嵌入排列模式的香农熵，normalize=True 除以 log(order!)。"""

    metadata = _metadata(
        "ts_permutation_entropy",
        "排列熵（order 阶嵌入排列模式熵）。",
        ["x", "window", "order", "delay", "normalize", "min_patterns"],
        unit="level",
        cost=5,
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, order: int = 3, delay: int = 1, normalize: bool = True, min_patterns: Any = None, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        ord_ = int(order)
        if not 2 <= ord_ <= 6:
            raise ValueError("order must be in [2, 6]")
        dl = int(delay)
        if dl < 1:
            raise ValueError("delay must be >= 1")
        norm = bool(normalize)
        min_p = 2 if min_patterns is None else max(2, int(min_patterns))

        def _fn(chunk: np.ndarray) -> float:
            codes = _permutation_codes(chunk, ord_, dl)
            if len(codes) < min_p:
                return np.nan
            counts: dict[int, int] = {}
            for c in codes:
                counts[c] = counts.get(c, 0) + 1
            return _entropy_from_counts(list(counts.values()), len(codes), math.factorial(ord_) if norm else None)

        return frame_like(x, map_rolling(x.to_numpy(dtype=float), w, _fn))


def _embedding_variance(values: np.ndarray) -> float:
    return float(np.var(values, ddof=0)) if len(values) > 1 else 0.0


def _embedding_range(values: np.ndarray) -> float:
    return float(np.max(values) - np.min(values))


@register_operator(
    name="ts_weighted_permutation_entropy",
    category="complexity",
    business_category="sequence_complexity",
    canonical="ts_weighted_permutation_entropy",
    source="sequence_complexity",
)
class TsWeightedPermutationEntropy(SeriesOperator):
    """加权排列熵：按嵌入振幅（方差/极差）加权，权重只作用于当前窗口。"""

    metadata = _metadata(
        "ts_weighted_permutation_entropy",
        "加权排列熵（weight='variance' 或 'range'）。",
        ["x", "window", "order", "delay", "weight", "normalize"],
        unit="level",
        cost=5,
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, order: int = 3, delay: int = 1, weight: str = "variance", normalize: bool = True, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        ord_ = int(order)
        if not 2 <= ord_ <= 6:
            raise ValueError("order must be in [2, 6]")
        dl = int(delay)
        if dl < 1:
            raise ValueError("delay must be >= 1")
        weight_kind = str(weight).lower()
        if weight_kind not in {"variance", "range"}:
            raise ValueError("weight must be 'variance' or 'range'")
        norm = bool(normalize)

        def _fn(chunk: np.ndarray) -> float:
            embed_len = (ord_ - 1) * dl + 1
            n = len(chunk)
            weights: list[float] = []
            codes: list[int] = []
            for i in range(n - embed_len + 1):
                idx = [i + d * dl for d in range(ord_)]
                vals = chunk[idx]
                if not np.isfinite(vals).all():
                    continue
                codes.append(_permutation_pattern(vals))
                weights.append(_embedding_variance(vals) if weight_kind == "variance" else _embedding_range(vals))
            if len(codes) < 2:
                return np.nan
            total_weight = float(sum(weights))
            if total_weight <= 0.0:
                return np.nan
            pattern_weight: dict[int, float] = {}
            for code, wgt in zip(codes, weights):
                pattern_weight[code] = pattern_weight.get(code, 0.0) + wgt
            value = 0.0
            for wgt in pattern_weight.values():
                p = wgt / total_weight
                if p > 0:
                    value -= p * math.log(p)
            if norm:
                value /= math.log(math.factorial(ord_))
            return float(value)

        return frame_like(x, map_rolling(x.to_numpy(dtype=float), w, _fn))


@register_operator(
    name="ts_permutation_transition_entropy",
    category="complexity",
    business_category="sequence_complexity",
    canonical="ts_permutation_transition_entropy",
    source="sequence_complexity",
)
class TsPermutationTransitionEntropy(SeriesOperator):
    """排列转移熵：相邻排列状态的转移条件熵，normalize 除以 log(状态数)。"""

    metadata = _metadata(
        "ts_permutation_transition_entropy",
        "排列转移熵（相邻排列转移条件熵）。",
        ["x", "window", "order", "delay", "normalize"],
        unit="level",
        cost=5,
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, order: int = 3, delay: int = 1, normalize: bool = True, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        ord_ = int(order)
        if not 2 <= ord_ <= 6:
            raise ValueError("order must be in [2, 6]")
        dl = int(delay)
        if dl < 1:
            raise ValueError("delay must be >= 1")
        norm = bool(normalize)

        def _fn(chunk: np.ndarray) -> float:
            codes = _permutation_codes(chunk, ord_, dl)
            if len(codes) < 3:
                return np.nan
            states = sorted(set(codes))
            if len(states) < 2:
                return np.nan
            state_index = {state: i for i, state in enumerate(states)}
            transitions = [[0.0] * len(states) for _ in range(len(states))]
            for prev, cur in zip(codes, codes[1:]):
                transitions[state_index[prev]][state_index[cur]] += 1.0
            cond = 0.0
            for row in transitions:
                row_sum = float(sum(row))
                if row_sum <= 0:
                    continue
                p_cur = row_sum / (len(codes) - 1)
                for count in row:
                    if count <= 0:
                        continue
                    p = count / row_sum
                    cond -= p_cur * p * math.log(p)
            if norm:
                cond /= math.log(len(states))
            return float(cond)

        return frame_like(x, map_rolling(x.to_numpy(dtype=float), w, _fn))


def _sample_entropy(run: np.ndarray, m: int, r: float) -> float:
    n = len(run)
    if n < m + 2:
        return np.nan
    # Count (m+1) and m matches under Chebyshev distance without self-matches.
    matches_m_plus_1 = 0
    matches_m = 0
    for i in range(n - m - 1):
        vi = run[i : i + m + 1]
        for j in range(i + 1, n - m):
            vj = run[j : j + m + 1]
            if np.max(np.abs(vi - vj)) <= r:
                matches_m_plus_1 += 1
    for i in range(n - m):
        vi = run[i : i + m]
        for j in range(i + 1, n - m + 1):
            vj = run[j : j + m]
            if np.max(np.abs(vi - vj)) <= r:
                matches_m += 1
    if matches_m == 0 or matches_m_plus_1 == 0:
        return np.nan
    return -math.log(matches_m_plus_1 / matches_m)


@register_operator(
    name="ts_sample_entropy",
    category="complexity",
    business_category="sequence_complexity",
    canonical="ts_sample_entropy",
    source="sequence_complexity",
)
class TsSampleEntropy(SeriesOperator):
    """样本熵：容差 = tolerance_scale × 窗口标准差；无匹配对或常数窗口返回 NaN。window 上限 120。"""

    metadata = _metadata(
        "ts_sample_entropy",
        "样本熵（Chebyshev 匹配，无匹配对返回 NaN）。",
        ["x", "window", "embedding_dim", "tolerance_scale", "min_periods"],
        unit="level",
        cost=7,
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, embedding_dim: int = 2, tolerance_scale: float = 0.2, min_periods: Any = None, **_: Any) -> pd.DataFrame:
        w = int(window)
        if not 2 <= w <= 120:
            raise ValueError("window must be in [2, 120]")
        m = int(embedding_dim)
        if not 1 <= m <= 4:
            raise ValueError("embedding_dim must be in [1, 4]")
        tol = float(tolerance_scale)
        if tol <= 0.0:
            raise ValueError("tolerance_scale must be > 0")
        min_p = (m + 2) if min_periods is None else max(m + 2, int(min_periods))

        def _fn(chunk: np.ndarray) -> float:
            run = _longest_finite_run(chunk)
            if run.size < min_p:
                return np.nan
            std = float(np.std(run, ddof=0))
            if std <= 1e-12:
                return np.nan
            return _sample_entropy(run, m, tol * std)

        return frame_like(x, map_rolling(x.to_numpy(dtype=float), w, _fn))


def _dfa_hurst(run: np.ndarray, min_scale: int, max_scale: int, n_scales: int) -> float:
    n = len(run)
    y = np.cumsum(run - np.mean(run))
    scales = np.unique(np.rint(np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales)).astype(int))
    scales = [s for s in scales if 2 <= s <= n // 2]
    if len(scales) < 2:
        return np.nan
    log_s: list[float] = []
    log_f: list[float] = []
    for s in scales:
        n_boxes = n // s
        if n_boxes < 2:
            continue
        residual_sq: list[float] = []
        for b in range(n_boxes):
            seg = y[b * s : (b + 1) * s]
            xs = np.arange(len(seg), dtype=float)
            coeff = np.polyfit(xs, seg, 1)
            trend = np.polyval(coeff, xs)
            residual_sq.append(np.mean((seg - trend) ** 2))
        f = float(np.sqrt(np.mean(residual_sq)))
        if f <= 1e-12:
            continue
        log_s.append(math.log(float(s)))
        log_f.append(math.log(f))
    if len(log_s) < 2:
        return np.nan
    slope, _ = np.polyfit(log_s, log_f, 1)
    return float(slope)


@register_operator(
    name="ts_hurst_dfa",
    category="complexity",
    business_category="sequence_complexity",
    canonical="ts_hurst_dfa",
    source="sequence_complexity",
)
class TsHurstDfa(SeriesOperator):
    """去趋势波动分析 Hurst 指数：log F(s) 对 log s 的斜率。window 上限 512。"""

    metadata = _metadata(
        "ts_hurst_dfa",
        "DFA Hurst 指数（累积去趋势均方根斜率）。",
        ["x", "window", "min_scale", "max_scale", "n_scales"],
        unit="level",
        cost=7,
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 120, min_scale: int = 4, max_scale: Any = None, n_scales: int = 6, **_: Any) -> pd.DataFrame:
        w = int(window)
        if not 2 <= w <= 512:
            raise ValueError("window must be in [2, 512]")
        min_s = max(2, int(min_scale))
        max_s = w // 4 if max_scale is None else int(max_scale)
        if max_s < min_s:
            max_s = min_s + 1
        ns = max(3, int(n_scales))

        def _fn(chunk: np.ndarray) -> float:
            run = _longest_finite_run(chunk)
            if run.size < max_s * 2 + 4:
                return np.nan
            return _dfa_hurst(run, min_s, min(max_s, run.size // 2), ns)

        return frame_like(x, map_rolling(x.to_numpy(dtype=float), w, _fn))


def _higuchi_fd(run: np.ndarray, k_max: int) -> float:
    n = len(run)
    log_k: list[float] = []
    log_l: list[float] = []
    for k in range(1, k_max + 1):
        lengths: list[float] = []
        for m in range(k):
            idx = list(range(m, n, k))
            if len(idx) < 2:
                continue
            diff_sum = 0.0
            for i in range(1, len(idx)):
                diff_sum += abs(run[idx[i]] - run[idx[i - 1]])
            norm = (n - 1) / (len(idx) * k)
            lengths.append(diff_sum * norm / k)
        if not lengths:
            continue
        lk = float(np.mean(lengths))
        if lk <= 1e-12:
            continue
        log_k.append(math.log(1.0 / k))
        log_l.append(math.log(lk))
    if len(log_k) < 2:
        return np.nan
    slope, _ = np.polyfit(log_k, log_l, 1)
    return float(slope)


@register_operator(
    name="ts_higuchi_fractal_dimension",
    category="complexity",
    business_category="sequence_complexity",
    canonical="ts_higuchi_fractal_dimension",
    source="sequence_complexity",
)
class TsHiguchiFractalDimension(SeriesOperator):
    """Higuchi 分形维数：log L(k) 对 log(1/k) 的斜率，衡量路径粗糙度。"""

    metadata = _metadata(
        "ts_higuchi_fractal_dimension",
        "Higuchi 分形维数（路径粗糙度）。",
        ["x", "window", "k_max"],
        unit="level",
        cost=7,
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 120, k_max: int = 8, **_: Any) -> pd.DataFrame:
        w = int(window)
        if not 2 <= w <= 512:
            raise ValueError("window must be in [2, 512]")
        km = int(k_max)
        if not 1 <= km <= 32:
            raise ValueError("k_max must be in [1, 32]")

        def _fn(chunk: np.ndarray) -> float:
            run = _longest_finite_run(chunk)
            if run.size < 2 * km + 2:
                return np.nan
            return _higuchi_fd(run, km)

        return frame_like(x, map_rolling(x.to_numpy(dtype=float), w, _fn))


def _variogram_slope(chunk: np.ndarray, max_lag: int, min_valid_lags: int) -> float:
    n = len(chunk)
    max_lag = min(max_lag, n - 1)
    log_lag: list[float] = []
    log_var: list[float] = []
    for k in range(1, max_lag + 1):
        diffs = chunk[k:] - chunk[: n - k]
        finite = diffs[np.isfinite(diffs)]
        if finite.size < 2:
            continue
        var = float(np.mean(finite * finite))
        if var <= 1e-12:
            continue
        log_lag.append(math.log(float(k)))
        log_var.append(math.log(var))
    if len(log_lag) < max(2, int(min_valid_lags)):
        return np.nan
    slope, _ = np.polyfit(log_lag, log_var, 1)
    return float(slope)


@register_operator(
    name="ts_variogram_slope",
    category="complexity",
    business_category="sequence_complexity",
    canonical="ts_variogram_slope",
    source="sequence_complexity",
)
class TsVariogramSlope(SeriesOperator):
    """变差函数斜率：log E[(x[t+k]-x[t])²] 对 log k 的斜率（尺度结构）。"""

    metadata = _metadata(
        "ts_variogram_slope",
        "变差函数斜率（对数变差对对数滞后回归）。",
        ["x", "window", "max_lag", "min_valid_lags"],
        unit="level",
        cost=3,
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, max_lag: int = 10, min_valid_lags: int = 4, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        ml = int(max_lag)
        if not 1 <= ml <= 30:
            raise ValueError("max_lag must be in [1, 30]")
        mvl = max(2, int(min_valid_lags))

        def _fn(chunk: np.ndarray) -> float:
            return _variogram_slope(chunk, ml, mvl)

        return frame_like(x, map_rolling(x.to_numpy(dtype=float), w, _fn))


def _autocorr_half_life(chunk: np.ndarray, max_lag: int, use_abs: bool, min_periods: int) -> float:
    n = len(chunk)
    max_lag = min(max_lag, n - 1)
    log_ac: list[float] = []
    lags: list[float] = []
    for k in range(1, max_lag + 1):
        x0 = chunk[: n - k]
        xk = chunk[k:]
        finite = np.isfinite(x0) & np.isfinite(xk)
        if int(finite.sum()) < max(2, int(min_periods)):
            continue
        a = x0[finite]
        b = xk[finite]
        var = float(np.var(a, ddof=0))
        if var <= 1e-12:
            continue
        ac = float(np.cov(a, b, ddof=0)[0, 1] / var)
        if not np.isfinite(ac):
            continue
        if use_abs:
            ac = abs(ac)
        if ac <= 1e-12:
            continue
        log_ac.append(math.log(ac))
        lags.append(float(k))
    if len(lags) < 2:
        return np.nan
    slope, _ = np.polyfit(lags, log_ac, 1)
    if slope >= 0.0:
        return np.nan
    return float(-math.log(2.0) / slope)


@register_operator(
    name="ts_autocorr_decay_half_life",
    category="complexity",
    business_category="sequence_complexity",
    canonical="ts_autocorr_decay_half_life",
    source="sequence_complexity",
)
class TsAutocorrDecayHalfLife(SeriesOperator):
    """自相关衰减半衰期：拟合多阶滞后自相关指数衰减，无衰减返回 NaN。"""

    metadata = _metadata(
        "ts_autocorr_decay_half_life",
        "自相关衰减半衰期（log ac 对 lag 回归，slope>=0 返回 NaN）。",
        ["x", "window", "max_lag", "use_abs", "min_periods"],
        unit="count",
        cost=2,
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, max_lag: int = 10, use_abs: bool = False, min_periods: int = 2, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        ml = int(max_lag)
        if not 1 <= ml <= 30:
            raise ValueError("max_lag must be in [1, 30]")
        mp = max(2, int(min_periods))
        abs_ = bool(use_abs)

        def _fn(chunk: np.ndarray) -> float:
            return _autocorr_half_life(chunk, ml, abs_, mp)

        return frame_like(x, map_rolling(x.to_numpy(dtype=float), w, _fn))


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS)
        | {
            "ts_permutation_entropy", "ts_weighted_permutation_entropy",
            "ts_permutation_transition_entropy", "ts_sample_entropy",
            "ts_hurst_dfa", "ts_higuchi_fractal_dimension",
            "ts_variogram_slope", "ts_autocorr_decay_half_life",
        }
    )
    # ``ts_permutation_entropy`` / ``ts_sample_entropy`` are also registered by
    # the ts_model.complexity research family (loaded earlier).  This module is
    # the reviewed production implementation, so pull those names off the
    # research surface to keep the extended/research partitions disjoint.
    _surface.RESEARCH_ONLY_CANONICALS = frozenset(
        set(_surface.RESEARCH_ONLY_CANONICALS)
        - {"ts_permutation_entropy", "ts_sample_entropy"}
    )
    for _canon in (
        "ts_permutation_entropy", "ts_weighted_permutation_entropy",
        "ts_permutation_transition_entropy", "ts_sample_entropy",
        "ts_hurst_dfa", "ts_higuchi_fractal_dimension",
        "ts_variogram_slope", "ts_autocorr_decay_half_life",
    ):
        register_polars_bridge(_canon)


_register_surface()
