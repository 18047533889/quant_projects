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

from cleaned_operators.base import (
    OperatorMetadata,
    ParamRole,
    ParamSpec,
    SeriesOperator,
    register_operator,
)
from cleaned_operators.closure.strict_scalar import strict_float, strict_int
from cleaned_operators.rolling_pack import (
    check_window,
    frame_like,
    map_rolling,
    register_polars_bridge,
    valid_values,
)


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int, param_specs: dict[str, Any] | None = None) -> OperatorMetadata:
    meta = OperatorMetadata(
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
    if param_specs:
        meta.param_specs = dict(param_specs)
    return meta


# Round-3 cross-cutting: estimator-resolution knobs (order / delay) are never
# an economic search dimension — declaring them ESTIMATOR_RESOLUTION and
# searchable=False keeps the mining grammar from optimizing over the embedding
# resolution instead of a real economic regularity.
_ORDINAL_KNOB_SPECS = {
    "order": ParamSpec(
        dtype=int, min=2, param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False
    ),
    "delay": ParamSpec(
        dtype=int, min=1, param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False
    ),
}
# M-10xx: estimator-resolution knobs (embedding_dim / max_scale / n_scales /
# k_max / max_lag / tolerance_scale) are declared ESTIMATOR_RESOLUTION +
# searchable=False; support floors (min_periods / min_valid_lags /
# min_patterns) are SUPPORT_POLICY — never free search dimensions.
_ESTIMATOR_INT_SPEC = ParamSpec(
    dtype=int, param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False
)
_SUPPORT_INT_SPEC = ParamSpec(
    dtype=int, param_role=ParamRole.SUPPORT_POLICY, searchable=False
)
_ESTIMATOR_FLOAT_SPEC = ParamSpec(
    dtype=float, param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False
)


def _trailing_finite_suffix(chunk: np.ndarray) -> np.ndarray:
    """Trailing contiguous finite suffix ending at the window's last row.

    Complexity estimators must never reach back across a gap into an older
    finite block: after a recent missing value only the trailing contiguous
    finite points are a valid sample for the *current* complexity (review
    R4-64).  Values are never re-connected across a gap.
    """
    n = len(chunk)
    j = n
    while j > 0 and not np.isfinite(chunk[j - 1]):
        j -= 1
    i = j
    while i > 0 and np.isfinite(chunk[i - 1]):
        i -= 1
    return chunk[i:j].astype(float)


def _permutation_pattern(values: np.ndarray) -> tuple[int, ...] | None:
    """Ordinal pattern of a no-tie embedding, or ``None`` when a tie exists.

    Review round-3 #44: ordinal ties DROP the embedding from the
    permutation-count state space.  Average-tie-rank patterns (e.g. (1.5, 1.5,
    3)) are a continuum, NOT a permutation state space, so normalizing by
    log(order!) would be mathematically inconsistent.  A block with any tie is
    excluded entirely.  (The ``tie_fraction`` fail-closed guard in the caller
    still reports windows where ties dominate, so a sparse no-tie sample is
    not silently measured.)
    """
    n = len(values)
    if np.unique(values).size < n:
        return None
    order = np.argsort(values, kind="mergesort")
    pattern = np.argsort(order, kind="mergesort")
    return tuple(int(x) for x in pattern)


# P1-14: stable-index tie-breaking manufactures an artificial ordering for tied
# values (0-return days, limit bars, discrete financials), so the permutation
# entropy of a heavily tied window measures the tie-break rule, not complexity.
# Windows whose embeddings are mostly tied fail closed to NaN.
_TIE_RATIO_FAIL_CLOSED = 0.5


def _permutation_codes(
    chunk: np.ndarray, order: int, delay: int
) -> tuple[list[tuple[tuple[int, ...], int]], float]:
    """Valid no-tie (pattern_key, start_index) embeddings + the tie-drop fraction.

    NaN embeddings are dropped, but the original start index is kept so a
    transition can only be formed between *genuinely adjacent* embeddings
    (audit R02): after an invalid embedding is removed, ``zip(codes[1:])`` must
    not bridge the gap into a transition.  Review round-3 #44: an embedding
    whose ordinal pattern contains a tie is DROPPED (the state space must stay
    a genuine permutation space).  ``tie_fraction`` is the share of all
    non-NaN embeddings dropped for a tie — the fail-closed guard (P1-14) uses
    it so a heavily tied window does not report a sparse no-tie sample.
    """
    codes: list[tuple[tuple[int, ...], int]] = []
    embed_len = (order - 1) * delay + 1
    n = len(chunk)
    n_total = 0
    n_tied = 0
    for i in range(n - embed_len + 1):
        idx = [i + d * delay for d in range(order)]
        vals = chunk[idx]
        if not np.isfinite(vals).all():
            continue
        n_total += 1
        pattern = _permutation_pattern(vals)
        if pattern is None:
            n_tied += 1
            continue
        codes.append((pattern, i))
    tie_fraction = (n_tied / n_total) if n_total > 0 else 1.0
    return codes, tie_fraction


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
    research_only=True,
)
class TsPermutationEntropy(SeriesOperator):
    """排列熵：窗口内连续嵌入排列模式的香农熵，normalize=True 除以 log(order!)。"""

    metadata = _metadata(
        "ts_permutation_entropy",
        "排列熵（order 阶嵌入排列模式熵）。",
        ["x", "window", "order", "delay", "normalize", "min_patterns"],
        unit="level",
        cost=5,
        param_specs=dict(
            _ORDINAL_KNOB_SPECS,
            normalize=ParamSpec(
                dtype=bool, choices=(True, False),
                param_role=ParamRole.POLICY, searchable=False,
            ),
            min_patterns=ParamSpec(
                dtype=int, min=2,
                param_role=ParamRole.SUPPORT_POLICY, searchable=False,
                default=None,
            ),
        ),
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, order: int = 3, delay: int = 1, normalize: bool = True, min_patterns: Any = None, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        ord_ = strict_int(order, "order", lower=2, upper=6)
        dl = strict_int(delay, "delay", lower=1)
        norm = bool(normalize)
        min_p = 2 if min_patterns is None else strict_int(min_patterns, "min_patterns", lower=2)

        def _fn(chunk: np.ndarray) -> float:
            coded, tie_frac = _permutation_codes(chunk, ord_, dl)
            if len(coded) < min_p:
                return np.nan
            if tie_frac > _TIE_RATIO_FAIL_CLOSED:
                return np.nan  # P1-14: ties dominate -> sparse no-tie sample
            counts: dict[tuple[int, ...], int] = {}
            for c, _ in coded:
                counts[c] = counts.get(c, 0) + 1
            return _entropy_from_counts(list(counts.values()), len(coded), math.factorial(ord_) if norm else None)

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
    research_only=True,
)
class TsWeightedPermutationEntropy(SeriesOperator):
    """加权排列熵：按嵌入振幅（方差/极差）加权，权重只作用于当前窗口。"""

    metadata = _metadata(
        "ts_weighted_permutation_entropy",
        "加权排列熵（weight='variance' 或 'range'）。",
        ["x", "window", "order", "delay", "weight", "normalize"],
        unit="level",
        cost=5,
        param_specs=dict(
            _ORDINAL_KNOB_SPECS,
            weight=ParamSpec(
                dtype=str, choices=("variance", "range"),
                param_role=ParamRole.POLICY, searchable=False,
            ),
            normalize=ParamSpec(
                dtype=bool, choices=(True, False),
                param_role=ParamRole.POLICY, searchable=False,
            ),
        ),
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, order: int = 3, delay: int = 1, weight: str = "variance", normalize: bool = True, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        ord_ = strict_int(order, "order", lower=2, upper=6)
        dl = strict_int(delay, "delay", lower=1)
        weight_kind = str(weight).lower()
        if weight_kind not in {"variance", "range"}:
            raise ValueError("weight must be 'variance' or 'range'")
        norm = bool(normalize)

        def _fn(chunk: np.ndarray) -> float:
            embed_len = (ord_ - 1) * dl + 1
            n = len(chunk)
            weights: list[float] = []
            codes: list[tuple[int, ...]] = []
            n_total = 0
            n_tied = 0
            for i in range(n - embed_len + 1):
                idx = [i + d * dl for d in range(ord_)]
                vals = chunk[idx]
                if not np.isfinite(vals).all():
                    continue
                n_total += 1
                pattern = _permutation_pattern(vals)
                if pattern is None:
                    n_tied += 1
                    continue  # review round-3 #44: ordinal tie -> drop embedding
                codes.append(pattern)
                weights.append(_embedding_variance(vals) if weight_kind == "variance" else _embedding_range(vals))
            tie_frac = (n_tied / n_total) if n_total > 0 else 1.0
            if len(codes) < 2:
                return np.nan
            if tie_frac > _TIE_RATIO_FAIL_CLOSED:
                return np.nan  # P1-14: tied values -> sparse no-tie sample
            total_weight = float(sum(weights))
            if total_weight <= 0.0:
                return np.nan
            pattern_weight: dict[tuple[int, ...], float] = {}
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
    research_only=True,
)
class TsPermutationTransitionEntropy(SeriesOperator):
    """排列转移熵：相邻排列状态的转移条件熵，normalize 除以 log(状态数)。"""

    metadata = _metadata(
        "ts_permutation_transition_entropy",
        "排列转移熵（相邻排列转移条件熵）。",
        ["x", "window", "order", "delay", "normalize"],
        unit="level",
        cost=5,
        param_specs=dict(
            _ORDINAL_KNOB_SPECS,
            normalize=ParamSpec(
                dtype=bool, choices=(True, False),
                param_role=ParamRole.POLICY, searchable=False,
            ),
        ),
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, order: int = 3, delay: int = 1, normalize: bool = True, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        ord_ = strict_int(order, "order", lower=2, upper=6)
        dl = strict_int(delay, "delay", lower=1)
        norm = bool(normalize)

        def _fn(chunk: np.ndarray) -> float:
            coded, tie_frac = _permutation_codes(chunk, ord_, dl)
            if len(coded) < 3:
                return np.nan
            if tie_frac > _TIE_RATIO_FAIL_CLOSED:
                return np.nan  # P1-14: tied values -> sparse no-tie sample
            states = sorted({c for c, _ in coded})
            if len(states) < 2:
                return np.nan
            state_index = {state: i for i, state in enumerate(states)}
            transitions = [[0.0] * len(states) for _ in range(len(states))]
            n_trans = 0
            for (prev, start), (cur, start_next) in zip(coded, coded[1:]):
                # Only genuinely adjacent embeddings form a transition (audit
                # R02): a dropped NaN/tie embedding must not turn two
                # non-adjacent patterns into a fake next-pattern link.
                # Embeddings are the sliding windows starting at i, i+1, ... —
                # adjacency is start index + 1 regardless of the
                # *intra-embedding* delay.  Requiring ``start_next == start +
                # dl`` rejected every transition for delay > 1 (a delay-spaced
                # walk skips embeddings), so those parameter combinations
                # returned all-NaN (review P0-11).
                if start_next != start + 1:
                    continue
                transitions[state_index[prev]][state_index[cur]] += 1.0
                n_trans += 1
            if n_trans < 2:
                return np.nan
            cond = 0.0
            for row in transitions:
                row_sum = float(sum(row))
                if row_sum <= 0:
                    continue
                p_cur = row_sum / n_trans
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
    research_only=True,
)
class TsSampleEntropy(SeriesOperator):
    """样本熵：容差 = tolerance_scale × 窗口标准差；无匹配对或常数窗口返回 NaN。window 上限 120。"""

    metadata = _metadata(
        "ts_sample_entropy",
        "样本熵（Chebyshev 匹配，无匹配对返回 NaN）。",
        ["x", "window", "embedding_dim", "tolerance_scale", "min_periods"],
        unit="level",
        cost=7,
        param_specs={
            "window": ParamSpec(dtype=int, min=2, max=120),
            "embedding_dim": ParamSpec(
                dtype=int, min=1, max=4,
                param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False,
            ),
            "tolerance_scale": ParamSpec(
                dtype=float, min=0.0,
                param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False,
            ),
            "min_periods": ParamSpec(
                dtype=int, min=3,
                param_role=ParamRole.SUPPORT_POLICY, searchable=False,
                default=None,
            ),
        },
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, embedding_dim: int = 2, tolerance_scale: float = 0.2, min_periods: Any = None, **_: Any) -> pd.DataFrame:
        # M-10xx: strict integer/float validation — fractional window/embedding
        # or a NaN/negative tolerance is rejected, never int()/float()-truncated.
        w = strict_int(window, "window", lower=2, upper=120)
        m = strict_int(embedding_dim, "embedding_dim", lower=1, upper=4)
        tol = strict_float(tolerance_scale, "tolerance_scale", lower=0.0)
        if tol <= 0.0:
            raise ValueError("tolerance_scale must be > 0")
        min_p = (m + 2) if min_periods is None else max(m + 2, strict_int(min_periods, "min_periods", lower=3))

        def _fn(chunk: np.ndarray) -> float:
            run = _trailing_finite_suffix(chunk)
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
    research_only=True,
)
class TsHurstDfa(SeriesOperator):
    """去趋势波动分析 Hurst 指数：log F(s) 对 log s 的斜率。window 上限 512。"""

    metadata = _metadata(
        "ts_hurst_dfa",
        "DFA Hurst 指数（累积去趋势均方根斜率）。",
        ["x", "window", "min_scale", "max_scale", "n_scales"],
        unit="level",
        cost=7,
        param_specs={
            "window": ParamSpec(dtype=int, min=2, max=512),
            "min_scale": ParamSpec(
                dtype=int, min=2,
                param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False,
            ),
            "max_scale": ParamSpec(
                dtype=int, min=2,
                param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False,
                default=None,
            ),
            "n_scales": ParamSpec(
                dtype=int, min=3,
                param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False,
            ),
        },
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 120, min_scale: int = 4, max_scale: Any = None, n_scales: int = 6, **_: Any) -> pd.DataFrame:
        # M-10xx: strict integer validation (no silent max(2, int(...)) clamp).
        w = strict_int(window, "window", lower=2, upper=512)
        min_s = strict_int(min_scale, "min_scale", lower=2)
        max_s = w // 4 if max_scale is None else strict_int(max_scale, "max_scale", lower=2)
        if max_s < min_s:
            max_s = min_s + 1
        ns = max(3, strict_int(n_scales, "n_scales", lower=3))

        def _fn(chunk: np.ndarray) -> float:
            run = _trailing_finite_suffix(chunk)
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
    research_only=True,
)
class TsHiguchiFractalDimension(SeriesOperator):
    """Higuchi 分形维数：log L(k) 对 log(1/k) 的斜率，衡量路径粗糙度。"""

    metadata = _metadata(
        "ts_higuchi_fractal_dimension",
        "Higuchi 分形维数（路径粗糙度）。",
        ["x", "window", "k_max"],
        unit="level",
        cost=7,
        param_specs={
            "window": ParamSpec(dtype=int, min=2, max=512),
            "k_max": ParamSpec(
                dtype=int, min=1, max=32,
                param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False,
            ),
        },
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 120, k_max: int = 8, **_: Any) -> pd.DataFrame:
        # M-10xx: strict integer validation (no int() truncation).
        w = strict_int(window, "window", lower=2, upper=512)
        km = strict_int(k_max, "k_max", lower=1, upper=32)

        def _fn(chunk: np.ndarray) -> float:
            run = _trailing_finite_suffix(chunk)
            if run.size < 2 * km + 2:
                return np.nan
            return _higuchi_fd(run, km)

        return frame_like(x, map_rolling(x.to_numpy(dtype=float), w, _fn))


def _variogram_slope(chunk: np.ndarray, max_lag: int, min_valid_lags: int) -> float:
    run = _trailing_finite_suffix(chunk)  # R4-64: never bridge a recent gap
    n = len(run)
    max_lag = min(max_lag, n - 1)
    log_lag: list[float] = []
    log_var: list[float] = []
    for k in range(1, max_lag + 1):
        diffs = run[k:] - run[: n - k]
        var = float(np.mean(diffs * diffs))
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
    research_only=True,
)
class TsVariogramSlope(SeriesOperator):
    """变差函数斜率：log E[(x[t+k]-x[t])²] 对 log k 的斜率（尺度结构）。"""

    metadata = _metadata(
        "ts_variogram_slope",
        "变差函数斜率（对数变差对对数滞后回归）。",
        ["x", "window", "max_lag", "min_valid_lags"],
        unit="level",
        cost=3,
        param_specs={
            "window": ParamSpec(dtype=int, min=2),
            "max_lag": ParamSpec(
                dtype=int, min=1, max=30,
                param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False,
            ),
            "min_valid_lags": ParamSpec(
                dtype=int, min=2,
                param_role=ParamRole.SUPPORT_POLICY, searchable=False,
            ),
        },
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, max_lag: int = 10, min_valid_lags: int = 4, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        ml = strict_int(max_lag, "max_lag", lower=1, upper=30)
        mvl = strict_int(min_valid_lags, "min_valid_lags", lower=2)

        def _fn(chunk: np.ndarray) -> float:
            return _variogram_slope(chunk, ml, mvl)

        return frame_like(x, map_rolling(x.to_numpy(dtype=float), w, _fn))


def _autocorr_half_life(chunk: np.ndarray, max_lag: int, use_abs: bool, min_periods: int) -> float:
    run = _trailing_finite_suffix(chunk)  # R4-64: never bridge a recent gap
    n = len(run)
    max_lag = min(max_lag, n - 1)
    log_ac: list[float] = []
    lags: list[float] = []
    for k in range(1, max_lag + 1):
        if n - k < max(2, int(min_periods)):
            continue
        a = run[: n - k]
        b = run[k:]
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
    research_only=True,
)
class TsAutocorrDecayHalfLife(SeriesOperator):
    """自相关衰减半衰期：拟合多阶滞后自相关指数衰减，无衰减返回 NaN。"""

    metadata = _metadata(
        "ts_autocorr_decay_half_life",
        "自相关衰减半衰期（log ac 对 lag 回归，slope>=0 返回 NaN）。",
        ["x", "window", "max_lag", "use_abs", "min_periods"],
        unit="count",
        cost=2,
        param_specs={
            "window": ParamSpec(dtype=int, min=2),
            "max_lag": ParamSpec(
                dtype=int, min=1, max=30,
                param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False,
            ),
            "use_abs": ParamSpec(
                dtype=bool, choices=(True, False),
                param_role=ParamRole.POLICY, searchable=False,
            ),
            "min_periods": ParamSpec(
                dtype=int, min=2,
                param_role=ParamRole.SUPPORT_POLICY, searchable=False,
            ),
        },
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, max_lag: int = 10, use_abs: bool = False, min_periods: int = 2, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        ml = strict_int(max_lag, "max_lag", lower=1, upper=30)
        mp = strict_int(min_periods, "min_periods", lower=2)
        abs_ = bool(use_abs)

        def _fn(chunk: np.ndarray) -> float:
            return _autocorr_half_life(chunk, ml, abs_, mp)

        return frame_like(x, map_rolling(x.to_numpy(dtype=float), w, _fn))


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({
            "ts_permutation_entropy", "ts_weighted_permutation_entropy",
            "ts_permutation_transition_entropy", "ts_sample_entropy",
            "ts_hurst_dfa", "ts_higuchi_fractal_dimension",
            "ts_variogram_slope", "ts_autocorr_decay_half_life",
        })
    # ``ts_permutation_entropy`` / ``ts_sample_entropy`` are also registered by
    # the ts_model.complexity research family (loaded earlier).  This module is
    # the reviewed production implementation, so pull those names off the
    # research surface to keep the extended/research partitions disjoint.
    # R9-P1-045: a live REMOVAL mutator — never a frozenset reassignment (the
    # subtraction is unrepresentable by the add-only extend mutator).
    _surface.retract_research_only({"ts_permutation_entropy", "ts_sample_entropy"})
    for _canon in (
        "ts_permutation_entropy", "ts_weighted_permutation_entropy",
        "ts_permutation_transition_entropy", "ts_sample_entropy",
        "ts_hurst_dfa", "ts_higuchi_fractal_dimension",
        "ts_variogram_slope", "ts_autocorr_decay_half_life",
    ):
        register_polars_bridge(_canon)


_register_surface()
