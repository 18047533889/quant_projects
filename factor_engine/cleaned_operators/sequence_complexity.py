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

from factor_engine.cleaned_operators.base import (
    OperatorMetadata,
    ParamRole,
    ParamSpec,
    SeriesOperator,
    register_operator,
)
from factor_engine.cleaned_operators.closure.strict_scalar import strict_float, strict_int
from factor_engine.cleaned_operators.rolling_pack import (
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
    source="sequence_complexity")
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
            window=ParamSpec(dtype=int, min=2, default=60, history_semantics="max_rows", param_role=ParamRole.HORIZON),
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

        return frame_like(x, _vec_column_permutation_entropy(
            x.to_numpy(dtype=float), w, ord_, dl, norm, min_p))


def _embedding_variance(values: np.ndarray) -> float:
    return float(np.var(values, ddof=0)) if len(values) > 1 else 0.0


def _embedding_range(values: np.ndarray) -> float:
    return float(np.max(values) - np.min(values))


@register_operator(
    name="ts_weighted_permutation_entropy",
    category="complexity",
    business_category="sequence_complexity",
    canonical="ts_weighted_permutation_entropy",
    source="sequence_complexity")
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
            window=ParamSpec(dtype=int, min=2, default=60, history_semantics="max_rows", param_role=ParamRole.HORIZON),
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
            finite = chunk[np.isfinite(chunk)]
            scale = float(np.max(np.abs(finite))) if finite.size else 0.0
            if not np.isfinite(scale) or scale == 0.0:
                return np.nan
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
                stable_vals = vals / scale
                weights.append(_embedding_variance(stable_vals) if weight_kind == "variance" else _embedding_range(stable_vals))
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

        return frame_like(x, _vec_column_weighted_permutation_entropy(
            x.to_numpy(dtype=float), w, ord_, dl, weight_kind, norm))


@register_operator(
    name="ts_permutation_transition_entropy",
    category="complexity",
    business_category="sequence_complexity",
    canonical="ts_permutation_transition_entropy",
    source="sequence_complexity")
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
            window=ParamSpec(dtype=int, min=2, default=60, history_semantics="max_rows", param_role=ParamRole.HORIZON),
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

        return frame_like(x, _vec_column_permutation_transition_entropy(
            x.to_numpy(dtype=float), w, ord_, dl, norm))


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
    # B uses the same extendable template starts as A; the terminal length-m
    # template has no following observation and cannot enter the denominator.
    for i in range(n - m - 1):
        vi = run[i : i + m]
        for j in range(i + 1, n - m):
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
    semantic_version="2.0",
    source="sequence_complexity")
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

        return frame_like(x, _vec_column_sample_entropy(
            x.to_numpy(dtype=float), w, m, tol))


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
        if not np.isfinite(f) or f <= 0.0:
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
    source="sequence_complexity")
class TsHurstDfa(SeriesOperator):
    """去趋势波动分析 Hurst 指数：log F(s) 对 log s 的斜率。window 上限 512。"""

    metadata = _metadata(
        "ts_hurst_dfa",
        "DFA Hurst 指数（累积去趋势均方根斜率）。",
        ["x", "window", "min_scale", "max_scale", "n_scales"],
        unit="level",
        cost=7,
        param_specs={
            "window": ParamSpec(dtype=int, min=2, max=512, default=120, history_semantics="max_rows", param_role=ParamRole.HORIZON),
            "min_scale": ParamSpec(dtype=int, min=2, default=4, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "max_scale": ParamSpec(dtype=int, min=2, default=None, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "n_scales": ParamSpec(dtype=int, min=3, default=6, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        },
    )
    # metadata.param_specs intentionally omitted - use canonical contract from pandas backend

    def _calculate_series(self, x: pd.DataFrame, window: int = 120, min_scale: int = 4, max_scale: Any = None, n_scales: int = 6, **_: Any) -> pd.DataFrame:
        # M-10xx: strict integer validation (no silent max(2, int(...)) clamp).
        w = strict_int(window, "window", lower=2, upper=512)
        min_s = strict_int(min_scale, "min_scale", lower=2)
        max_s = w // 4 if max_scale is None else strict_int(max_scale, "max_scale", lower=2)
        if max_s <= min_s:
            raise ValueError("max_scale must be greater than min_scale")
        if max_s > w // 2:
            raise ValueError("max_scale must be <= window // 2")
        ns = strict_int(n_scales, "n_scales", lower=3)

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
            norm = (n - 1) / ((len(idx) - 1) * k)
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
    source="sequence_complexity")
class TsHiguchiFractalDimension(SeriesOperator):
    """Higuchi 分形维数：log L(k) 对 log(1/k) 的斜率，衡量路径粗糙度。"""

    metadata = _metadata(
        "ts_higuchi_fractal_dimension",
        "Higuchi 分形维数（路径粗糙度）。",
        ["x", "window", "k_max"],
        unit="level",
        cost=7,
        param_specs={
            "window": ParamSpec(dtype=int, min=2, max=512, default=120, history_semantics="max_rows", param_role=ParamRole.HORIZON),
            "k_max": ParamSpec(dtype=int, min=1, max=32, default=8, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        },
    )
    # metadata.param_specs intentionally omitted - use canonical contract from pandas backend

    def _calculate_series(self, x: pd.DataFrame, window: int = 120, k_max: int = 8, **_: Any) -> pd.DataFrame:
        # M-10xx: strict integer validation (no int() truncation).
        w = strict_int(window, "window", lower=2, upper=512)
        km = strict_int(k_max, "k_max", lower=1, upper=32)

        def _fn(chunk: np.ndarray) -> float:
            run = _trailing_finite_suffix(chunk)
            if run.size < 2 * km + 2:
                return np.nan
            return _higuchi_fd(run, km)

        return frame_like(x, _vec_column_higuchi_fd(
            x.to_numpy(dtype=float), w, km))


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
    source="sequence_complexity")
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

        return frame_like(x, _vec_column_variogram_slope(
            x.to_numpy(dtype=float), w, ml, mvl))


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
    source="sequence_complexity")
class TsAutocorrDecayHalfLife(SeriesOperator):
    """自相关衰减半衰期：拟合多阶滞后自相关指数衰减，无衰减返回 NaN。"""

    metadata = _metadata(
        "ts_autocorr_decay_half_life",
        "自相关衰减半衰期（log ac 对 lag 回归，slope>=0 返回 NaN）。",
        ["x", "window", "max_lag", "use_abs", "min_periods"],
        unit="count",
        cost=2,
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=60, history_semantics="max_rows", param_role=ParamRole.HORIZON),
            "max_lag": ParamSpec(dtype=int, min=2, max=30, default=10, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "use_abs": ParamSpec(dtype=bool, default=False, searchable=False, param_role=ParamRole.POLICY),
            "min_periods": ParamSpec(dtype=int, min=2, default=2, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        },
    )
    # metadata.param_specs intentionally omitted - use canonical contract from pandas backend

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, max_lag: int = 10, use_abs: bool = False, min_periods: int = 2, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        ml = strict_int(max_lag, "max_lag", lower=2, upper=30)
        mp = strict_int(min_periods, "min_periods", lower=2)
        abs_ = bool(use_abs)

        return frame_like(x, _vec_column_autocorr_half_life(
            x.to_numpy(dtype=float), w, ml, abs_, mp))




# ---------------------------------------------------------------------------
# R61 vectorized columns (no per-window Python loop)
# ---------------------------------------------------------------------------

def _finite_run_bounds(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each row t: (start, end] of the trailing contiguous finite run that
    ``_trailing_finite_suffix`` would return for the window ending at t,
    assuming the window starts at 0 (callers clip the start to the window)."""
    n = vals.size
    finite = np.isfinite(vals)
    idx = np.where(finite, np.arange(n), -1)
    last_fin = np.maximum.accumulate(idx)
    nan_idx = np.where(~finite, np.arange(n), -1)
    last_nan = np.maximum.accumulate(nan_idx)
    start = np.maximum(last_nan + 1, 0)
    return start, last_fin


def _win_run_lo_hi(vals: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per row t: (s, k, lf) — trailing finite run of window
    [max(0, t-w+1), t] occupies absolute indices [s, lf] with length k.
    Mirrors _trailing_finite_suffix (trailing NaNs are trimmed)."""
    n = vals.size
    finite = np.isfinite(vals)
    idx = np.where(finite, np.arange(n), -1)
    last_fin = np.maximum.accumulate(idx)
    nan_idx = np.where(~finite, np.arange(n), -1)
    last_nan = np.maximum.accumulate(nan_idx)
    t = np.arange(n)
    lo = np.maximum(t - w + 1, 0)
    lf = np.maximum(last_fin, -1)
    s = np.maximum(last_nan[np.maximum(lf, 0)] + 1, lo)
    s = np.where(lf >= 0, s, lo)
    k = np.where(lf >= s, lf - s + 1, 0)
    return s, k, lf


def _cumsum_safe(x: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(np.where(np.isfinite(x), x, 0.0))))


def _masked_slope(x: np.ndarray, y: np.ndarray, valid: np.ndarray, min_pts: int) -> np.ndarray:
    """Row-wise least-squares slope of y on x over valid entries; NaN when a
    row has fewer than ``min_pts`` valid entries or a zero x-dispersion."""
    vd = valid.astype(np.float64)
    cnt = vd.sum(axis=1)
    xc = np.where(valid, x, 0.0)
    yc = np.where(valid, y, 0.0)
    sx = (xc * vd[:, None] if x.ndim == 2 else xc).sum(axis=1)
    # xc already zeroed on invalid; re-zero y too
    yc = np.where(valid, y, 0.0)
    sx = xc.sum(axis=1)
    sy = yc.sum(axis=1)
    sxx = (xc * xc).sum(axis=1)
    syy = (yc * yc).sum(axis=1)
    sxy = (xc * yc).sum(axis=1)
    cnt_safe = np.maximum(cnt, 1.0)
    cov = sxy - sx * sy / cnt_safe
    var = sxx - sx * sx / cnt_safe
    ok = cnt >= min_pts
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(ok & (var > 0), cov / np.where(var > 0, var, 1.0), np.nan)
    return slope


def _vec_permutation_core(vals: np.ndarray, w: int, order: int, delay: int):
    """Shared ordinal-embedding machinery.

    Returns (s_hi, s_lo, total, n_total, n_tied, uniq, Hc, code_good, good)
    where for window end t: embedding starts i in [lo(t), hi(t)] with
    hi(t) = t - e + 1, lo(t) = max(0, t - w + 1);
    total = good-embedding count, n_total/n_tied = finite/tied counts;
    Hc[:, g] = cumsum one-hot of pattern codes over embedding starts
    (Hc[:, g+1] - Hc[:, lo] = per-window pattern counts).
    """
    from numpy.lib.stride_tricks import sliding_window_view
    e = (order - 1) * delay + 1
    n = vals.size
    n_emb = max(n - e + 1, 0)
    n_pat = math.factorial(order)
    if n_emb <= 0:
        return None
    E = sliding_window_view(vals, e)[:, ::delay]          # (n_emb, order)
    valid = np.isfinite(E).all(axis=1)
    srt = np.sort(E, axis=1)
    tied = valid & (np.diff(srt, axis=1) == 0).any(axis=1)
    good = valid & ~tied
    # stable rank pattern for good rows (no ties -> any argsort = rank perm)
    pat = np.argsort(np.argsort(np.where(good[:, None], E, 0.0), axis=1, kind="stable"), axis=1, kind="stable")
    powers = (order ** np.arange(order)).astype(np.int64)
    code_all = pat @ powers
    code_good = np.where(good, code_all, -1)
    uniq, inv = np.unique(code_good[good], return_inverse=True) if good.any() else (np.empty(0, np.int64), np.empty(0, np.int64))
    n_u = uniq.size
    T_rows = n  # windows indexed by end t in [0, n)
    # embedding-index -> window-end coverage handled by caller via ranges
    Hc = np.zeros((n_u, n_emb + 1))
    if n_u:
        Hc[inv, np.arange(n_emb)[good] + 1] = 1.0
    Hc = np.cumsum(Hc, axis=1)
    Gc = np.concatenate(([0.0], np.cumsum(good.astype(np.float64))))
    Vc = np.concatenate(([0.0], np.cumsum(valid.astype(np.float64))))
    Tc = np.concatenate(([0.0], np.cumsum(tied.astype(np.float64))))
    return dict(e=e, n_emb=n_emb, n_u=n_u, Hc=Hc, Gc=Gc, Vc=Vc, Tc=Tc,
                good=good, inv=inv, uniq=uniq, n_pat=n_pat)


def _vec_win_ranges(core, vals: np.ndarray, w: int):
    e = core["e"]; n = vals.size
    t = np.arange(n)
    hi = t - e + 1                     # last embedding start (inclusive)
    lo = np.maximum(t - w + 1, 0)
    okw = hi >= lo                     # window contains >= 1 embedding
    hi_c = np.clip(hi + 1, 0, core["n_emb"])   # cumsum index (exclusive)
    lo_c = np.clip(lo, 0, core["n_emb"])
    return lo_c, hi_c, okw


def _vec_column_permutation_entropy(panel: np.ndarray, w: int, order: int, delay: int,
                                    normalize: bool, min_p: int) -> np.ndarray:
    rows, cols = panel.shape
    out = np.full((rows, cols), np.nan)
    norm = math.log(math.factorial(order)) if normalize else None
    for c in range(cols):
        vals = panel[:, c]
        core = _vec_permutation_core(vals, w, order, delay)
        if core is None or core["n_u"] == 0:
            continue
        lo_c, hi_c, okw = _vec_win_ranges(core, vals, w)
        counts = core["Hc"][:, hi_c] - core["Hc"][:, lo_c]        # (n_u, rows)
        total = core["Gc"][hi_c] - core["Gc"][lo_c]
        n_total = core["Vc"][hi_c] - core["Vc"][lo_c]
        n_tied = core["Tc"][hi_c] - core["Tc"][lo_c]
        with np.errstate(divide="ignore", invalid="ignore"):
            tie_frac = np.where(n_total > 0, n_tied / np.maximum(n_total, 1.0), 1.0)
            p = np.where(counts > 0, counts / np.maximum(total, 1.0), 0.0)
            ent = -np.sum(np.where(counts > 0, p * np.log(np.maximum(p, 1e-300)), 0.0), axis=0)
        ok = okw & (total >= min_p) & (tie_frac <= _TIE_RATIO_FAIL_CLOSED) & (total >= 2)
        if norm is not None:
            ent = ent / norm
        out[:, c] = np.where(ok, ent, np.nan)
    return out


def _vec_column_weighted_permutation_entropy(panel: np.ndarray, w: int, order: int, delay: int,
                                             weight_kind: str, normalize: bool) -> np.ndarray:
    rows, cols = panel.shape
    out = np.full((rows, cols), np.nan)
    norm = math.log(math.factorial(order)) if normalize else None
    for c in range(cols):
        vals = panel[:, c]
        core = _vec_permutation_core(vals, w, order, delay)
        if core is None or core["n_u"] == 0:
            continue
        e = core["e"]
        n_emb = core["n_emb"]
        E = np.lib.stride_tricks.sliding_window_view(vals, e)[:, ::delay]
        # The authority weights each embedding by var/range of E_i/scale_t
        # (window max|finite|).  That factor is constant within a window and
        # cancels in p = w/sum(w), so a single COLUMN-level scale C gives the
        # identical entropy while avoiding var(1e300-scale data) overflow
        # (contract test: scale stability at 1e300).
        fin_abs = np.abs(vals[np.isfinite(vals)])
        col_scale = float(fin_abs.max()) if fin_abs.size else 0.0
        if col_scale > 0 and np.isfinite(col_scale):
            En = E / col_scale
        else:
            En = E
        with np.errstate(all="ignore"):
            if weight_kind == "variance":
                wt = En.var(axis=1, ddof=0)
            else:
                wt = En.max(axis=1) - En.min(axis=1)
        wt = np.where(core["good"], wt, 0.0)
        # per-pattern weighted sums via one-hot cumsum over embedding starts
        Wc = np.zeros((core["n_u"], n_emb + 1))
        np.add.at(Wc, (core["inv"], np.arange(n_emb)[core["good"]] + 1), wt[core["good"]])
        Wc = np.cumsum(Wc, axis=1)
        lo_c, hi_c, okw = _vec_win_ranges(core, vals, w)
        wcnt = Wc[:, hi_c] - Wc[:, lo_c]                          # (n_u, rows)
        total = core["Gc"][hi_c] - core["Gc"][lo_c]
        n_total = core["Vc"][hi_c] - core["Vc"][lo_c]
        n_tied = core["Tc"][hi_c] - core["Tc"][lo_c]
        W = wcnt.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            tie_frac = np.where(n_total > 0, n_tied / np.maximum(n_total, 1.0), 1.0)
            p = np.where(wcnt > 0, wcnt / np.maximum(W, 1e-300), 0.0)
            ent = -np.sum(np.where(wcnt > 0, p * np.log(np.maximum(p, 1e-300)), 0.0), axis=0)
        ok = okw & (total >= 2) & (tie_frac <= _TIE_RATIO_FAIL_CLOSED) & (W > 0)
        if norm is not None:
            ent = ent / norm
        out[:, c] = np.where(ok, ent, np.nan)
    return out


def _vec_column_permutation_transition_entropy(panel: np.ndarray, w: int, order: int,
                                               delay: int, normalize: bool) -> np.ndarray:
    rows, cols = panel.shape
    out = np.full((rows, cols), np.nan)
    for c in range(cols):
        vals = panel[:, c]
        core = _vec_permutation_core(vals, w, order, delay)
        if core is None or core["n_u"] == 0:
            continue
        e = core["e"]; n_emb = core["n_emb"]; n_u = core["n_u"]
        good = core["good"]; inv = core["inv"]
        lo_c, hi_c, okw = _vec_win_ranges(core, vals, w)
        total = core["Gc"][hi_c] - core["Gc"][lo_c]
        n_total = core["Vc"][hi_c] - core["Vc"][lo_c]
        n_tied = core["Tc"][hi_c] - core["Tc"][lo_c]
        # states present per window (patterns among good embeddings)
        counts = core["Hc"][:, hi_c] - core["Hc"][:, lo_c]        # (n_u, rows)
        n_states = (counts > 0).sum(axis=0).astype(np.float64)
        # transitions between adjacent good embeddings
        adj = good[:-1] & good[1:]                                 # pair start i (n_emb-1)
        inv_full = np.full(n_emb, -1, dtype=np.int64)
        inv_full[good] = inv
        pc = inv_full[:-1] * n_u + inv_full[1:]
        # per-pair cumsum one-hot + per-prev-state cumsum
        pair_ids, pair_inv = np.unique(pc[adj], return_inverse=True) if adj.any() else (np.empty(0, np.int64), np.empty(0, np.int64))
        P = np.zeros((pair_ids.size, max(n_emb, 1)))
        if pair_ids.size:
            P[pair_inv, np.arange(n_emb - 1)[adj]] = 1.0
            Pc = np.concatenate((np.zeros((pair_ids.size, 1)), np.cumsum(P, axis=1)), axis=1)
        else:
            Pc = np.zeros((0, n_emb + 1))
        Rc = np.zeros((n_u, n_emb + 1))
        np.add.at(Rc, (inv_full[:-1][adj], np.arange(n_emb - 1)[adj] + 1), 1.0)
        Rc = np.cumsum(Rc, axis=1)
        # window t covers pair starts i in [lo, hi-1]  (need i+1 <= t-e+1)
        t = np.arange(vals.size)
        hi_pair = np.clip(t - e, 0, max(n_emb - 1, 0)) + 1
        lo_pair = np.clip(np.maximum(t - w + 1, 0), 0, max(n_emb - 1, 0))
        n_trans = Rc.sum(axis=0)[hi_pair] - Rc.sum(axis=0)[lo_pair] if n_u else np.zeros(vals.size)
        pair_cnt = Pc[:, hi_pair] - Pc[:, lo_pair] if pair_ids.size else np.zeros((0, vals.size))
        prev_id = (pair_ids // n_u).astype(np.int64)
        cnt_pc = np.where(pair_cnt > 0, pair_cnt, 0.0)
        sum_cnt_log = np.sum(np.where(pair_cnt > 0, cnt_pc * np.log(np.maximum(cnt_pc, 1e-300)), 0.0), axis=0)
        rowsum = np.zeros((n_u, vals.size))
        if pair_ids.size:
            np.add.at(rowsum, prev_id, np.maximum(pair_cnt, 0.0))
        sum_row_log = np.sum(np.where(rowsum > 0, rowsum * np.log(np.maximum(rowsum, 1e-300)), 0.0), axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            tie_frac = np.where(n_total > 0, n_tied / np.maximum(n_total, 1.0), 1.0)
            cond = -(sum_cnt_log - sum_row_log) / np.maximum(n_trans, 1.0)
            normv = np.where(n_states > 0, np.log(np.maximum(n_states, 1.0)), np.nan)
            cond = cond / normv if normalize else cond
        ok = okw & (total >= 3) & (tie_frac <= _TIE_RATIO_FAIL_CLOSED) & (n_states >= 2) & (n_trans >= 2)
        out[:, c] = np.where(ok, cond, np.nan)
    return out


def _vec_column_sample_entropy(panel: np.ndarray, w: int, m: int, tol_scale: float) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view
    rows, cols = panel.shape
    out = np.full((rows, cols), np.nan)
    for c in range(cols):
        vals = panel[:, c]
        n = vals.size
        s, k, lf = _win_run_lo_hi(vals, w)
        # per-window mean/std of the run (variable start s, end lf)
        C1 = _cumsum_safe(vals)
        C2 = _cumsum_safe(vals * vals)
        lo = s
        hi1 = np.clip(lf + 1, 0, n)
        cnt = k.astype(np.float64)
        s1 = C1[hi1] - C1[lo]
        s2 = C2[hi1] - C2[lo]
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = s1 / np.maximum(cnt, 1.0)
            var = s2 / np.maximum(cnt, 1.0) - mean * mean
            var = np.maximum(var, 0.0)
            std = np.sqrt(var)
            r = tol_scale * std
        ok_win = (cnt >= m + 2) & (std > 1e-12)
        if not ok_win.any():
            continue
        E1 = sliding_window_view(vals, m + 1)          # starts 0..n-m-1
        E0 = sliding_window_view(vals, m)              # starts 0..n-m
        t_ok = np.where(ok_win)[0]
        matches_a = np.zeros(n)
        matches_b = np.zeros(n)
        CH = 128
        for a0 in range(0, t_ok.size, CH):
            ts = t_ok[a0:a0 + CH]
            st = s[ts]
            kt = cnt[ts]
            rt = r[ts]
            # local template slots p in [0, w-m-2]; absolute start = st+p
            L = w - m                                   # max local starts for (m+1)
            ploc = np.arange(L)
            ia = st[:, None] + ploc[None, :]            # (B, L)
            valid_slot = (ploc[None, :] <= (kt[:, None] - m - 2)) & (ia < n - m)
            ia_c = np.clip(ia, 0, n - m - 1)
            A1 = E1[ia_c]                               # (B, L, m+1)
            A0 = E0[np.clip(ia, 0, n - m)]              # (B, L, m)
            # distances between all slot pairs
            dA = np.abs(A1[:, :, None, :] - A1[:, None, :, :]).max(axis=3)   # (B, L, L)
            dB = np.abs(A0[:, :, None, :] - A0[:, None, :, :]).max(axis=3)
            pi = ploc[None, :, None]
            pj = ploc[None, None, :]
            fit_i = pi <= (kt[:, None, None] - m - 2)
            fit_j = pj <= (kt[:, None, None] - m - 1)
            pair_ok = fit_i & fit_j & (pj > pi) & (ia[:, :, None] < n) & (ia[:, None, :] < n)
            ma = (pair_ok & (dA <= rt[:, None, None])).sum(axis=(1, 2))
            mb = (pair_ok & (dB <= rt[:, None, None])).sum(axis=(1, 2))
            matches_a[ts] = ma
            matches_b[ts] = mb
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.log(matches_a / matches_b)
        ent = np.where((matches_a > 0) & (matches_b > 0), ent, np.nan)
        out[:, c] = np.where(ok_win, ent, np.nan)
    return out


def _vec_column_autocorr_half_life(panel: np.ndarray, w: int, max_lag: int, use_abs: bool,
                                   min_periods: int) -> np.ndarray:
    rows, cols = panel.shape
    out = np.full((rows, cols), np.nan)
    lags = np.arange(1, max_lag + 1, dtype=np.float64)
    for c in range(cols):
        vals = panel[:, c]
        n = vals.size
        # ac is shift-invariant; centering once removes the catastrophic
        # cancellation in E[x*y]-E[x]E[y] for large-mean series.
        fin = vals[np.isfinite(vals)]
        vals = vals - (float(np.mean(fin)) if fin.size else 0.0)
        s, k, lf = _win_run_lo_hi(vals, w)
        ac_mat = np.full((n, max_lag), np.nan)
        cnt_mat = np.zeros((n, max_lag))
        for qi, q in enumerate(range(1, max_lag + 1)):
            if q >= w:
                continue
            D = vals[q:] * vals[:-q]           # x[i]*x[i+q], index i
            Dl = vals[:-q]
            Dr = vals[q:]
            Dll = Dl * Dl
            Drr = Drr_ = Dr * Dr
            CD = _cumsum_safe(D); CDl = _cumsum_safe(Dl); CDr = _cumsum_safe(Dr)
            CDll = _cumsum_safe(Dll); CDrr = _cumsum_safe(Dll if False else Drr)
            # sums over i in [s, lf-q]; cumsum has n-q+1 entries
            hi = np.clip(lf - q + 1, 0, n - q)
            lo = np.clip(s, 0, n - q)
            cnt = np.clip(k - q, 0, None).astype(np.float64)
            sD = CD[hi] - CD[lo]
            sL = CDl[hi] - CDl[lo]
            sR = CDr[hi] - CDr[lo]
            sLL = CDll[hi] - CDll[lo]
            sRR = CDrr[hi] - CDrr[lo]
            with np.errstate(divide="ignore", invalid="ignore"):
                csafe = np.maximum(cnt, 1.0)
                va = sLL / csafe - (sL / csafe) ** 2
                va = np.maximum(va, 0.0)
                cov = sD / csafe - (sL / csafe) * (sR / csafe)
                ac = np.where(va > 1e-12, cov / np.where(va > 1e-12, va, 1.0), np.nan)
            if use_abs:
                ac = np.abs(ac)
            # authority: `if ac <= 1e-12: continue` — SIGNED comparison, so
            # negative autocorrelations are skipped unless use_abs.
            valid = (cnt >= max(2, min_periods)) & (cnt >= 2) & np.isfinite(ac) & (ac > 1e-12) & (va > 1e-12)
            ac_mat[:, qi] = np.where(valid, ac, np.nan)
            cnt_mat[:, qi] = np.where(valid, 1.0, 0.0)
        # regression log(ac) ~ lag per row
        with np.errstate(divide="ignore", invalid="ignore"):
            ly = np.log(np.maximum(ac_mat, 1e-300))
        slope = _masked_slope(np.broadcast_to(lags, ac_mat.shape), ly, np.isfinite(ac_mat), 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            hl = np.where(slope < 0, -math.log(2.0) / np.where(slope < 0, slope, -1.0), np.nan)
        out[:, c] = hl
    return out


def _vec_column_variogram_slope(panel: np.ndarray, w: int, max_lag: int, min_valid_lags: int) -> np.ndarray:
    rows, cols = panel.shape
    out = np.full((rows, cols), np.nan)
    lags = np.arange(1, max_lag + 1, dtype=np.float64)
    for c in range(cols):
        vals = panel[:, c]
        n = vals.size
        s, k, lf = _win_run_lo_hi(vals, w)
        lv_mat = np.full((n, max_lag), np.nan)
        for q in range(1, max_lag + 1):
            if q >= w:
                continue
            D2 = (vals[q:] - vals[:-q]) ** 2
            CD2 = _cumsum_safe(D2)
            hi = np.clip(lf - q + 1, 0, n - q)
            lo = np.clip(s, 0, n - q)
            cnt = np.clip(k - q, 0, None).astype(np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                v = (CD2[hi] - CD2[lo]) / np.maximum(cnt, 1.0)
            valid = (cnt >= 1) & (v > 1e-12) & np.isfinite(v)
            lv_mat[:, q - 1] = np.where(valid, np.log(np.maximum(v, 1e-300)), np.nan)
        # authority regresses log(var) on log(lag)
        slope = _masked_slope(np.broadcast_to(np.log(lags), lv_mat.shape), lv_mat, np.isfinite(lv_mat), max(2, min_valid_lags))
        out[:, c] = slope
    return out


def _vec_column_higuchi_fd(panel: np.ndarray, w: int, km: int) -> np.ndarray:
    rows, cols = panel.shape
    out = np.full((rows, cols), np.nan)
    for c in range(cols):
        vals = panel[:, c]
        n = vals.size
        s, k, lf = _win_run_lo_hi(vals, w)
        logk_list = []
        lk_mat = []
        valid_k = []
        for kp in range(1, km + 1):
            if kp >= w:
                continue
            D = np.abs(vals[kp:] - vals[:-kp])   # position i: |x[i+kp]-x[i]|
            # residue-cumsum table: R[rr, g] = sum of finite D[i] for i%kp==rr, i<g
            R = np.zeros((kp, n + 1))
            for rr in range(kp):
                mask = (np.arange(n - kp) % kp) == rr
                R[rr, 1:n - kp + 1] = np.cumsum(np.where(mask, np.where(np.isfinite(D), D, 0.0), 0.0))
            hi = np.clip(lf - kp + 1, 0, n)      # exclusive end over i in [s, lf-kp]
            lo = np.clip(s, 0, n)
            mgrid = np.arange(kp)[None, :]
            # len(idx) = #{i_loc = m, m+kp, ... < k_t} = ceil((k_t-m)/kp)
            with np.errstate(invalid="ignore"):
                L = np.ceil(np.maximum(k[:, None] - mgrid, 0) / kp)
            m_ok = L >= 2                        # need >= 2 subsample points
            npts = np.maximum(L - 1.0, 1.0)      # path segments = len(idx)-1
            path = np.zeros((n, kp))
            for m in range(kp):
                rr = (s + m) % kp
                path[:, m] = R[rr, hi] - R[rr, lo]
            with np.errstate(divide="ignore", invalid="ignore"):
                norm = (k[:, None] - 1) / (npts * kp)
                Lm = path * norm / kp
            lk = np.sum(np.where(m_ok, Lm, 0.0), axis=1) / np.maximum(m_ok.sum(axis=1), 1.0)
            n_m = m_ok.sum(axis=1)
            okk = (k >= 2 * km + 2) & (n_m >= 1) & (lk > 1e-12)
            logk_list.append(math.log(1.0 / kp))
            lk_mat.append(np.where(okk, np.log(np.maximum(lk, 1e-300)), np.nan))
            valid_k.append(okk)
        if not lk_mat:
            continue
        X = np.stack([np.full(n, v) for v in logk_list], axis=1)
        Y = np.stack(lk_mat, axis=1)
        V = np.stack(valid_k, axis=1)
        slope = _masked_slope(X, Y, V, 2)
        out[:, c] = slope
    return out


def _register_surface() -> None:
    import factor_engine.cleaned_operators.operator_surface as _surface

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
