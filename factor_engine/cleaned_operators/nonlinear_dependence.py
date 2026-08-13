# -*- coding: utf-8 -*-
"""Nonlinear dependence operators (2026-08 final pack, group 1).

Adds distance correlation/covariance, quantile-histogram mutual information
(and its lagged form) and fixed-q upper/lower tail coexceedance probabilities
(``ts_upper_tail_coexceedance_probability`` /
``ts_lower_tail_coexceedance_probability``; the legacy
``ts_upper_tail_dependence`` / ``ts_lower_tail_dependence`` names resolve as
deprecated aliases).  All are causal daily-panel transforms.

Contract notes
--------------
* Aligned pairs only: ``x[t]`` and ``y[t]`` are paired at the same position;
  NaN is never dropped and then re-compressed.
* ``min_periods`` counts *valid pairs*; below it the output is NaN.
* ``ts_lagged_mutual_information.window`` counts the ALIGNED PAIRS used in the
  statistic, so its raw history requirement is ``window + lag`` bars and the
  parameter means the same thing across lags (window=60,lag=1 and
  window=60,lag=10 both report 60 aligned pairs).
* Constant windows and degenerate quantiles return NaN (fail-closed).
* Deterministic: the MI estimator is rank-quantile-binned (no randomness).

M-180/M-182 — kernel-Granger / HSIC family (relocated by the reconciler)
------------------------------------------------------------------------
The live kernel-Granger / HSIC canonicals are registered from the *research*
surface: ``ts_kernel_granger_score`` / ``ts_residualized_hsic`` in
``research_spectral.py`` and ``ts_hsic`` in ``dependence_ext.py``.  The dead
legacy names ``ts_kernel_granger_causality`` / ``ts_kernel_granger_oos`` /
``ts_hsic_dependence`` were removed from the contract registries — they must NOT
be re-registered as aliases.  ``ts_kernel_granger_score`` uses a BLOCKED
out-of-sample train/test split and is a DIAGNOSTIC_STRUCTURE score, not a
next-return predictor:

* training data strictly before the test window — the trailing window is split
  ``train = [0, train_end)``, ``test = [train_end, window)`` with
  ``train_end = int(0.7 * n_available)``;
* the scaler (per-block mean/std) and the RBF bandwidth (median pairwise
  distance) are selected on the TRAINING partition only — the test partition is
  never touched by any fit statistic, so ``ln(MSE_restricted / MSE_full)``
  carries no in-sample / training-residual leakage.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import (
    OperatorMetadata,
    ParamSpec,
    RelationalParamSpec,
    SeriesOperator,
    register_operator,
)
from cleaned_operators.rolling_pack import (
    aligned_pairs,
    check_window,
    frame_like,
    map_pair_rolling,
    register_polars_bridge,
)


def _metadata(name: str, description: str, params: list[str], *, unit: str) -> OperatorMetadata:
    # R11 #18 unit-algebra honesty: algebraic output units (``same_as:`` /
    # ``unit(...)`` / ``sqrt(...)`` / ``dimensionless``) are propagated to
    # ``output_unit`` so the catalog / typed search see the real output
    # dimension instead of an opaque ``level`` tag.  ``ts_distance_cov`` uses
    # the sqrt-product form ``sqrt(unit(x)*unit(y))`` (review #18): distance
    # covariance is a scale-laden statistic, and declaring the exact output
    # dimension makes cross-scale mixing (price x turnover vs return x amount)
    # visible in the metadata instead of silently changing with input scale.
    output_unit = unit if (
        unit.startswith("same_as:") or unit.startswith("unit(")
        or unit.startswith("sqrt(") or unit == "dimensionless"
    ) else None
    return OperatorMetadata(
        name=name,
        category="time_series_risk",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "time_series_risk", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", "domain:dependence",
            f"unit:{unit}", "cost:5",
        ],
        output_unit=output_unit,
    )


def _double_center(a: np.ndarray) -> np.ndarray:
    d = np.abs(a[:, None] - a[None, :])
    return d - d.mean(axis=0, keepdims=True) - d.mean(axis=1, keepdims=True) + d.mean()


def _distance_corr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (distance_correlation, distance_covariance) in [0, 1] / level."""
    n = a.size
    if n < 4:
        return np.nan, np.nan
    A = _double_center(a)
    B = _double_center(b)
    dcov2 = np.where(float(n * n) != 0, float(np.sum(A * B)) / float(n * n), np.nan)
    dvar_x2 = np.where(float(n * n) != 0, float(np.sum(A * A)) / float(n * n), np.nan)
    dvar_y2 = np.where(float(n * n) != 0, float(np.sum(B * B)) / float(n * n), np.nan)
    if dvar_x2 <= 1e-12 or dvar_y2 <= 1e-12:
        return np.nan, np.nan
    dcov = float(np.sqrt(max(dcov2, 0.0)))
    denom = float(np.sqrt(np.sqrt(dvar_x2) * np.sqrt(dvar_y2)))
    if denom <= 1e-12:
        return np.nan, dcov
    dcorr = np.where(denom))) != 0, float(min(1.0, max(0.0, dcov / denom))), np.nan)
    return dcorr, dcov


@register_operator(
    name="ts_distance_corr",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_distance_corr",
    source="nonlinear_dependence",
)
class TsDistanceCorr(SeriesOperator):
    """距离相关：同时捕捉线性与非线性依赖，常数/过短窗口返回 NaN，输出 [0,1]。"""

    metadata = _metadata(
        "ts_distance_corr",
        "距离相关（distance correlation），仅使用同位置有效配对。",
        ["x", "y", "window", "min_periods"],
        unit="ratio",
    )
    # R11 #16: ``min_periods`` is a real lower bound, not a clamped minimum —
    # values 2..10 used to be silently coerced to 10, manufacturing a fake
    # parameter interval.  A value < 10 now raises at binding.
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2),
        "min_periods": ParamSpec(dtype=int, min=10),
    }

    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, window: int = 40, min_periods: int = 10, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        mp = min_periods  # strict ParamSpec(dtype=int, min=10) — no silent clamp
        xv = x.to_numpy(dtype=float)
        yv = y.to_numpy(dtype=float)

        def _fn(a: np.ndarray, b: np.ndarray) -> float:
            pa, pb = aligned_pairs(a, b)
            if pa.size < mp:
                return np.nan
            return _distance_corr(pa, pb)[0]

        return frame_like(x, map_pair_rolling(xv, yv, w, _fn))


@register_operator(
    name="ts_distance_cov",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_distance_cov",
    source="nonlinear_dependence",
)
class TsDistanceCov(SeriesOperator):
    """距离协方差：距离相关的底层尺度量，允许挖掘规模相关非线性联动。"""

    metadata = _metadata(
        "ts_distance_cov",
        "距离协方差（distance covariance）。",
        ["x", "y", "window", "min_periods"],
        # R11 #18: distance covariance is scale-laden — its dimension is
        # sqrt(unit(x) * unit(y)), NOT a plain ``level``.  Declaring the exact
        # output dimension (the sqrt-product algebraic form) makes cross-scale
        # mixes (price x turnover vs return x amount) visible in the metadata
        # instead of silently scaling the output with the inputs.
        unit="sqrt(unit(x)*unit(y))",
    )
    # R11 #16: ``min_periods`` is a real lower bound, not a clamped minimum —
    # values 2..10 used to be silently coerced to 10, manufacturing a fake
    # parameter interval.  A value < 10 now raises at binding.
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2),
        "min_periods": ParamSpec(dtype=int, min=10),
    }

    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, window: int = 40, min_periods: int = 10, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        mp = min_periods  # strict ParamSpec(dtype=int, min=10) — no silent clamp
        xv = x.to_numpy(dtype=float)
        yv = y.to_numpy(dtype=float)

        def _fn(a: np.ndarray, b: np.ndarray) -> float:
            pa, pb = aligned_pairs(a, b)
            if pa.size < mp:
                return np.nan
            return _distance_corr(pa, pb)[1]

        return frame_like(x, map_pair_rolling(xv, yv, w, _fn))


def _value_bins(values: np.ndarray, bins: int) -> np.ndarray:
    """Assign values to quantile buckets by value (window-local edges only).

    Equal values always land in the same bucket (``side='right'``).  Buckets may
    stay empty under heavy ties — empty bins contribute 0 to the contingency and
    are handled stably; edges are recomputed from the current window only.
    """
    cuts = np.quantile(values, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    bucket = np.searchsorted(cuts, values, side="right")
    return np.clip(bucket.astype(np.int64), 0, bins - 1)


def _quantile_hist_mi(
    a: np.ndarray,
    b: np.ndarray,
    bins: int,
    normalized: bool,
    bias_correction: bool = True,
) -> float:
    """Rank-quantile histogram mutual information (plug-in, base-e / nats).

    ``bias_correction`` controls the Miller–Madow finite-sample bias
    correction — it is an OPTION, not always on (audit #19).  When enabled the
    correction term is ``(r_occ - 1)·(c_occ - 1) / (2n)`` where ``r_occ`` /
    ``c_occ`` are the numbers of *occupied* row / column marginal cells (R5
    P1-39(a)): with ties / empty bins the effective support is smaller than
    ``bins``, so the textbook ``(bins-1)^2/(2n)`` term over-corrects.  The
    corrected value is floored at 0 (the plug-in estimate is always
    non-negative; a negative correction is a finite-sample artefact).
    """
    n = a.size
    if n < 2:
        return np.nan
    ba = _value_bins(a, bins)
    bb = _value_bins(b, bins)
    cont = np.zeros((bins, bins), dtype=np.float64)
    for i in range(n):
        cont[ba[i], bb[i]] += 1.0
    p = cont / n if n > 0 else np.nan
    p_row = p.sum(axis=1, keepdims=True)
    p_col = p.sum(axis=0, keepdims=True)
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if p[i, j] > 0 and p_row[i, 0] > 0 and p_col[0, j] > 0:
                mi += p[i, j] * np.log(p[i, j] / (p_row[i, 0] * p_col[0, j]))
    if bias_correction:
        r_occ = int((p_row[:, 0] > 0).sum())
        c_occ = int((p_col[0, :] > 0).sum())
        mi = np.where((2.0 * n)) != 0, max(0.0, mi - float((r_occ - 1) * (c_occ - 1)) / (2.0 * n)), np.nan)
    hx = -float(np.sum(p_row * np.log(np.where(p_row > 0, p_row, 1.0))))
    hy = -float(np.sum(p_col * np.log(np.where(p_col > 0, p_col, 1.0))))
    # A (near-)constant marginal carries no information: fail closed to NaN.
    if hx <= 1e-12 or hy <= 1e-12:
        return np.nan
    if not normalized:
        return float(mi)
    denom = min(hx, hy)
    if denom <= 1e-12:
        return np.nan
    return np.where(denom))) != 0, float(min(1.0, max(0.0, mi / denom))), np.nan)


@register_operator(
    name="ts_mutual_information",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_mutual_information",
    source="nonlinear_dependence",
)
class TsMutualInformation(SeriesOperator):
    """互信息（分位数直方图估计）：normalized=True 输出 [0,1] 无量纲；否则为 nats。"""

    metadata = _metadata(
        "ts_mutual_information",
        "互信息（rank 分位数直方图 + 可选 Miller–Madow 有限样本校正）。"
        " 原始输出单位为 nats；normalized=True 时按 min(Hx,Hy) 归一为 [0,1] 无量纲比值。",
        ["x", "y", "window", "estimator", "bins", "min_periods", "normalized", "bias_correction"],
        unit="nats",
    )
    metadata.param_specs = {
        "bias_correction": ParamSpec(dtype=bool, choices=(True, False)),
        # R11 #17: strict bool, matching bias_correction — ``bool(normalized)``
        # used to accept 1/2/"False"/-1 via Python truthiness, manufacturing
        # false search-space duplicates.  Only True/False pass the gate.
        "normalized": ParamSpec(dtype=bool, choices=(True, False)),
        "window": ParamSpec(dtype=int, min=2),
        "bins": ParamSpec(dtype=int, min=2, max=10),
        # R11 #16: ``min_periods`` is a real lower bound, not a clamped minimum —
        # values 2..10 used to be silently coerced to 10, manufacturing a fake
        # parameter interval.  A value < 10 now raises at binding.
        "min_periods": ParamSpec(dtype=int, min=10),
    }

    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, window: int = 40, estimator: str = "quantile_hist", bins: int = 5, min_periods: int = 10, normalized: bool = True, bias_correction: bool = True, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        if str(estimator).lower() != "quantile_hist":
            raise ValueError("estimator must be 'quantile_hist' (deterministic)")
        nb = int(bins)
        if not 2 <= nb <= 10:
            raise ValueError("bins must be in [2, 10]")
        mp = min_periods  # strict ParamSpec(dtype=int, min=10) — no silent clamp
        norm = normalized  # strict bool ParamSpec — no bool() truthiness coercion
        bc = bias_correction
        xv = x.to_numpy(dtype=float)
        yv = y.to_numpy(dtype=float)

        def _fn(a: np.ndarray, b: np.ndarray) -> float:
            pa, pb = aligned_pairs(a, b)
            if pa.size < mp:
                return np.nan
            return _quantile_hist_mi(pa, pb, nb, norm, bias_correction=bc)

        return frame_like(x, map_pair_rolling(xv, yv, w, _fn))


@register_operator(
    name="ts_lagged_mutual_information",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_lagged_mutual_information",
    source="nonlinear_dependence",
)
class TsLaggedMutualInformation(SeriesOperator):
    """滞后互信息：MI(x[t-lag], y[t])，lag 必须为非负整数，输出单位为 nats。"""

    metadata = _metadata(
        "ts_lagged_mutual_information",
        "滞后互信息 MI(x[t-lag], y[t])（rank 分位数直方图，nats）。",
        ["x", "y", "window", "lag", "bins", "min_periods", "bias_correction"],
        unit="nats",
    )
    metadata.param_specs = {
        "bias_correction": ParamSpec(dtype=bool, choices=(True, False)),
        # R11 #15: strict-integer lag — ``int(lag)`` used to truncate 1.9 -> 1,
        # manufacturing false search-space duplicates.  Non-integer finite values
        # raise; lag >= 1 (a 0-lag MI is the synchronous ts_mutual_information).
        "lag": ParamSpec(dtype=int, min=1),
        "window": ParamSpec(dtype=int, min=2),
        "bins": ParamSpec(dtype=int, min=2, max=10),
        # R11 #16: ``min_periods`` is a real lower bound, not a clamped minimum —
        # values 2..10 used to be silently coerced to 10, manufacturing a fake
        # parameter interval.  A value < 10 now raises at binding.
        "min_periods": ParamSpec(dtype=int, min=10),
    }
    # R11 #14/#15: ``window`` counts the ALIGNED PAIRS actually used in the
    # statistic, so the raw history requirement is ``window + lag`` bars and the
    # parameter means the same thing across lags (window=60,lag=1 and
    # window=60,lag=10 now use the SAME 60 aligned pairs).  A lag >= window
    # would leave zero aligned pairs -> infeasible at the call boundary.
    metadata.relational_specs = [
        RelationalParamSpec(
            "lag < window",
            message="lag must be < window (window counts aligned pairs; raw history = window + lag)",
        ),
    ]

    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, window: int = 40, lag: int = 1, bins: int = 5, min_periods: int = 10, bias_correction: bool = True, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        lag_v = lag  # strict-integer ParamSpec(dtype=int, min=1) — never int()-truncated
        if lag_v < 1:
            raise ValueError("lag must be >= 1")
        nb = int(bins)
        if not 2 <= nb <= 10:
            raise ValueError("bins must be in [2, 10]")
        mp = min_periods  # strict ParamSpec(dtype=int, min=10) — no silent clamp
        bc = bias_correction
        xv = x.to_numpy(dtype=float)
        yv = y.to_numpy(dtype=float)
        # R11 #14: raw history = window + lag bars so every lag reports exactly
        # ``window`` aligned pairs (the rolling kernel consumes window+lag raw
        # bars, then the lag-shift pairs x[..:raw-lag] with y[lag:..]).
        raw_w = w + lag_v

        def _fn(a: np.ndarray, b: np.ndarray) -> float:
            n = a.size
            if n < raw_w:
                return np.nan
            x_lead = a[: n - lag_v]
            y_lag = b[lag_v:]
            pa, pb = aligned_pairs(x_lead, y_lag)
            if pa.size < mp:
                return np.nan
            return _quantile_hist_mi(pa, pb, nb, False, bias_correction=bc)

        return frame_like(x, map_pair_rolling(xv, yv, raw_w, _fn))


def _fractional_tail_membership(vals: np.ndarray, quantile: float, direction: str) -> np.ndarray:
    """Fractional tail-membership weights in [0, 1] for a fixed quantile ``q``.

    ``quantile`` is the q LEVEL (same convention as the operator's ``q``:
    ``(0.5, 1)`` for ``upper``, ``(0, 0.5)`` for ``lower``).  The tail is the
    strict members plus a FRACTION of the boundary-tie group so the effective
    tail mass equals the target tail size exactly:

    * upper: ``count(v > Q(q)) + f * count(v == Q(q)) = (1 - q) * n``
    * lower: ``count(v < Q(q)) + f * count(v == Q(q)) = q * n``

    where ``Q(q) = np.quantile(vals, q)`` and ``f`` is the boundary fraction.
    On tie-free (continuous) data the boundary group is empty and the weights
    collapse to the strict 0/1 indicator.  The fractional split keeps the
    effective tail mass symmetric for discrete A-share-style data (0 returns,
    limit up/down) where the old strict/inclusive pair ``x > Q`` vs ``x <= Q``
    was asymmetric (R11 P1 tie asymmetry).
    """
    v = np.asarray(vals, dtype=float)
    n = v.size
    if n == 0:
        return np.zeros(0, dtype=float)
    thr = float(np.quantile(v, quantile))
    if direction == "upper":
        strict = v > thr
        target = (1.0 - quantile) * n
    elif direction == "lower":
        strict = v < thr
        target = quantile * n
    else:
        raise ValueError("direction must be 'upper' or 'lower'")
    boundary = v == thr
    weights = np.zeros(n, dtype=float)
    weights[strict] = 1.0
    n_boundary = int(boundary.sum())
    if n_boundary:
        n_strict = float(weights.sum())
        # Clamp to [0, 1]: under a pathological quantile rank the boundary group
        # may not span the whole gap to the target (treat it fail-closed).
        fraction = np.where(n_boundary, 0.0, 1.0)) != 0, float(np.clip((target - n_strict) / n_boundary, 0.0, 1.0)), np.nan)
        weights[boundary] = fraction
    return weights


def _tail_dependence(a: np.ndarray, b: np.ndarray, q: float, upper: bool, min_tail_count: int) -> float:
    """P(b in tail | a in tail) — directional, conditioned on ``a`` (source).

    ``a`` is the conditioning / source series (operator param ``x``), ``b`` the
    target series (operator param ``y``).  This is NOT a symmetric copula
    dependence: it answers "given the SOURCE is in its tail, how likely is the
    TARGET to be in its own tail".

    Fractional boundary-tie membership (R11 P1): both tails use the SAME
    ``_fractional_tail_membership`` weights so the effective tail mass on each
    side equals the target tail size even under heavy ties.  The joint is the
    weighted product ``x_w * y_w`` and the denominator is the source weight sum.
    """
    n = a.size
    if n < 4:
        return np.nan
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    direction = "upper" if upper else "lower"
    x_w = _fractional_tail_membership(a, q, direction)
    y_w = _fractional_tail_membership(b, q, direction)
    denom = float(x_w.sum())
    if denom < max(2, int(min_tail_count)):
        return np.nan
    joint = float(np.sum(x_w * y_w))
    return np.where(denom != 0, joint / denom, np.nan)


@register_operator(
    name="ts_upper_tail_coexceedance_probability",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_upper_tail_coexceedance_probability",
    source="nonlinear_dependence",
)
class TsUpperTailDependence(SeriesOperator):
    """固定 q 的上尾同超概率 P(y > Qy(q) | x > Qx(q))，有方向：以 source x 为条件。

    这是固定 q 的尾部同超概率（tail coexceedance probability），不是渐近
    尾部依赖系数 λ = lim_{q→1} P(y > Fy⁻¹(q) | x > Fx⁻¹(q))。旧名
    ``ts_upper_tail_dependence`` 保留为弃用别名。
    方向约定 (R5 P1-39(b))：输出是 P(y > Qy(q) | x > Qx(q))，即以 x（source/
    条件变量）的右尾为条件，衡量 target y 是否跟随。不是对称的 copula
    依赖；交换 x/y 会得到不同的数值。
    """

    metadata = _metadata(
        "ts_upper_tail_coexceedance_probability",
        "固定 q 的上尾同超概率（tail coexceedance probability）"
        " P(y > Qy(q) | x > Qx(q))：以 source x 的 q 分位数为条件、target y 跟随"
        " 的条件概率；是固定 q 的同超概率，而非渐近尾部依赖系数 λ；旧名"
        " ts_upper_tail_dependence 为弃用别名。条件样本不足返回 NaN。",
        ["x", "y", "window", "q", "min_tail_count"],
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, window: int = 60, q: float = 0.9, min_tail_count: int = 5, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        quantile = float(q)
        # R5 P1-39(c): upper tail is only meaningful past the median.
        if not 0.5 < quantile < 1.0:
            raise ValueError("q must be in (0.5, 1) for the upper tail")
        min_tail = max(2, int(min_tail_count))
        xv = x.to_numpy(dtype=float)
        yv = y.to_numpy(dtype=float)

        def _fn(a: np.ndarray, b: np.ndarray) -> float:
            pa, pb = aligned_pairs(a, b)
            return _tail_dependence(pa, pb, quantile, True, min_tail)

        return frame_like(x, map_pair_rolling(xv, yv, w, _fn))


@register_operator(
    name="ts_lower_tail_coexceedance_probability",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_lower_tail_coexceedance_probability",
    source="nonlinear_dependence",
)
class TsLowerTailDependence(SeriesOperator):
    """固定 q 的下尾同超概率 P(y ≤ Qy(q) | x ≤ Qx(q))，有方向：以 source x 为条件。

    这是固定 q 的尾部同超概率（tail coexceedance probability），不是渐近
    尾部依赖系数 λ = lim_{q→0} P(y < Fy⁻¹(q) | x < Fx⁻¹(q))。旧名
    ``ts_lower_tail_dependence`` 保留为弃用别名。
    方向约定 (R5 P1-39(b))：输出是 P(y ≤ Qy(q) | x ≤ Qx(q))，即以 x（source/
    条件变量）的左尾为条件，衡量 target y 是否跟随。不是对称的 copula
    依赖；交换 x/y 会得到不同的数值。
    """

    metadata = _metadata(
        "ts_lower_tail_coexceedance_probability",
        "固定 q 的下尾同超概率（tail coexceedance probability）"
        " P(y ≤ Qy(q) | x ≤ Qx(q))：以 source x 的 q 分位数为条件、target y 跟随"
        " 的条件概率；是固定 q 的同超概率，而非渐近尾部依赖系数 λ；旧名"
        " ts_lower_tail_dependence 为弃用别名。条件样本不足返回 NaN。",
        ["x", "y", "window", "q", "min_tail_count"],
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, window: int = 60, q: float = 0.1, min_tail_count: int = 5, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        quantile = float(q)
        # R5 P1-39(c): lower tail is only meaningful below the median.
        if not 0.0 < quantile < 0.5:
            raise ValueError("q must be in (0, 0.5) for the lower tail")
        min_tail = max(2, int(min_tail_count))
        xv = x.to_numpy(dtype=float)
        yv = y.to_numpy(dtype=float)

        def _fn(a: np.ndarray, b: np.ndarray) -> float:
            pa, pb = aligned_pairs(a, b)
            return _tail_dependence(pa, pb, quantile, False, min_tail)

        return frame_like(x, map_pair_rolling(xv, yv, w, _fn))


_NEW_CANONICALS = (
    "ts_upper_tail_coexceedance_probability",
    "ts_lower_tail_coexceedance_probability",
)
# Round-11 integration: every canonical registered by this module (the renamed
# coexceedance pair plus the pre-existing dependence operators) must stay
# classified EXTENDED — the static surface partition gate counts every
# registered operator against the partitions.
_MODULE_CANONICALS = (
    "ts_distance_corr",
    "ts_distance_cov",
    "ts_mutual_information",
    "ts_lagged_mutual_information",
    "ts_upper_tail_coexceedance_probability",
    "ts_lower_tail_coexceedance_probability",
)
# R11 P1 honest naming: these operators compute a FIXED-q tail coexceedance
# probability, NOT the asymptotic tail-dependence coefficient λ.  The canonical
# names now say what they compute; the old names remain as deprecated resolving
# aliases so existing recipes keep loading.
_DEPRECATED_ALIASES = {
    "ts_upper_tail_dependence": "ts_upper_tail_coexceedance_probability",
    "ts_lower_tail_dependence": "ts_lower_tail_coexceedance_probability",
}


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface
    from cleaned_operators.registry import OperatorRegistry

    # Round-11 integration: the whole module's canonical set must be classified
    # EXTENDED — WS-E's rename replaced the old full union with only the new
    # names, silently de-classifying the pre-existing dependence operators
    # (ts_distance_corr / ts_mutual_information / …) and breaking the static
    # surface partition gate.
    _surface.extend_extended_only(set(_MODULE_CANONICALS))
    for _old, _new in _DEPRECATED_ALIASES.items():
        try:
            OperatorRegistry.register_alias(_old, _new)
        except (KeyError, ValueError):
            pass  # already registered
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
