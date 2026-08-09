# -*- coding: utf-8 -*-
"""Nonlinear dependence operators (2026-08 final pack, group 1).

Adds distance correlation/covariance, quantile-histogram mutual information
(and its lagged form) and upper/lower tail dependence.  All are causal
daily-panel transforms.

Contract notes
--------------
* Aligned pairs only: ``x[t]`` and ``y[t]`` are paired at the same position;
  NaN is never dropped and then re-compressed.
* ``min_periods`` counts *valid pairs*; below it the output is NaN.
* Constant windows and degenerate quantiles return NaN (fail-closed).
* Deterministic: the MI estimator is rank-quantile-binned (no randomness).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, ParamSpec, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import (
    aligned_pairs,
    check_window,
    frame_like,
    map_pair_rolling,
    register_polars_bridge,
)


def _metadata(name: str, description: str, params: list[str], *, unit: str) -> OperatorMetadata:
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
    dcov2 = float(np.sum(A * B)) / float(n * n)
    dvar_x2 = float(np.sum(A * A)) / float(n * n)
    dvar_y2 = float(np.sum(B * B)) / float(n * n)
    if dvar_x2 <= 1e-12 or dvar_y2 <= 1e-12:
        return np.nan, np.nan
    dcov = float(np.sqrt(max(dcov2, 0.0)))
    denom = float(np.sqrt(np.sqrt(dvar_x2) * np.sqrt(dvar_y2)))
    if denom <= 1e-12:
        return np.nan, dcov
    dcorr = float(min(1.0, max(0.0, dcov / denom)))
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

    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, window: int = 40, min_periods: int = 10, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        mp = max(10, int(min_periods))
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
        unit="level",
    )

    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, window: int = 40, min_periods: int = 10, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        mp = max(10, int(min_periods))
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
    p = cont / n
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
        mi = max(0.0, mi - float((r_occ - 1) * (c_occ - 1)) / (2.0 * n))
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
    return float(min(1.0, max(0.0, mi / denom)))


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
        "window": ParamSpec(dtype=int, min=2),
        "bins": ParamSpec(dtype=int, min=2, max=10),
    }

    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, window: int = 40, estimator: str = "quantile_hist", bins: int = 5, min_periods: int = 10, normalized: bool = True, bias_correction: bool = True, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        if str(estimator).lower() != "quantile_hist":
            raise ValueError("estimator must be 'quantile_hist' (deterministic)")
        nb = int(bins)
        if not 2 <= nb <= 10:
            raise ValueError("bins must be in [2, 10]")
        mp = max(10, int(min_periods))
        norm = bool(normalized)
        bc = bool(bias_correction)
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
        "window": ParamSpec(dtype=int, min=2),
        "bins": ParamSpec(dtype=int, min=2, max=10),
    }

    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, window: int = 40, lag: int = 1, bins: int = 5, min_periods: int = 10, bias_correction: bool = True, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        lag_v = int(lag)
        if lag_v < 0:
            raise ValueError("lag must be a non-negative integer")
        nb = int(bins)
        if not 2 <= nb <= 10:
            raise ValueError("bins must be in [2, 10]")
        mp = max(10, int(min_periods))
        bc = bool(bias_correction)
        xv = x.to_numpy(dtype=float)
        yv = y.to_numpy(dtype=float)

        def _fn(a: np.ndarray, b: np.ndarray) -> float:
            n = a.size
            if n <= lag_v:
                return np.nan
            x_lead = a[: n - lag_v]
            y_lag = b[lag_v:]
            pa, pb = aligned_pairs(x_lead, y_lag)
            if pa.size < mp:
                return np.nan
            return _quantile_hist_mi(pa, pb, nb, False, bias_correction=bc)

        return frame_like(x, map_pair_rolling(xv, yv, w, _fn))


def _tail_dependence(a: np.ndarray, b: np.ndarray, q: float, upper: bool, min_tail_count: int) -> float:
    """P(b in tail | a in tail) — directional, conditioned on ``a`` (source).

    ``a`` is the conditioning / source series (operator param ``x``), ``b`` the
    target series (operator param ``y``).  This is NOT a symmetric copula
    dependence: it answers "given the SOURCE is in its tail, how likely is the
    TARGET to be in its own tail".
    """
    n = a.size
    if n < 4:
        return np.nan
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    if upper:
        x_thr = float(np.quantile(a, q))
        y_thr = float(np.quantile(b, q))
        cond = a > x_thr
        joint = (a > x_thr) & (b > y_thr)
    else:
        x_thr = float(np.quantile(a, q))
        y_thr = float(np.quantile(b, q))
        cond = a <= x_thr
        joint = (a <= x_thr) & (b <= y_thr)
    count_cond = int(cond.sum())
    if count_cond < max(2, int(min_tail_count)):
        return np.nan
    return float(joint.sum()) / float(count_cond)


@register_operator(
    name="ts_upper_tail_dependence",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_upper_tail_dependence",
    source="nonlinear_dependence",
)
class TsUpperTailDependence(SeriesOperator):
    """上尾条件概率 P(y 在右尾 | x 在右尾)，有方向：以 source x 为条件。

    方向约定 (R5 P1-39(b))：输出是 P(y > Qy(q) | x > Qx(q))，即以 x（source/
    条件变量）的右尾为条件，衡量 target y 是否跟随。不是对称的 copula
    依赖；交换 x/y 会得到不同的数值。
    """

    metadata = _metadata(
        "ts_upper_tail_dependence",
        "方向性上尾依赖 P(target y 右尾 | source x 右尾)，条件样本不足返回 NaN。",
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
    name="ts_lower_tail_dependence",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_lower_tail_dependence",
    source="nonlinear_dependence",
)
class TsLowerTailDependence(SeriesOperator):
    """下尾条件概率 P(y 在左尾 | x 在左尾)，有方向：以 source x 为条件。

    方向约定 (R5 P1-39(b))：输出是 P(y ≤ Qy(q) | x ≤ Qx(q))，即以 x（source/
    条件变量）的左尾为条件，衡量 target y 是否跟随。不是对称的 copula
    依赖；交换 x/y 会得到不同的数值。
    """

    metadata = _metadata(
        "ts_lower_tail_dependence",
        "方向性下尾依赖 P(target y 左尾 | source x 左尾)，条件样本不足返回 NaN。",
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


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({
            "ts_distance_corr", "ts_distance_cov", "ts_mutual_information",
            "ts_lagged_mutual_information", "ts_upper_tail_dependence",
            "ts_lower_tail_dependence",
        })
    for _canon in (
        "ts_distance_corr", "ts_distance_cov", "ts_mutual_information",
        "ts_lagged_mutual_information", "ts_upper_tail_dependence",
        "ts_lower_tail_dependence",
    ):
        register_polars_bridge(_canon)


_register_surface()
