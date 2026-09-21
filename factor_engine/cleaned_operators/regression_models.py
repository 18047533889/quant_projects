# -*- coding: utf-8 -*-
"""Model-type rolling operators (2026-08 P2 expansion).

Robust / ridge / quantile regression residuals, AR coefficient, variance
ratio and structural-shift scores.  All kernels are causal rolling per-column
operations over ``timestamp x instrument`` panels.

Honest-naming review (R11 round-2):
- ``ts_variance_ratio`` -> ``ts_variance_ratio_proxy`` (a VR PROXY over
  differenced price/log-price LEVELS, not the Lo–MacKinlay estimator);
  ``ts_lo_mackinlay_vr`` / ``ts_lo_mackinlay_z`` are the proper overlapping
  Lo–MacKinlay estimator and its heteroskedasticity-robust z-statistic.
- ``ts_cusum_break_score`` -> ``ts_cumulative_deviation_score`` (a heuristic
  max-absolute standardized cumulative deviation, NOT a CUSUM test statistic).
- ``ts_huber_regression_resid`` / ``ts_ridge_regression_resid`` split into
  explicit ``*_in_sample_resid`` and ``*_predictive_resid`` canonicals; the
  old names remain as deprecated aliases of the in-sample variants.
- ``ts_level_shift_score`` / ``ts_vol_shift_score`` preserve PHYSICAL time:
  the trailing contiguous finite run is split at the window midpoint, never
  drop-finite-compressed across a NaN gap.
"""
from __future__ import annotations

from typing import Any
from factor_engine.cleaned_operators.parameter_validation import strict_integer, strict_finite_scalar

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from factor_engine.cleaned_operators.base import OperatorMetadata, ParamRole, ParamSpec, SeriesOperator, register_operator
from factor_engine.cleaned_operators.common.daily_panel import _aligned
from factor_engine.cleaned_operators.ts_model._rolling_core import (
    _HUBER_DELTA,
    huber_fit,
    pinball_quantile_fit,
    ridge_fit,
)

# Model-audit Phase 4 (search-space hygiene): explicit ParamSpec declarations.
# ``window`` is the alpha horizon (HORIZON, searched); ``min_periods`` is a
# statistical-support floor; ``alpha`` is the ridge shrinkage (numerical
# policy); ``lag`` in ``ts_ar_coefficient`` is the AR lag being estimated — the
# economic mechanism, so it is searched (M-115/M-162/M-170).
#
# P1 (window-semantics governance): ``window`` is a MAX LOOKBACK, not a strict
# full window; ``warmup_policy`` is catalog-visible (part of the semantic
# identity), with ``"full"`` opting into a strict full-history floor.
#
# P1 (Huber / Ridge estimator-policy governance): Huber's ``delta`` is a FIXED,
# versioned NUMERICAL policy (``ts_model._rolling_core._HUBER_DELTA``, M-055) —
# it is NOT an operator parameter, never searchable, and any change must ship as
# a new semantic version.  Ridge's ``alpha`` IS an exposed scalar: it is
# declared NUMERICAL + searchable=False (never a continuous AlphaProbe/LLM
# search dimension), and ``_RIDGE_ALPHA_CERTIFIED_PRESETS`` is the reviewed
# preset domain a role-aware search / LLM may sample from — it is a documented
# governance list, not a hard runtime rejection of arbitrary recipe values.
_HUBER_DELTA_POLICY = _HUBER_DELTA
_HUBER_DELTA_ROLE = ParamRole.NUMERICAL
_RIDGE_ALPHA_CERTIFIED_PRESETS = (0.0, 0.1, 0.5, 1.0)
_REGRESSION_PARAM_SPECS: dict[str, ParamSpec] = {
    "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON, searchable=True),
    "min_periods": ParamSpec(dtype=int, min=1, param_role=ParamRole.SUPPORT_POLICY, searchable=False),
    "alpha": ParamSpec(dtype=float, min=0.0, param_role=ParamRole.NUMERICAL, searchable=False),
    "lag": ParamSpec(dtype=int, min=1, param_role=ParamRole.ECONOMIC, searchable=True),
    "warmup_policy": ParamSpec(dtype=str, choices=("expanding", "full"),
                               param_role=ParamRole.POLICY, searchable=False),
}
_AR_WINDOW_SEMANTICS = "max_lookback"


_REMAINING_MODEL_NAMES = {
    "ts_quantile_regression_slope", "ts_variance_ratio_proxy", "ts_lo_mackinlay_vr",
    "ts_lo_mackinlay_z", "ts_cumulative_deviation_score", "ts_level_shift_score", "ts_vol_shift_score",
}

def _remaining_specs(name):
    variance=name in {"ts_variance_ratio_proxy","ts_lo_mackinlay_vr","ts_lo_mackinlay_z"}
    floor=7 if name=="ts_variance_ratio_proxy" else 6 if variance else 4 if name in {"ts_level_shift_score","ts_vol_shift_score"} else 3
    specs={
        "window":ParamSpec(dtype=int,min=20 if name=="ts_level_shift_score" else floor,default=60 if variance else 20,param_role=ParamRole.HORIZON),
        "min_periods":ParamSpec(dtype=int,min=1,default=10 if variance else 5,param_role=ParamRole.SUPPORT_POLICY,searchable=False),
    }
    if variance:
        specs["q"]=ParamSpec(dtype=int,min=2,default=5,param_role=ParamRole.HORIZON)
    elif name=="ts_quantile_regression_slope":
        specs["q"]=ParamSpec(dtype=float,min=np.nextafter(0.,1.),max=np.nextafter(1.,0.),default=.5,param_role=ParamRole.ECONOMIC)
    return specs

def _remaining_window(window,min_periods,floor,*,levels_extra=0):
    w=strict_integer(window,"window",minimum=floor+levels_extra)
    mp=max(floor,strict_integer(min_periods,"min_periods",minimum=1))
    if mp+levels_extra>w:
        raise ValueError("min_periods and estimator support must fit within window")
    return w,mp

def _scale_finite(values):
    finite=values[np.isfinite(values)]
    scale=np.max(np.abs(finite)) if finite.size else 0.
    return values/scale if scale>0 else values


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    domain: str,
    unit: str,
    output_unit: str | None = None,
    input_units: dict[str, str] | None = None,
    compatible_units: dict[str, tuple[str, ...]] | None = None,
    param_specs: dict[str, ParamSpec] | None = None,
    diagnostic_only: bool = False,
) -> OperatorMetadata:
    tags = [
        "time_series_regression", "daily", "pit_safe", "causal", "typed_v2",
        f"signature:{','.join(params)}->series", f"domain:{domain}",
        f"unit:{unit}", "cost:2",
    ]
    if diagnostic_only:
        tags.append("diagnostic_only")
    return OperatorMetadata(
        name=name,
        category="time_series_regression",
        description=description,
        param_names=params,
        panel_params=tuple(p for p in params if p in {"x","y"}) if name in _REMAINING_MODEL_NAMES else (),
        scalar_params=tuple(p for p in params if p not in {"x","y"}) if name in _REMAINING_MODEL_NAMES else (),
        return_type="series",
        param_specs={k: v for k, v in (_remaining_specs(name) if name in _REMAINING_MODEL_NAMES else (param_specs or {})).items() if k in params},
        tags=tags,
        output_unit=output_unit,
        input_units=dict(input_units or {}),
        compatible_units=dict(compatible_units or {}),
    )


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def _trailing_contiguous(chunk: np.ndarray) -> np.ndarray:
    """Longest trailing contiguous finite run ending at the current row.

    A missing value breaks the time axis: rows on either side of a NaN are never
    re-paired as adjacent (no drop-finite/reconnect).  A NaN at the current row
    yields an empty block so the caller emits NaN (fail-closed) instead of
    silently reusing the last valid history (review P1-123).
    """
    n = chunk.shape[0]
    if n == 0 or not np.isfinite(chunk[-1]):
        return chunk[:0]
    end = n
    while end > 0 and np.isfinite(chunk[end - 1]):
        end -= 1
    return chunk[end:]


def _trailing_run_halves(segment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split the trailing contiguous run at the WINDOW's physical midpoint.

    Structural-change operators (level/vol shift) compare the first vs second
    half of the WINDOW in physical time (review R11 #12).  A NaN breaks the time
    axis, so only the trailing contiguous finite run ending at the current row
    participates — rows on the far side of a gap are never re-paired with rows
    on this side (no drop-finite compression of the whole window).  The split is
    anchored at the window's physical midpoint: ``first`` holds the run's rows
    that fall in the first physical half of the window, ``second`` the rest.  An
    empty half means the "before" (or "after") state is unobservable in physical
    time and the caller emits NaN (fail-closed).
    """
    vals = _trailing_contiguous(segment)
    if vals.size == 0:
        return vals[:0], vals[:0]
    n = segment.shape[0]
    gap = n - vals.size          # leading rows dropped by the trailing-run cut
    mid = n // 2                 # physical midpoint of the window (offset)
    first_len = max(0, min(mid, n) - gap)
    first_len = min(first_len, vals.size)
    return vals[:first_len], vals[first_len:]


def _rolling_apply_2d(values: np.ndarray, window: int, fn: Any, min_periods: int = 1) -> np.ndarray:
    rows, cols = values.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            start = max(0, row - window + 1)
            out[row, col] = fn(values[start : row + 1, col])
    return out


def _regression_resid(
    y: np.ndarray,
    x: np.ndarray,
    *,
    method: str,
    min_periods: int,
    alpha: float = 0.0,
    predictive: bool = False,
) -> float:
    """Residual at the CURRENT row of ``y`` on ``x`` (``predictive=False``), or
    the out-of-sample residual at the current row after fitting on the
    PRECEDING window only (``predictive=True``).

    ``predictive=True`` fits on ``[t-W, t-1]`` (the trailing window without the
    current observation) and scores observation ``t``: the fitted β has NOT seen
    ``(x_t, y_t)``, so the residual is a genuine innovation (no look-ahead).
    ``predictive=False`` keeps the legacy in-sample behaviour where the fit
    includes the current row.
    """
    if predictive:
        fit_y, fit_x = y[:-1], x[:-1]
    else:
        fit_y, fit_x = y, x
    valid = np.isfinite(fit_y) & np.isfinite(fit_x)
    if valid.sum() < max(min_periods, 3):
        return np.nan
    xs = fit_x[valid]
    ys = fit_y[valid]
    if np.std(xs) <= 0.0:
        return np.nan
    design = np.column_stack([np.ones(len(xs)), xs])
    if method == "ols":
        beta, *_ = np.linalg.lstsq(design, ys, rcond=None)
    elif method == "ridge":
        beta = ridge_fit(design, ys, alpha, has_intercept=True)
        if beta is None:
            return np.nan
    elif method == "huber":
        beta = _huber_fit(design, ys)
        if beta is None:
            return np.nan
    else:
        raise ValueError(f"unknown method: {method}")
    # 当前样本残差（最后一行）
    y_cur = y[-1]
    x_cur = x[-1]
    if not np.isfinite(y_cur) or not np.isfinite(x_cur):
        return np.nan
    return float(y_cur - (beta[0] + beta[1] * x_cur))


def _huber_fit(
    design: np.ndarray,
    ys: np.ndarray,
    *,
    delta: float = _HUBER_DELTA,
    iterations: int = 100,
) -> np.ndarray | None:
    """Compatibility wrapper around the single shared Huber kernel."""
    return huber_fit(design, ys, delta=delta, iterations=iterations)


@register_operator(
    name="ts_huber_regression_in_sample_resid",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_huber_regression_in_sample_resid",
    source="regression_models",
    status="experimental",
)
class TsHuberRegressionInSampleResid(SeriesOperator):
    """Huber 稳健回归 IN-SAMPLE 残差（拟合含当前样本 (x_t, y_t)）。

    The fitted β has already seen the current observation, so the residual is
    self-explanatory, not an innovation.  DIAGNOSTIC use only — auto-mining
    should prefer ``ts_huber_regression_predictive_resid`` (no look-ahead).
    The legacy name ``ts_huber_regression_resid`` resolves here as a deprecated
    alias.
    """

    metadata = _metadata(
        "ts_huber_regression_in_sample_resid",
        "Huber 稳健回归 in-sample 残差（拟合含当前样本；诊断用，自动挖掘请用 predictive 变体）。Huber delta 为固定 NUMERICAL 政策（_HUBER_DELTA=1.345，versioned，不可搜索）。",
        ["y", "x", "window", "min_periods"],
        domain="price_volume",
        unit="level",
        param_specs=_REGRESSION_PARAM_SPECS,
        diagnostic_only=True,
    )

    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, window: int = 20, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        y, x = _aligned(y, x)
        w = int(window)
        mp = max(3, int(min_periods))
        yv = y.to_numpy(dtype=float)
        xv = x.to_numpy(dtype=float)
        rows, cols = yv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(0, row - w + 1)
                out[row, col] = _regression_resid(yv[start : row + 1, col], xv[start : row + 1, col], method="huber", min_periods=mp)
        return _frame_like(y, out)


@register_operator(
    name="ts_huber_regression_predictive_resid",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_huber_regression_predictive_resid",
    source="regression_models",
    status="experimental",
)
class TsHuberRegressionPredictiveResid(SeriesOperator):
    """Huber 稳健回归 PREDICTIVE 残差（拟合 [t-W, t-1]，评分 t）。

    The fitted β has NOT seen ``(x_t, y_t)``, so the residual is a genuine
    innovation (no look-ahead).  Auto-mining should prefer this predictive
    variant over the in-sample diagnostic.
    """

    metadata = _metadata(
        "ts_huber_regression_predictive_resid",
        "Huber 稳健回归 predictive 残差（拟合不含当前样本；无前视，自动挖掘首选）。Huber delta 为固定 NUMERICAL 政策（_HUBER_DELTA=1.345，versioned，不可搜索）。",
        ["y", "x", "window", "min_periods"],
        domain="price_volume",
        unit="level",
        param_specs=_REGRESSION_PARAM_SPECS,
    )

    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, window: int = 20, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        y, x = _aligned(y, x)
        w = int(window)
        mp = max(3, int(min_periods))
        yv = y.to_numpy(dtype=float)
        xv = x.to_numpy(dtype=float)
        rows, cols = yv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                # PREDICTIVE maturity fix: need W+1 rows [t-W, t] so [:-1] gives
                # exactly W prior observations [t-W, t-1] for fitting.
                start = max(0, row - w)
                out[row, col] = _regression_resid(yv[start : row + 1, col], xv[start : row + 1, col], method="huber", min_periods=mp, predictive=True)
        return _frame_like(y, out)


@register_operator(
    name="ts_ridge_regression_in_sample_resid",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_ridge_regression_in_sample_resid",
    source="regression_models",
    status="experimental",
)
class TsRidgeRegressionInSampleResid(SeriesOperator):
    """Ridge 回归 IN-SAMPLE 残差（拟合含当前样本 (x_t, y_t)）。

    The fitted β has already seen the current observation, so the residual is
    self-explanatory, not an innovation.  DIAGNOSTIC use only — auto-mining
    should prefer ``ts_ridge_regression_predictive_resid`` (no look-ahead).
    The legacy name ``ts_ridge_regression_resid`` resolves here as a deprecated
    alias.
    """

    metadata = _metadata(
        "ts_ridge_regression_in_sample_resid",
        "Ridge 回归 in-sample 残差（拟合含当前样本；诊断用，自动挖掘请用 predictive 变体）。alpha=NUMERICAL 政策（searchable=False，不可连续搜索；certified presets 见 _RIDGE_ALPHA_CERTIFIED_PRESETS）。",
        ["y", "x", "window", "alpha", "min_periods"],
        domain="price_volume",
        unit="level",
        param_specs=_REGRESSION_PARAM_SPECS,
        diagnostic_only=True,
    )

    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, window: int = 20, alpha: float = 0.1, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        y, x = _aligned(y, x)
        w = int(window)
        mp = max(3, int(min_periods))
        a = float(alpha)
        if a < 0.0:
            raise ValueError("alpha must be non-negative")
        yv = y.to_numpy(dtype=float)
        xv = x.to_numpy(dtype=float)
        rows, cols = yv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(0, row - w + 1)
                out[row, col] = _regression_resid(yv[start : row + 1, col], xv[start : row + 1, col], method="ridge", min_periods=mp, alpha=a)
        return _frame_like(y, out)


@register_operator(
    name="ts_ridge_regression_predictive_resid",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_ridge_regression_predictive_resid",
    source="regression_models",
    status="experimental",
)
class TsRidgeRegressionPredictiveResid(SeriesOperator):
    """Ridge 回归 PREDICTIVE 残差（拟合 [t-W, t-1]，评分 t）。

    The fitted β has NOT seen ``(x_t, y_t)``, so the residual is a genuine
    innovation (no look-ahead).  Auto-mining should prefer this predictive
    variant over the in-sample diagnostic.
    """

    metadata = _metadata(
        "ts_ridge_regression_predictive_resid",
        "Ridge 回归 predictive 残差（拟合不含当前样本；无前视，自动挖掘首选）。alpha=NUMERICAL 政策（searchable=False，不可连续搜索；certified presets 见 _RIDGE_ALPHA_CERTIFIED_PRESETS）。",
        ["y", "x", "window", "alpha", "min_periods"],
        domain="price_volume",
        unit="level",
        param_specs=_REGRESSION_PARAM_SPECS,
    )

    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, window: int = 20, alpha: float = 0.1, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        y, x = _aligned(y, x)
        w = int(window)
        mp = max(3, int(min_periods))
        a = float(alpha)
        if a < 0.0:
            raise ValueError("alpha must be non-negative")
        yv = y.to_numpy(dtype=float)
        xv = x.to_numpy(dtype=float)
        rows, cols = yv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                # PREDICTIVE maturity fix: need W+1 rows [t-W, t] so [:-1] gives
                # exactly W prior observations [t-W, t-1] for fitting.
                start = max(0, row - w)
                out[row, col] = _regression_resid(yv[start : row + 1, col], xv[start : row + 1, col], method="ridge", min_periods=mp, alpha=a, predictive=True)
        return _frame_like(y, out)


@register_operator(
    name="ts_quantile_regression_slope",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_quantile_regression_slope",
    source="regression_models",
    status="experimental",
)
class TsQuantileRegressionSlope(SeriesOperator):
    """分位数回归斜率（单分位，pinball-loss 线性规划）。"""

    metadata = _metadata(
        "ts_quantile_regression_slope",
        "分位数回归斜率。",
        ["y", "x", "window", "q", "min_periods"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, window: int = 20, q: float = 0.5, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        y, x = _aligned(y, x)
        w, mp = _remaining_window(window,min_periods,3,levels_extra=0)
        quantile = strict_finite_scalar(q,"q")
        if not (0.0 < quantile < 1.0):
            raise ValueError("q must be in (0,1)")
        yv = y.to_numpy(dtype=float)
        xv = x.to_numpy(dtype=float)
        rows, cols = yv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(0, row - w + 1)
                out[row, col] = _quantile_slope(yv[start : row + 1, col], xv[start : row + 1, col], quantile, mp)
        return _frame_like(y, out)


def _quantile_slope(y: np.ndarray, x: np.ndarray, q: float, min_periods: int) -> float:
    """True pinball-loss quantile slope (Koenker–Bassett LP)."""
    valid = np.isfinite(y) & np.isfinite(x)
    if valid.sum() < max(min_periods, 3):
        return np.nan
    xs = x[valid]
    ys = y[valid]
    if np.std(xs) <= 0.0:
        return np.nan
    design = np.column_stack([np.ones(len(xs)), xs])
    beta = pinball_quantile_fit(design, ys, float(q))
    if beta is None:
        return np.nan
    return float(beta[1])


@register_operator(
    name="ts_ar_coefficient",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_ar_coefficient",
    source="regression_models",
    status="experimental",
)
class TsArCoefficient(SeriesOperator):
    """AR(lag) 系数：x_t 对 x_{t-lag} 的回归斜率。

    IN-SAMPLE DESCRIPTIVE (audit M-040): the window used to compute the
    covariance includes the current row, so the reported coefficient is a
    trailing descriptive AR estimate over ``[t-w+1, t]`` — NOT a strictly-prior
    fit.  It contains no future data (PIT-safe) but the current observation is a
    member of the fitted window, so the output must be read as in-sample state,
    not a predictive coefficient.  The strictly-prior variant that fits only on
    rows ``<= t-1`` is ``ts_ar_prior_coeff``.
    """

    metadata = _metadata(
        "ts_ar_coefficient",
        "AR(lag) 回归系数（IN-SAMPLE：拟合窗口含当前样本，描述性状态，非预测；严格截至 t-1 版本用 ts_ar_prior_coeff）。window=max lookback（非严格满窗），min_effective_obs=min_periods，warmup_policy=expanding 渐进输出。",
        ["x", "window", "lag", "min_periods", "warmup_policy"],
        domain="price_volume",
        unit="ratio",
        param_specs=_REGRESSION_PARAM_SPECS,
    )
    metadata.window_semantics = _AR_WINDOW_SEMANTICS

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, lag: int = 1, min_periods: int = 5,
                          warmup_policy: str = "expanding", **_: Any) -> pd.DataFrame:
        if warmup_policy not in ("expanding", "full"):
            raise ValueError(f"warmup_policy must be 'expanding' or 'full', got {warmup_policy!r}")
        w = int(window)
        lg = max(1, int(lag))
        mp = max(3, int(min_periods))
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                # P1: strict full-history floor when warmup_policy="full".
                if warmup_policy == "full" and row < w - 1:
                    continue
                # ``window`` counts source observations, matching the public
                # trailing-window contract and the other AR implementation.
                # Lag construction therefore yields ``window-lag`` pairs.
                start = max(0, row - w + 1)
                if row - start < lg:
                    # insufficient history to form even one lag pair
                    continue
                # IN-SAMPLE (audit M-040): the segment is ``[row-w+1, row]`` —
                # it INCLUDES the current row, so the fitted AR(lag) covariance
                # is a trailing descriptive estimate, NOT a strict-prior fit.
                # ``ts_ar_prior_coeff`` (ar_meanrev) is the strictly t-1 variant.
                segment = xv[start : row + 1, col]
                current = segment[lg:]
                lagged = segment[:-lg]
                valid = np.isfinite(current) & np.isfinite(lagged)
                if valid.sum() < mp or np.std(lagged[valid]) <= 0.0:
                    continue
                cov = float(np.mean((current[valid] - np.mean(current[valid])) * (lagged[valid] - np.mean(lagged[valid]))))
                var = float(np.var(lagged[valid]))
                out[row, col] = cov / var if var > 0.0 else np.nan
        return _frame_like(x, out)




# ---------------------------------------------------------------------------
# R62 vectorised kernels for the three remaining per-row loops.
#
# ``_scale_finite`` (divide the segment by its largest finite magnitude) and
# the trailing-contiguous cohort are both preserved bit-for-bit, but as one
# batch per column: a single ``(n, w)`` sliding window matrix holds
# ``xv[max(0,t-w+1) : t+1]``, the trailing run is the last ``L`` columns, and
# every mean / variance / lag-shift is a masked row reduction.  Scaling each
# row by its own window magnitude is what the authority does, so even
# knife-edge panels (exactly-equal increments) keep the same underflow pattern.
# ---------------------------------------------------------------------------
def _scaled_window(col: np.ndarray, fin: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """``_scale_finite`` segment matrix + trailing contiguous run length."""
    n = col.size
    pad = np.concatenate([np.full(w - 1, np.nan, dtype=float), col])
    win = sliding_window_view(pad, w)[:n]        # row t -> x[max(0,t-w+1) .. t]
    with np.errstate(invalid="ignore", divide="ignore"):
        mag = np.where(np.isfinite(win), np.abs(win), 0.0).max(axis=1)
        sw = win / np.where(mag > 0.0, mag, 1.0)[:, None]
    ar = np.arange(n)
    last_bad = np.maximum.accumulate(np.where(fin, -1, ar))
    run = np.minimum(np.where(fin, ar - last_bad, 0), w)
    return sw, run


def _iter_run_groups(run: np.ndarray):
    """Yield ``(rows, L)`` groups of equal trailing-run length, largest first.

    Every row is reduced on its own *compacted* ``(k, L)`` trailing-run block
    rather than on the padded ``(k, w)`` window.  NumPy's pairwise summation
    groups by position, so a padded row reproduces the last-bit rounding of a
    length-``w`` array instead of the length-``L`` array the authority sums —
    and on knife-edge windows (exactly-equal increments, true variance ~1e-34)
    that difference is amplified into O(1) relative error.  Batching by ``L``
    is the only layout whose rounding is bit-identical to HEAD's per-row call.
    """
    nz = np.flatnonzero(run)
    if nz.size == 0:
        return
    r = run[nz]
    order = np.argsort(r, kind="stable")
    srt, rs = nz[order], r[order]
    cut = np.flatnonzero(np.concatenate(([True], rs[1:] != rs[:-1])))
    ends = np.concatenate((cut[1:], [rs.size]))
    for a, b in zip(cut, ends):
        yield srt[a:b], int(rs[a])


def _variance_ratio_proxy_col(col: np.ndarray, w: int, mp: int, q: int) -> np.ndarray:
    """``ts_variance_ratio_proxy`` kernel for one column (batched by run length)."""
    n = col.size
    res = np.full(n, np.nan, dtype=float)
    fin = np.isfinite(col)
    if n == 0 or not np.any(fin):
        return res
    sw, run = _scaled_window(col, fin, w)
    for idx, L in _iter_run_groups(run):
        # gates mirror the authority exactly: vals.size < mp / rets.size < mp /
        # qrets.size < 2 (== ``L - q < 2``).
        if L < mp or L - 1 < mp or L - q < 2:
            continue
        vals = sw[idx, w - L:]
        var1 = np.var(vals[:, 1:] - vals[:, :-1], axis=1)
        varq = np.var(vals[:, q:] - vals[:, :-q], axis=1)
        good = var1 > 0.0
        if not good.any():
            continue
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            val = varq / (q * np.where(good, var1, 1.0)) - 1.0
        res[idx[good]] = val[good]
    return res


def _lo_mackinlay_z_col(col: np.ndarray, w: int, mp: int, q: int) -> np.ndarray:
    """``ts_lo_mackinlay_z`` kernel for one column (batched by run length)."""
    n = col.size
    res = np.full(n, np.nan, dtype=float)
    fin = np.isfinite(col)
    if n == 0 or not np.any(fin) or q >= w:
        return res
    sw, run = _scaled_window(col, fin, w)
    weights = [(2.0 * (q - k) / q) ** 2 for k in range(1, int(q))]
    for idx, L in _iter_run_groups(run):
        nr = L - 1                      # number of 1-period returns
        # ``vals.size < mp`` and the ``rets.size < periods + 1`` feasibility gate.
        if L < mp or nr < q + 1:
            continue
        vals = sw[idx, w - L:]
        rets = vals[:, 1:] - vals[:, :-1]
        mu = rets.sum(axis=1) / nr
        dev = rets - mu[:, None]
        denom = (dev * dev).sum(axis=1)
        # ``_lo_mackinlay_vr`` uses the (n-1) denominator; <= 0 fails closed.
        sigma_a2 = denom / (nr - 1.0)
        good = sigma_a2 > 0.0
        if not good.any():
            continue
        # overlapping q-period return sums == np.convolve(rets, ones(q), "valid")
        qsum = sliding_window_view(rets, q, axis=1).sum(axis=2)
        m = q * (nr - q + 1) * (1.0 - q / nr)
        qdev = qsum - q * mu[:, None]
        sigma_c2 = (qdev * qdev).sum(axis=1) / m
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            vr = sigma_c2 / np.where(good, sigma_a2, 1.0)
        good &= np.isfinite(vr)
        if not good.any():
            continue
        den2 = denom * denom
        dev2 = dev * dev
        theta = np.zeros(rets.shape[0], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for k, wt in zip(range(1, int(q)), weights):
                num = (dev2[:, : nr - k] * dev2[:, k:]).sum(axis=1)
                theta += wt * (num / den2)
            z = (vr - 1.0) / np.sqrt(theta)
        good &= (theta > 0.0) & (m > 0.0)
        res[idx[good]] = z[good]
    return res


def _level_shift_score_col(col: np.ndarray, w: int, mp: int) -> np.ndarray:
    """``ts_level_shift_score`` kernel for one column (physical-midpoint split)."""
    n = col.size
    res = np.full(n, np.nan, dtype=float)
    fin = np.isfinite(col)
    if n == 0 or not np.any(fin):
        return res
    win, run = _scaled_window(col, fin, w)
    ar = np.arange(n)
    seg = np.minimum(ar + 1, w)
    lo = ar - seg + 1
    s = ar - run + 1
    first_len = np.clip(lo + seg // 2 - s, 0, run)
    second_len = run - first_len
    cols = np.arange(w)
    start = w - run
    in_run = cols[None, :] >= start[:, None]
    m_first = in_run & (cols[None, :] < (start + first_len)[:, None])
    m_second = cols[None, :] >= (start + first_len)[:, None]
    safe = np.maximum(run, 1)
    mu = np.where(in_run, win, 0.0).sum(axis=1) / safe
    sd = np.sqrt(np.where(in_run, (win - mu[:, None]) ** 2, 0.0).sum(axis=1) / safe)
    mf = np.where(m_first, win, 0.0).sum(axis=1) / np.maximum(first_len, 1)
    ms = np.where(m_second, win, 0.0).sum(axis=1) / np.maximum(second_len, 1)
    good = (run >= mp) & (first_len > 0) & (second_len > 0) & (sd > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = (ms - mf) / np.where(sd > 0.0, sd, 1.0)
    res[good] = val[good]
    return res


def _column_kernel(xv: np.ndarray, fn) -> np.ndarray:
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        out[:, c] = fn(xv[:, c])
    return out

def _lo_mackinlay_vr(rets: np.ndarray, q: int) -> float:
    """Overlapping Lo–MacKinlay variance-ratio estimator VR(q).

    ``rets`` is the trailing 1-period return series (diff of a price/log-price
    LEVEL input).  With ``r_t(q) = sum_{i=0}^{q-1} r_{t-i}`` and the LM
    finite-sample correction ``m = q*(n-q+1)*(1 - q/n)``,

        VR(q) = sigma_c^2(q) / sigma_a^2,
        sigma_a^2  = (1/(n-1)) sum (r_t - mu)^2,
        sigma_c^2  = (1/m)      sum (r_t(q) - q*mu)^2,

    so VR(q) == 1 for a random walk (drift-adjusted, overlapping returns).
    Sign convention (audit M-045): VR(q) > 1 means positive serial correlation
    (q-period variance grows more than linearly) -> a TRENDING tendency;
    VR(q) < 1 means negative serial correlation -> a MEAN-REVERSION tendency.
    Requires ``n > q`` (at least q+1 returns) and a positive 1-period variance.
    """
    n = rets.size
    if n < 2:
        return np.nan
    mu = float(np.mean(rets))
    sigma_a2 = float(np.sum((rets - mu) ** 2) / (n - 1))
    if sigma_a2 <= 0.0:
        return np.nan
    qrets = np.convolve(rets, np.ones(q, dtype=float), mode="valid")
    m = q * (n - q + 1) * (1.0 - q / n)
    if m <= 0.0:
        return np.nan
    sigma_c2 = float(np.sum((qrets - q * mu) ** 2) / m)
    return sigma_c2 / sigma_a2


def _lo_mackinlay_z(rets: np.ndarray, q: int) -> float:
    """Heteroskedasticity-robust Lo–MacKinlay z-statistic z*(q).

    ``z*(q) = (VR(q) - 1) / sqrt(theta_2)`` with the heteroskedasticity-robust
    variance ``theta_2 = sum_{k=1}^{q-1} (2(q-k)/q)^2 * delta_k`` and
    ``delta_k = sum (r_t-mu)^2 (r_{t-k}-mu)^2 / (sum (r_t-mu)^2)^2``
    (Lo–MacKinlay 1988, eq. 14/17).  ``VR`` comes from :func:`_lo_mackinlay_vr`.

    Sign convention (audit M-045): ``z > 0`` means ``VR > 1`` -> positive serial
    correlation -> TRENDING tendency; ``z < 0`` means ``VR < 1`` -> negative
    serial correlation -> MEAN-REVERSION tendency.
    """
    vr = _lo_mackinlay_vr(rets, q)
    if not np.isfinite(vr):
        return np.nan
    n = rets.size
    mu = float(np.mean(rets))
    dev = rets - mu
    denom = float(np.sum(dev ** 2))
    if denom <= 0.0:
        return np.nan
    theta = 0.0
    for k in range(1, q):
        delta_k = float(np.sum(dev[k:] ** 2 * dev[:-k] ** 2)) / (denom * denom)
        theta += (2.0 * (q - k) / q) ** 2 * delta_k
    if theta <= 0.0:
        return np.nan
    return (vr - 1.0) / float(np.sqrt(theta))


@register_operator(
    name="ts_variance_ratio_proxy",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_variance_ratio_proxy",
    source="regression_models",
    status="experimental",
)
class TsVarianceRatioProxy(SeriesOperator):
    """方差比代理：Var(q-period diff) / (q * Var(1-period diff)) - 1。

    HEURISTIC PROXY, NOT the full Lo–MacKinlay estimator/test statistic: it
    differences the input LEVELS and compares the q-period return variance to q
    times the 1-period return variance.  The input semantic contract is a
    PRICE / LOG-PRICE LEVEL (``PriceLevel``/``LogPriceLevel``); the typed layer
    reads that contract and REJECTS a return-typed input (fail-closed), because
    a Return fed here would silently become ΔReturn (a double difference) rather
    than a variance ratio.  Use ``ts_lo_mackinlay_vr`` / ``ts_lo_mackinlay_z``
    for the proper overlapping estimator and its z-test.  The legacy name
    ``ts_variance_ratio`` resolves here as a deprecated alias.

    Sign convention (audit M-045): a positive proxy (``Var(q)/q/Var(1) > 1``) ->
    positive serial correlation -> TRENDING tendency; a negative proxy ->
    MEAN-REVERSION tendency.
    """

    metadata = _metadata(
        "ts_variance_ratio_proxy",
        "方差比代理 Var(q)/q/Var(1)-1（启发式；要求价格/对数价格水平输入，typed 层拒绝收益输入）。",
        ["x", "window", "q", "min_periods"],
        domain="price_volume",
        unit="ratio",
        input_units={"x": "price_level_or_log_price_level"},
        compatible_units={"x": ("price_level", "log_price_level")},
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, q: int = 5, min_periods: int = 10, **_: Any) -> pd.DataFrame:
        w, mp = _remaining_window(window,min_periods,6,levels_extra=1)
        if w < 1:
            raise ValueError("window must be >= 1")
        periods = strict_integer(q,"q",minimum=2)
        # Feasibility validation (review P1-123): the q-period return needs at
        # least 2 overlapping q-period windows inside the trailing window, and a
        # degenerate q<2 used to be silently clamped.
        if periods < 2:
            raise ValueError("q must be >= 2")
        if periods > w-2:
            raise ValueError("q must be <= window - 2")
        xv = x.to_numpy(dtype=float)
        out = _column_kernel(xv, lambda col: _variance_ratio_proxy_col(col, w, mp, periods))
        return _frame_like(x, out)

@register_operator(
    name="ts_lo_mackinlay_vr",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_lo_mackinlay_vr",
    source="regression_models",
    status="experimental",
)
class TsLoMackinlayVr(SeriesOperator):
    """Lo–MacKinlay 重叠方差比 VR(q)（drift-adjusted，有限样本校正 m）。

    The proper overlapping Lo–MacKinlay variance-ratio estimator over q-period
    returns (VR == 1 for a random walk), computed on a trailing contiguous run
    of a PRICE / LOG-PRICE LEVEL input.  Deterministic.

    Sign convention (audit M-045): VR(q) > 1 -> positive serial correlation ->
    TRENDING tendency; VR(q) < 1 -> negative serial correlation ->
    MEAN-REVERSION tendency.  ``ts_lo_mackinlay_z`` is the heteroskedasticity-
    robust z-statistic of the same sign convention.
    """

    metadata = _metadata(
        "ts_lo_mackinlay_vr",
        "Lo–MacKinlay 重叠方差比 VR(q)（要求价格/对数价格水平输入，typed 层拒绝收益输入）。",
        ["x", "window", "q", "min_periods"],
        domain="price_volume",
        unit="ratio",
        input_units={"x": "price_level_or_log_price_level"},
        compatible_units={"x": ("price_level", "log_price_level")},
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, q: int = 5, min_periods: int = 10, **_: Any) -> pd.DataFrame:
        w, mp = _remaining_window(window,min_periods,6,levels_extra=0)
        if w < 1:
            raise ValueError("window must be >= 1")
        periods = strict_integer(q,"q",minimum=2)
        if periods < 2:
            raise ValueError("q must be >= 2")
        if periods > w-2:
            raise ValueError("q must be <= window - 2")
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(0, row - w + 1)
                segment = _scale_finite(xv[start : row + 1, col])
                vals = _trailing_contiguous(segment)
                if vals.size < mp:
                    continue
                rets = np.diff(vals)
                # overlapping q-period estimator needs at least q+1 returns
                if rets.size < periods + 1:
                    continue
                out[row, col] = _lo_mackinlay_vr(rets, periods)
        return _frame_like(x, out)


@register_operator(
    name="ts_lo_mackinlay_z",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_lo_mackinlay_z",
    source="regression_models",
    status="experimental",
)
class TsLoMackinlayZ(SeriesOperator):
    """Lo–MacKinlay 异方差稳健 z 检验统计量 z*(q)。

    ``(VR(q) - 1) / sqrt(theta_2)`` with the heteroskedasticity-robust variance
    ``theta_2`` — the z-statistic associated with :func:`_lo_mackinlay_vr`.
    Deterministic.  Sign convention (audit M-045): ``z > 0`` (``VR > 1``) ->
    positive serial correlation -> TRENDING tendency; ``z < 0`` (``VR < 1``) ->
    negative serial correlation -> MEAN-REVERSION tendency (Lo–MacKinlay).
    """

    metadata = _metadata(
        "ts_lo_mackinlay_z",
        "Lo–MacKinlay 异方差稳健 z 统计量 z*(q)（要求价格/对数价格水平输入，typed 层拒绝收益输入）。",
        ["x", "window", "q", "min_periods"],
        domain="price_volume",
        unit="ratio",
        input_units={"x": "price_level_or_log_price_level"},
        compatible_units={"x": ("price_level", "log_price_level")},
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, q: int = 5, min_periods: int = 10, **_: Any) -> pd.DataFrame:
        w, mp = _remaining_window(window,min_periods,6,levels_extra=0)
        if w < 1:
            raise ValueError("window must be >= 1")
        periods = strict_integer(q,"q",minimum=2)
        if periods < 2:
            raise ValueError("q must be >= 2")
        if periods > w-2:
            raise ValueError("q must be <= window - 2")
        xv = x.to_numpy(dtype=float)
        out = _column_kernel(xv, lambda col: _lo_mackinlay_z_col(col, w, mp, periods))
        return _frame_like(x, out)

@register_operator(
    name="ts_cumulative_deviation_score",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_cumulative_deviation_score",
    source="regression_models",
    status="experimental",
)
class TsCumulativeDeviationScore(SeriesOperator):
    """累计标准化偏差最大绝对值：max | cumsum((x - mean)/std) |。

    HEURISTIC factor, NOT a standard CUSUM change-point test statistic.  It
    standardises the window (subtract mean, divide by std), takes the running
    cumulative sum, and reports the maximum absolute value of the path — a
    level jump after the window mean makes the standardized deviations share a
    sign and the cumulative path grow.  The legacy name ``ts_cusum_break_score``
    resolves here as a deprecated alias.
    """

    metadata = _metadata(
        "ts_cumulative_deviation_score",
        "累计标准化偏差最大绝对值（启发式，非标准 CUSUM 检验统计量）。",
        ["x", "window", "min_periods"],
        domain="price_volume",
        unit="level",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w, mp = _remaining_window(window,min_periods,3,levels_extra=0)
        if w < 1:
            raise ValueError("window must be >= 1")
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(0, row - w + 1)
                segment = _scale_finite(xv[start : row + 1, col])
                # Trailing contiguous run: no drop-finite/reconnect (P1-123).
                vals = _trailing_contiguous(segment)
                if vals.size < mp:
                    continue
                mean = float(np.mean(vals))
                sd = float(np.std(vals))
                if sd <= 0.0:
                    continue
                # 到当前为止的累计标准化偏差。窗口总偏差对自身均值为 0，
                # 因此取路径上 |cumsum| 的最大值（末尾值无信息量）。
                running = np.cumsum((vals - mean) / sd)
                out[row, col] = float(np.max(np.abs(running)))
        return _frame_like(x, out)


@register_operator(
    name="ts_level_shift_score",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_level_shift_score",
    source="regression_models",
    status="experimental",
)
class TsLevelShiftScore(SeriesOperator):
    """水平位移得分：后半与前半窗口均值差，除以滚动标准差。

    Structural-change operator: physical time must be preserved.  Only the
    trailing contiguous finite run is used, split at the WINDOW's physical
    midpoint (no drop-finite compression across a NaN gap).  When the "before"
    or "after" half is unobservable in physical time the score is NaN.
    """

    metadata = _metadata(
        "ts_level_shift_score",
        "前后半窗口均值差 / 滚动标准差（物理时间中点切分，NaN 不跨缺口压缩）。",
        ["x", "window", "min_periods"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        window = strict_integer(window,"window",minimum=20)
        w, mp = _remaining_window(window,min_periods,4,levels_extra=0)
        xv = x.to_numpy(dtype=float)
        out = _column_kernel(xv, lambda col: _level_shift_score_col(col, w, mp))
        return _frame_like(x, out)

@register_operator(
    name="ts_vol_shift_score",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_vol_shift_score",
    source="regression_models",
    status="experimental",
)
class TsVolShiftScore(SeriesOperator):
    """波动率位移得分：后半与前半窗口标准差的对数比。

    Structural-change operator: physical time must be preserved.  Only the
    trailing contiguous finite run is used, split at the WINDOW's physical
    midpoint (no drop-finite compression across a NaN gap).  When the "before"
    or "after" half is unobservable in physical time (or too short to form a
    standard deviation) the score is NaN.
    """

    metadata = _metadata(
        "ts_vol_shift_score",
        "log(后半 std / 前半 std)（物理时间中点切分，NaN 不跨缺口压缩）。",
        ["x", "window", "min_periods"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w, mp = _remaining_window(window,min_periods,4,levels_extra=0)
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(0, row - w + 1)
                segment = _scale_finite(xv[start : row + 1, col])
                # Trailing contiguous run: a NaN must not re-pair rows on either
                # side of a gap (review R11 #12).
                vals = _trailing_contiguous(segment)
                if vals.size < mp:
                    continue
                first, second = _trailing_run_halves(segment)
                if first.size < 2 or second.size < 2:
                    continue
                sd1 = float(np.std(first))
                sd2 = float(np.std(second))
                if sd1 <= 0.0 or sd2 <= 0.0:
                    continue
                out[row, col] = float(np.log(sd2 / sd1))
        return _frame_like(x, out)


# ---------------------------------------------------------------------------
# R11 round-2 honest naming: every canonical registered by this module must stay
# classified EXTENDED (the static surface partition gate counts every registered
# operator against the partitions), and the legacy names resolve as deprecated
# aliases so existing recipes keep loading.
# ---------------------------------------------------------------------------
_MODULE_CANONICALS = (
    "ts_huber_regression_in_sample_resid",
    "ts_huber_regression_predictive_resid",
    "ts_ridge_regression_in_sample_resid",
    "ts_ridge_regression_predictive_resid",
    "ts_quantile_regression_slope",
    "ts_ar_coefficient",
    "ts_variance_ratio_proxy",
    "ts_lo_mackinlay_vr",
    "ts_lo_mackinlay_z",
    "ts_cumulative_deviation_score",
    "ts_level_shift_score",
    "ts_vol_shift_score",
)
# Legacy spellings -> new canonical (deprecated resolving aliases).
_DEPRECATED_ALIASES = {
    "ts_variance_ratio": "ts_variance_ratio_proxy",
    "ts_cusum_break_score": "ts_cumulative_deviation_score",
    "ts_huber_regression_resid": "ts_huber_regression_in_sample_resid",
    "ts_ridge_regression_resid": "ts_ridge_regression_in_sample_resid",
}


def _register_surface_and_aliases() -> None:
    from factor_engine.cleaned_operators.operator_surface import (
        extend_extended_only,
        retract_extended_only,
    )
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    # The legacy spellings are no longer ACTUAL canonicals (they now resolve as
    # aliases), so they must leave the static extended partition or the
    # layer-governance partition gate rejects them as inactive.  The new
    # canonicals (including the unchanged quantile/AR/level/vol ops) join it.
    retract_extended_only(set(_DEPRECATED_ALIASES))
    extend_extended_only(set(_MODULE_CANONICALS))
    for alias, canonical in _DEPRECATED_ALIASES.items():
        try:
            OperatorRegistry.register_alias(alias, canonical)
        except (KeyError, ValueError):
            pass  # already registered / not yet resolvable


_register_surface_and_aliases()
