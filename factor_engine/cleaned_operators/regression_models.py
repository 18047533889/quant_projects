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

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.common.daily_panel import _aligned
from cleaned_operators.ts_model._rolling_core import pinball_quantile_fit


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
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="time_series_regression",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "time_series_regression", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", f"domain:{domain}",
            f"unit:{unit}", "cost:2",
        ],
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
        ridge = alpha * np.eye(design.shape[1])
        ridge[0, 0] = 0.0  # 不惩罚截距
        beta, *_ = np.linalg.lstsq(design.T @ design + ridge, design.T @ ys, rcond=None)
    elif method == "huber":
        beta = _huber_fit(design, ys)
    else:
        raise ValueError(f"unknown method: {method}")
    # 当前样本残差（最后一行）
    y_cur = y[-1]
    x_cur = x[-1]
    if not np.isfinite(y_cur) or not np.isfinite(x_cur):
        return np.nan
    return float(y_cur - (beta[0] + beta[1] * x_cur))


def _huber_fit(design: np.ndarray, ys: np.ndarray, *, delta: float = 1.345, iterations: int = 5) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(design, ys, rcond=None)
    for _ in range(iterations):
        resid = ys - design @ beta
        scale = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        if scale <= 0.0:
            scale = np.std(resid)
        if scale <= 0.0:
            break
        z = resid / scale
        abs_z = np.abs(z)
        with np.errstate(divide="ignore", invalid="ignore"):
            weight = np.where(abs_z <= delta, 1.0, delta / abs_z)
        # Huber IRLS minimizes sum(w_i e_i^2).  Weighted least squares solves
        # the normal equations with sqrt(w_i) applied to both design and
        # response (same convention as ``daily_panel._ols_residual``).
        # Weighting design/y by w (not sqrt(w)) minimized sum(w_i^2 e_i^2),
        # which over-shrinks outlier rows (review P0-01).
        root_w = np.sqrt(weight)
        wdesign = design * root_w[:, None]
        beta, *_ = np.linalg.lstsq(wdesign, ys * root_w, rcond=None)
    return beta


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
        "Huber 稳健回归 in-sample 残差（拟合含当前样本；诊断用，自动挖掘请用 predictive 变体）。",
        ["y", "x", "window", "min_periods"],
        domain="price_volume",
        unit="level",
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
        "Huber 稳健回归 predictive 残差（拟合不含当前样本；无前视，自动挖掘首选）。",
        ["y", "x", "window", "min_periods"],
        domain="price_volume",
        unit="level",
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
        "Ridge 回归 in-sample 残差（拟合含当前样本；诊断用，自动挖掘请用 predictive 变体）。",
        ["y", "x", "window", "alpha", "min_periods"],
        domain="price_volume",
        unit="level",
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
        "Ridge 回归 predictive 残差（拟合不含当前样本；无前视，自动挖掘首选）。",
        ["y", "x", "window", "alpha", "min_periods"],
        domain="price_volume",
        unit="level",
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
        w = int(window)
        mp = max(3, int(min_periods))
        quantile = float(q)
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
    """AR(lag) 系数：x_t 对 x_{t-lag} 的回归斜率。"""

    metadata = _metadata(
        "ts_ar_coefficient",
        "AR(lag) 回归系数。",
        ["x", "window", "lag", "min_periods"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, lag: int = 1, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w = int(window)
        lg = max(1, int(lag))
        mp = max(3, int(min_periods))
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(0, row - w + 1)
                if row - lg < start:
                    continue
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


def _lo_mackinlay_vr(rets: np.ndarray, q: int) -> float:
    """Overlapping Lo–MacKinlay variance-ratio estimator VR(q).

    ``rets`` is the trailing 1-period return series (diff of a price/log-price
    LEVEL input).  With ``r_t(q) = sum_{i=0}^{q-1} r_{t-i}`` and the LM
    finite-sample correction ``m = q*(n-q+1)*(1 - q/n)``,

        VR(q) = sigma_c^2(q) / sigma_a^2,
        sigma_a^2  = (1/(n-1)) sum (r_t - mu)^2,
        sigma_c^2  = (1/m)      sum (r_t(q) - q*mu)^2,

    so VR(q) == 1 for a random walk (drift-adjusted, overlapping returns).
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
        w = int(window)
        if w < 1:
            raise ValueError("window must be >= 1")
        periods = int(q)
        # Feasibility validation (review P1-123): the q-period return needs at
        # least 2 overlapping q-period windows inside the trailing window, and a
        # degenerate q<2 used to be silently clamped.
        if periods < 2:
            raise ValueError("q must be >= 2")
        if periods >= w:
            raise ValueError("q must be < window")
        mp = max(6, int(min_periods))
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(0, row - w + 1)
                segment = xv[start : row + 1, col]
                # Trailing contiguous run: a gap must not re-pair values that
                # were not temporally adjacent (review P1-123).
                vals = _trailing_contiguous(segment)
                if vals.size < mp:
                    continue
                rets = np.diff(vals)
                if rets.size < mp:
                    continue
                var1 = float(np.var(rets))
                if var1 <= 0.0:
                    continue
                # q-period returns
                n = rets.size
                qrets = np.array([vals[i + periods] - vals[i] for i in range(0, n - periods + 1)])
                if qrets.size < 2:
                    continue
                varq = float(np.var(qrets))
                out[row, col] = varq / (periods * var1) - 1.0
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
        w = int(window)
        if w < 1:
            raise ValueError("window must be >= 1")
        periods = int(q)
        if periods < 2:
            raise ValueError("q must be >= 2")
        if periods >= w:
            raise ValueError("q must be < window")
        mp = max(6, int(min_periods))
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(0, row - w + 1)
                segment = xv[start : row + 1, col]
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
    Deterministic; positive values indicate mean reversion, negative values
    indicate trending (Lo–MacKinlay sign convention).
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
        w = int(window)
        if w < 1:
            raise ValueError("window must be >= 1")
        periods = int(q)
        if periods < 2:
            raise ValueError("q must be >= 2")
        if periods >= w:
            raise ValueError("q must be < window")
        mp = max(6, int(min_periods))
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(0, row - w + 1)
                segment = xv[start : row + 1, col]
                vals = _trailing_contiguous(segment)
                if vals.size < mp:
                    continue
                rets = np.diff(vals)
                if rets.size < periods + 1:
                    continue
                out[row, col] = _lo_mackinlay_z(rets, periods)
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
        w = int(window)
        if w < 1:
            raise ValueError("window must be >= 1")
        mp = max(3, int(min_periods))
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(0, row - w + 1)
                segment = xv[start : row + 1, col]
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
        w = int(window)
        mp = max(4, int(min_periods))
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(0, row - w + 1)
                segment = xv[start : row + 1, col]
                # Trailing contiguous run: a NaN must not re-pair rows on either
                # side of a gap (review R11 #12).
                vals = _trailing_contiguous(segment)
                if vals.size < mp:
                    continue
                first, second = _trailing_run_halves(segment)
                if first.size == 0 or second.size == 0:
                    continue
                sd = float(np.std(vals))
                if sd <= 0.0:
                    continue
                out[row, col] = float(np.mean(second) - np.mean(first)) / sd
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
        w = int(window)
        mp = max(4, int(min_periods))
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(0, row - w + 1)
                segment = xv[start : row + 1, col]
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
    from cleaned_operators.operator_surface import (
        extend_extended_only,
        retract_extended_only,
    )
    from cleaned_operators.registry import OperatorRegistry

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
