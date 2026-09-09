# -*- coding: utf-8 -*-
"""Rolling multi-variable / robust / quantile regression operators (P0).

Daily panels in, daily panels out.  Each kernel is a rolling regression per
(instrument, date).  Timing is controlled by ``fit_lag``:

* ``fit_lag=0`` (the legacy ``*_coeff`` / ``*_resid`` family) trains each window
  on the rows *including* the current row and reports the in-sample coefficient
  / residual.  This is a descriptive self-fit (no future data is used), NOT a
  causal "current slot" — the current observation is a member of the training
  set.  These operators are stamped ``diagnostic_only`` / in-sample and hidden
  from default mining; the causal counterparts below are the advertised
  replacement.
* ``fit_lag>=1`` (the ``*_prior`` / ``*_forecast_error`` family) trains strictly
  on rows before the current one (``<= t-1``) and reports the out-of-sample
  coefficient / forecast error at the current row — a genuinely causal slot.

All fits degrade to NaN rather than fabricate values.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import ParamRole, ParamSpec, SeriesOperator, register_operator
from factor_engine.cleaned_operators.ts_model._rolling_core import (
    aligned,
    build_design,
    FitResult,
    FitStatus,
    fit_result,
    frame_like,
    huber_fit,
    metadata,
    ols_fit,
    pinball_quantile_fit,
    quantile_fit,
    record_current_fit,
    ridge_fit,
    rolling_fit,
)

# The historic IRLS ``quantile_fit`` is an expectile fit; the ``expectile_*``
# operators registered below share the exact same kernel with an honest name.
# ``pinball_quantile_fit`` is the true quantile-regression kernel (LP) used by
# the ``ts_quantile_*`` operators.
from factor_engine.cleaned_operators.ts_model._rolling_core import expectile_fit  # noqa: E402

_CANONICALS: list[str] = []
_MULTI_CONFIGURED_HISTORY_CANONICALS: list[str] = []

# Model-audit Phase 4 (search-space hygiene): explicit ParamSpec declarations.
# ``window`` is the alpha horizon (HORIZON, searched); ``coefficient_index`` is
# an estimator selector; ``min_periods`` is a statistical-support floor; and
# ``add_intercept`` is boolean governance — none of the latter three is a
# full-resolution search dimension (M-115/M-162/M-170).  ``q``/``q_high``/``q_low``
# ARE the economic tail mechanism in quantile/expectile regression, so they are
# declared ECONOMIC and searched.
_MULTI_PARAM_SPECS: dict[str, ParamSpec] = {
    "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON, searchable=True),
    "coefficient_index": ParamSpec(dtype=int, min=0, param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False),
    "min_periods": ParamSpec(dtype=int, min=1, param_role=ParamRole.SUPPORT_POLICY, searchable=False),
    "add_intercept": ParamSpec(dtype=bool, param_role=ParamRole.POLICY, searchable=False),
    # P1: window semantics governance — ``window`` is a MAX LOOKBACK, NOT a
    # strict full window; ``warmup_policy`` is catalog-visible (part of the
    # semantic identity), with ``"full"`` opting into a strict full-history floor.
    "warmup_policy": ParamSpec(dtype=str, choices=("expanding", "full"),
                               param_role=ParamRole.POLICY, searchable=False),
}
# P1 window-semantics contract for the rolling multi regression family.
_MULTI_WINDOW_SEMANTICS = "max_lookback"
_MULTI_WARMUP_POLICY = "expanding"
_MULTI_FULL_WARMUP = "full"
# Model-audit 2026-08-13: raised from "n_coeffs + 1" to "5 * n_coeffs" to prevent
# fitting a 4-feature model on 6 observations (overfitting risk). User can override
# via explicit min_periods if needed.
_MULTI_MIN_EFFECTIVE_OBS_EXPR = "max(min_periods, 5 * n_coeffs)"


def validate_multi_configured_history(
    window: int, min_periods: int, feature_count: int, add_intercept: bool,
    *, fit_lag: int = 0,
) -> int:
    n_coeffs = int(feature_count) + (1 if add_intercept else 0)
    required_window = max(int(min_periods), 5 * n_coeffs)
    if int(window) < required_window:
        raise ValueError(
            "INSUFFICIENT_CONFIGURED_HISTORY: "
            f"window={window} requires at least {required_window} for "
            f"feature_count={feature_count}, add_intercept={bool(add_intercept)}"
        )
    return required_window + max(0, int(fit_lag))

# P1 (no-intercept R² definition governance): the R² reported by every
# ``ts_*_regression_r2*`` canonical uses the CENTERED total sum of squares
# ``SS_tot = sum((y - mean(y))**2)`` REGARDLESS of ``add_intercept`` — a single,
# versioned, well-defined convention.  No separate through-the-origin uncentered
# R² (``1 - sum(e**2)/sum(y**2)``) is exposed; changing to it would silently
# alter every no-intercept output, so any such change must ship as a new
# semantic version.  The adjusted-R² degrees of freedom pair with the intercept
# convention: ``df_total = n - 1`` (centered) and
# ``df_error = n - p - (1 if add_intercept else 0)`` (``p`` = slope features,
# intercept = 1 when fitted, else 0).
_R2_DEFINITION = "centered_ss_tot"
_R2_ADJ_TOTAL_DF_EXPR = "n - 1"
_R2_ADJ_ERROR_DF_EXPR = "n - p - (1 if add_intercept else 0)"
_QUANTILE_PARAM_SPECS: dict[str, ParamSpec] = {
    "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON, searchable=True),
    "q": ParamSpec(dtype=float, min=0.0, max=1.0, param_role=ParamRole.ECONOMIC, searchable=True),
    "min_periods": ParamSpec(dtype=int, min=1, param_role=ParamRole.SUPPORT_POLICY, searchable=False),
}
_QUANTILE_SPREAD_PARAM_SPECS: dict[str, ParamSpec] = {
    "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON, searchable=True),
    "q_high": ParamSpec(dtype=float, min=0.0, max=1.0, param_role=ParamRole.ECONOMIC, searchable=True),
    "q_low": ParamSpec(dtype=float, min=0.0, max=1.0, param_role=ParamRole.ECONOMIC, searchable=True),
    "min_periods": ParamSpec(dtype=int, min=1, param_role=ParamRole.SUPPORT_POLICY, searchable=False),
}


def _gather_features(args: tuple[Any, ...], n: int) -> list[pd.DataFrame]:
    out: list[pd.DataFrame] = []
    for i in range(n):
        x = args[i]
        if x is not None:
            out.append(x)
    return out


def _build_fit(fit_fn: Callable[..., Any], extra: Any, add_intercept: bool) -> Callable[..., Any]:
    """Bind the per-window fit callable.

    Ridge needs to know whether the design starts with an intercept so it only
    exempts the intercept column from the L2 penalty.  ``extra`` carries the
    ridge ``alpha`` (or any other scalar hyper-parameter of ``fit_fn``).
    """
    if fit_fn is ridge_fit:
        return lambda d, v: ridge_fit(d, v, extra, has_intercept=bool(add_intercept))
    if extra is not None:
        return lambda d, v: fit_fn(d, v, extra)
    return fit_fn


def _multi_regression(
    y: pd.DataFrame,
    features: list[pd.DataFrame],
    window: int,
    min_periods: int,
    add_intercept: bool,
    fit_fn: Callable[..., Any],
    extra: Any,
    stat: str,
    coeff_index: int,
    *,
    fit_lag: int = 0,
    stability_k: int = 0,
    warmup_policy: str = "expanding",
) -> pd.DataFrame:
    """Rolling multi-variable regression over ``(row, col)`` panels.

    ``fit_lag`` is the number of rows between the end of the training window
    and the row whose statistic is reported.  ``fit_lag=0`` (legacy) trains on
    the window *including* the current row and reports the in-sample residual.
    ``fit_lag>=1`` trains on rows strictly before the current one and reports
    the out-of-sample forecast error at the current row.  ``stability_k>0``
    additionally reports the standard deviation of the fitted coefficient over
    the last ``stability_k`` consecutive fits (each ending ``fit_lag`` rows
    before its row).

    Window semantics (P1): ``window`` is a MAX LOOKBACK, not a strict full
    window.  With ``warmup_policy="expanding"`` (default) the kernel emits as
    soon as ``max(min_periods, 5 * n_coeffs)`` valid rows are present inside the
    trailing window.  With ``warmup_policy="full"`` the trailing window must be
    completely observed (``fit_end >= window - 1``) before any output.
    """
    if not features:
        raise ValueError("at least one feature panel is required")
    if warmup_policy not in ("expanding", "full"):
        raise ValueError(f"warmup_policy must be 'expanding' or 'full', got {warmup_policy!r}")
    frames = [y] + features
    aligned_frames = aligned(*frames)
    y = aligned_frames[0]
    feats = aligned_frames[1:]
    yv = y.to_numpy(dtype=float)
    rows, cols = yv.shape
    xs = [f.to_numpy(dtype=float) for f in feats]
    n_coeffs = len(feats) + (1 if add_intercept else 0)
    if stat == "coeff" and (coeff_index < 0 or coeff_index >= n_coeffs):
        raise ValueError(f"coefficient_index {coeff_index} out of range [0, {n_coeffs})")
    if stability_k > 0 and stat != "coeff":
        raise ValueError("stability_k>0 requires stat='coeff'")
    fit = _build_fit(fit_fn, extra, add_intercept)
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    # Model-audit 2026-08-13: ratio-based floor prevents overfitting (e.g., 4-feature
    # fit on 6 obs). User min_periods still honored if set higher.
    mp = max(int(min_periods), 5 * n_coeffs)
    lag = max(0, int(fit_lag))
    validate_multi_configured_history(
        w, min_periods, len(feats), add_intercept, fit_lag=lag
    )
    k = max(0, int(stability_k))

    def record(
        result: FitResult, *, col: int, start: int, fit_end: int, row: int,
    ) -> None:
        record_current_fit(
            result,
            instrument=y.columns[col],
            window_start=y.index[start],
            window_end=y.index[fit_end],
            output_row=y.index[row],
            fit_cutoff=y.index[fit_end],
            maturity_cutoff=y.index[fit_end],
        )

    for col in range(cols):
        ycol = yv[:, col]
        xcols = [x[:, col] for x in xs]
        for row in range(rows):
            fit_end = row - lag
            if fit_end < 0:
                continue
            # P1: strict full-history floor when warmup_policy="full".
            if warmup_policy == "full" and fit_end < w - 1:
                continue
            start = max(0, fit_end - w + 1)
            seg_y = ycol[start : fit_end + 1]
            seg_xs = [x[start : fit_end + 1] for x in xcols]
            valid = np.isfinite(seg_y)
            for x in seg_xs:
                valid &= np.isfinite(x)
            if valid.sum() < mp:
                record(FitResult(None, FitStatus(False, "insufficient_sample")),
                       col=col, start=start, fit_end=fit_end, row=row)
                continue
            vy = seg_y[valid]
            vxs = [x[valid] for x in seg_xs]
            if any(np.std(vx) <= 0.0 for vx in vxs):
                record(FitResult(None, FitStatus(False, "singular")),
                       col=col, start=start, fit_end=fit_end, row=row)
                continue
            design = build_design(vxs, add_intercept)
            result = fit_result(fit, design, vy)
            record(result, col=col, start=start, fit_end=fit_end, row=row)
            b = result.value
            if b is None:
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                pred = design @ b
                e = vy - pred
            if stat == "coeff":
                if k > 0:
                    coeffs = []
                    for j in range(k):
                        fe = row - lag - j
                        if fe < 0:
                            break
                        s2 = max(0, fe - w + 1)
                        sy = ycol[s2 : fe + 1]
                        sx = [x[s2 : fe + 1] for x in xcols]
                        v2 = np.isfinite(sy)
                        for x in sx:
                            v2 &= np.isfinite(x)
                        if v2.sum() < mp:
                            record(FitResult(None, FitStatus(False, "insufficient_sample")),
                                   col=col, start=s2, fit_end=fe, row=row)
                            break
                        vy2 = sy[v2]
                        vx2 = [x[v2] for x in sx]
                        if any(np.std(vx2) <= 0.0 for vx2 in vx2):
                            record(FitResult(None, FitStatus(False, "singular")),
                                   col=col, start=s2, fit_end=fe, row=row)
                            break
                        d2 = build_design(vx2, add_intercept)
                        result2 = fit_result(fit, d2, vy2)
                        record(result2, col=col, start=s2, fit_end=fe, row=row)
                        b2 = result2.value
                        if b2 is None:
                            break
                        coeffs.append(float(b2[coeff_index]))
                    if len(coeffs) >= 2:
                        out[row, col] = float(np.std(coeffs))
                else:
                    out[row, col] = float(b[coeff_index])
            elif stat == "resid":
                if np.isfinite(ycol[row]):
                    cur_xs = [x[row] for x in xcols]
                    terms = ([1.0] if add_intercept else []) + cur_xs
                    out[row, col] = float(ycol[row] - float(np.dot(terms, b)))
            elif stat == "resid_z":
                if np.isfinite(ycol[row]):
                    ddof = max(design.shape[1], 1)
                    if len(e) > ddof:
                        sd = float(np.sqrt(np.sum(e * e) / max(len(e) - ddof, 1)))
                    else:
                        sd = np.nan
                    if sd is not None and np.isfinite(sd) and sd > 0.0:
                        cur_xs = [x[row] for x in xcols]
                        terms = ([1.0] if add_intercept else []) + cur_xs
                        resid = float(ycol[row] - float(np.dot(terms, b)))
                        out[row, col] = resid / sd
            elif stat in ("r2", "r2_adj"):
                ss_res = float(np.sum(e * e))
                # P1 (R² definition governance, versioned): ``_R2_DEFINITION ==
                # "centered_ss_tot"`` — SS_tot is the CENTERED total sum of
                # squares ``sum((y - mean(y))**2)`` REGARDLESS of add_intercept.
                # No separate through-the-origin uncentered R²
                # (``1 - sum(e**2)/sum(y**2)``) is exposed.
                ss_tot = float(np.sum((vy - np.mean(vy)) ** 2))
                if ss_tot > 0.0:
                    # P1-88: R² is only guaranteed non-negative when an intercept
                    # is fitted.  A through-the-origin (no-intercept) fit can be
                    # arbitrarily worse than predicting the mean, so R² must not
                    # be clipped to 0 — that would manufacture a false fit floor.
                    r2 = float(1.0 - ss_res / ss_tot)
                    if stat == "r2_adj":
                        n = len(vy)
                        # P1-88: the predictor count is the number of slope
                        # features.  The intercept is a fitted parameter, not a
                        # predictor, so it must not be double-counted in the
                        # adjusted-R² penalty (the old code used
                        # ``design.shape[1]`` which included the intercept and
                        # penalised one extra degree of freedom).
                        p = len(feats)
                        # P1 (df pair with the intercept convention, versioned):
                        # with an intercept the error df are n - p - 1; without
                        # one (regression through the origin) they are n - p.
                        # The centered SS_tot fixes the total df at n - 1.
                        denom = n - p - (1 if add_intercept else 0)
                        out[row, col] = float(1.0 - (1.0 - r2) * (n - 1.0) / max(denom, 1.0))
                    else:
                        out[row, col] = r2
    return frame_like(y, out)


# P1-89: honest input_units / output_unit for the multi-variable regression
# family.  beta (a slope) has unit(y)/unit(x); an intercept coefficient has
# unit(y); a raw residual / forecast error has unit(y); a standardised residual
# is a dimensionless z-score; R² / adjusted R² are dimensionless.
_DEFAULT_INPUT_UNITS = {
    "y": "target",
    "x1": "predictor", "x2": "predictor", "x3": "predictor", "x4": "predictor",
}
_DEFAULT_OUTPUT_UNIT = {
    "coeff": "unit(y)/unit(x) (intercept: unit(y))",
    "resid": "unit(y)",
    "resid_z": "z_score",
    "r2": "r2",
    "r2_adj": "r2",
}


def _register_multi(name: str, description: str, fit_fn: Callable[..., Any], extra: Any, stat: str, unit: str,
                    *, fit_lag: int = 0, stability_k: int = 0, cost: int = 5,
                    input_units: dict[str, str] | None = None,
                    output_unit: str | None = None,
                    diagnostic_only: bool = False):
    _MULTI_CONFIGURED_HISTORY_CANONICALS.append(name)
    _desc = f"{description}（window=max lookback，非严格满窗；min_effective_obs=max(min_periods, 5*系数数)；warmup_policy={_MULTI_WARMUP_POLICY} 渐进输出）"
    if stat in ("r2", "r2_adj"):
        _desc += f"（R² 定义 versioned：{_R2_DEFINITION}，加截距与不加截距统一 centered SS_tot）"

    @register_operator(
        name=name,
        category="time_series_regression",
        business_category="time_series_regression",
        canonical=name,
        source="ts_model.dynamic_regression",
        backend="pandas_numpy",
        status="experimental",
        semantic_version="3.0" if name.startswith("ts_ridge_regression_") else "2.0",
    )
    class _MultiOp(SeriesOperator):
        metadata = metadata(
            name, _desc,
            ["y", "x1", "x2", "x3", "x4", "window", "coefficient_index", "min_periods", "add_intercept", "warmup_policy"],
            unit=unit, cost=cost,
            input_units=input_units if input_units is not None else _DEFAULT_INPUT_UNITS,
            output_unit=output_unit if output_unit is not None else _DEFAULT_OUTPUT_UNIT.get(stat),
            diagnostic_only=diagnostic_only,
            param_specs=_MULTI_PARAM_SPECS,
        )
        # P1: window = max lookback (documented, machine-readable contract).
        metadata.window_semantics = _MULTI_WINDOW_SEMANTICS
        if stat != "coeff":
            metadata.deprecated_ignored_params = ("coefficient_index",)

        def _calculate_series(self, y, x1=None, x2=None, x3=None, x4=None,
                              window=60, coefficient_index=1, min_periods=10, add_intercept=True,
                              warmup_policy="expanding", **_):
            feats = _gather_features((x1, x2, x3, x4), 4)
            return _multi_regression(
                y, feats, int(window), int(min_periods), bool(add_intercept),
                fit_fn, extra, stat, int(coefficient_index),
                fit_lag=int(fit_lag), stability_k=int(stability_k),
                warmup_policy=str(warmup_policy),
            )

        def validate_params(self, y, x1=None, x2=None, x3=None, x4=None,
                            window=60, coefficient_index=1, min_periods=10,
                            add_intercept=True, warmup_policy="expanding", **_):
            feature_count = sum(x is not None for x in (x1, x2, x3, x4))
            validate_multi_configured_history(
                window, min_periods, feature_count, add_intercept, fit_lag=fit_lag
            )
            return True

    return _MultiOp


_register_multi(
    "ts_multi_regression_coeff", "多变量滚动回归指定系数（in-sample，训练窗口含当前观测；因果 t-1 版本用 ts_multi_regression_coeff_prior）。",
    ols_fit, None, "coeff", "level", diagnostic_only=True,
)
_register_multi(
    "ts_multi_regression_resid", "多变量滚动回归当前残差（in-sample，训练窗口含当前观测）。",
    ols_fit, None, "resid", "level", diagnostic_only=True,
)
_register_multi(
    "ts_multi_regression_resid_z", "多变量滚动回归标准化残差（in-sample）。",
    ols_fit, None, "resid_z", "level", diagnostic_only=True,
)
_register_multi(
    "ts_multi_regression_r2", "多变量滚动回归 R²（in-sample）。",
    ols_fit, None, "r2", "r2", diagnostic_only=True,
)
_register_multi(
    "ts_huber_regression_coeff", "Huber 稳健回归斜率（in-sample，训练窗口含当前观测；因果 t-1 版本用 ts_huber_regression_coeff_prior）。",
    huber_fit, None, "coeff", "level", diagnostic_only=True,
)
_register_multi(
    "ts_huber_regression_resid_z", "Huber 稳健回归标准化残差（in-sample）。",
    huber_fit, None, "resid_z", "level", diagnostic_only=True,
)
_register_multi(
    "ts_ridge_regression_coeff", "岭回归系数（in-sample，训练窗口含当前观测；因果 t-1 版本用 ts_ridge_regression_coeff_prior）。",
    ridge_fit, 0.1, "coeff", "level", diagnostic_only=True,
)
_register_multi(
    "ts_ridge_regression_resid_z", "岭回归标准化残差。",
    ridge_fit, 0.1, "resid_z", "level", diagnostic_only=True,
)

# ---------------------------------------------------------------------------
# Prior-window (out-of-sample) variants.  ``fit_lag=1`` trains each window on
# rows strictly before the current one and reports the current row's coefficient
# / forecast error against that historical fit, so the output is a true
# out-of-sample anomaly rather than an in-sample residual.
#
# M-051: every ``*_prior`` / ``*_forecast_error`` variant below is a strict
# prior fit: the training window ends at ``t-1`` (fit_lag=1).  None of them has
# an explicit MODEL_TIMING_CONTRACTS entry — they rely on the name-driven
# auto-generated default, which is a research hint only (R34 P0-025).  The
# reconciler must add explicit ``fit_cutoff_offset=1`` timing entries for each
# of these names (the name-based default happens to be correct for them, but
# production admission requires an explicit reviewed contract).
# ---------------------------------------------------------------------------
_register_multi(
    "ts_multi_regression_coeff_prior", "多变量回归系数（截至 t-1 训练）。",
    ols_fit, None, "coeff", "level", fit_lag=1,
)
_register_multi(
    "ts_multi_regression_forecast_error", "多变量回归当前样本相对历史拟合的预测误差。",
    ols_fit, None, "resid", "level", fit_lag=1,
)
_register_multi(
    "ts_multi_regression_forecast_error_z", "多变量回归预测误差 / 历史训练窗口残差标准差。",
    ols_fit, None, "resid_z", "level", fit_lag=1,
)
_register_multi(
    "ts_multi_regression_r2_prior", "多变量回归训练窗口 R²（截至 t-1）。",
    ols_fit, None, "r2", "r2", fit_lag=1,
)
_register_multi(
    "ts_multi_regression_adjusted_r2_prior", "多变量回归训练窗口调整 R²（截至 t-1）。",
    ols_fit, None, "r2_adj", "r2", fit_lag=1,
)
_register_multi(
    "ts_multi_regression_coeff_stability",
    "多变量回归系数在最近 K 个滚动窗口的标准差（因果 model-state alpha：衡量截至 t-1 的连续历史拟合中系数的稳定性，非诊断）。",
    ols_fit, None, "coeff", "level", fit_lag=1, stability_k=5, cost=7,
)
_register_multi(
    "ts_huber_regression_coeff_prior", "Huber 稳健回归斜率（截至 t-1 训练）。",
    huber_fit, None, "coeff", "level", fit_lag=1,
)
_register_multi(
    "ts_huber_regression_forecast_error", "Huber 回归预测误差（截至 t-1 训练）。",
    huber_fit, None, "resid", "level", fit_lag=1,
)
_register_multi(
    "ts_huber_regression_forecast_error_z", "Huber 回归预测误差 / 历史残差标准差。",
    huber_fit, None, "resid_z", "level", fit_lag=1,
)
_register_multi(
    "ts_ridge_regression_coeff_prior", "岭回归系数（截至 t-1 训练）。",
    ridge_fit, 0.1, "coeff", "level", fit_lag=1,
)
_register_multi(
    "ts_ridge_regression_forecast_error", "岭回归预测误差（截至 t-1 训练）。",
    ridge_fit, 0.1, "resid", "level", fit_lag=1,
)
_register_multi(
    "ts_ridge_regression_forecast_error_z", "岭回归预测误差 / 历史残差标准差。",
    ridge_fit, 0.1, "resid_z", "level", fit_lag=1,
)


def _single_regression(
    y: pd.DataFrame,
    x: pd.DataFrame,
    window: int,
    min_periods: int,
    add_intercept: bool,
    fit_fn: Callable[..., Any],
    extra: Any,
    stat: str,
    q: float,
    coeff_index: int,
    *,
    fit_lag: int = 0,
) -> pd.DataFrame:
    y, x = aligned(y, x)
    return _multi_regression(
        y, [x], int(window), int(min_periods), bool(add_intercept),
        fit_fn, extra, stat, int(coeff_index), fit_lag=int(fit_lag),
    )


def _expectile_op(name: str, description: str, stat: str, *, fit_lag: int = 0, diagnostic_only: bool = False):
    """Register an explicitly-named expectile regression operator.

    The underlying kernel is the IRLS asymmetric-weighted least squares that is
    historically called ``quantile_*`` in this codebase; the ``expectile_*``
    names carry the honest mathematical label (expectile, not pinball quantile).
    """
    @register_operator(
        name=name,
        category="time_series_regression",
        business_category="time_series_regression",
        canonical=name,
        source="ts_model.dynamic_regression",
        backend="pandas_numpy",
        status="experimental",
    )
    class _ExpectileOp(SeriesOperator):
        metadata = metadata(
            name, description, ["y", "x", "window", "q", "min_periods"], unit="level",
            input_units={"y": "target", "x": "predictor"},
            output_unit="unit(y)/unit(x) (intercept: unit(y))" if stat == "coeff" else "unit(y)",
            diagnostic_only=diagnostic_only,
            param_specs=_QUANTILE_PARAM_SPECS,
        )

        def _calculate_series(self, y, x, window=60, q=0.5, min_periods=10, **_):
            qv = float(q)
            if not (0.0 < qv < 1.0):
                raise ValueError("q must be in (0, 1)")
            return _single_regression(y, x, int(window), int(min_periods), True,
                                      expectile_fit, qv, stat, qv, 1, fit_lag=int(fit_lag))

    _CANONICALS.append(name)
    return _ExpectileOp


_expectile_op("ts_expectile_regression_coeff", "expectile 回归斜率（IRLS 非对称加权最小二乘；in-sample，训练窗口含当前观测，因果 t-1 版本用 ts_expectile_regression_coeff_prior）。", "coeff", diagnostic_only=True)
_expectile_op("ts_expectile_regression_resid", "expectile 回归当前残差（in-sample）。", "resid", diagnostic_only=True)
# M-051: the two ``*_prior`` / ``*_forecast_error`` expectile variants below are
# strict prior fits (fit_lag=1, window ends at t-1).  They have no explicit
# MODEL_TIMING_CONTRACTS entry and rely on the auto-generated name-based default
# (research hint only, R34 P0-025) — the reconciler must add explicit
# ``fit_cutoff_offset=1`` timing entries.
_expectile_op("ts_expectile_regression_coeff_prior", "expectile 回归斜率（截至 t-1 训练）。", "coeff", fit_lag=1)
_expectile_op("ts_expectile_regression_forecast_error", "expectile 回归预测误差（截至 t-1 训练）。", "resid", fit_lag=1)


@register_operator(
    name="ts_expectile_beta_spread",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_expectile_beta_spread",
    source="ts_model.dynamic_regression",
    backend="pandas_numpy",
    status="experimental",
)
class TsExpectileBetaSpread(SeriesOperator):
    """高 expectile Beta 减 低 expectile Beta：上涨/下跌尾部非对称响应。

    IN-SAMPLE (audit M-050): the training window includes the current
    observation (fit_lag=0), so the spread is a descriptive self-fit, not a
    causal t-1 signal — tagged ``diagnostic_only`` and hidden from default
    mining.
    """

    metadata = metadata(
        "ts_expectile_beta_spread", "expectile(q_high) - expectile(q_low)（in-sample，训练窗口含当前观测，诊断用）。",
        ["y", "x", "window", "q_high", "q_low", "min_periods"], unit="level",
        input_units={"y": "target", "x": "predictor"},
        output_unit="unit(y)/unit(x)",
        diagnostic_only=True,
    )

    def _calculate_series(self, y, x, window=60, q_high=0.9, q_low=0.1, min_periods=10, **_):
        qh, ql = float(q_high), float(q_low)
        if not (0.0 < ql < qh < 1.0):
            raise ValueError("q_low < q_high must hold in (0, 1)")
        high = _single_regression(y, x, int(window), int(min_periods), True,
                                  expectile_fit, qh, "coeff", qh, 1)
        low = _single_regression(y, x, int(window), int(min_periods), True,
                                 expectile_fit, ql, "coeff", ql, 1)
        return high - low


@register_operator(
    name="ts_quantile_regression_coeff",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_quantile_regression_coeff",
    source="ts_model.dynamic_regression",
    backend="pandas_numpy",
    status="experimental",
)
class TsQuantileRegressionCoeff(SeriesOperator):
    """条件分位数回归斜率（pinball-loss LP，Koenker–Bassett）。

    In-sample: the training window includes the current observation.  The causal
    t-1 slot is ``ts_quantile_regression_coeff_prior``.
    """

    metadata = metadata(
        "ts_quantile_regression_coeff", "分位数回归斜率（pinball LP；in-sample，训练窗口含当前观测，因果 t-1 版本用 ts_quantile_regression_coeff_prior）。",
        ["y", "x", "window", "q", "min_periods"], unit="level",
        input_units={"y": "target", "x": "predictor"},
        output_unit="unit(y)/unit(x) (intercept: unit(y))",
        diagnostic_only=True,
    )

    def _calculate_series(self, y, x, window=60, q=0.5, min_periods=10, **_):
        quantile = float(q)
        if not (0.0 < quantile < 1.0):
            raise ValueError("q must be in (0, 1)")
        return _single_regression(y, x, int(window), int(min_periods), True,
                                  pinball_quantile_fit, quantile, "coeff", quantile, 1)


@register_operator(
    name="ts_quantile_regression_coeff_prior",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_quantile_regression_coeff_prior",
    source="ts_model.dynamic_regression",
    backend="pandas_numpy",
    status="experimental",
)
# M-051: this is a strict prior fit (fit_lag=1, window ends at t-1) with no
# explicit MODEL_TIMING_CONTRACTS entry — the name-driven auto-generated default
# (fit_cutoff_offset=1) is a research hint only (R34 P0-025).  The reconciler
# must add an explicit ``fit_cutoff_offset=1`` timing contract.
class TsQuantileRegressionCoeffPrior(SeriesOperator):
    """条件分位数回归斜率，截至 t-1 训练（pinball-loss LP，因果槽位）。

    Each window trains strictly on rows before the current one and reports the
    coefficient at the current row, so the output is the causal out-of-sample
    beta — the in-sample variant is ``ts_quantile_regression_coeff``.
    """

    metadata = metadata(
        "ts_quantile_regression_coeff_prior", "分位数回归斜率（pinball LP，截至 t-1 训练）。",
        ["y", "x", "window", "q", "min_periods"], unit="level",
        input_units={"y": "target", "x": "predictor"},
        output_unit="unit(y)/unit(x) (intercept: unit(y))",
        param_specs=_QUANTILE_PARAM_SPECS,
    )

    def _calculate_series(self, y, x, window=60, q=0.5, min_periods=10, **_):
        quantile = float(q)
        if not (0.0 < quantile < 1.0):
            raise ValueError("q must be in (0, 1)")
        return _single_regression(y, x, int(window), int(min_periods), True,
                                  pinball_quantile_fit, quantile, "coeff", quantile, 1,
                                  fit_lag=1)


@register_operator(
    name="ts_quantile_regression_resid",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_quantile_regression_resid",
    source="ts_model.dynamic_regression",
    backend="pandas_numpy",
    status="experimental",
)
class TsQuantileRegressionResid(SeriesOperator):
    """当前值相对条件分位数（pinball LP）预测的偏差。"""

    metadata = metadata(
        "ts_quantile_regression_resid", "分位数回归残差（pinball LP）。",
        ["y", "x", "window", "q", "min_periods"], unit="level",
        input_units={"y": "target", "x": "predictor"},
        output_unit="unit(y)",
        diagnostic_only=True,
    )

    def _calculate_series(self, y, x, window=60, q=0.5, min_periods=10, **_):
        quantile = float(q)
        if not (0.0 < quantile < 1.0):
            raise ValueError("q must be in (0, 1)")
        return _single_regression(y, x, int(window), int(min_periods), True,
                                  pinball_quantile_fit, quantile, "resid", quantile, 1)


@register_operator(
    name="ts_quantile_beta_spread",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_quantile_beta_spread",
    source="ts_model.dynamic_regression",
    backend="pandas_numpy",
    status="experimental",
)
class TsQuantileBetaSpread(SeriesOperator):
    """高分位 Beta 减 低分位 Beta（pinball LP）：上涨/下跌尾部非对称响应。

    IN-SAMPLE (audit M-050): the training window includes the current
    observation (fit_lag=0), so the spread is a descriptive self-fit, not a
    causal t-1 signal — tagged ``diagnostic_only`` and hidden from default
    mining.  The strictly-prior causal variant is
    ``ts_quantile_beta_spread_prior``.
    """

    metadata = metadata(
        "ts_quantile_beta_spread", "beta(q_high) - beta(q_low)（pinball LP，in-sample，训练窗口含当前观测，诊断用；prior 版本用 ts_quantile_beta_spread_prior）。",
        ["y", "x", "window", "q_high", "q_low", "min_periods"], unit="level",
        input_units={"y": "target", "x": "predictor"},
        output_unit="unit(y)/unit(x)",
        diagnostic_only=True,
    )

    def _calculate_series(self, y, x, window=60, q_high=0.9, q_low=0.1, min_periods=10, **_):
        qh, ql = float(q_high), float(q_low)
        if not (0.0 < ql < qh < 1.0):
            raise ValueError("q_low < q_high must hold in (0, 1)")
        high = _single_regression(y, x, int(window), int(min_periods), True,
                                  pinball_quantile_fit, qh, "coeff", qh, 1)
        low = _single_regression(y, x, int(window), int(min_periods), True,
                                 pinball_quantile_fit, ql, "coeff", ql, 1)
        return high - low


# M-057: NEW CANONICAL — ``ts_quantile_beta_spread_prior``.  The reconciler must
# add surface + layer_governance registration for this name (it is registered
# here via the module's ``_CANONICALS`` / ``extend_extended_only`` idiom, but the
# static surface partition and layer-governance stores are shared files that
# need the new name added).
@register_operator(
    name="ts_quantile_beta_spread_prior",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_quantile_beta_spread_prior",
    source="ts_model.dynamic_regression",
    backend="pandas_numpy",
    status="experimental",
)
class TsQuantileBetaSpreadPrior(SeriesOperator):
    """高分位 Beta 减 低分位 Beta（pinball LP），严格截至 t-1 拟合的 prior 版。

    Identical quantile-beta spread to ``ts_quantile_beta_spread`` (high-quantile
    beta minus low-quantile beta, pinball LP) but each beta is fit strictly on
    rows before the current one (fit_lag=1): the training window ends at ``t-1``
    and the reported spread is the causal prior-window spread, not an in-sample
    self-fit.  The in-sample diagnostic variant is ``ts_quantile_beta_spread``.
    """

    metadata = metadata(
        "ts_quantile_beta_spread_prior", "beta(q_high) - beta(q_low)（pinball LP，严格截至 t-1 拟合的 prior 版）。",
        ["y", "x", "window", "q_high", "q_low", "min_periods"], unit="level",
        input_units={"y": "target", "x": "predictor"},
        output_unit="unit(y)/unit(x)",
        param_specs=_QUANTILE_SPREAD_PARAM_SPECS,
    )

    def _calculate_series(self, y, x, window=60, q_high=0.9, q_low=0.1, min_periods=10, **_):
        qh, ql = float(q_high), float(q_low)
        if not (0.0 < ql < qh < 1.0):
            raise ValueError("q_low < q_high must hold in (0, 1)")
        high = _single_regression(y, x, int(window), int(min_periods), True,
                                  pinball_quantile_fit, qh, "coeff", qh, 1, fit_lag=1)
        low = _single_regression(y, x, int(window), int(min_periods), True,
                                 pinball_quantile_fit, ql, "coeff", ql, 1, fit_lag=1)
        return high - low


_CANONICALS.extend(
    [
        "ts_multi_regression_coeff",
        "ts_multi_regression_resid",
        "ts_multi_regression_resid_z",
        "ts_multi_regression_r2",
        "ts_huber_regression_coeff",
        "ts_huber_regression_resid_z",
        "ts_ridge_regression_coeff",
        "ts_ridge_regression_resid_z",
        "ts_quantile_regression_coeff",
        "ts_quantile_regression_coeff_prior",
        "ts_quantile_regression_resid",
        "ts_quantile_beta_spread",
        "ts_quantile_beta_spread_prior",
        "ts_multi_regression_coeff_prior",
        "ts_multi_regression_forecast_error",
        "ts_multi_regression_forecast_error_z",
        "ts_multi_regression_r2_prior",
        "ts_multi_regression_adjusted_r2_prior",
        "ts_multi_regression_coeff_stability",
        "ts_huber_regression_coeff_prior",
        "ts_huber_regression_forecast_error",
        "ts_huber_regression_forecast_error_z",
        "ts_ridge_regression_coeff_prior",
        "ts_ridge_regression_forecast_error",
        "ts_ridge_regression_forecast_error_z",
        "ts_expectile_regression_coeff",
        "ts_expectile_regression_resid",
        "ts_expectile_regression_coeff_prior",
        "ts_expectile_regression_forecast_error",
        "ts_expectile_beta_spread",
    ]
)

import factor_engine.cleaned_operators.operator_surface as _surface  # noqa: E402

_surface.extend_extended_only(set(_CANONICALS))
