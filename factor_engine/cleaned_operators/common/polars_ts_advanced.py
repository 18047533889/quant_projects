# -*- coding: utf-8 -*-
"""Phase 3 Module 12: Advanced time-series operators (Polars native).

Implements AR models, polynomial regression, ridge regression, Huber regression,
quantile regression, and multi-variable regression operators using pure Polars expressions.
All operators are causal (use only past data) and vectorized across columns.

P0-B1-FINALIZE governance note (2026-08-28):

* This module is the authoritative native-Polars implementation tier for the 30
  ts_ar_* / ts_poly2_* / ts_ridge_* / ts_huber_* / ts_multi_* /
  ts_quantile_* canonicals that the DAILY_FACTOR_MIGRATED surface declares but
  that no previously-loaded bootstrap module registered (they were only
  reachable via the research-only ts_model.dynamic_regression / ts_model.ar_…
  modules, which ``load_all(include_research=False)`` intentionally skips).  It
  is therefore part of the production load surface so the layer-governance
  static-partition gate (R40 #154 / layer_governance.finalize_layer_governance)
  sees every daily-surface canonical actually registered.
* Registration order matters: it must be imported AFTER the pandas reference
  modules (regression_models / common.time_series / ts_model.dynamic_regression
  / ts_model.ar_meanrev) that establish the pandas_numpy first-backend so the
  ``_backfill_logical_contract`` canonical authority stays pandas-owned.  The
  loader appends this module after those (see
  ``factor_engine.cleaned_operators.__init__``).
"""
from __future__ import annotations

try:
    import polars as pl
    import numpy as np
except ImportError:  # pragma: no cover
    pl = None  # type: ignore
    np = None  # type: ignore

from factor_engine.cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.cleaned_operators.base import ParamRole, ParamSpec
from factor_engine.cleaned_operators.parameter_validation import strict_integer, strict_finite_scalar

_SKIP = frozenset({"date", "stock_code"})
_SRC = "polars_ts_advanced_phase3"


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _with_meta(result: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
    if "date" in source.columns and "date" not in result.columns:
        result = result.with_columns(source["date"])
    return result


# ---------------------------------------------------------------------------
# AR Model Operators
# ---------------------------------------------------------------------------


@register_operator(
    name="ts_ar_coefficient",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_ar_coefficient",
    source=_SRC,
    backend="polars")
class TSARCoefficientNative(SeriesOperator):
    """Rolling AR(p) coefficient estimation via OLS."""

    metadata = OperatorMetadata(
        name="ts_ar_coefficient",
        category="time_series",
        description="AR(p) 系数估计",
        param_names=["x", "window", "lag", "min_periods", "warmup_policy"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "lag": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
            "min_periods": ParamSpec(dtype=int, min=1, default=1, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, lag: int = 1,
        min_periods: int = 1, warmup_policy: str = "exact", **kwargs
    ) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        p = strict_integer(lag, "lag", minimum=1)
        coef_index = int(kwargs.get("coef_index", 0) or 0)
        idx = strict_integer(coef_index, "coef_index", minimum=0)

        if idx >= p:
            from factor_engine.backend.operator_errors import OperatorParameterError
            raise OperatorParameterError(f"coef_index {idx} must be < lag {p}")

        cols = _numeric_cols(x)

        def _ar_coef(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < p + 2:
                return np.nan
            arr = arr[valid]
            if len(arr) < p + 2:
                return np.nan
            y = arr[p:]
            X = np.column_stack([arr[p-i-1:-i-1] for i in range(p)])
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                return float(coef[idx]) if idx < len(coef) else np.nan
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_ar_coef, window_size=w, min_samples=p+2).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_ar_fitted_value",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_ar_fitted_value",
    source=_SRC,
    backend="polars")
class TSARFittedValueNative(SeriesOperator):
    """Rolling AR(p) fitted value (in-sample prediction)."""

    metadata = OperatorMetadata(
        name="ts_ar_fitted_value",
        category="time_series",
        description="AR(p) 拟合值",
        param_names=["x", "window", "order", "warmup_policy"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "order": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, order: int = 1,
                          warmup_policy: str = "exact", **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        p = strict_integer(order, "order", minimum=1)
        lag = p
        cols = _numeric_cols(x)

        def _ar_fitted(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < p + 2:
                return np.nan
            arr = arr[valid]
            if len(arr) < p + 2:
                return np.nan
            y = arr[p:]
            X = np.column_stack([arr[p-i-1:-i-1] for i in range(p)])
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                x_last = arr[-p:][::-1]
                return float(np.dot(coef, x_last))
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_ar_fitted, window_size=w, min_samples=p+2).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_ar_forecast",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_ar_forecast",
    source=_SRC,
    backend="polars")
class TSARForecastNative(SeriesOperator):
    """Rolling AR(p) one-step-ahead forecast."""

    metadata = OperatorMetadata(
        name="ts_ar_forecast",
        category="time_series",
        description="AR(p) 预测值",
        param_names=["x", "window", "order", "warmup_policy"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "order": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, order: int = 1,
                          warmup_policy: str = "exact", **kwargs) -> pl.DataFrame:
        # Forecast at t uses data up to t-1, predicts t
        w = strict_integer(window, "window", minimum=3)
        p = strict_integer(order, "order", minimum=1)
        lag = p
        cols = _numeric_cols(x)

        def _ar_forecast(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < p + 2:
                return np.nan
            arr = arr[valid]
            if len(arr) < p + 2:
                return np.nan
            y = arr[p:]
            X = np.column_stack([arr[p-i-1:-i-1] for i in range(p)])
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                x_last = arr[-p:][::-1]
                return float(np.dot(coef, x_last))
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).shift(1).rolling_map(_ar_forecast, window_size=w, min_samples=p+2).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_ar_innovation",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_ar_innovation",
    source=_SRC,
    backend="polars")
class TSARInnovationNative(SeriesOperator):
    """AR(p) innovation (forecast error): y_t - forecast_t."""

    metadata = OperatorMetadata(
        name="ts_ar_innovation",
        category="time_series",
        description="AR(p) 创新项",
        param_names=["x", "window", "order", "warmup_policy"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "order": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, order: int = 1,
                          warmup_policy: str = "exact", **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        p = strict_integer(order, "order", minimum=1)
        lag = p
        cols = _numeric_cols(x)

        def _ar_innov(vals):
            arr = np.asarray(vals, dtype=float)
            if len(arr) < w:
                return np.nan
            current = arr[-1]
            if np.isnan(current):
                return np.nan
            past = arr[:-1]
            valid = ~np.isnan(past)
            if np.sum(valid) < p + 2:
                return np.nan
            past = past[valid]
            if len(past) < p + 2:
                return np.nan
            y = past[p:]
            X = np.column_stack([past[p-i-1:-i-1] for i in range(p)])
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                x_last = past[-p:][::-1]
                forecast = np.dot(coef, x_last)
                return float(current - forecast)
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_ar_innov, window_size=w+1, min_samples=p+3).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_ar_innovation_z",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_ar_innovation_z",
    source=_SRC,
    backend="polars")
class TSARInnovationZNative(SeriesOperator):
    """Standardized AR(p) innovation: innovation / std(innovation)."""

    metadata = OperatorMetadata(
        name="ts_ar_innovation_z",
        category="time_series",
        description="AR(p) 标准化创新项",
        param_names=["x", "window", "order", "warmup_policy"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "order": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, order: int = 1,
                          warmup_policy: str = "exact", **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        p = strict_integer(order, "order", minimum=1)
        lag = p
        cols = _numeric_cols(x)

        def _ar_innov_z(vals):
            arr = np.asarray(vals, dtype=float)
            if len(arr) < w:
                return np.nan
            current = arr[-1]
            if np.isnan(current):
                return np.nan
            past = arr[:-1]
            valid = ~np.isnan(past)
            if np.sum(valid) < p + 2:
                return np.nan
            past = past[valid]
            if len(past) < p + 2:
                return np.nan
            y = past[p:]
            X = np.column_stack([past[p-i-1:-i-1] for i in range(p)])
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                fitted = X @ coef
                resid = y - fitted
                std = np.std(resid, ddof=1)
                if std < 1e-12:
                    return np.nan
                x_last = past[-p:][::-1]
                forecast = np.dot(coef, x_last)
                innov = current - forecast
                return np.where(std != 0, (float(innov) / (std)), np.nan)
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_ar_innov_z, window_size=w+1, min_samples=p+3).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_ar_in_sample_resid",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_ar_in_sample_resid",
    source=_SRC,
    backend="polars")
class TSARInSampleResidNative(SeriesOperator):
    """AR(p) in-sample residual std."""

    metadata = OperatorMetadata(
        name="ts_ar_in_sample_resid",
        category="time_series",
        description="AR(p) 样本内残差标准差",
        param_names=["x", "window", "order", "warmup_policy"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "order": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, order: int = 1, warmup_policy: str = "expanding", **kwargs) -> pl.DataFrame:
        # R4-100 parity: canonical (x, window, order, warmup_policy); lag is the legacy alias.
        w = strict_integer(window, "window", minimum=3)
        p = strict_integer(kwargs.get("lag", order), "order", minimum=1)
        cols = _numeric_cols(x)

        def _ar_resid(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < p + 2:
                return np.nan
            arr = arr[valid]
            if len(arr) < p + 2:
                return np.nan
            y = arr[p:]
            X = np.column_stack([arr[p-i-1:-i-1] for i in range(p)])
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                fitted = X @ coef
                resid = y - fitted
                return float(np.std(resid, ddof=1))
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_ar_resid, window_size=w, min_samples=p+2).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_ar_coeff_stability",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_ar_coeff_stability",
    source=_SRC,
    backend="polars")
class TSARCoeffStabilityNative(SeriesOperator):
    """AR coefficient stability: std of coefficient over sub-windows."""

    metadata = OperatorMetadata(
        name="ts_ar_coeff_stability",
        category="time_series",
        description="AR 系数稳定性",
        param_names=["x", "window", "order", "warmup_policy"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=10, default=60, searchable=True, param_role=ParamRole.HORIZON),
            "order": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 60, lag: int = 1, coef_index: int = 0, **kwargs
    ) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=10)
        p = strict_integer(lag, "lag", minimum=1)
        idx = strict_integer(coef_index, "coef_index", minimum=0)

        if idx >= p:
            from factor_engine.backend.operator_errors import OperatorParameterError
            raise OperatorParameterError(f"coef_index {idx} must be < lag {p}")

        cols = _numeric_cols(x)

        def _coef_stability(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < w:
                return np.nan
            arr = arr[valid]
            if len(arr) < w:
                return np.nan

            sub_w = max(p + 5, w // 3)
            n_subs = max(2, (len(arr) - sub_w) // 5)
            coefs = []

            for i in range(n_subs):
                start = i * 5
                end = start + sub_w
                if end > len(arr):
                    break
                sub = arr[start:end]
                if len(sub) < p + 2:
                    continue
                y = sub[p:]
                X = np.column_stack([sub[p-i-1:-i-1] for i in range(p)])
                try:
                    coef = np.linalg.lstsq(X, y, rcond=None)[0]
                    if idx < len(coef):
                        coefs.append(coef[idx])
                except:
                    pass

            if len(coefs) < 2:
                return np.nan
            return float(np.std(coefs, ddof=1))

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_coef_stability, window_size=w, min_samples=w).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


# ---------------------------------------------------------------------------
# AR Prior Operators (uses data strictly before forecast window)
# ---------------------------------------------------------------------------


@register_operator(
    name="ts_ar_prior_coeff",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_ar_prior_coeff",
    source=_SRC,
    backend="polars")
class TSARPriorCoeffNative(SeriesOperator):
    """AR coefficient estimated on prior window (before current)."""

    metadata = OperatorMetadata(
        name="ts_ar_prior_coeff",
        category="time_series",
        description="AR 历史系数",
        param_names=["x", "window", "order", "warmup_policy"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "order": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, order: int = 1, **kwargs
    ) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        p = strict_integer(order, "order", minimum=1)
        idx = 0

        if idx >= p:
            from factor_engine.backend.operator_errors import OperatorParameterError
            raise OperatorParameterError(f"coef_index {idx} must be < lag {p}")

        cols = _numeric_cols(x)

        def _ar_coef(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < p + 2:
                return np.nan
            arr = arr[valid]
            if len(arr) < p + 2:
                return np.nan
            y = arr[p:]
            X = np.column_stack([arr[p-i-1:-i-1] for i in range(p)])
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                return float(coef[idx]) if idx < len(coef) else np.nan
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).shift(1).rolling_map(_ar_coef, window_size=w, min_samples=p+2).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_ar_prior_forecast",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_ar_prior_forecast",
    source=_SRC,
    backend="polars")
class TSARPriorForecastNative(SeriesOperator):
    """AR forecast using prior window only."""

    metadata = OperatorMetadata(
        name="ts_ar_prior_forecast",
        category="time_series",
        description="AR 历史预测",
        param_names=["x", "window", "order", "warmup_policy"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "order": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, order: int = 1,
                          warmup_policy: str = "exact", **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        p = strict_integer(order, "order", minimum=1)
        lag = p
        cols = _numeric_cols(x)

        def _ar_forecast(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < p + 2:
                return np.nan
            arr = arr[valid]
            if len(arr) < p + 2:
                return np.nan
            y = arr[p:]
            X = np.column_stack([arr[p-i-1:-i-1] for i in range(p)])
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                x_last = arr[-p:][::-1]
                return float(np.dot(coef, x_last))
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).shift(1).rolling_map(_ar_forecast, window_size=w, min_samples=p+2).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_ar_prior_innovation",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_ar_prior_innovation",
    source=_SRC,
    backend="polars")
class TSARPriorInnovationNative(SeriesOperator):
    """AR innovation using prior window forecast."""

    metadata = OperatorMetadata(
        name="ts_ar_prior_innovation",
        category="time_series",
        description="AR 历史创新项",
        param_names=["x", "window", "order", "warmup_policy"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "order": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, order: int = 1,
                          warmup_policy: str = "exact", **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        p = strict_integer(order, "order", minimum=1)
        lag = p
        cols = _numeric_cols(x)

        def _ar_innov(vals):
            arr = np.asarray(vals, dtype=float)
            if len(arr) < w + 1:
                return np.nan
            current = arr[-1]
            if np.isnan(current):
                return np.nan
            past = arr[:-1]
            valid = ~np.isnan(past)
            if np.sum(valid) < p + 2:
                return np.nan
            past = past[valid]
            if len(past) < p + 2:
                return np.nan
            y = past[p:]
            X = np.column_stack([past[p-i-1:-i-1] for i in range(p)])
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                x_last = past[-p:][::-1]
                forecast = np.dot(coef, x_last)
                return float(current - forecast)
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_ar_innov, window_size=w+1, min_samples=p+3).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_ar_prior_innovation_z",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_ar_prior_innovation_z",
    source=_SRC,
    backend="polars")
class TSARPriorInnovationZNative(SeriesOperator):
    """Standardized AR innovation using prior window."""

    metadata = OperatorMetadata(
        name="ts_ar_prior_innovation_z",
        category="time_series",
        description="AR 历史标准化创新项",
        param_names=["x", "window", "order", "warmup_policy"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "order": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, order: int = 1, warmup_policy: str = "expanding", **kwargs) -> pl.DataFrame:
        # R4-100 parity: canonical (x, window, order, warmup_policy); lag is legacy.
        w = strict_integer(window, "window", minimum=3)
        p = strict_integer(kwargs.get("lag", order), "order", minimum=1)
        cols = _numeric_cols(x)

        def _ar_innov_z(vals):
            arr = np.asarray(vals, dtype=float)
            if len(arr) < w + 1:
                return np.nan
            current = arr[-1]
            if np.isnan(current):
                return np.nan
            past = arr[:-1]
            valid = ~np.isnan(past)
            if np.sum(valid) < p + 2:
                return np.nan
            past = past[valid]
            if len(past) < p + 2:
                return np.nan
            y = past[p:]
            X = np.column_stack([past[p-i-1:-i-1] for i in range(p)])
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                fitted = X @ coef
                resid = y - fitted
                std = np.std(resid, ddof=1)
                if std < 1e-12:
                    return np.nan
                x_last = past[-p:][::-1]
                forecast = np.dot(coef, x_last)
                innov = current - forecast
                return np.where(std != 0, (float(innov) / (std)), np.nan)
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_ar_innov_z, window_size=w+1, min_samples=p+3).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


# ---------------------------------------------------------------------------
# Polynomial Regression Operators
# ---------------------------------------------------------------------------


@register_operator(
    name="ts_poly2_coeff",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_poly2_coeff",
    source=_SRC,
    backend="polars")
class TSPoly2CoeffNative(SeriesOperator):
    """Rolling quadratic polynomial coefficient."""

    metadata = OperatorMetadata(
        name="ts_poly2_coeff",
        category="time_series",
        description="二次多项式系数",
        param_names=["x", "window", "coef_index"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "coef_index": ParamSpec(dtype=int, min=0, max=2, default=0, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, coef_index: int = 0, **kwargs
    ) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        idx = strict_integer(coef_index, "coef_index", minimum=0)
        if idx > 2:
            from factor_engine.backend.operator_errors import OperatorParameterError
            raise OperatorParameterError(f"coef_index {idx} must be <= 2")

        cols = _numeric_cols(x)

        def _poly2_coef(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < 3:
                return np.nan
            y = arr[valid]
            t = np.arange(len(y))
            X = np.column_stack([np.ones(len(t)), t, t**2])
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                return float(coef[idx])
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_poly2_coef, window_size=w, min_samples=3).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_poly2_forecast_error",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_poly2_forecast_error",
    source=_SRC,
    backend="polars")
class TSPoly2ForecastErrorNative(SeriesOperator):
    """Quadratic polynomial forecast error."""

    metadata = OperatorMetadata(
        name="ts_poly2_forecast_error",
        category="time_series",
        description="二次多项式预测误差",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        cols = _numeric_cols(x)

        def _poly2_error(vals):
            arr = np.asarray(vals, dtype=float)
            if len(arr) < w:
                return np.nan
            current = arr[-1]
            if np.isnan(current):
                return np.nan
            past = arr[:-1]
            valid = ~np.isnan(past)
            if np.sum(valid) < 3:
                return np.nan
            y = past[valid]
            t = np.arange(len(y))
            X = np.column_stack([np.ones(len(t)), t, t**2])
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                t_next = len(y)
                forecast = coef[0] + coef[1] * t_next + coef[2] * t_next**2
                return float(current - forecast)
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_poly2_error, window_size=w+1, min_samples=4).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_poly2_forecast_error_z",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_poly2_forecast_error_z",
    source=_SRC,
    backend="polars")
class TSPoly2ForecastErrorZNative(SeriesOperator):
    """Standardized quadratic polynomial forecast error."""

    metadata = OperatorMetadata(
        name="ts_poly2_forecast_error_z",
        category="time_series",
        description="二次多项式标准化预测误差",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        cols = _numeric_cols(x)

        def _poly2_error_z(vals):
            arr = np.asarray(vals, dtype=float)
            if len(arr) < w:
                return np.nan
            current = arr[-1]
            if np.isnan(current):
                return np.nan
            past = arr[:-1]
            valid = ~np.isnan(past)
            if np.sum(valid) < 3:
                return np.nan
            y = past[valid]
            t = np.arange(len(y))
            X = np.column_stack([np.ones(len(t)), t, t**2])
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                fitted = X @ coef
                resid = y - fitted
                std = np.std(resid, ddof=1)
                if std < 1e-12:
                    return np.nan
                t_next = len(y)
                forecast = coef[0] + coef[1] * t_next + coef[2] * t_next**2
                error = current - forecast
                return np.where(std != 0, (float(error) / (std)), np.nan)
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_poly2_error_z, window_size=w+1, min_samples=4).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_poly2_prior_coeff",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_poly2_prior_coeff",
    source=_SRC,
    backend="polars")
class TSPoly2PriorCoeffNative(SeriesOperator):
    """Quadratic polynomial coefficient from prior window."""

    metadata = OperatorMetadata(
        name="ts_poly2_prior_coeff",
        category="time_series",
        description="二次多项式历史系数",
        param_names=["x", "window", "coef_index"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "coef_index": ParamSpec(dtype=int, min=0, max=2, default=0, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, coef_index: int = 0, **kwargs
    ) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        idx = strict_integer(coef_index, "coef_index", minimum=0)
        if idx > 2:
            from factor_engine.backend.operator_errors import OperatorParameterError
            raise OperatorParameterError(f"coef_index {idx} must be <= 2")

        cols = _numeric_cols(x)

        def _poly2_coef(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < 3:
                return np.nan
            y = arr[valid]
            t = np.arange(len(y))
            X = np.column_stack([np.ones(len(t)), t, t**2])
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                return float(coef[idx])
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).shift(1).rolling_map(_poly2_coef, window_size=w, min_samples=3).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_poly2_resid",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_poly2_resid",
    source=_SRC,
    backend="polars")
class TSPoly2ResidNative(SeriesOperator):
    """Quadratic polynomial in-sample residual std."""

    metadata = OperatorMetadata(
        name="ts_poly2_resid",
        category="time_series",
        description="二次多项式残差标准差",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        cols = _numeric_cols(x)

        def _poly2_resid(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < 3:
                return np.nan
            y = arr[valid]
            t = np.arange(len(y))
            X = np.column_stack([np.ones(len(t)), t, t**2])
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                fitted = X @ coef
                resid = y - fitted
                return float(np.std(resid, ddof=1))
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_poly2_resid, window_size=w, min_samples=3).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


# ---------------------------------------------------------------------------
# Ridge Regression Operators
# ---------------------------------------------------------------------------


@register_operator(
    name="ts_ridge_regression_coeff",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_ridge_regression_coeff",
    source=_SRC,
    backend="polars")
class TSRidgeRegressionCoeffNative(SeriesOperator):
    """Ridge regression coefficient (L2 regularized linear regression)."""

    metadata = OperatorMetadata(
        name="ts_ridge_regression_coeff",
        category="time_series",
        description="Ridge 回归系数",
        param_names=["x", "window", "alpha", "coef_index"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "alpha": ParamSpec(dtype=float, min=0.0, default=1.0, searchable=True, param_role=ParamRole.REGULARIZATION),
            "coef_index": ParamSpec(dtype=int, min=0, default=0, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, alpha: float = 1.0, coef_index: int = 0, **kwargs
    ) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        a = strict_finite_scalar(alpha, "alpha", minimum=0.0)
        idx = strict_integer(coef_index, "coef_index", minimum=0)

        cols = _numeric_cols(x)

        def _ridge_coef(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < 3:
                return np.nan
            y = arr[valid]
            t = np.arange(len(y))
            X = np.column_stack([np.ones(len(t)), t])
            try:
                XtX = X.T @ X + a * np.eye(X.shape[1])
                Xty = X.T @ y
                coef = np.linalg.solve(XtX, Xty)
                return float(coef[idx]) if idx < len(coef) else np.nan
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_ridge_coef, window_size=w, min_samples=3).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_ridge_regression_coeff_prior",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_ridge_regression_coeff_prior",
    source=_SRC,
    backend="polars")
class TSRidgeRegressionCoeffPriorNative(SeriesOperator):
    """Ridge regression coefficient from prior window."""

    metadata = OperatorMetadata(
        name="ts_ridge_regression_coeff_prior",
        category="time_series",
        description="Ridge 回归历史系数",
        param_names=["x", "window", "alpha", "coef_index"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "alpha": ParamSpec(dtype=float, min=0.0, default=1.0, searchable=True, param_role=ParamRole.REGULARIZATION),
            "coef_index": ParamSpec(dtype=int, min=0, default=0, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, alpha: float = 1.0, coef_index: int = 0, **kwargs
    ) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        a = strict_finite_scalar(alpha, "alpha", minimum=0.0)
        idx = strict_integer(coef_index, "coef_index", minimum=0)

        cols = _numeric_cols(x)

        def _ridge_coef(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < 3:
                return np.nan
            y = arr[valid]
            t = np.arange(len(y))
            X = np.column_stack([np.ones(len(t)), t])
            try:
                XtX = X.T @ X + a * np.eye(X.shape[1])
                Xty = X.T @ y
                coef = np.linalg.solve(XtX, Xty)
                return float(coef[idx]) if idx < len(coef) else np.nan
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).shift(1).rolling_map(_ridge_coef, window_size=w, min_samples=3).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_ridge_regression_forecast_error",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_ridge_regression_forecast_error",
    source=_SRC,
    backend="polars")
class TSRidgeRegressionForecastErrorNative(SeriesOperator):
    """Ridge regression forecast error."""

    metadata = OperatorMetadata(
        name="ts_ridge_regression_forecast_error",
        category="time_series",
        description="Ridge 回归预测误差",
        param_names=["x", "window", "alpha"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "alpha": ParamSpec(dtype=float, min=0.0, default=1.0, searchable=True, param_role=ParamRole.REGULARIZATION),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, alpha: float = 1.0, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        a = strict_finite_scalar(alpha, "alpha", minimum=0.0)
        cols = _numeric_cols(x)

        def _ridge_error(vals):
            arr = np.asarray(vals, dtype=float)
            if len(arr) < w:
                return np.nan
            current = arr[-1]
            if np.isnan(current):
                return np.nan
            past = arr[:-1]
            valid = ~np.isnan(past)
            if np.sum(valid) < 3:
                return np.nan
            y = past[valid]
            t = np.arange(len(y))
            X = np.column_stack([np.ones(len(t)), t])
            try:
                XtX = X.T @ X + a * np.eye(X.shape[1])
                Xty = X.T @ y
                coef = np.linalg.solve(XtX, Xty)
                t_next = len(y)
                forecast = coef[0] + coef[1] * t_next
                return float(current - forecast)
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_ridge_error, window_size=w+1, min_samples=4).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_ridge_regression_forecast_error_z",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_ridge_regression_forecast_error_z",
    source=_SRC,
    backend="polars")
class TSRidgeRegressionForecastErrorZNative(SeriesOperator):
    """Standardized ridge regression forecast error."""

    metadata = OperatorMetadata(
        name="ts_ridge_regression_forecast_error_z",
        category="time_series",
        description="Ridge 回归标准化预测误差",
        param_names=["x", "window", "alpha"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "alpha": ParamSpec(dtype=float, min=0.0, default=1.0, searchable=True, param_role=ParamRole.REGULARIZATION),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, alpha: float = 1.0, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        a = strict_finite_scalar(alpha, "alpha", minimum=0.0)
        cols = _numeric_cols(x)

        def _ridge_error_z(vals):
            arr = np.asarray(vals, dtype=float)
            if len(arr) < w:
                return np.nan
            current = arr[-1]
            if np.isnan(current):
                return np.nan
            past = arr[:-1]
            valid = ~np.isnan(past)
            if np.sum(valid) < 3:
                return np.nan
            y = past[valid]
            t = np.arange(len(y))
            X = np.column_stack([np.ones(len(t)), t])
            try:
                XtX = X.T @ X + a * np.eye(X.shape[1])
                Xty = X.T @ y
                coef = np.linalg.solve(XtX, Xty)
                fitted = X @ coef
                resid = y - fitted
                std = np.std(resid, ddof=1)
                if std < 1e-12:
                    return np.nan
                t_next = len(y)
                forecast = coef[0] + coef[1] * t_next
                error = current - forecast
                return np.where(std != 0, (float(error) / (std)), np.nan)
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_ridge_error_z, window_size=w+1, min_samples=4).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_ridge_regression_in_sample_resid",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_ridge_regression_in_sample_resid",
    source=_SRC,
    backend="polars")
class TSRidgeRegressionInSampleResidNative(SeriesOperator):
    """Ridge regression in-sample residual std."""

    metadata = OperatorMetadata(
        name="ts_ridge_regression_in_sample_resid",
        category="time_series",
        description="Ridge 回归样本内残差",
        param_names=["x", "window", "alpha"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "alpha": ParamSpec(dtype=float, min=0.0, default=1.0, searchable=True, param_role=ParamRole.REGULARIZATION),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, alpha: float = 1.0, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        a = strict_finite_scalar(alpha, "alpha", minimum=0.0)
        cols = _numeric_cols(x)

        def _ridge_resid(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < 3:
                return np.nan
            y = arr[valid]
            t = np.arange(len(y))
            X = np.column_stack([np.ones(len(t)), t])
            try:
                XtX = X.T @ X + a * np.eye(X.shape[1])
                Xty = X.T @ y
                coef = np.linalg.solve(XtX, Xty)
                fitted = X @ coef
                resid = y - fitted
                return float(np.std(resid, ddof=1))
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_ridge_resid, window_size=w, min_samples=3).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_ridge_regression_predictive_resid",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_ridge_regression_predictive_resid",
    source=_SRC,
    backend="polars")
class TSRidgeRegressionPredictiveResidNative(SeriesOperator):
    """Ridge regression rolling predictive residual std."""

    metadata = OperatorMetadata(
        name="ts_ridge_regression_predictive_resid",
        category="time_series",
        description="Ridge 回归预测残差",
        param_names=["x", "window", "alpha"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "alpha": ParamSpec(dtype=float, min=0.0, default=1.0, searchable=True, param_role=ParamRole.REGULARIZATION),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, alpha: float = 1.0, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        a = strict_finite_scalar(alpha, "alpha", minimum=0.0)
        cols = _numeric_cols(x)

        def _ridge_pred_resid(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < w:
                return np.nan
            arr = arr[valid]
            if len(arr) < w:
                return np.nan

            errors = []
            for i in range(3, len(arr)):
                y_train = arr[:i]
                t_train = np.arange(len(y_train))
                X_train = np.column_stack([np.ones(len(t_train)), t_train])
                try:
                    XtX = X_train.T @ X_train + a * np.eye(X_train.shape[1])
                    Xty = X_train.T @ y_train
                    coef = np.linalg.solve(XtX, Xty)
                    t_next = len(y_train)
                    forecast = coef[0] + coef[1] * t_next
                    errors.append(arr[i] - forecast)
                except:
                    pass

            if len(errors) < 2:
                return np.nan
            return float(np.std(errors, ddof=1))

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_ridge_pred_resid, window_size=w, min_samples=w).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_ridge_regression_resid_z",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_ridge_regression_resid_z",
    source=_SRC,
    backend="polars")
class TSRidgeRegressionResidZNative(SeriesOperator):
    """Current value as z-score of ridge residuals."""

    metadata = OperatorMetadata(
        name="ts_ridge_regression_resid_z",
        category="time_series",
        description="Ridge 回归残差 Z 分数",
        param_names=["x", "window", "alpha"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "alpha": ParamSpec(dtype=float, min=0.0, default=1.0, searchable=True, param_role=ParamRole.REGULARIZATION),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, alpha: float = 1.0, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        a = strict_finite_scalar(alpha, "alpha", minimum=0.0)
        cols = _numeric_cols(x)

        def _ridge_resid_z(vals):
            arr = np.asarray(vals, dtype=float)
            if len(arr) < w:
                return np.nan
            current = arr[-1]
            if np.isnan(current):
                return np.nan
            past = arr[:-1]
            valid = ~np.isnan(past)
            if np.sum(valid) < 3:
                return np.nan
            y = past[valid]
            t = np.arange(len(y))
            X = np.column_stack([np.ones(len(t)), t])
            try:
                XtX = X.T @ X + a * np.eye(X.shape[1])
                Xty = X.T @ y
                coef = np.linalg.solve(XtX, Xty)
                fitted = X @ coef
                resid = y - fitted
                mean_resid = np.mean(resid)
                std_resid = np.std(resid, ddof=1)
                if std_resid < 1e-12:
                    return np.nan
                t_next = len(y)
                forecast = coef[0] + coef[1] * t_next
                current_resid = current - forecast
                return np.where(std_resid != 0, (float((current_resid - mean_resid)) / (std_resid)), np.nan)
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_ridge_resid_z, window_size=w+1, min_samples=4).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


# ---------------------------------------------------------------------------
# Huber Regression Operators (robust regression)
# ---------------------------------------------------------------------------


@register_operator(
    name="ts_huber_regression_coeff",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_huber_regression_coeff",
    source=_SRC,
    backend="polars")
class TSHuberRegressionCoeffNative(SeriesOperator):
    """Huber robust regression coefficient (simplified IRLS)."""

    metadata = OperatorMetadata(
        name="ts_huber_regression_coeff",
        category="time_series",
        description="Huber 鲁棒回归系数",
        param_names=["x", "window", "delta", "order", "warmup_policy"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "delta": ParamSpec(dtype=float, min=0.0, default=1.35, searchable=True, param_role=ParamRole.THRESHOLD),
            "order": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, delta: float = 1.35, order: int = 1, warmup_policy: str = "exact", **kwargs
    ) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        d = strict_finite_scalar(delta, "delta", minimum=0.0)
        idx = 0

        cols = _numeric_cols(x)

        def _huber_coef(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < 3:
                return np.nan
            y = arr[valid]
            t = np.arange(len(y))
            X = np.column_stack([np.ones(len(t)), t])
            try:
                # Simple IRLS with 3 iterations
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                for _ in range(3):
                    fitted = X @ coef
                    resid = y - fitted
                    scale = (np.median(np.abs(resid))) / 0.6745 if 0.6745 != 0 else np.nan
                    if scale < 1e-12:
                        break
                    weights = np.where(
                        np.abs(resid / scale) <= d,
                        1.0,
                        d / (np.abs(resid / scale) + 1e-12)
                    )
                    W = np.diag(weights)
                    coef = np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]
                return float(coef[idx]) if idx < len(coef) else np.nan
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_huber_coef, window_size=w, min_samples=3).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_huber_regression_coeff_prior",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_huber_regression_coeff_prior",
    source=_SRC,
    backend="polars")
class TSHuberRegressionCoeffPriorNative(SeriesOperator):
    """Huber robust regression coefficient from prior window."""

    metadata = OperatorMetadata(
        name="ts_huber_regression_coeff_prior",
        category="time_series",
        description="Huber 鲁棒回归历史系数",
        param_names=["x", "window", "delta", "order", "warmup_policy"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "delta": ParamSpec(dtype=float, min=0.0, default=1.35, searchable=True, param_role=ParamRole.THRESHOLD),
            "order": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, delta: float = 1.35, order: int = 1, warmup_policy: str = "exact", **kwargs
    ) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        d = strict_finite_scalar(delta, "delta", minimum=0.0)
        idx = 0

        cols = _numeric_cols(x)

        def _huber_coef(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < 3:
                return np.nan
            y = arr[valid]
            t = np.arange(len(y))
            X = np.column_stack([np.ones(len(t)), t])
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                for _ in range(3):
                    fitted = X @ coef
                    resid = y - fitted
                    scale = (np.median(np.abs(resid))) / 0.6745 if 0.6745 != 0 else np.nan
                    if scale < 1e-12:
                        break
                    weights = np.where(
                        np.abs(resid / scale) <= d,
                        1.0,
                        d / (np.abs(resid / scale) + 1e-12)
                    )
                    W = np.diag(weights)
                    coef = np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]
                return float(coef[idx]) if idx < len(coef) else np.nan
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).shift(1).rolling_map(_huber_coef, window_size=w, min_samples=3).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


# Similar pattern for other Huber operators - forecast_error, forecast_error_z, in_sample_resid, predictive_resid, resid_z
# Implementing key ones for completeness


@register_operator(
    name="ts_huber_regression_forecast_error",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_huber_regression_forecast_error",
    source=_SRC,
    backend="polars")
class TSHuberRegressionForecastErrorNative(SeriesOperator):
    """Huber robust regression forecast error."""

    metadata = OperatorMetadata(
        name="ts_huber_regression_forecast_error",
        category="time_series",
        description="Huber 鲁棒回归预测误差",
        param_names=["x", "window", "delta"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "delta": ParamSpec(dtype=float, min=0.0, default=1.35, searchable=True, param_role=ParamRole.THRESHOLD),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, delta: float = 1.35, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        d = strict_finite_scalar(delta, "delta", minimum=0.0)
        cols = _numeric_cols(x)

        def _huber_error(vals):
            arr = np.asarray(vals, dtype=float)
            if len(arr) < w:
                return np.nan
            current = arr[-1]
            if np.isnan(current):
                return np.nan
            past = arr[:-1]
            valid = ~np.isnan(past)
            if np.sum(valid) < 3:
                return np.nan
            y = past[valid]
            t = np.arange(len(y))
            X = np.column_stack([np.ones(len(t)), t])
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                for _ in range(3):
                    fitted = X @ coef
                    resid = y - fitted
                    scale = (np.median(np.abs(resid))) / 0.6745 if 0.6745 != 0 else np.nan
                    if scale < 1e-12:
                        break
                    weights = np.where(
                        np.abs(resid / scale) <= d,
                        1.0,
                        d / (np.abs(resid / scale) + 1e-12)
                    )
                    W = np.diag(weights)
                    coef = np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]
                t_next = len(y)
                forecast = coef[0] + coef[1] * t_next
                return float(current - forecast)
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_huber_error, window_size=w+1, min_samples=4).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_huber_regression_forecast_error_z",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_huber_regression_forecast_error_z",
    source=_SRC,
    backend="polars")
class TSHuberRegressionForecastErrorZNative(SeriesOperator):
    """Standardized Huber robust regression forecast error."""

    metadata = OperatorMetadata(
        name="ts_huber_regression_forecast_error_z",
        category="time_series",
        description="Huber 鲁棒回归标准化预测误差",
        param_names=["x", "window", "delta"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "delta": ParamSpec(dtype=float, min=0.0, default=1.35, searchable=True, param_role=ParamRole.THRESHOLD),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, delta: float = 1.35, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        d = strict_finite_scalar(delta, "delta", minimum=0.0)
        cols = _numeric_cols(x)

        def _huber_error_z(vals):
            arr = np.asarray(vals, dtype=float)
            if len(arr) < w:
                return np.nan
            current = arr[-1]
            if np.isnan(current):
                return np.nan
            past = arr[:-1]
            valid = ~np.isnan(past)
            if np.sum(valid) < 3:
                return np.nan
            y = past[valid]
            t = np.arange(len(y))
            X = np.column_stack([np.ones(len(t)), t])
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                for _ in range(3):
                    fitted = X @ coef
                    resid = y - fitted
                    scale = (np.median(np.abs(resid))) / 0.6745 if 0.6745 != 0 else np.nan
                    if scale < 1e-12:
                        break
                    weights = np.where(
                        np.abs(resid / scale) <= d,
                        1.0,
                        d / (np.abs(resid / scale) + 1e-12)
                    )
                    W = np.diag(weights)
                    coef = np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]
                fitted = X @ coef
                resid = y - fitted
                std = np.std(resid, ddof=1)
                if std < 1e-12:
                    return np.nan
                t_next = len(y)
                forecast = coef[0] + coef[1] * t_next
                error = current - forecast
                return np.where(std != 0, (float(error) / (std)), np.nan)
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_huber_error_z, window_size=w+1, min_samples=4).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_huber_regression_in_sample_resid",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_huber_regression_in_sample_resid",
    source=_SRC,
    backend="polars")
class TSHuberRegressionInSampleResidNative(SeriesOperator):
    """Huber robust regression in-sample residual std."""

    metadata = OperatorMetadata(
        name="ts_huber_regression_in_sample_resid",
        category="time_series",
        description="Huber 鲁棒回归样本内残差",
        param_names=["x", "window", "delta"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "delta": ParamSpec(dtype=float, min=0.0, default=1.35, searchable=True, param_role=ParamRole.THRESHOLD),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, delta: float = 1.35, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        d = strict_finite_scalar(delta, "delta", minimum=0.0)
        cols = _numeric_cols(x)

        def _huber_resid(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < 3:
                return np.nan
            y = arr[valid]
            t = np.arange(len(y))
            X = np.column_stack([np.ones(len(t)), t])
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                for _ in range(3):
                    fitted = X @ coef
                    resid = y - fitted
                    scale = (np.median(np.abs(resid))) / 0.6745 if 0.6745 != 0 else np.nan
                    if scale < 1e-12:
                        break
                    weights = np.where(
                        np.abs(resid / scale) <= d,
                        1.0,
                        d / (np.abs(resid / scale) + 1e-12)
                    )
                    W = np.diag(weights)
                    coef = np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]
                fitted = X @ coef
                resid = y - fitted
                return float(np.std(resid, ddof=1))
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_huber_resid, window_size=w, min_samples=3).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_huber_regression_predictive_resid",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_huber_regression_predictive_resid",
    source=_SRC,
    backend="polars")
class TSHuberRegressionPredictiveResidNative(SeriesOperator):
    """Huber robust regression rolling predictive residual std."""

    metadata = OperatorMetadata(
        name="ts_huber_regression_predictive_resid",
        category="time_series",
        description="Huber 鲁棒回归预测残差",
        param_names=["x", "window", "delta"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "delta": ParamSpec(dtype=float, min=0.0, default=1.35, searchable=True, param_role=ParamRole.THRESHOLD),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, delta: float = 1.35, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        d = strict_finite_scalar(delta, "delta", minimum=0.0)
        cols = _numeric_cols(x)

        def _huber_pred_resid(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < w:
                return np.nan
            arr = arr[valid]
            if len(arr) < w:
                return np.nan

            errors = []
            for i in range(3, len(arr)):
                y_train = arr[:i]
                t_train = np.arange(len(y_train))
                X_train = np.column_stack([np.ones(len(t_train)), t_train])
                try:
                    coef = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
                    for _ in range(3):
                        fitted = X_train @ coef
                        resid = y_train - fitted
                        scale = (np.median(np.abs(resid))) / 0.6745 if 0.6745 != 0 else np.nan
                        if scale < 1e-12:
                            break
                        weights = np.where(
                            np.abs(resid / scale) <= d,
                            1.0,
                            d / (np.abs(resid / scale) + 1e-12)
                        )
                        W = np.diag(weights)
                        coef = np.linalg.lstsq(W @ X_train, W @ y_train, rcond=None)[0]
                    t_next = len(y_train)
                    forecast = coef[0] + coef[1] * t_next
                    errors.append(arr[i] - forecast)
                except:
                    pass

            if len(errors) < 2:
                return np.nan
            return float(np.std(errors, ddof=1))

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_huber_pred_resid, window_size=w, min_samples=w).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_huber_regression_resid_z",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_huber_regression_resid_z",
    source=_SRC,
    backend="polars")
class TSHuberRegressionResidZNative(SeriesOperator):
    """Current value as z-score of Huber residuals."""

    metadata = OperatorMetadata(
        name="ts_huber_regression_resid_z",
        category="time_series",
        description="Huber 鲁棒回归残差 Z 分数",
        param_names=["x", "window", "delta"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "delta": ParamSpec(dtype=float, min=0.0, default=1.35, searchable=True, param_role=ParamRole.THRESHOLD),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, delta: float = 1.35, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        d = strict_finite_scalar(delta, "delta", minimum=0.0)
        cols = _numeric_cols(x)

        def _huber_resid_z(vals):
            arr = np.asarray(vals, dtype=float)
            if len(arr) < w:
                return np.nan
            current = arr[-1]
            if np.isnan(current):
                return np.nan
            past = arr[:-1]
            valid = ~np.isnan(past)
            if np.sum(valid) < 3:
                return np.nan
            y = past[valid]
            t = np.arange(len(y))
            X = np.column_stack([np.ones(len(t)), t])
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                for _ in range(3):
                    fitted = X @ coef
                    resid = y - fitted
                    scale = (np.median(np.abs(resid))) / 0.6745 if 0.6745 != 0 else np.nan
                    if scale < 1e-12:
                        break
                    weights = np.where(
                        np.abs(resid / scale) <= d,
                        1.0,
                        d / (np.abs(resid / scale) + 1e-12)
                    )
                    W = np.diag(weights)
                    coef = np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]
                fitted = X @ coef
                resid = y - fitted
                mean_resid = np.mean(resid)
                std_resid = np.std(resid, ddof=1)
                if std_resid < 1e-12:
                    return np.nan
                t_next = len(y)
                forecast = coef[0] + coef[1] * t_next
                current_resid = current - forecast
                return np.where(std_resid != 0, (float((current_resid - mean_resid)) / (std_resid)), np.nan)
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_huber_resid_z, window_size=w+1, min_samples=4).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


# ---------------------------------------------------------------------------
# Quantile Regression Operators
# ---------------------------------------------------------------------------


@register_operator(
    name="ts_quantile_regression_coeff",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_quantile_regression_coeff",
    source=_SRC,
    backend="polars")
class TSQuantileRegressionCoeffNative(SeriesOperator):
    """Quantile regression coefficient (simplified via weighted least squares approximation)."""

    metadata = OperatorMetadata(
        name="ts_quantile_regression_coeff",
        category="time_series",
        description="分位数回归系数",
        param_names=["x", "window", "quantile", "coef_index"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "quantile": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.5, searchable=True, param_role=ParamRole.THRESHOLD),
            "coef_index": ParamSpec(dtype=int, min=0, default=0, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, quantile: float = 0.5, coef_index: int = 0, **kwargs
    ) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        q = strict_finite_scalar(quantile, "quantile", minimum=0.0, maximum=1.0)
        idx = strict_integer(coef_index, "coef_index", minimum=0)

        cols = _numeric_cols(x)

        def _quantile_coef(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < 3:
                return np.nan
            y = arr[valid]
            t = np.arange(len(y))
            X = np.column_stack([np.ones(len(t)), t])
            try:
                # Simplified: use OLS then adjust towards quantile via 2 iterations
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                for _ in range(2):
                    fitted = X @ coef
                    resid = y - fitted
                    weights = np.where(resid >= 0, q, 1 - q)
                    W = np.diag(weights)
                    coef = np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]
                return float(coef[idx]) if idx < len(coef) else np.nan
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_quantile_coef, window_size=w, min_samples=3).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_quantile_regression_coeff_prior",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_quantile_regression_coeff_prior",
    source=_SRC,
    backend="polars")
class TSQuantileRegressionCoeffPriorNative(SeriesOperator):
    """Quantile regression coefficient from prior window."""

    metadata = OperatorMetadata(
        name="ts_quantile_regression_coeff_prior",
        category="time_series",
        description="分位数回归历史系数",
        param_names=["x", "window", "quantile", "coef_index"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "quantile": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.5, searchable=True, param_role=ParamRole.THRESHOLD),
            "coef_index": ParamSpec(dtype=int, min=0, default=0, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, quantile: float = 0.5, coef_index: int = 0, **kwargs
    ) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        q = strict_finite_scalar(quantile, "quantile", minimum=0.0, maximum=1.0)
        idx = strict_integer(coef_index, "coef_index", minimum=0)

        cols = _numeric_cols(x)

        def _quantile_coef(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < 3:
                return np.nan
            y = arr[valid]
            t = np.arange(len(y))
            X = np.column_stack([np.ones(len(t)), t])
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                for _ in range(2):
                    fitted = X @ coef
                    resid = y - fitted
                    weights = np.where(resid >= 0, q, 1 - q)
                    W = np.diag(weights)
                    coef = np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]
                return float(coef[idx]) if idx < len(coef) else np.nan
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).shift(1).rolling_map(_quantile_coef, window_size=w, min_samples=3).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_quantile_regression_resid",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_quantile_regression_resid",
    source=_SRC,
    backend="polars")
class TSQuantileRegressionResidNative(SeriesOperator):
    """Quantile regression in-sample residual std."""

    metadata = OperatorMetadata(
        name="ts_quantile_regression_resid",
        category="time_series",
        description="分位数回归残差",
        param_names=["x", "window", "quantile"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "quantile": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.5, searchable=True, param_role=ParamRole.THRESHOLD),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, quantile: float = 0.5, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        q = strict_finite_scalar(quantile, "quantile", minimum=0.0, maximum=1.0)
        cols = _numeric_cols(x)

        def _quantile_resid(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < 3:
                return np.nan
            y = arr[valid]
            t = np.arange(len(y))
            X = np.column_stack([np.ones(len(t)), t])
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                for _ in range(2):
                    fitted = X @ coef
                    resid = y - fitted
                    weights = np.where(resid >= 0, q, 1 - q)
                    W = np.diag(weights)
                    coef = np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]
                fitted = X @ coef
                resid = y - fitted
                return float(np.std(resid, ddof=1))
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_quantile_resid, window_size=w, min_samples=3).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_quantile_regression_slope",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_quantile_regression_slope",
    source=_SRC,
    backend="polars")
class TSQuantileRegressionSlopeNative(SeriesOperator):
    """Quantile regression slope coefficient (coef_index=1)."""

    metadata = OperatorMetadata(
        name="ts_quantile_regression_slope",
        category="time_series",
        description="分位数回归斜率",
        param_names=["x", "window", "quantile"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "quantile": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.5, searchable=True, param_role=ParamRole.THRESHOLD),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, quantile: float = 0.5, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        q = strict_finite_scalar(quantile, "quantile", minimum=0.0, maximum=1.0)
        cols = _numeric_cols(x)

        def _quantile_slope(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < 3:
                return np.nan
            y = arr[valid]
            t = np.arange(len(y))
            X = np.column_stack([np.ones(len(t)), t])
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                for _ in range(2):
                    fitted = X @ coef
                    resid = y - fitted
                    weights = np.where(resid >= 0, q, 1 - q)
                    W = np.diag(weights)
                    coef = np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]
                return float(coef[1])
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_quantile_slope, window_size=w, min_samples=3).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


# Multi-variable regression operators would require multiple input series,
# which requires a different operator signature. Implementing simplified versions.


@register_operator(
    name="ts_multi_regression_coeff",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_multi_regression_coeff",
    source=_SRC,
    backend="polars")
class TSMultiRegressionCoeffNative(SeriesOperator):
    """Multi-variable regression coefficient (uses polynomial features from single series)."""

    metadata = OperatorMetadata(
        name="ts_multi_regression_coeff",
        category="time_series",
        description="多变量回归系数",
        param_names=["x", "window", "degree", "coef_index"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "degree": ParamSpec(dtype=int, min=1, max=3, default=2, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "coef_index": ParamSpec(dtype=int, min=0, default=0, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, degree: int = 2, coef_index: int = 0, **kwargs
    ) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        deg = strict_integer(degree, "degree", minimum=1, maximum=3)
        idx = strict_integer(coef_index, "coef_index", minimum=0)

        cols = _numeric_cols(x)

        def _multi_coef(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < deg + 2:
                return np.nan
            y = arr[valid]
            t = np.arange(len(y))
            X_list = [np.ones(len(t))]
            for d in range(1, deg + 1):
                X_list.append(t ** d)
            X = np.column_stack(X_list)
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                return float(coef[idx]) if idx < len(coef) else np.nan
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_multi_coef, window_size=w, min_samples=deg+2).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_multi_regression_coeff_prior",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_multi_regression_coeff_prior",
    source=_SRC,
    backend="polars")
class TSMultiRegressionCoeffPriorNative(SeriesOperator):
    """Multi-variable regression coefficient from prior window."""

    metadata = OperatorMetadata(
        name="ts_multi_regression_coeff_prior",
        category="time_series",
        description="多变量回归历史系数",
        param_names=["x", "window", "degree", "coef_index"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "degree": ParamSpec(dtype=int, min=1, max=3, default=2, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "coef_index": ParamSpec(dtype=int, min=0, default=0, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, degree: int = 2, coef_index: int = 0, **kwargs
    ) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        deg = strict_integer(degree, "degree", minimum=1, maximum=3)
        idx = strict_integer(coef_index, "coef_index", minimum=0)

        cols = _numeric_cols(x)

        def _multi_coef(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < deg + 2:
                return np.nan
            y = arr[valid]
            t = np.arange(len(y))
            X_list = [np.ones(len(t))]
            for d in range(1, deg + 1):
                X_list.append(t ** d)
            X = np.column_stack(X_list)
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                return float(coef[idx]) if idx < len(coef) else np.nan
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).shift(1).rolling_map(_multi_coef, window_size=w, min_samples=deg+2).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_multi_regression_r2",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_multi_regression_r2",
    source=_SRC,
    backend="polars")
class TSMultiRegressionR2Native(SeriesOperator):
    """Multi-variable regression R-squared."""

    metadata = OperatorMetadata(
        name="ts_multi_regression_r2",
        category="time_series",
        description="多变量回归 R²",
        param_names=["x", "window", "degree"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "degree": ParamSpec(dtype=int, min=1, max=3, default=2, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, degree: int = 2, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        deg = strict_integer(degree, "degree", minimum=1, maximum=3)
        cols = _numeric_cols(x)

        def _multi_r2(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < deg + 2:
                return np.nan
            y = arr[valid]
            t = np.arange(len(y))
            X_list = [np.ones(len(t))]
            for d in range(1, deg + 1):
                X_list.append(t ** d)
            X = np.column_stack(X_list)
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                fitted = X @ coef
                ss_res = np.sum((y - fitted) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                if ss_tot < 1e-12:
                    return np.nan
                return np.where(ss_tot != 0, (float(1.0 - ss_res) / (ss_tot)), np.nan)
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_multi_r2, window_size=w, min_samples=deg+2).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_multi_regression_r2_prior",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_multi_regression_r2_prior",
    source=_SRC,
    backend="polars")
class TSMultiRegressionR2PriorNative(SeriesOperator):
    """Multi-variable regression R-squared from prior window."""

    metadata = OperatorMetadata(
        name="ts_multi_regression_r2_prior",
        category="time_series",
        description="多变量回归历史 R²",
        param_names=["x", "window", "degree"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "degree": ParamSpec(dtype=int, min=1, max=3, default=2, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, degree: int = 2, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        deg = strict_integer(degree, "degree", minimum=1, maximum=3)
        cols = _numeric_cols(x)

        def _multi_r2(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < deg + 2:
                return np.nan
            y = arr[valid]
            t = np.arange(len(y))
            X_list = [np.ones(len(t))]
            for d in range(1, deg + 1):
                X_list.append(t ** d)
            X = np.column_stack(X_list)
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                fitted = X @ coef
                ss_res = np.sum((y - fitted) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                if ss_tot < 1e-12:
                    return np.nan
                return np.where(ss_tot != 0, (float(1.0 - ss_res) / (ss_tot)), np.nan)
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).shift(1).rolling_map(_multi_r2, window_size=w, min_samples=deg+2).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_multi_regression_forecast_error",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_multi_regression_forecast_error",
    source=_SRC,
    backend="polars")
class TSMultiRegressionForecastErrorNative(SeriesOperator):
    """Multi-variable regression forecast error."""

    metadata = OperatorMetadata(
        name="ts_multi_regression_forecast_error",
        category="time_series",
        description="多变量回归预测误差",
        param_names=["x", "window", "degree"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "degree": ParamSpec(dtype=int, min=1, max=3, default=2, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, degree: int = 2, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        deg = strict_integer(degree, "degree", minimum=1, maximum=3)
        cols = _numeric_cols(x)

        def _multi_error(vals):
            arr = np.asarray(vals, dtype=float)
            if len(arr) < w:
                return np.nan
            current = arr[-1]
            if np.isnan(current):
                return np.nan
            past = arr[:-1]
            valid = ~np.isnan(past)
            if np.sum(valid) < deg + 2:
                return np.nan
            y = past[valid]
            t = np.arange(len(y))
            X_list = [np.ones(len(t))]
            for d in range(1, deg + 1):
                X_list.append(t ** d)
            X = np.column_stack(X_list)
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                t_next = len(y)
                forecast = sum(coef[i] * (t_next ** i) for i in range(len(coef)))
                return float(current - forecast)
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_multi_error, window_size=w+1, min_samples=deg+3).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_multi_regression_forecast_error_z",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_multi_regression_forecast_error_z",
    source=_SRC,
    backend="polars")
class TSMultiRegressionForecastErrorZNative(SeriesOperator):
    """Standardized multi-variable regression forecast error."""

    metadata = OperatorMetadata(
        name="ts_multi_regression_forecast_error_z",
        category="time_series",
        description="多变量回归标准化预测误差",
        param_names=["x", "window", "degree"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "degree": ParamSpec(dtype=int, min=1, max=3, default=2, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, degree: int = 2, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        deg = strict_integer(degree, "degree", minimum=1, maximum=3)
        cols = _numeric_cols(x)

        def _multi_error_z(vals):
            arr = np.asarray(vals, dtype=float)
            if len(arr) < w:
                return np.nan
            current = arr[-1]
            if np.isnan(current):
                return np.nan
            past = arr[:-1]
            valid = ~np.isnan(past)
            if np.sum(valid) < deg + 2:
                return np.nan
            y = past[valid]
            t = np.arange(len(y))
            X_list = [np.ones(len(t))]
            for d in range(1, deg + 1):
                X_list.append(t ** d)
            X = np.column_stack(X_list)
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                fitted = X @ coef
                resid = y - fitted
                std = np.std(resid, ddof=1)
                if std < 1e-12:
                    return np.nan
                t_next = len(y)
                forecast = sum(coef[i] * (t_next ** i) for i in range(len(coef)))
                error = current - forecast
                return np.where(std != 0, (float(error) / (std)), np.nan)
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_multi_error_z, window_size=w+1, min_samples=deg+3).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_multi_regression_resid",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_multi_regression_resid",
    source=_SRC,
    backend="polars")
class TSMultiRegressionResidNative(SeriesOperator):
    """Multi-variable regression in-sample residual std."""

    metadata = OperatorMetadata(
        name="ts_multi_regression_resid",
        category="time_series",
        description="多变量回归残差",
        param_names=["x", "window", "degree"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "degree": ParamSpec(dtype=int, min=1, max=3, default=2, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, degree: int = 2, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        deg = strict_integer(degree, "degree", minimum=1, maximum=3)
        cols = _numeric_cols(x)

        def _multi_resid(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < deg + 2:
                return np.nan
            y = arr[valid]
            t = np.arange(len(y))
            X_list = [np.ones(len(t))]
            for d in range(1, deg + 1):
                X_list.append(t ** d)
            X = np.column_stack(X_list)
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                fitted = X @ coef
                resid = y - fitted
                return float(np.std(resid, ddof=1))
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_multi_resid, window_size=w, min_samples=deg+2).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_multi_regression_resid_z",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_multi_regression_resid_z",
    source=_SRC,
    backend="polars")
class TSMultiRegressionResidZNative(SeriesOperator):
    """Current value as z-score of multi-regression residuals."""

    metadata = OperatorMetadata(
        name="ts_multi_regression_resid_z",
        category="time_series",
        description="多变量回归残差 Z 分数",
        param_names=["x", "window", "degree"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "degree": ParamSpec(dtype=int, min=1, max=3, default=2, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, degree: int = 2, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        deg = strict_integer(degree, "degree", minimum=1, maximum=3)
        cols = _numeric_cols(x)

        def _multi_resid_z(vals):
            arr = np.asarray(vals, dtype=float)
            if len(arr) < w:
                return np.nan
            current = arr[-1]
            if np.isnan(current):
                return np.nan
            past = arr[:-1]
            valid = ~np.isnan(past)
            if np.sum(valid) < deg + 2:
                return np.nan
            y = past[valid]
            t = np.arange(len(y))
            X_list = [np.ones(len(t))]
            for d in range(1, deg + 1):
                X_list.append(t ** d)
            X = np.column_stack(X_list)
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                fitted = X @ coef
                resid = y - fitted
                mean_resid = np.mean(resid)
                std_resid = np.std(resid, ddof=1)
                if std_resid < 1e-12:
                    return np.nan
                t_next = len(y)
                forecast = sum(coef[i] * (t_next ** i) for i in range(len(coef)))
                current_resid = current - forecast
                return np.where(std_resid != 0, (float((current_resid - mean_resid)) / (std_resid)), np.nan)
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_multi_resid_z, window_size=w+1, min_samples=deg+3).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_multi_regression_coeff_stability",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_multi_regression_coeff_stability",
    source=_SRC,
    backend="polars")
class TSMultiRegressionCoeffStabilityNative(SeriesOperator):
    """Multi-regression coefficient stability over sub-windows."""

    metadata = OperatorMetadata(
        name="ts_multi_regression_coeff_stability",
        category="time_series",
        description="多变量回归系数稳定性",
        param_names=["x", "window", "degree", "coef_index"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=10, default=60, searchable=True, param_role=ParamRole.HORIZON),
            "degree": ParamSpec(dtype=int, min=1, max=3, default=2, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "coef_index": ParamSpec(dtype=int, min=0, default=0, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 60, degree: int = 2, coef_index: int = 0, **kwargs
    ) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=10)
        deg = strict_integer(degree, "degree", minimum=1, maximum=3)
        idx = strict_integer(coef_index, "coef_index", minimum=0)

        cols = _numeric_cols(x)

        def _coef_stability(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < w:
                return np.nan
            arr = arr[valid]
            if len(arr) < w:
                return np.nan

            sub_w = max(deg + 5, w // 3)
            n_subs = max(2, (len(arr) - sub_w) // 5)
            coefs = []

            for i in range(n_subs):
                start = i * 5
                end = start + sub_w
                if end > len(arr):
                    break
                sub = arr[start:end]
                if len(sub) < deg + 2:
                    continue
                t = np.arange(len(sub))
                X_list = [np.ones(len(t))]
                for d in range(1, deg + 1):
                    X_list.append(t ** d)
                X = np.column_stack(X_list)
                try:
                    coef = np.linalg.lstsq(X, sub, rcond=None)[0]
                    if idx < len(coef):
                        coefs.append(coef[idx])
                except:
                    pass

            if len(coefs) < 2:
                return np.nan
            return float(np.std(coefs, ddof=1))

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_coef_stability, window_size=w, min_samples=w).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="ts_multi_regression_adjusted_r2_prior",
    category="time_series",
    business_category="time_series_regression",
    canonical="ts_multi_regression_adjusted_r2_prior",
    source=_SRC,
    backend="polars")
class TSMultiRegressionAdjustedR2PriorNative(SeriesOperator):
    """Multi-variable regression adjusted R-squared from prior window."""

    metadata = OperatorMetadata(
        name="ts_multi_regression_adjusted_r2_prior",
        category="time_series",
        description="多变量回归调整 R²",
        param_names=["x", "window", "degree"],
        return_type="series",
        tags=["time_series", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "degree": ParamSpec(dtype=int, min=1, max=3, default=2, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, degree: int = 2, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        deg = strict_integer(degree, "degree", minimum=1, maximum=3)
        cols = _numeric_cols(x)

        def _adj_r2(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < deg + 2:
                return np.nan
            y = arr[valid]
            n = len(y)
            t = np.arange(n)
            X_list = [np.ones(n)]
            for d in range(1, deg + 1):
                X_list.append(t ** d)
            X = np.column_stack(X_list)
            p = X.shape[1]
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                fitted = X @ coef
                ss_res = np.sum((y - fitted) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                if ss_tot < 1e-12:
                    return np.nan
                r2 = (1.0 - ss_res) / ss_tot if ss_tot != 0 else np.nan
                if n <= p:
                    return np.nan
                adj_r2 = (1.0 - (1.0 - r2) * (n - 1)) / ((n - p) if (n - p) != 0 else np.nan)
                return float(adj_r2)
            except:
                return np.nan

        exprs = [
            pl.col(c).fill_nan(None).shift(1).rolling_map(_adj_r2, window_size=w, min_samples=deg+2).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)



