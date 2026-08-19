# -*- coding: utf-8 -*-
"""Time-series rolling/correlation operators - Polars native implementations (Phase 2, Module 8).

All operators use TRUE Polars expressions only - no pandas fallback, no NumPy.
"""
from __future__ import annotations

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.base import ParamRole, ParamSpec
from backend.contracts import ExecutionKind, PhysicalImplementationSpec

_PAIRWISE_CANONICALS = frozenset({
    "ts_corr",
    "ts_cov",
    "ts_regression_slope",
    "ts_regression_intercept",
    "ts_regression_resid",
    "ts_regression_r2",
})


def _register_rolling_operator(**kwargs):
    """Register only the repaired pairwise surface during normal bootstrap."""
    if kwargs.get("canonical") in _PAIRWISE_CANONICALS:
        return register_operator(**kwargs)
    return lambda cls: cls

_SKIP = frozenset({"date", "stock_code"})
_SRC = "factor_dsl_polars_native"


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _require_pairwise_columns(
    left: pl.DataFrame, right: pl.DataFrame | None, *, operator: str
) -> list[str]:
    """Fail closed when a wide pairwise operand lacks counterpart columns."""
    if right is None:
        raise ValueError(f"{operator} requires a counterpart panel")
    left_cols = _numeric_cols(left)
    right_cols = set(_numeric_cols(right))
    missing = [column for column in left_cols if column not in right_cols]
    if missing:
        sample = missing[:5]
        raise ValueError(
            f"{operator}: counterpart panel missing {len(missing)} column(s); "
            f"left_count={len(left_cols)}, right_count={len(right_cols)}, "
            f"missing_sample={sample!r}"
        )
    return left_cols


def _with_meta(result: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
    if "date" in source.columns and "date" not in result.columns:
        result = result.with_columns(source["date"])
    return result


def _pairwise_rolling_moments(
    x_col: pl.Expr,
    y_col: pl.Expr,
    *,
    window: int,
    min_periods: int,
) -> dict[str, pl.Expr]:
    """Return direct-window moments over finite ``(x, y)`` pairs only."""
    valid = x_col.is_finite().fill_null(False) & y_col.is_finite().fill_null(False)
    x_valid = pl.when(valid).then(x_col).otherwise(None)
    y_valid = pl.when(valid).then(y_col).otherwise(None)
    # Build each active window horizontally and center it on that window's first
    # finite pair.  A fixed dataset-wide anchor can become numerically useless
    # after it rolls out; the per-window translation keeps all moment arithmetic
    # on the scale of the observations currently being compared.
    x_window = [x_valid.shift(lag) for lag in range(window)]
    y_window = [y_valid.shift(lag) for lag in range(window)]
    x_anchor = pl.coalesce(x_window)
    y_anchor = pl.coalesce(y_window)
    x_centered = [value - x_anchor for value in x_window]
    y_centered = [value - y_anchor for value in y_window]
    count = pl.sum_horizontal(
        [value.is_not_null().cast(pl.Float64) for value in x_window]
    )
    sum_x = pl.sum_horizontal(x_centered)
    sum_y = pl.sum_horizontal(y_centered)
    sum_xx = pl.sum_horizontal([value * value for value in x_centered])
    sum_yy = pl.sum_horizontal([value * value for value in y_centered])
    sum_xy = pl.sum_horizontal(
        [left * right for left, right in zip(x_centered, y_centered)]
    )
    ready = count >= min_periods
    # Re-center a second time around the window mean.  Expanding
    # ``sum((x-anchor) * (y-anchor)) - sum_x * sum_y / count`` loses low-order
    # covariance when observations have a large common offset.  The direct
    # residual products retain those low-order differences while keeping the
    # anchor local to each active window.
    mean_x = x_anchor + sum_x / count
    mean_y = y_anchor + sum_y / count
    centered_x = [value - mean_x for value in x_window]
    centered_y = [value - mean_y for value in y_window]
    stable_sum_xx = pl.sum_horizontal([value * value for value in centered_x])
    stable_sum_yy = pl.sum_horizontal([value * value for value in centered_y])
    stable_sum_xy = pl.sum_horizontal(
        [left * right for left, right in zip(centered_x, centered_y)]
    )

    # Through-origin fits need uncentered sums, but compute them from the
    # same per-window anchor to avoid squaring large absolute offsets.
    raw_xx = count * x_anchor * x_anchor + 2 * x_anchor * sum_x + sum_xx
    raw_xy = (
        count * x_anchor * y_anchor
        + x_anchor * sum_y
        + y_anchor * sum_x
        + sum_xy
    )
    raw_yy = count * y_anchor * y_anchor + 2 * y_anchor * sum_y + sum_yy

    return {
        "count": count,
        "mean_x": pl.when(ready).then(mean_x).otherwise(None),
        "mean_y": pl.when(ready).then(mean_y).otherwise(None),
        "cross": pl.when(ready).then(stable_sum_xy).otherwise(None),
        "ss_x": pl.when(ready).then(stable_sum_xx).otherwise(None),
        "ss_y": pl.when(ready).then(stable_sum_yy).otherwise(None),
        "raw_xx": pl.when(ready).then(raw_xx).otherwise(None),
        "raw_xy": pl.when(ready).then(raw_xy).otherwise(None),
        "raw_yy": pl.when(ready).then(raw_yy).otherwise(None),
        "x_endpoint": x_valid,
        "y_endpoint": y_valid,
    }


# ---------------------------------------------------------------------------
# Rolling correlation & covariance
# ---------------------------------------------------------------------------


@_register_rolling_operator(
    name="ts_corr",
    category="time_series",
    business_category="time_series",
    canonical="ts_corr",
    source=_SRC,
    backend="polars",
)
class TSCorrNative(SeriesOperator):
    """Rolling correlation between x and y."""

    _physical_spec = PhysicalImplementationSpec(
        canonical="ts_corr", backend="polars",
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
        supports_lazy=False, materializes_full_panel=True, requires_sorted=True,
        supports_nulls=True, supports_nan=False, supports_inf=False,
        implementation_source_hash="cleaned_operators.common.polars_ts_rolling:TSCorrNative:v1",
        emitter_identity="polars.rolling.pairwise_moments:v1",
        parameter_domain_hash="ts_corr.x:dataframe,y:dataframe,window:int:min=2:default=20,min_periods:int:min=1:nullable:default=2",
        semantic_contract_hash="ts_corr:pairwise_finite:v1",
    )

    metadata = OperatorMetadata(
        name="ts_corr",
        category="time_series",
        description="滚动相关系数",
        param_names=["x", "y", "window", "min_periods"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame | None = None,
                         window: int = 20, min_periods: int = 2, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        min_p = 2 if min_periods is None else strict_integer(min_periods, "min_periods", minimum=1)
        if min_p > w:
            raise ValueError("min_periods must be <= window")
        cols = _require_pairwise_columns(x, y, operator="ts_corr")

        exprs = []
        for c in cols:
            x_col = pl.col(c)
            y_col = y[c]

            moments = _pairwise_rolling_moments(x_col, y_col, window=w, min_periods=min_p)
            corr = pl.when((moments["ss_x"] <= 0) | (moments["ss_y"] <= 0)).then(
                None
            ).otherwise(moments["cross"] / (moments["ss_x"] * moments["ss_y"]).sqrt())
            exprs.append(corr.alias(c))

        return x.lazy().with_columns(exprs).collect()


@_register_rolling_operator(
    name="ts_cov",
    category="time_series",
    business_category="time_series",
    canonical="ts_cov",
    source=_SRC,
    backend="polars",
)
class TSCovNative(SeriesOperator):
    """Rolling covariance between x and y."""

    _physical_spec = PhysicalImplementationSpec(
        canonical="ts_cov", backend="polars",
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
        supports_lazy=False, materializes_full_panel=True, requires_sorted=True,
        supports_nulls=True, supports_nan=False, supports_inf=False,
        implementation_source_hash="cleaned_operators.common.polars_ts_rolling:TSCovNative:v1",
        emitter_identity="polars.rolling.pairwise_moments:v1",
        parameter_domain_hash="ts_cov.x:dataframe,y:dataframe,window:int:min=2:default=20,ddof:int:min=0:default=1,min_periods:int:min=1:nullable:default=2",
        semantic_contract_hash="ts_cov:pairwise_finite:ddof:v1",
    )

    metadata = OperatorMetadata(
        name="ts_cov",
        category="time_series",
        description="滚动协方差",
        param_names=["x", "y", "window", "ddof", "min_periods"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame | None = None,
                         window: int = 20, ddof: int = 1,
                         min_periods: int | None = None, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        ddof_val = strict_integer(ddof, "ddof", minimum=0)

        # Default min_periods to 2 if not specified (need at least 2 pairs)
        if min_periods is None:
            min_p = 2
        else:
            min_p = strict_integer(min_periods, "min_periods", minimum=1)
        if min_p > w:
            raise ValueError("min_periods must be <= window")

        cols = _require_pairwise_columns(x, y, operator="ts_cov")

        exprs = []
        for c in cols:
            x_col = pl.col(c)
            y_col = y[c]

            moments = _pairwise_rolling_moments(
                x_col, y_col, window=w, min_periods=min_p
            )
            cov = pl.when(moments["count"] > ddof_val).then(
                moments["cross"] / (moments["count"] - ddof_val)
            ).otherwise(None)

            exprs.append(cov.alias(c))

        return x.lazy().with_columns(exprs).collect()


@_register_rolling_operator(
    name="ts_ewm_corr",
    category="time_series",
    business_category="time_series",
    canonical="ts_ewm_corr",
    source=_SRC,
    backend="polars",
)
class TSEwmCorrNative(SeriesOperator):
    """Exponentially weighted correlation."""

    metadata = OperatorMetadata(
        name="ts_ewm_corr",
        category="time_series",
        description="指数加权相关系数",
        param_names=["x", "y", "window"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame | None = None,
                         window: int = 20, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        alpha = 2.0 / (w + 1)

        cols = _require_pairwise_columns(x, y, operator="ts_ewm_corr")

        exprs = []
        for c in cols:
            x_raw = pl.col(c)
            y_raw = y[c]
            valid = x_raw.is_finite().fill_null(False) & y_raw.is_finite().fill_null(False)
            x_col = pl.when(valid).then(x_raw).otherwise(None)
            y_col = pl.when(valid).then(y_raw).otherwise(None)

            x_mean = x_col.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)
            y_mean = y_col.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)

            x_var = ((x_col - x_mean) ** 2).ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)
            y_var = ((y_col - y_mean) ** 2).ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)
            cov = ((x_col - x_mean) * (y_col - y_mean)).ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)

            x_std = x_var.sqrt()
            y_std = y_var.sqrt()

            corr = pl.when((x_std == 0) | (y_std == 0) | x_std.is_null() | y_std.is_null()).then(None).otherwise(cov / (x_std * y_std))
            exprs.append(corr.alias(c))

        return x.lazy().with_columns(exprs).collect()


def _validate_regression_params(window: int, min_periods: int | None) -> tuple[int, int]:
    from cleaned_operators.parameter_validation import strict_integer

    w = strict_integer(window, "window", minimum=2)
    mp = 3 if min_periods is None else strict_integer(min_periods, "min_periods", minimum=1)
    if mp > w:
        raise ValueError("min_periods must be <= window")
    return w, mp


@_register_rolling_operator(
    name="ts_regression_slope",
    category="time_series",
    business_category="time_series",
    canonical="ts_regression_slope",
    source=_SRC,
    backend="polars",
)
class TSRegressionSlopeNative(SeriesOperator):
    """Rolling regression slope (beta): cov(x,y) / var(x)."""

    _physical_spec = PhysicalImplementationSpec(
        canonical="ts_regression_slope", backend="polars",
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
        supports_lazy=False, materializes_full_panel=True, requires_sorted=True,
        supports_nulls=True, supports_nan=False, supports_inf=False,
        implementation_source_hash="cleaned_operators.common.polars_ts_rolling:TSRegressionSlopeNative:v1",
        emitter_identity="polars.rolling.pairwise_regression:v1",
        parameter_domain_hash="ts_regression_slope.y:dataframe,x:dataframe,window:int:min=2:default=20,lag:int:min=0:nullable,retval:enum:slope|beta|intercept|resid|residual|r2|r_squared:nullable,min_periods:int:min=1:nullable:default=3,add_intercept:bool:default=true",
        semantic_contract_hash="ts_regression_slope:pairwise_finite:intercept:v1",
    )

    metadata = OperatorMetadata(
        name="ts_regression_slope",
        category="time_series",
        description="滚动回归斜率",
        param_names=["y", "x", "window", "lag", "retval", "min_periods", "add_intercept"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame | None = None,
                         window: int = 20, *legacy_args, lag=None, retval=None, min_periods: int | None = None, add_intercept=True, **kwargs) -> pl.DataFrame:
        w, mp = _validate_regression_params(window, min_periods)
        cols = _require_pairwise_columns(y, x, operator="ts_regression_slope")

        if legacy_args:
            if len(legacy_args) > 4:
                raise TypeError(
                    "ts_regression_slope accepts at most lag, retval, "
                    "min_periods and add_intercept"
                )
            if len(legacy_args) == 1 and isinstance(legacy_args[0], bool):
                add_intercept = legacy_args[0]
            else:
                lag = legacy_args[0]
                if len(legacy_args) >= 2:
                    retval = legacy_args[1]
                if len(legacy_args) >= 3:
                    min_periods = legacy_args[2]
                if len(legacy_args) == 4:
                    add_intercept = legacy_args[3]
                w, mp = _validate_regression_params(window, min_periods)
        lag_i = 0 if lag is None else int(lag)
        if lag_i < 0:
            from backend.operator_errors import FutureReferenceError
            raise FutureReferenceError(
                f"ts_regression_slope: negative lag {lag_i} references future data"
            )
        if lag_i:
            x = x.shift(lag_i)

        output = str(retval or "slope").lower()
        if output == "beta":
            output = "slope"
        exprs = []
        for c in cols:
            y_col = pl.col(c)
            x_col = x[c]
            moments = _pairwise_rolling_moments(x_col, y_col, window=w, min_periods=mp)
            if add_intercept:
                slope = pl.when(moments["ss_x"] <= 0).then(None).otherwise(
                    moments["cross"] / moments["ss_x"]
                )
                intercept = moments["mean_y"] - slope * moments["mean_x"]
                resid = (moments["y_endpoint"] - moments["mean_y"]) - slope * (
                    moments["x_endpoint"] - moments["mean_x"]
                )
                r2 = pl.when((moments["ss_x"] <= 0) | (moments["ss_y"] <= 0)).then(None).otherwise(
                    (moments["cross"] ** 2) / (moments["ss_x"] * moments["ss_y"])
                )
            else:
                slope = pl.when(moments["raw_xx"] <= 0).then(None).otherwise(
                    moments["raw_xy"] / moments["raw_xx"]
                )
                intercept = pl.lit(0.0)
                resid = moments["y_endpoint"] - slope * moments["x_endpoint"]
                sse = moments["raw_yy"] - 2 * slope * moments["raw_xy"] + slope * slope * moments["raw_xx"]
                r2 = pl.when(moments["raw_yy"] <= 0).then(None).otherwise(1 - sse / moments["raw_yy"])
            selected = {"slope": slope, "intercept": intercept, "resid": resid, "residual": resid, "r2": r2, "r_squared": r2}.get(output)
            if selected is None:
                raise ValueError(f"unsupported regression retval: {retval!r}")
            exprs.append(selected.alias(c))

        return y.with_columns(exprs)


@_register_rolling_operator(
    name="ts_regression_intercept",
    category="time_series",
    business_category="time_series",
    canonical="ts_regression_intercept",
    source=_SRC,
    backend="polars",
)
class TSRegressionInterceptNative(SeriesOperator):
    """Rolling regression intercept: mean(y) - slope * mean(x)."""

    _physical_spec = PhysicalImplementationSpec(
        canonical="ts_regression_intercept", backend="polars",
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
        supports_lazy=False, materializes_full_panel=True, requires_sorted=True,
        supports_nulls=True, supports_nan=False, supports_inf=False,
        implementation_source_hash="cleaned_operators.common.polars_ts_rolling:TSRegressionInterceptNative:v1",
        emitter_identity="polars.rolling.pairwise_regression:v1",
        parameter_domain_hash="ts_regression_intercept.y:dataframe,x:dataframe,window:int:min=2:default=20,min_periods:int:min=1:nullable:default=3,add_intercept:bool:default=true",
        semantic_contract_hash="ts_regression_intercept:pairwise_finite:intercept:v1",
    )

    metadata = OperatorMetadata(
        name="ts_regression_intercept",
        category="time_series",
        description="滚动回归截距",
        param_names=["y", "x", "window", "min_periods", "add_intercept"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame | None = None,
                         window: int = 20, min_periods: int | None = None, add_intercept: bool = True, **kwargs) -> pl.DataFrame:
        w, mp = _validate_regression_params(window, min_periods)
        cols = _require_pairwise_columns(y, x, operator="ts_regression_intercept")


        exprs = []
        for c in cols:
            y_col = pl.col(c)
            x_col = x[c]

            moments = _pairwise_rolling_moments(x_col, y_col, window=w, min_periods=mp)
            intercept = pl.when(
                moments["raw_xx"].is_null() | (moments["raw_xx"] <= 0)
            ).then(None).otherwise(0.0)
            if add_intercept:
                centered_slope = pl.when(moments["ss_x"] <= 0).then(None).otherwise(
                    moments["cross"] / moments["ss_x"]
                )
                intercept = moments["mean_y"] - centered_slope * moments["mean_x"]
            exprs.append(intercept.alias(c))

        return y.with_columns(exprs)


@_register_rolling_operator(
    name="ts_regression_resid",
    category="time_series",
    business_category="time_series",
    canonical="ts_regression_resid",
    source=_SRC,
    backend="polars",
)
class TSRegressionResidNative(SeriesOperator):
    """Rolling regression residual: y - (intercept + slope * x)."""

    _physical_spec = PhysicalImplementationSpec(
        canonical="ts_regression_resid", backend="polars",
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
        supports_lazy=False, materializes_full_panel=True, requires_sorted=True,
        supports_nulls=True, supports_nan=False, supports_inf=False,
        implementation_source_hash="cleaned_operators.common.polars_ts_rolling:TSRegressionResidNative:v1",
        emitter_identity="polars.rolling.pairwise_regression:v1",
        parameter_domain_hash="ts_regression_resid.y:dataframe,x:dataframe,window:int:min=2:default=20,min_periods:int:min=1:nullable:default=3,add_intercept:bool:default=true",
        semantic_contract_hash="ts_regression_resid:pairwise_finite:intercept:v1",
    )

    metadata = OperatorMetadata(
        name="ts_regression_resid",
        category="time_series",
        description="滚动回归残差",
        param_names=["y", "x", "window", "min_periods", "add_intercept"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame | None = None,
                         window: int = 20, min_periods: int | None = None, add_intercept: bool = True, **kwargs) -> pl.DataFrame:
        w, mp = _validate_regression_params(window, min_periods)
        cols = _require_pairwise_columns(y, x, operator="ts_regression_resid")


        exprs = []
        for c in cols:
            y_col = pl.col(c)
            x_col = x[c]

            moments = _pairwise_rolling_moments(x_col, y_col, window=w, min_periods=mp)
            if add_intercept:
                slope = pl.when(moments["ss_x"] <= 0).then(None).otherwise(
                    moments["cross"] / moments["ss_x"]
                )
                resid = (moments["y_endpoint"] - moments["mean_y"]) - slope * (
                    moments["x_endpoint"] - moments["mean_x"]
                )
            else:
                slope = pl.when(moments["raw_xx"] <= 0).then(None).otherwise(
                    moments["raw_xy"] / moments["raw_xx"]
                )
                resid = moments["y_endpoint"] - slope * moments["x_endpoint"]

            exprs.append(resid.alias(c))

        return y.with_columns(exprs)


@_register_rolling_operator(
    name="ts_regression_r2",
    category="time_series",
    business_category="time_series",
    canonical="ts_regression_r2",
    source=_SRC,
    backend="polars",
)
class TSRegressionR2Native(SeriesOperator):
    """Rolling regression R-squared."""

    _physical_spec = PhysicalImplementationSpec(
        canonical="ts_regression_r2", backend="polars",
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
        supports_lazy=False, materializes_full_panel=True, requires_sorted=True,
        supports_nulls=True, supports_nan=False, supports_inf=False,
        implementation_source_hash="cleaned_operators.common.polars_ts_rolling:TSRegressionR2Native:v1",
        emitter_identity="polars.rolling.pairwise_regression:v1",
        parameter_domain_hash="ts_regression_r2.y:dataframe,x:dataframe,window:int:min=2:default=20,min_periods:int:min=1:nullable:default=3,add_intercept:bool:default=true",
        semantic_contract_hash="ts_regression_r2:pairwise_finite:intercept:v1",
    )

    metadata = OperatorMetadata(
        name="ts_regression_r2",
        category="time_series",
        description="滚动回归R平方",
        param_names=["y", "x", "window", "min_periods", "add_intercept"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame | None = None,
                         window: int = 20, min_periods: int | None = None, add_intercept: bool = True, **kwargs) -> pl.DataFrame:
        w, mp = _validate_regression_params(window, min_periods)
        cols = _require_pairwise_columns(y, x, operator="ts_regression_r2")


        exprs = []
        for c in cols:
            y_col = pl.col(c)
            x_col = x[c]

            moments = _pairwise_rolling_moments(x_col, y_col, window=w, min_periods=mp)
            if add_intercept:
                r2 = pl.when((moments["ss_x"] <= 0) | (moments["ss_y"] <= 0)).then(None).otherwise(
                    (moments["cross"] ** 2) / (moments["ss_x"] * moments["ss_y"])
                )
            else:
                slope = pl.when(moments["raw_xx"] <= 0).then(None).otherwise(
                    moments["raw_xy"] / moments["raw_xx"]
                )
                sse = moments["raw_yy"] - 2 * slope * moments["raw_xy"] + slope * slope * moments["raw_xx"]
                r2 = pl.when(moments["raw_yy"] <= 0).then(None).otherwise(1 - sse / moments["raw_yy"])
            exprs.append(r2.alias(c))

        return y.with_columns(exprs)


# ---------------------------------------------------------------------------
# Decay & Trend
# ---------------------------------------------------------------------------


@_register_rolling_operator(
    name="ts_decay_linear",
    category="time_series",
    business_category="time_series",
    canonical="ts_decay_linear",
    source=_SRC,
    backend="polars",
)
class TSDecayLinearNative(SeriesOperator):
    """Linear decay weighted average."""

    metadata = OperatorMetadata(
        name="ts_decay_linear",
        category="time_series",
        description="线性衰减加权",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(x)

        # Linear decay: weight[i] = w - i for i in 0..w-1
        weight_sum = (w * (w + 1)) / 2.0

        exprs = []
        for c in cols:
            weighted_sum = pl.lit(0.0)
            for i in range(w):
                weight = w - i
                weighted_sum = weighted_sum + pl.col(c).shift(i) * weight

            exprs.append((weighted_sum / weight_sum).alias(c))

        return x.lazy().with_columns(exprs).collect()


@_register_rolling_operator(
    name="ts_trend_slope",
    category="time_series",
    business_category="time_series",
    canonical="ts_trend_slope",
    source=_SRC,
    backend="polars",
)
class TSTrendSlopeNative(SeriesOperator):
    """Rolling linear trend slope (regression against time index)."""

    metadata = OperatorMetadata(
        name="ts_trend_slope",
        category="time_series",
        description="滚动趋势斜率",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)

        # Use physical row positions, not a fixed 0..w-1 grid. This keeps
        # missing observations from becoming adjacent after filtering and lets
        # prefix windows use their actual number of observations.
        row_name = "__ts_row"
        while row_name in x.columns:
            row_name = f"_{row_name}"
        frame = x.with_row_index(row_name)
        row = pl.col(row_name).cast(pl.Float64)
        exprs = []
        for c in cols:
            y = pl.col(c).cast(pl.Float64)
            valid = y.is_finite().fill_null(False)
            y = pl.when(valid).then(y).otherwise(None)
            valid = valid.cast(pl.Float64)
            n = valid.rolling_sum(window_size=w, min_samples=1)
            sum_t = (row * valid).rolling_sum(window_size=w, min_samples=1)
            sum_y = y.rolling_sum(window_size=w, min_samples=1)
            sum_tt = (row * row * valid).rolling_sum(window_size=w, min_samples=1)
            sum_ty = (row * y).rolling_sum(window_size=w, min_samples=1)

            # Center independently within each trailing window and fail closed
            # until two finite observations provide a non-zero time variance.
            centered_tt = sum_tt - (sum_t * sum_t) / n
            centered_ty = sum_ty - (sum_t * sum_y) / n
            exprs.append(
                pl.when((n >= 2) & (centered_tt > 0.0))
                .then(centered_ty / centered_tt)
                .otherwise(None)
                .alias(c)
            )

        return frame.lazy().with_columns(exprs).drop(row_name).collect()


@_register_rolling_operator(
    name="ts_monotonicity",
    category="time_series",
    business_category="time_series",
    canonical="ts_monotonicity",
    source=_SRC,
    backend="polars",
)
class TSMonotonicityNative(SeriesOperator):
    """Rolling monotonicity: fraction of consecutive increases."""

    metadata = OperatorMetadata(
        name="ts_monotonicity",
        category="time_series",
        description="滚动单调性",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)

        exprs = []
        for c in cols:
            delta = pl.col(c) - pl.col(c).shift(1)
            is_increase = (delta > 0).cast(pl.Float64)
            mono = is_increase.rolling_mean(window_size=w, min_samples=1)
            exprs.append(mono.alias(c))

        return x.lazy().with_columns(exprs).collect()


@_register_rolling_operator(
    name="ts_expanding_rank",
    category="time_series",
    business_category="time_series",
    canonical="ts_expanding_rank",
    source=_SRC,
    backend="polars",
)
class TSExpandingRankNative(SeriesOperator):
    """Expanding percentile rank from start."""

    metadata = OperatorMetadata(
        name="ts_expanding_rank",
        category="time_series",
        description="扩展窗口百分位排名",
        param_names=["x"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)

        exprs = []
        for c in cols:
            # expanding rank
            rank = pl.col(c).fill_nan(None).rank(method="average")
            count = pl.col(c).fill_nan(None).is_not_null().cast(pl.Float64).cum_sum()
            pct_rank = pl.when(count == 0).then(None).otherwise(rank / count)
            exprs.append(pct_rank.alias(c))

        return x.lazy().with_columns(exprs).collect()


# <CONTINUATION_MARKER_TS_ROLLING>
