# -*- coding: utf-8 -*-
"""Time-series rolling/correlation operators - Polars native implementations (Phase 2, Module 8).

Most operators use TRUE Polars expressions only - no pandas fallback, no NumPy.
EXCEPTION (R20-P0-EWM-PAIRWISE): ``TSEwmCorrNative``/``TSEwmCovNative`` delegate
per column to the pandas ``ewm`` reference kernel and are declared
``POLARS_PANDAS_DELEGATE`` (see the EWM block comment mid-file).
"""
from __future__ import annotations

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore

from factor_engine.cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.cleaned_operators.base import ParamRole, ParamSpec
from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
from factor_engine.backend.evidence_provenance import semantic_hashes_for

_PAIRWISE_CANONICALS = frozenset({
    "ts_corr",
    "ts_cov",
    "ts_regression_slope",
    "ts_regression_intercept",
    "ts_regression_resid",
    "ts_regression_r2",
    "ts_ewm_corr",
    "ts_ewm_cov",
})


def _register_rolling_operator(**kwargs):
    """Register only the repaired pairwise surface during normal bootstrap."""
    if kwargs.get("canonical") in _PAIRWISE_CANONICALS:
        return register_operator(**kwargs)
    return lambda cls: cls

_SKIP = frozenset({"date", "stock_code"})
_SRC = "factor_dsl_polars_native"

_REGRESSION_CONTRACT_SPECS = {
    "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    "min_periods": ParamSpec(dtype=int, min=1, default=None, searchable=False,
                             param_role=ParamRole.SUPPORT_POLICY),
    "add_intercept": ParamSpec(dtype=bool, default=True, searchable=False,
                               param_role=ParamRole.POLICY),
}


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


def _pairwise_staged_moments(
    x_series: pl.Series,
    y_series: pl.Series,
    *,
    window: int,
    min_periods: int,
) -> pl.DataFrame:
    """Materialize stable pairwise moments with an O(rows * min(window, rows)) workspace."""
    frame = pl.DataFrame({"__x": x_series, "__y": y_series}).with_columns(
        pl.when(
            pl.col("__x").is_finite().fill_null(False)
            & pl.col("__y").is_finite().fill_null(False)
        ).then(pl.col("__x")).otherwise(None).alias("__xv"),
        pl.when(
            pl.col("__x").is_finite().fill_null(False)
            & pl.col("__y").is_finite().fill_null(False)
        ).then(pl.col("__y")).otherwise(None).alias("__yv"),
    )
    active_window = min(window, max(1, frame.height))
    xw = [f"__xw{i}" for i in range(active_window)]
    yw = [f"__yw{i}" for i in range(active_window)]
    frame = frame.with_columns(
        [pl.col("__xv").shift(i).alias(xw[i]) for i in range(active_window)]
        + [pl.col("__yv").shift(i).alias(yw[i]) for i in range(active_window)]
    ).with_columns(
        pl.coalesce([pl.col(c) for c in xw]).alias("__xa"),
        pl.coalesce([pl.col(c) for c in yw]).alias("__ya"),
        pl.sum_horizontal(
            [pl.col(c).is_not_null().cast(pl.Float64) for c in xw]
        ).alias("__n"),
    )
    xc = [f"__xc{i}" for i in range(active_window)]
    yc = [f"__yc{i}" for i in range(active_window)]
    frame = frame.with_columns(
        [(pl.col(xw[i]) - pl.col("__xa")).alias(xc[i]) for i in range(active_window)]
        + [(pl.col(yw[i]) - pl.col("__ya")).alias(yc[i]) for i in range(active_window)]
    ).with_columns(
        pl.sum_horizontal([pl.col(c) for c in xc]).alias("__sx"),
        pl.sum_horizontal([pl.col(c) for c in yc]).alias("__sy"),
        pl.sum_horizontal([pl.col(c) * pl.col(c) for c in xc]).alias("__sxx"),
        pl.sum_horizontal([pl.col(c) * pl.col(c) for c in yc]).alias("__syy"),
        pl.sum_horizontal(
            [pl.col(xc[i]) * pl.col(yc[i]) for i in range(active_window)]
        ).alias("__sxy"),
    ).with_columns(
        (pl.col("__xa") + pl.col("__sx") / pl.col("__n")).alias("__mx"),
        (pl.col("__ya") + pl.col("__sy") / pl.col("__n")).alias("__my"),
        (
            pl.col("__n") * pl.col("__xa") * pl.col("__xa")
            + 2 * pl.col("__xa") * pl.col("__sx") + pl.col("__sxx")
        ).alias("__raw_xx"),
        (
            pl.col("__n") * pl.col("__ya") * pl.col("__ya")
            + 2 * pl.col("__ya") * pl.col("__sy") + pl.col("__syy")
        ).alias("__raw_yy"),
        (
            pl.col("__n") * pl.col("__xa") * pl.col("__ya")
            + pl.col("__xa") * pl.col("__sy")
            + pl.col("__ya") * pl.col("__sx") + pl.col("__sxy")
        ).alias("__raw_xy"),
    ).select([
        "__xv", "__yv", "__n", "__mx", "__my",
        "__raw_xx", "__raw_xy", "__raw_yy", *xw, *yw,
    ])
    xr = [f"__xr{i}" for i in range(active_window)]
    yr = [f"__yr{i}" for i in range(active_window)]
    frame = frame.with_columns(
        [(pl.col(xw[i]) - pl.col("__mx")).alias(xr[i]) for i in range(active_window)]
        + [(pl.col(yw[i]) - pl.col("__my")).alias(yr[i]) for i in range(active_window)]
    ).with_columns(
        pl.sum_horizontal([pl.col(c) * pl.col(c) for c in xr]).alias("__ssx"),
        pl.sum_horizontal([pl.col(c) * pl.col(c) for c in yr]).alias("__ssy"),
        pl.sum_horizontal(
            [pl.col(xr[i]) * pl.col(yr[i]) for i in range(active_window)]
        ).alias("__cross"),
    )
    ready = pl.col("__n") >= min_periods
    return frame.select(
        "__xv", "__yv", "__n",
        pl.when(ready).then(pl.col("__mx")).otherwise(None).alias("__mx"),
        pl.when(ready).then(pl.col("__my")).otherwise(None).alias("__my"),
        pl.when(ready).then(pl.col("__ssx")).otherwise(None).alias("__ssx"),
        pl.when(ready).then(pl.col("__ssy")).otherwise(None).alias("__ssy"),
        pl.when(ready).then(pl.col("__cross")).otherwise(None).alias("__cross"),
        pl.when(ready).then(pl.col("__raw_xx")).otherwise(None).alias("__raw_xx"),
        pl.when(ready).then(pl.col("__raw_xy")).otherwise(None).alias("__raw_xy"),
        pl.when(ready).then(pl.col("__raw_yy")).otherwise(None).alias("__raw_yy"),
    )


def _pairwise_staged_stat(
    x_series: pl.Series,
    y_series: pl.Series,
    *,
    window: int,
    min_periods: int,
    statistic: str,
    ddof: int = 1,
) -> pl.Series:
    frame = _pairwise_staged_moments(
        x_series, y_series, window=window, min_periods=min_periods
    )
    if statistic == "corr":
        value = pl.when(
            (pl.col("__ssx") > 0) & (pl.col("__ssy") > 0)
        ).then(
            # Divide by each norm separately: multiplying squared moments
            # first overflows/underflows even when correlation is representable.
            pl.col("__cross") / pl.col("__ssx").sqrt() / pl.col("__ssy").sqrt()
        ).otherwise(None)
    elif statistic == "cov":
        value = pl.when(pl.col("__n") > ddof).then(
            pl.col("__cross") / (pl.col("__n") - ddof)
        ).otherwise(None)
    else:
        raise ValueError(f"unsupported staged statistic: {statistic}")
    return frame.select(value.alias("__result")).get_column("__result")


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
        implementation_source_hash="cleaned_operators.common.polars_ts_rolling:TSCorrNative:v3",
        emitter_identity="polars.rolling.pairwise_moments.staged:v2",
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
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        min_p = 2 if min_periods is None else strict_integer(min_periods, "min_periods", minimum=1)
        if min_p > w:
            raise ValueError("min_periods must be <= window")
        cols = _require_pairwise_columns(x, y, operator="ts_corr")

        result = pl.DataFrame({
            c: _pairwise_staged_stat(
                x[c], y[c], window=w, min_periods=min_p, statistic="corr"
            )
            for c in cols
        })
        return x.with_columns([result.get_column(c).alias(c) for c in cols])


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
        implementation_source_hash="cleaned_operators.common.polars_ts_rolling:TSCovNative:v2",
        emitter_identity="polars.rolling.pairwise_moments.staged:v2",
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
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

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

        result = pl.DataFrame({
            c: _pairwise_staged_stat(
                x[c], y[c], window=w, min_periods=min_p,
                statistic="cov", ddof=ddof_val,
            )
            for c in cols
        })
        return x.with_columns([result.get_column(c).alias(c) for c in cols])


# ---------------------------------------------------------------------------
# EWM pairwise correlation / covariance (R20-P0-EWM-PAIRWISE)
#
# Reference semantics: pandas ``Series.ewm(span=window, adjust=False,
# ignore_na=False).corr/cov(other)`` — the SAME reference the polars long
# emitter pins (``backend/polars_expr_emitter._ewm_binary_map_groups``) and the
# stateful runtime reproduces (``stateful_runtime._ewm_moment_segment``).
#
# Contract (documented, verified against pandas 2.3.3):
#   * alpha = 2/(window+1) via ``span=window`` (span mapping).
#   * adjust=False, ignore_na=False, min_periods=2 (pandas ewmcov is NaN below
#     two valid pairs by construction: the bias correction denominator
#     sum_wt^2 - sum_wt2 is zero at T=1).
#   * cov bias: pandas ``.cov()`` default bias=False (unbiased);
#     corr internally uses the biased streams — both are pandas built-ins, so
#     delegation reproduces them exactly.
#   * pairwise-finite: a row is valid only when BOTH operands are finite;
#     Inf/-Inf is INVALID (pre-masked to NaN — pandas would otherwise
#     propagate Inf as a valid observation). Invalid rows do not update the
#     EWM state and the output carries the previous value (pandas behavior).
#   * zero-variance stream -> corr is NaN (pandas; the null guard survives).
#   * Polars has no ``ewm_cov``/``ewm_corr`` expression and the pandas
#     recursion (gap-decayed sum_wt/sum_wt2 state machine) cannot be expressed
#     faithfully as nested ``ewm_mean`` expressions — the pre-fix kernel tried
#     exactly that (EWM means of same-row EWM-centered products) and diverged
#     from pandas on cov from row 0 (e.g. 0.197 vs 0.674 at window=6) and at
#     every NaN hole.  This kernel therefore delegates per column to the
#     pandas reference and is declared POLARS_PANDAS_DELEGATE — it is NOT
#     Polars-native (see ``_physical_spec``).
# ---------------------------------------------------------------------------


def _ewm_pairwise_columns(
    x: pl.DataFrame, y: pl.DataFrame, *, window: int, corr: bool
) -> dict[str, "pl.Series"]:
    """Per-column pandas EWM corr/cov over pairwise-finite observations."""
    import numpy as np
    import pandas as pd

    def _finite_pd(series: "pl.Series") -> pd.Series:
        arr = np.asarray(series.to_numpy(), dtype=float)
        # Nulls arrive as NaN; Inf is contract-invalid and pre-masked so the
        # pandas kernel (which treats Inf as valid) never sees it.
        arr = np.where(np.isfinite(arr), arr, np.nan)
        return pd.Series(arr, dtype=float)

    results: dict[str, pl.Series] = {}
    for column in _numeric_cols(x):
        xs = _finite_pd(x[column])
        ys = _finite_pd(y[column])
        # Pairwise-finite: a row with either operand invalid is dropped from
        # BOTH streams (pandas ewmcov skips the pair; masking y into x keeps
        # the marginal streams consistent with the pair stream).
        pair = xs.notna() & ys.notna()
        xs = xs.where(pair, np.nan)
        ys = ys.where(pair, np.nan)
        ewm = xs.ewm(span=window, adjust=False, ignore_na=False, min_periods=2)
        out = ewm.corr(ys) if corr else ewm.cov(ys)
        arr = out.to_numpy()
        # pandas NaN (warmup / zero variance / invalid prefix) becomes a polars
        # null — the null contract the rest of this module observes.
        values = [None if not np.isfinite(value) else float(value) for value in arr]
        results[column] = pl.Series(name=column, values=values, dtype=pl.Float64)
    return results


@_register_rolling_operator(
    name="ts_ewm_corr",
    category="time_series",
    business_category="time_series",
    canonical="ts_ewm_corr",
    source=_SRC,
    backend="polars",
)
class TSEwmCorrNative(SeriesOperator):
    """Exponentially weighted correlation (pandas ewm(span, adjust=False)).

    Delegates per column to the pandas reference — see the EWM block comment
    above for the full contract.  NOT Polars-native.
    """

    _physical_spec = PhysicalImplementationSpec(
        canonical="ts_ewm_corr", backend="polars",
        execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,
        supports_lazy=False, materializes_full_panel=True, requires_sorted=True,
        stateful=True,
        supports_nulls=True, supports_nan=False, supports_inf=False,
        implementation_source_hash="cleaned_operators.common.polars_ts_rolling:TSEwmCorrNative:v2",
        emitter_identity="pandas.ewm.pairwise_corr:v1",
        parameter_domain_hash="ts_ewm_corr.x:dataframe,y:dataframe,window:int:min=2:default=20",
        semantic_contract_hash=semantic_hashes_for("ts_ewm_corr")["semantic_contract_hash"],
        notes="Per-column pandas ewm(span=window, adjust=False, ignore_na=False, "
              "min_periods=2) corr; Inf pre-masked to NaN (pairwise-finite).",
    )

    metadata = OperatorMetadata(
        name="ts_ewm_corr",
        category="time_series",
        description="指数加权相关系数",
        param_names=["x", "y", "window"],
        return_type="series",
        tags=["time_series", "polars", "pandas_delegate"],
        param_specs={"window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON)},
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame | None = None,
                         window: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        # ``span`` is the alias the pairwise EWM contract exposes on the pandas
        # side; the registry validates it against the window ParamSpec but does
        # not forward it, so honor it explicitly here.
        if kwargs.get("span") is not None:
            window = kwargs["span"]
        w = strict_integer(window, "window", minimum=2)

        _require_pairwise_columns(x, y, operator="ts_ewm_corr")

        columns = _ewm_pairwise_columns(x, y, window=w, corr=True)
        return x.with_columns([columns[c].alias(c) for c in _numeric_cols(x)])


@_register_rolling_operator(
    name="ts_ewm_cov",
    category="time_series",
    business_category="time_series",
    canonical="ts_ewm_cov",
    source=_SRC,
    backend="polars",
)
class TSEwmCovNative(SeriesOperator):
    """Exponentially weighted covariance (pandas ewm(span, adjust=False)).

    Delegates per column to the pandas reference — see the EWM block comment
    above for the full contract.  NOT Polars-native.
    """

    _physical_spec = PhysicalImplementationSpec(
        canonical="ts_ewm_cov", backend="polars",
        execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,
        supports_lazy=False, materializes_full_panel=True, requires_sorted=True,
        stateful=True,
        supports_nulls=True, supports_nan=False, supports_inf=False,
        implementation_source_hash="cleaned_operators.common.polars_ts_rolling:TSEwmCovNative:v2",
        emitter_identity="pandas.ewm.pairwise_cov:v1",
        parameter_domain_hash="ts_ewm_cov.x:dataframe,y:dataframe,window:int:min=2:default=20",
        semantic_contract_hash=semantic_hashes_for("ts_ewm_cov")["semantic_contract_hash"],
        notes="Per-column pandas ewm(span=window, adjust=False, ignore_na=False, "
              "min_periods=2) cov (bias=False, unbiased); Inf pre-masked to NaN "
              "(pairwise-finite).",
    )

    metadata = OperatorMetadata(
        name="ts_ewm_cov",
        category="time_series",
        description="指数加权协方差",
        param_names=["x", "y", "window"],
        return_type="series",
        tags=["time_series", "polars", "pandas_delegate"],
        param_specs={"window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON)},
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame | None = None,
                         window: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        if kwargs.get("span") is not None:
            window = kwargs["span"]
        w = strict_integer(window, "window", minimum=2)

        _require_pairwise_columns(x, y, operator="ts_ewm_cov")

        columns = _ewm_pairwise_columns(x, y, window=w, corr=False)
        return x.with_columns([columns[c].alias(c) for c in _numeric_cols(x)])


def _staged_regression_output(
    x_series: pl.Series,
    y_series: pl.Series,
    *,
    window: int,
    min_periods: int,
    output: str,
    add_intercept: bool,
) -> pl.Series:
    moments = _pairwise_staged_moments(
        x_series, y_series, window=window, min_periods=min_periods
    )
    if add_intercept:
        slope = pl.when(pl.col("__ssx") > 0).then(
            pl.col("__cross") / pl.col("__ssx")
        ).otherwise(None)
        intercept = pl.col("__my") - slope * pl.col("__mx")
        resid = (pl.col("__yv") - pl.col("__my")) - slope * (
            pl.col("__xv") - pl.col("__mx")
        )
        r2 = pl.when(
            (pl.col("__ssx") > 0) & (pl.col("__ssy") > 0)
        ).then(
            (pl.col("__cross") / pl.col("__ssx").sqrt() / pl.col("__ssy").sqrt()) ** 2
        ).otherwise(None)
    else:
        slope = pl.when(pl.col("__raw_xx") > 0).then(
            pl.col("__raw_xy") / pl.col("__raw_xx")
        ).otherwise(None)
        intercept = pl.when(pl.col("__raw_xx") > 0).then(0.0).otherwise(None)
        resid = pl.col("__yv") - slope * pl.col("__xv")
        sse = (
            pl.col("__raw_yy") - 2 * slope * pl.col("__raw_xy")
            + slope * slope * pl.col("__raw_xx")
        )
        r2 = pl.when(pl.col("__raw_yy") > 0).then(
            1 - sse / pl.col("__raw_yy")
        ).otherwise(None)
    selected = {
        "slope": slope, "intercept": intercept, "resid": resid,
        "residual": resid, "r2": r2, "r_squared": r2,
    }.get(output)
    if selected is None:
        raise ValueError(f"unsupported regression retval: {output!r}")
    return moments.select(selected.alias("__result")).get_column("__result")


def _validate_regression_params(window: int, min_periods: int | None) -> tuple[int, int]:
    from factor_engine.cleaned_operators.parameter_validation import strict_integer

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
        implementation_source_hash="cleaned_operators.common.polars_ts_rolling:TSRegressionSlopeNative:v3",
        emitter_identity="polars.rolling.pairwise_regression.staged:v2",
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
            from factor_engine.backend.operator_errors import FutureReferenceError
            raise FutureReferenceError(
                f"ts_regression_slope: negative lag {lag_i} references future data"
            )
        if lag_i:
            x = x.shift(lag_i)

        output = str(retval or "slope").lower()
        if output == "beta":
            output = "slope"
        result = pl.DataFrame({
            c: _staged_regression_output(
                x[c], y[c], window=w, min_periods=mp,
                output=output, add_intercept=bool(add_intercept),
            )
            for c in cols
        })
        return y.with_columns([result.get_column(c).alias(c) for c in cols])


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
        implementation_source_hash="cleaned_operators.common.polars_ts_rolling:TSRegressionInterceptNative:v2",
        emitter_identity="polars.rolling.pairwise_regression.staged:v2",
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
        param_specs=_REGRESSION_CONTRACT_SPECS,
        panel_params=("y", "x"),
        scalar_params=("window", "min_periods", "add_intercept"),
    )

    def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame | None = None,
                         window: int = 20, min_periods: int | None = None, add_intercept: bool = True, **kwargs) -> pl.DataFrame:
        w, mp = _validate_regression_params(window, min_periods)
        cols = _require_pairwise_columns(y, x, operator="ts_regression_intercept")


        result = pl.DataFrame({
            c: _staged_regression_output(
                x[c], y[c], window=w, min_periods=mp,
                output="intercept", add_intercept=bool(add_intercept),
            )
            for c in cols
        })
        return y.with_columns([result.get_column(c).alias(c) for c in cols])


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
        implementation_source_hash="cleaned_operators.common.polars_ts_rolling:TSRegressionResidNative:v2",
        emitter_identity="polars.rolling.pairwise_regression.staged:v2",
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
        param_specs=_REGRESSION_CONTRACT_SPECS,
        panel_params=("y", "x"),
        scalar_params=("window", "min_periods", "add_intercept"),
    )

    def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame | None = None,
                         window: int = 20, min_periods: int | None = None, add_intercept: bool = True, **kwargs) -> pl.DataFrame:
        w, mp = _validate_regression_params(window, min_periods)
        cols = _require_pairwise_columns(y, x, operator="ts_regression_resid")


        result = pl.DataFrame({
            c: _staged_regression_output(
                x[c], y[c], window=w, min_periods=mp,
                output="resid", add_intercept=bool(add_intercept),
            )
            for c in cols
        })
        return y.with_columns([result.get_column(c).alias(c) for c in cols])


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
        implementation_source_hash="cleaned_operators.common.polars_ts_rolling:TSRegressionR2Native:v3",
        emitter_identity="polars.rolling.pairwise_regression.staged:v2",
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
        param_specs=_REGRESSION_CONTRACT_SPECS,
        panel_params=("y", "x"),
        scalar_params=("window", "min_periods", "add_intercept"),
    )

    def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame | None = None,
                         window: int = 20, min_periods: int | None = None, add_intercept: bool = True, **kwargs) -> pl.DataFrame:
        w, mp = _validate_regression_params(window, min_periods)
        cols = _require_pairwise_columns(y, x, operator="ts_regression_r2")


        result = pl.DataFrame({
            c: _staged_regression_output(
                x[c], y[c], window=w, min_periods=mp,
                output="r2", add_intercept=bool(add_intercept),
            )
            for c in cols
        })
        return y.with_columns([result.get_column(c).alias(c) for c in cols])


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
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

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
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

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
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

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
