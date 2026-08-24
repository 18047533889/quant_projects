# -*- coding: utf-8 -*-
"""Cross-backend edge semantics found by the full operator audit."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.operator_audit_fixes import (
    _ols_residual_strict,
    _pd_regression_r2,
)
from factor_engine.cleaned_operators.overhaul.base import (
    EPS,
    PandasFunctionOperator,
    PolarsFunctionOperator,
    aligned_pd,
    frame_pd,
    pl,
    pl_base_with,
    pl_cols,
    positive_int,
    window_params,
)
from factor_engine.cleaned_operators.overhaul.regression import _rolling_regression
from factor_engine.cleaned_operators.registry import OperatorRegistry


def _register(
    canonical: str,
    *,
    pandas_fn=None,
    polars_fn=None,
    pandas_source: str = "operator_overhaul_audited",
    polars_source: str = "operator_overhaul_native_polars",
) -> None:
    catalog = OperatorRegistry._catalog.get(canonical, {})
    params = list(catalog.get("param_names") or [])
    category = str(catalog.get("business_category") or catalog.get("scope") or "audited")
    description = str(catalog.get("description") or canonical)
    status = str(catalog.get("status") or "production")
    if pandas_fn is not None:
        OperatorRegistry.register(
            PandasFunctionOperator(
                canonical, category, params, description, pandas_fn
            ),
            canonical=canonical,
            backend="pandas_numpy",
            source=pandas_source,
            status=status,
            backend_explicit=True,
        )
    if pl is not None and polars_fn is not None:
        OperatorRegistry.register(
            PolarsFunctionOperator(
                canonical, category, params, description, polars_fn
            ),
            canonical=canonical,
            backend="polars",
            source=polars_source,
            status=status,
            backend_explicit=True,
        )


def _finite_frame(x: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        np.isfinite(x.to_numpy(dtype=float)),
        index=x.index,
        columns=x.columns,
    )


# ---------------------------------------------------------------------------
# Cross-sectional bucket finite-value parity
# ---------------------------------------------------------------------------

def pd_cs_bucket(x, buckets=10, ascending=True, **_):
    count = positive_int(buckets, "buckets")
    finite = _finite_frame(x)
    clean = x.where(finite)
    rank = clean.rank(
        axis=1,
        method="average",
        ascending=bool(ascending),
        na_option="keep",
    )
    valid_count = finite.sum(axis=1).astype(float)
    rank01 = rank.sub(1).div((valid_count - 1).replace(0, np.nan), axis=0)
    single = valid_count.eq(1)
    if single.any():
        rank01.loc[single] = (
            finite.loc[single].astype(float) * 0.5
        ).where(finite.loc[single])
    return (np.floor(rank01 * count) + 1).clip(1, count).where(finite)


# ---------------------------------------------------------------------------
# Exact order-statistic and regression compatibility parameters
# ---------------------------------------------------------------------------

def pd_nth_value(x, window, n=1, order="largest", min_periods=None, **_):
    w = positive_int(window, "window")
    nth = positive_int(n, "n")
    if nth > w:
        raise ValueError("n must satisfy 1 <= n <= window")
    mp = nth if min_periods is None else positive_int(min_periods, "min_periods")
    if mp > w:
        raise ValueError("min_periods must not exceed window")
    order = str(order).lower()
    if order not in {"largest", "smallest"}:
        raise ValueError("order must be 'largest' or 'smallest'")
    values = x.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    for col in range(values.shape[1]):
        for row in range(values.shape[0]):
            sample = values[max(0, row - w + 1) : row + 1, col]
            sample = np.sort(sample[np.isfinite(sample)])
            if sample.size >= max(mp, nth):
                out[row, col] = (
                    sample[-nth] if order == "largest" else sample[nth - 1]
                )
    return frame_pd(x, out)


def ts_regression_compat_strict(
    y: pd.DataFrame,
    x: pd.DataFrame,
    window: int,
    *legacy_args,
    min_periods: int | None = None,
    add_intercept: bool = True,
    lag: int | None = None,
    retval: str | None = None,
    **_,
) -> pd.DataFrame:
    if len(legacy_args) > 2:
        raise TypeError(
            "ts_regression accepts at most legacy lag and retval arguments"
        )
    if legacy_args:
        lag = legacy_args[0]
    if len(legacy_args) == 2:
        retval = str(legacy_args[1])

    w = positive_int(window, "window")
    lag_i = 0 if lag is None else int(lag)
    if lag is not None and float(lag) != float(lag_i):
        raise ValueError("lag must be an integer")
    if lag_i < 0:
        raise ValueError("lag must be non-negative; negative lag is future data")
    mp = None if min_periods is None else positive_int(
        min_periods, "min_periods"
    )
    if mp is not None and mp > w:
        raise ValueError("min_periods must not exceed window")
    if lag_i:
        x = x.shift(lag_i)

    requested = str(retval or "slope").lower()
    output = {
        "slope": "slope",
        "beta": "slope",
        "intercept": "intercept",
        "residual": "resid",
        "resid": "resid",
        "r_squared": "r2",
        "r2": "r2",
        "tstat": "tstat",
        "t_stat": "tstat",
    }.get(requested)
    if output is None:
        raise ValueError(f"unsupported regression retval: {retval!r}")
    if output == "r2":
        return _pd_regression_r2(y, x, w, mp, bool(add_intercept))
    return _rolling_regression(
        y, x, w, mp, bool(add_intercept), output
    )


# ---------------------------------------------------------------------------
# Safe-division parameter contract
# ---------------------------------------------------------------------------

def pd_div_or_null(x, y, epsilon=EPS, **_):
    x, y = aligned_pd(x, y)
    eps = float(epsilon)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("epsilon must be finite and positive")
    xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
    valid = np.isfinite(xv) & np.isfinite(yv) & (np.abs(yv) > eps)
    out = np.full(xv.shape, np.nan, dtype=float)
    np.divide(xv, yv, out=out, where=valid)
    out[~np.isfinite(out)] = np.nan
    return frame_pd(x, out)


if pl is not None:

    def pl_div_or_null(x, y, epsilon=EPS, **_):
        eps = float(epsilon)
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("epsilon must be finite and positive")
        cols = [col for col in pl_cols(x) if col in y.columns]
        return x.with_columns(
            [
                pl.when(
                    x[col].cast(pl.Float64, strict=False).is_finite()
                    & y[col].cast(pl.Float64, strict=False).is_finite()
                    & (y[col].cast(pl.Float64, strict=False).abs() > eps)
                )
                .then(
                    x[col].cast(pl.Float64, strict=False)
                    / y[col].cast(pl.Float64, strict=False)
                )
                .otherwise(None)
                .alias(col)
                for col in cols
            ]
        )


# ---------------------------------------------------------------------------
# Strict residual minimum-observation contract
# ---------------------------------------------------------------------------

def _exact_min_obs(value: Any | None, coefficients: int) -> int | None:
    if value is None:
        return None
    out = positive_int(value, "min_obs")
    if out <= coefficients:
        raise ValueError("min_obs must exceed fitted coefficient count")
    return out


def pd_cs_multi_resid(y, *features, add_intercept=True, min_obs=None, **_):
    if not features:
        raise ValueError("cs_multi_resid requires at least one exposure")
    aligned = aligned_pd(y, *features)
    target, exposures = aligned[0], aligned[1:]
    coefficients = len(exposures) + (1 if bool(add_intercept) else 0)
    exact_min = _exact_min_obs(min_obs, coefficients)
    out = np.full(target.shape, np.nan, dtype=float)
    for row in range(target.shape[0]):
        out[row] = _ols_residual_strict(
            target.iloc[row].to_numpy(dtype=float),
            [item.iloc[row].to_numpy(dtype=float) for item in exposures],
            weights=None,
            add_intercept=bool(add_intercept),
            min_obs=exact_min,
        )
    return frame_pd(target, out)


def pd_cs_wls_resid(y, x, weight, add_intercept=True, min_obs=5, **_):
    y, x, weight = aligned_pd(y, x, weight)
    coefficients = 1 + (1 if bool(add_intercept) else 0)
    exact_min = _exact_min_obs(min_obs, coefficients)
    out = np.full(y.shape, np.nan, dtype=float)
    for row in range(y.shape[0]):
        out[row] = _ols_residual_strict(
            y.iloc[row].to_numpy(dtype=float),
            [x.iloc[row].to_numpy(dtype=float)],
            weights=weight.iloc[row].to_numpy(dtype=float),
            add_intercept=bool(add_intercept),
            min_obs=exact_min,
        )
    return frame_pd(y, out)


# ---------------------------------------------------------------------------
# Weighted cross-sectional operators, including native Polars
# ---------------------------------------------------------------------------

def _weighted_stats(values, weights):
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return None
    total = float(weights[valid].sum())
    if not np.isfinite(total) or total <= EPS:
        return None
    mean = float(np.dot(values[valid], weights[valid]) / total)
    variance = float(
        np.dot((values[valid] - mean) ** 2, weights[valid]) / total
    )
    return mean, max(variance, 0.0), valid


def pd_weighted(x, weight, mode):
    x, weight = aligned_pd(x, weight)
    xv, wv = x.to_numpy(dtype=float), weight.to_numpy(dtype=float)
    out = np.full(xv.shape, np.nan, dtype=float)
    for row in range(xv.shape[0]):
        stats = _weighted_stats(xv[row], wv[row])
        if stats is None:
            continue
        mean, variance, valid = stats
        if mode == "mean":
            out[row, valid] = mean
        elif mode == "demean":
            out[row, valid] = xv[row, valid] - mean
        elif variance > EPS:
            out[row, valid] = (xv[row, valid] - mean) / np.sqrt(variance)
    return frame_pd(x, out)


def pd_group_weighted(x, group, weight, mode):
    x, group, weight = aligned_pd(x, group, weight)
    xv = x.to_numpy(dtype=float)
    gv = group.to_numpy(dtype=object)
    wv = weight.to_numpy(dtype=float)
    out = np.full(xv.shape, np.nan, dtype=float)
    for row in range(xv.shape[0]):
        labels = pd.unique(pd.Series(gv[row], dtype=object).dropna())
        for label in labels:
            members = pd.Series(gv[row], dtype=object).eq(label).to_numpy()
            stats = _weighted_stats(xv[row, members], wv[row, members])
            if stats is None:
                continue
            mean, variance, valid_local = stats
            indices = np.flatnonzero(members)[valid_local]
            if mode == "mean":
                out[row, indices] = mean
            elif variance > EPS:
                out[row, indices] = (
                    xv[row, indices] - mean
                ) / np.sqrt(variance)
    return frame_pd(x, out)


if pl is not None:

    def _long(frame, cols, value_name):
        return (
            frame.select(cols)
            .with_row_index("_row")
            .unpivot(
                index="_row",
                on=cols,
                variable_name="_inst",
                value_name=value_name,
            )
        )

    def _weighted_long(x, weight, group=None):
        cols = [col for col in pl_cols(x) if col in weight.columns]
        if group is not None:
            cols = [col for col in cols if col in group.columns]
        long = _long(x, cols, "_x").join(
            _long(weight, cols, "_w"),
            on=["_row", "_inst"],
            how="left",
        )
        if group is not None:
            long = long.join(
                _long(group, cols, "_group"),
                on=["_row", "_inst"],
                how="left",
            )
        return cols, long.with_columns(
            pl.col("_x").cast(pl.Float64, strict=False),
            pl.col("_w").cast(pl.Float64, strict=False),
        )

    def _weighted_wide(x, cols, long, mode, keys):
        valid = (
            pl.col("_x").is_finite()
            & pl.col("_w").is_finite()
            & (pl.col("_w") > 0)
        )
        if "_group" in keys:
            valid &= pl.col("_group").is_not_null()
        staged = long.with_columns(
            pl.when(valid).then(pl.col("_w")).otherwise(None).alias("_vw"),
            pl.when(valid)
            .then(pl.col("_x") * pl.col("_w"))
            .otherwise(None)
            .alias("_vwx"),
        ).with_columns(
            pl.col("_vw").sum().over(keys).alias("_sum_w"),
            pl.col("_vwx").sum().over(keys).alias("_sum_wx"),
        ).with_columns(
            (pl.col("_sum_wx") / pl.col("_sum_w")).alias("_mean")
        )
        if mode == "mean":
            out_expr = pl.when(valid & (pl.col("_sum_w") > EPS)).then(
                pl.col("_mean")
            )
        elif mode == "demean":
            out_expr = pl.when(valid & (pl.col("_sum_w") > EPS)).then(
                pl.col("_x") - pl.col("_mean")
            )
        else:
            staged = staged.with_columns(
                pl.when(valid)
                .then(pl.col("_w") * (pl.col("_x") - pl.col("_mean")).pow(2))
                .otherwise(None)
                .alias("_variance_term")
            ).with_columns(
                (
                    pl.col("_variance_term").sum().over(keys)
                    / pl.col("_sum_w")
                ).alias("_variance")
            )
            out_expr = pl.when(
                valid
                & (pl.col("_sum_w") > EPS)
                & (pl.col("_variance") > EPS)
            ).then(
                (pl.col("_x") - pl.col("_mean"))
                / pl.col("_variance").sqrt()
            )
        wide = (
            staged.with_columns(out_expr.otherwise(None).alias("_out"))
            .pivot(
                on="_inst",
                index="_row",
                values="_out",
                aggregate_function="first",
            )
            .sort("_row")
        )
        return pl_base_with(x, {col: wide[col] for col in cols})

    def pl_weighted(x, weight, mode):
        cols, long = _weighted_long(x, weight)
        return _weighted_wide(x, cols, long, mode, ["_row"])

    def pl_group_weighted(x, group, weight, mode):
        cols, long = _weighted_long(x, weight, group)
        return _weighted_wide(
            x, cols, long, mode, ["_row", "_group"]
        )


def apply_edge_semantic_fixes() -> None:
    _register("cs_bucket", pandas_fn=pd_cs_bucket)
    _register("ts_nth_value", pandas_fn=pd_nth_value)
    _register(
        "ts_regression_slope",
        pandas_fn=ts_regression_compat_strict,
        pandas_source="operator_overhaul_compat",
    )
    _register(
        "div_or_null",
        pandas_fn=pd_div_or_null,
        polars_fn=pl_div_or_null if pl is not None else None,
        pandas_source="semantic_hardening",
        polars_source="semantic_hardening",
    )
    _register("cs_multi_resid", pandas_fn=pd_cs_multi_resid)
    _register("cs_wls_resid", pandas_fn=pd_cs_wls_resid)

    weighted = {
        "cs_weighted_mean": (
            lambda x, w, **_: pd_weighted(x, w, "mean"),
            lambda x, w, **_: pl_weighted(x, w, "mean"),
        ),
        "cs_weighted_demean": (
            lambda x, w, **_: pd_weighted(x, w, "demean"),
            lambda x, w, **_: pl_weighted(x, w, "demean"),
        ),
        "cs_weighted_zscore": (
            lambda x, w, **_: pd_weighted(x, w, "zscore"),
            lambda x, w, **_: pl_weighted(x, w, "zscore"),
        ),
        "group_weighted_mean": (
            lambda x, g, w, **_: pd_group_weighted(x, g, w, "mean"),
            lambda x, g, w, **_: pl_group_weighted(x, g, w, "mean"),
        ),
        "group_weighted_zscore": (
            lambda x, g, w, **_: pd_group_weighted(x, g, w, "zscore"),
            lambda x, g, w, **_: pl_group_weighted(x, g, w, "zscore"),
        ),
    }
    for canonical, (pandas_fn, polars_fn) in weighted.items():
        _register(
            canonical,
            pandas_fn=pandas_fn,
            polars_fn=polars_fn if pl is not None else None,
            pandas_source="operator_overhaul_audited",
            polars_source="operator_overhaul_native_polars",
        )


apply_edge_semantic_fixes()
