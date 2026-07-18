# -*- coding: utf-8 -*-
"""Batch/checkpoint convergence fixes for Wilder stateful indicators."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators.overhaul.base import (
    EPS,
    PandasFunctionOperator,
    PolarsFunctionOperator,
    aligned_pd,
    frame_pd,
    pl,
    pl_base_with,
    pl_cols,
    positive_int,
)
from cleaned_operators.registry import OperatorRegistry
from stateful_runtime import _adx_segment, _atr_segment, _rsi_segment


PANDAS_SOURCE = "composite_fastpath_primitives"
POLARS_SOURCE = "composite_fastpath_native_polars"


def _register(canonical, pandas_fn, polars_fn=None):
    catalog = OperatorRegistry._catalog.get(canonical, {})
    params = list(catalog.get("param_names") or [])
    description = str(catalog.get("description") or canonical)
    category = str(catalog.get("business_category") or "technical_signal")
    OperatorRegistry.register(
        PandasFunctionOperator(canonical, category, params, description, pandas_fn),
        canonical=canonical,
        backend="pandas_numpy",
        source=PANDAS_SOURCE,
        status=str(catalog.get("status") or "production"),
        backend_explicit=True,
    )
    if pl is not None and polars_fn is not None:
        OperatorRegistry.register(
            PolarsFunctionOperator(canonical, category, params, description, polars_fn),
            canonical=canonical,
            backend="polars",
            source=POLARS_SOURCE,
            status=str(catalog.get("status") or "production"),
            backend_explicit=True,
        )


def pd_rsi_wilder(x, window=14, **_):
    w = positive_int(window, "window")
    values = x.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    for col in range(values.shape[1]):
        out[:, col], _ = _rsi_segment(values[:, col], {}, w)
    return frame_pd(x, out)


def pd_atr_wilder(high, low, close, window=14, **_):
    high, low, close = aligned_pd(high, low, close)
    w = positive_int(window, "window")
    out = np.full(close.shape, np.nan, dtype=float)
    for col in range(close.shape[1]):
        out[:, col], _ = _atr_segment(
            high.iloc[:, col].to_numpy(dtype=float),
            low.iloc[:, col].to_numpy(dtype=float),
            close.iloc[:, col].to_numpy(dtype=float),
            {},
            w,
        )
    return frame_pd(close, out)


def pd_adx(high, low, close, window=14, **_):
    high, low, close = aligned_pd(high, low, close)
    w = positive_int(window, "window")
    out = np.full(close.shape, np.nan, dtype=float)
    for col in range(close.shape[1]):
        out[:, col], _ = _adx_segment(
            high.iloc[:, col].to_numpy(dtype=float),
            low.iloc[:, col].to_numpy(dtype=float),
            close.iloc[:, col].to_numpy(dtype=float),
            {},
            w,
        )
    return frame_pd(close, out)


if pl is not None:

    def _finite(expr):
        return expr.is_not_null() & expr.is_finite()

    def _clean(expr):
        return pl.when(_finite(expr)).then(expr.cast(pl.Float64)).otherwise(None)

    def _wilder_seeded(expr, window):
        """Native expression matching the checkpoint SMA-seeded recurrence."""
        w = positive_int(window, "window")
        clean = _clean(expr)
        valid = clean.is_not_null()
        count = valid.cast(pl.Int64).cum_sum()
        running_sum = pl.when(valid).then(clean).otherwise(0.0).cum_sum()
        seeded_input = (
            pl.when(count < w)
            .then(None)
            .when(valid & (count == w))
            .then(running_sum / float(w))
            .when(valid & (count > w))
            .then(clean)
            .otherwise(None)
        )
        smoothed = seeded_input.ewm_mean(
            alpha=1.0 / float(w),
            adjust=False,
            min_samples=1,
            ignore_nulls=True,
        )
        return pl.when(valid & (count >= w)).then(smoothed).otherwise(None)

    def _true_range_expr():
        previous_close = pl.col("_c").forward_fill().shift(1)
        return (
            pl.when(
                pl.col("_h").is_null()
                | pl.col("_l").is_null()
                | pl.col("_c").is_null()
            )
            .then(None)
            .when(previous_close.is_null())
            .then(pl.col("_h") - pl.col("_l"))
            .otherwise(
                pl.max_horizontal(
                    pl.col("_h") - pl.col("_l"),
                    (pl.col("_h") - previous_close).abs(),
                    (pl.col("_l") - previous_close).abs(),
                )
            )
        )

    def pl_rsi_wilder(x, window=14, **_):
        w = positive_int(window, "window")
        replacements = {}
        for col in pl_cols(x):
            temp = pl.DataFrame({"_raw": x[col]}).with_columns(
                _clean(pl.col("_raw")).alias("_close")
            ).with_columns(
                pl.col("_close").forward_fill().shift(1).alias("_previous")
            ).with_columns(
                pl.when(pl.col("_close").is_not_null())
                .then(pl.col("_close") - pl.col("_previous"))
                .otherwise(None)
                .alias("_delta")
            ).with_columns(
                pl.when(pl.col("_delta").is_null())
                .then(None)
                .when(pl.col("_delta") > 0)
                .then(pl.col("_delta"))
                .otherwise(0.0)
                .alias("_gain"),
                pl.when(pl.col("_delta").is_null())
                .then(None)
                .when(pl.col("_delta") < 0)
                .then(-pl.col("_delta"))
                .otherwise(0.0)
                .alias("_loss"),
            ).with_columns(
                _wilder_seeded(pl.col("_gain"), w).alias("_avg_gain"),
                _wilder_seeded(pl.col("_loss"), w).alias("_avg_loss"),
            ).with_columns(
                (
                    pl.col("_avg_gain")
                    / pl.when(pl.col("_avg_loss").abs() > EPS)
                    .then(pl.col("_avg_loss"))
                    .otherwise(None)
                ).alias("_ratio")
            ).with_columns(
                (
                    pl.when(
                        (pl.col("_avg_loss") == 0)
                        & (pl.col("_avg_gain") > 0)
                    )
                    .then(100.0)
                    .when(
                        (pl.col("_avg_gain") == 0)
                        & (pl.col("_avg_loss") > 0)
                    )
                    .then(0.0)
                    .when(
                        (pl.col("_avg_gain") == 0)
                        & (pl.col("_avg_loss") == 0)
                    )
                    .then(50.0)
                    .otherwise(100.0 - 100.0 / (1.0 + pl.col("_ratio")))
                ).alias("_rsi")
            )
            replacements[col] = temp.select(
                pl.when(pl.col("_close").is_not_null())
                .then(pl.col("_rsi"))
                .otherwise(None)
                .alias(col)
            )[col]
        return pl_base_with(x, replacements)

    def _ohlc_temp(high, low, close, col):
        return pl.DataFrame(
            {"_high_raw": high[col], "_low_raw": low[col], "_close_raw": close[col]}
        ).with_columns(
            _clean(pl.col("_high_raw")).alias("_h"),
            _clean(pl.col("_low_raw")).alias("_l"),
            _clean(pl.col("_close_raw")).alias("_c"),
        )

    def pl_atr_wilder(high, low, close, window=14, **_):
        w = positive_int(window, "window")
        replacements = {}
        for col in [
            name for name in pl_cols(close)
            if name in high.columns and name in low.columns
        ]:
            temp = _ohlc_temp(high, low, close, col).with_columns(
                _true_range_expr().alias("_tr")
            ).with_columns(
                _wilder_seeded(pl.col("_tr"), w).alias(col)
            )
            replacements[col] = temp[col]
        return pl_base_with(close, replacements)

    def pl_adx(high, low, close, window=14, **_):
        w = positive_int(window, "window")
        replacements = {}
        for col in [
            name for name in pl_cols(close)
            if name in high.columns and name in low.columns
        ]:
            temp = _ohlc_temp(high, low, close, col).with_columns(
                pl.col("_h").forward_fill().shift(1).alias("_previous_h"),
                pl.col("_l").forward_fill().shift(1).alias("_previous_l"),
                _true_range_expr().alias("_tr"),
            ).with_columns(
                (pl.col("_h") - pl.col("_previous_h")).alias("_up"),
                (pl.col("_previous_l") - pl.col("_l")).alias("_down"),
            ).with_columns(
                (
                    pl.when(
                        pl.col("_h").is_null()
                        | pl.col("_l").is_null()
                        | pl.col("_c").is_null()
                    )
                    .then(None)
                    .when(
                        pl.col("_previous_h").is_null()
                        | pl.col("_previous_l").is_null()
                    )
                    .then(0.0)
                    .when(
                        (pl.col("_up") > pl.col("_down"))
                        & (pl.col("_up") > 0)
                    )
                    .then(pl.col("_up"))
                    .otherwise(0.0)
                ).alias("_plus"),
                (
                    pl.when(
                        pl.col("_h").is_null()
                        | pl.col("_l").is_null()
                        | pl.col("_c").is_null()
                    )
                    .then(None)
                    .when(
                        pl.col("_previous_h").is_null()
                        | pl.col("_previous_l").is_null()
                    )
                    .then(0.0)
                    .when(
                        (pl.col("_down") > pl.col("_up"))
                        & (pl.col("_down") > 0)
                    )
                    .then(pl.col("_down"))
                    .otherwise(0.0)
                ).alias("_minus"),
            ).with_columns(
                _wilder_seeded(pl.col("_tr"), w).alias("_atr"),
                _wilder_seeded(pl.col("_plus"), w).alias("_plus_sm"),
                _wilder_seeded(pl.col("_minus"), w).alias("_minus_sm"),
            ).with_columns(
                pl.when(pl.col("_atr").abs() > EPS)
                .then(100.0 * pl.col("_plus_sm") / pl.col("_atr"))
                .otherwise(0.0)
                .alias("_plus_di"),
                pl.when(pl.col("_atr").abs() > EPS)
                .then(100.0 * pl.col("_minus_sm") / pl.col("_atr"))
                .otherwise(0.0)
                .alias("_minus_di"),
            ).with_columns(
                (pl.col("_plus_di") + pl.col("_minus_di")).alias("_denom")
            ).with_columns(
                pl.when(pl.col("_atr").is_null())
                .then(None)
                .when(pl.col("_denom") > EPS)
                .then(
                    100.0
                    * (pl.col("_plus_di") - pl.col("_minus_di")).abs()
                    / pl.col("_denom")
                )
                .otherwise(0.0)
                .alias("_dx")
            ).with_columns(
                _wilder_seeded(pl.col("_dx"), w).alias(col)
            )
            replacements[col] = temp[col]
        return pl_base_with(close, replacements)


_register(
    "RSI_WILDER",
    pd_rsi_wilder,
    pl_rsi_wilder if pl is not None else None,
)
_register(
    "ATR_WILDER",
    pd_atr_wilder,
    pl_atr_wilder if pl is not None else None,
)
_register(
    "ADX",
    pd_adx,
    pl_adx if pl is not None else None,
)
