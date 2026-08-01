# -*- coding: utf-8 -*-
"""Final native-Polars parity fixes and strict COS/fiscal layers."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Imported here because this module is deliberately the last runtime layer in
# cleaned_operators.load_all. Their registrations therefore override historical
# implementations without widening the bootstrap surface.
from cleaned_operators import fiscal_strict as _fiscal_strict  # noqa: F401
from cleaned_operators import layer_cos_fundamental as _cos_fundamental  # noqa: F401
from cleaned_operators import layer_weighted as _weighted  # noqa: F401
from cleaned_operators import layer_topk_compat as _topk_compat  # noqa: F401
from cleaned_operators.overhaul.base import (
    EPS,
    PandasFunctionOperator,
    PolarsFunctionOperator,
    frame_pd,
    pl,
    pl_base_with,
    pl_cols,
    pl_finite,
    positive_int,
)
from cleaned_operators.registry import OperatorRegistry


def pl_adx_strict(high, low, close, window=14, **_):
    """Wilder ADX with the same missing previous-close seed as pandas."""
    w = positive_int(window, "window")
    replacements = {}
    for col in [c for c in pl_cols(high) if c in low.columns and c in close.columns]:
        temp = pl.DataFrame({"h": high[col], "l": low[col], "c": close[col]})
        previous_close = pl.col("c").shift(1)
        raw_plus = pl.col("h") - pl.col("h").shift(1)
        raw_minus = pl.col("l").shift(1) - pl.col("l")
        true_range = pl.when(previous_close.is_null()).then(None).otherwise(
            pl.max_horizontal(
                pl.col("h") - pl.col("l"),
                (pl.col("h") - previous_close).abs(),
                (pl.col("l") - previous_close).abs(),
            )
        )
        plus = pl.when((raw_plus > raw_minus) & (raw_plus > 0)).then(raw_plus).otherwise(0.0)
        minus = pl.when((raw_minus > raw_plus) & (raw_minus > 0)).then(raw_minus).otherwise(0.0)
        staged = temp.with_columns(
            true_range.ewm_mean(alpha=1.0 / w, adjust=False, min_samples=w).alias("atr"),
            plus.ewm_mean(alpha=1.0 / w, adjust=False, min_samples=w).alias("pdm"),
            minus.ewm_mean(alpha=1.0 / w, adjust=False, min_samples=w).alias("mdm"),
        ).with_columns(
            (
                100.0
                * pl.col("pdm")
                / pl.when(pl.col("atr").abs() > EPS).then(pl.col("atr")).otherwise(None)
            ).alias("pdi"),
            (
                100.0
                * pl.col("mdm")
                / pl.when(pl.col("atr").abs() > EPS).then(pl.col("atr")).otherwise(None)
            ).alias("mdi"),
        ).with_columns(
            (
                100.0
                * (pl.col("pdi") - pl.col("mdi")).abs()
                / pl.when((pl.col("pdi") + pl.col("mdi")).abs() > EPS)
                .then(pl.col("pdi") + pl.col("mdi"))
                .otherwise(None)
            ).alias("dx")
        )
        replacements[col] = staged.select(
            pl.col("dx")
            .ewm_mean(alpha=1.0 / w, adjust=False, min_samples=w)
            .alias("value")
        )["value"]
    return pl_base_with(high, replacements)


def pd_days_since_inclusive(condition: pd.DataFrame, max_lookback=None, **_):
    """Distance to latest true observation; max_lookback is an inclusive distance."""
    limit = None if max_lookback is None else positive_int(max_lookback, "max_lookback")
    values = condition.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    for col in range(values.shape[1]):
        last = -1
        for row, value in enumerate(values[:, col]):
            if np.isfinite(value) and value != 0:
                last = row
            if last >= 0:
                distance = row - last
                if limit is None or distance <= limit:
                    out[row, col] = float(distance)
    return frame_pd(condition, out)


def pl_days_since_inclusive(condition, max_lookback=None, **_):
    limit = None if max_lookback is None else positive_int(max_lookback, "max_lookback")
    replacements = {}
    for col in pl_cols(condition):
        temp = pl.DataFrame({"c": condition[col]}).with_row_index("i")
        true = pl_finite("c") & (pl.col("c") != 0)
        last = pl.when(true).then(pl.col("i")).otherwise(None).forward_fill()
        distance = pl.col("i").cast(pl.Float64) - last.cast(pl.Float64)
        if limit is not None:
            distance = pl.when(distance <= limit).then(distance).otherwise(None)
        replacements[col] = temp.select(distance.alias("v"))["v"]
    return pl_base_with(condition, replacements)


OperatorRegistry.register(
    PandasFunctionOperator(
        "ts_days_since",
        "time_series_condition",
        ["condition", "max_lookback"],
        "distance to latest true observation; max_lookback is inclusive",
        pd_days_since_inclusive,
    ),
    canonical="ts_days_since",
    backend="pandas_numpy",
    source="layer_composite_fixes",
    status="production",
    backend_explicit=True,
)

if pl is not None:
    OperatorRegistry.register(
        PolarsFunctionOperator(
            "ADX",
            "technical_signal",
            ["high", "low", "close", "window"],
            "Wilder ADX with strict pandas-compatible missing seed",
            pl_adx_strict,
        ),
        canonical="ADX",
        backend="polars",
        source="layer_governance_native_polars",
        status="production",
        backend_explicit=True,
    )
    OperatorRegistry.register(
        PolarsFunctionOperator(
            "ts_days_since",
            "time_series_condition",
            ["condition", "max_lookback"],
            "distance to latest true observation; max_lookback is inclusive",
            pl_days_since_inclusive,
        ),
        canonical="ts_days_since",
        backend="polars",
        source="layer_governance_native_polars",
        status="production",
        backend_explicit=True,
    )
