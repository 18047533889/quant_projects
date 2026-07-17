# -*- coding: utf-8 -*-
"""Final native-Polars parity fixes for retained fused composites."""
from __future__ import annotations

from cleaned_operators.overhaul.base import EPS, PolarsFunctionOperator, pl, pl_base_with, pl_cols, positive_int
from cleaned_operators.registry import OperatorRegistry


def pl_adx_strict(high, low, close, window=14, **_):
    """Wilder ADX with the same missing previous-close seed as pandas.

    Polars ``max_horizontal`` ignores null inputs, while NumPy ``maximum`` in the
    pandas reference propagates the missing previous close.  Explicitly nulling
    True Range whenever previous close is unavailable keeps warm-up and interior
    gaps identical without leaving Polars execution.
    """
    w = positive_int(window, "window")
    replacements = {}
    for col in [c for c in pl_cols(high) if c in low.columns and c in close.columns]:
        temp = pl.DataFrame({"h": high[col], "l": low[col], "c": close[col]})
        previous_close = pl.col("c").shift(1)
        raw_plus = pl.col("h") - pl.col("h").shift(1)
        raw_minus = pl.col("l").shift(1) - pl.col("l")
        true_range = (
            pl.when(previous_close.is_null())
            .then(None)
            .otherwise(
                pl.max_horizontal(
                    pl.col("h") - pl.col("l"),
                    (pl.col("h") - previous_close).abs(),
                    (pl.col("l") - previous_close).abs(),
                )
            )
        )
        plus = pl.when((raw_plus > raw_minus) & (raw_plus > 0)).then(raw_plus).otherwise(0.0)
        minus = pl.when((raw_minus > raw_plus) & (raw_minus > 0)).then(raw_minus).otherwise(0.0)
        staged = temp.with_columns(
            true_range.ewm_mean(alpha=1.0 / w, adjust=False, min_samples=w).alias("atr"),
            plus.ewm_mean(alpha=1.0 / w, adjust=False, min_samples=w).alias("pdm"),
            minus.ewm_mean(alpha=1.0 / w, adjust=False, min_samples=w).alias("mdm"),
        ).with_columns(
            (100.0 * pl.col("pdm") / pl.when(pl.col("atr").abs() > EPS).then(pl.col("atr")).otherwise(None)).alias("pdi"),
            (100.0 * pl.col("mdm") / pl.when(pl.col("atr").abs() > EPS).then(pl.col("atr")).otherwise(None)).alias("mdi"),
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
            pl.col("dx").ewm_mean(alpha=1.0 / w, adjust=False, min_samples=w).alias("value")
        )["value"]
    return pl_base_with(high, replacements)


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
