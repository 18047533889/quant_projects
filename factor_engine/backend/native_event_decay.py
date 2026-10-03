"""Polars expression collector for FP-compatible lagged event decay.

The pandas-facing FE recipe may prepare asset identifiers and row positions;
the numeric recurrence, gap segmentation, warmup, and extreme-value handling
are evaluated as Polars expressions here.
"""
from __future__ import annotations

import math
import numbers

import numpy as np


def collect_event_decay(positions, asset_ids, values, *, alpha: float,
                        min_periods: int):
    """Return native Polars event-decay outputs keyed by original positions."""
    import polars as pl

    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, numbers.Real):
        raise ValueError("alpha must be finite and in (0, 1]")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be finite and in (0, 1]")
    if (isinstance(min_periods, (bool, np.bool_))
            or not isinstance(min_periods, numbers.Integral)
            or min_periods < 1):
        raise ValueError("min_periods must be a positive integer")

    positions = np.asarray(positions, dtype=np.int64)
    asset_ids = np.asarray(asset_ids, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    if not (positions.ndim == asset_ids.ndim == values.ndim == 1
            and len(positions) == len(asset_ids) == len(values)):
        raise ValueError("positions, asset_ids, and values must be aligned vectors")
    if len(positions) == 0:
        return pl.DataFrame({
            "pos": pl.Series([], dtype=pl.Int64),
            "event_decay": pl.Series([], dtype=pl.Float64),
        })

    frame = pl.DataFrame({
        "pos": positions,
        "asset": asset_ids,
        "value": values,
    }).sort(["asset", "pos"])
    x = pl.col("value")
    finite = x.is_finite().fill_null(False)
    frame = frame.with_columns(
        pl.when(finite).then(x).otherwise(None).alias("_clean")
    ).with_columns(
        pl.col("_clean").shift(1).over("asset", order_by="pos").alias("_lag")
    ).with_columns(
        pl.col("_lag").is_null().cast(pl.UInt32).cum_sum()
        .over("asset", order_by="pos").alias("_run")
    )
    lag = pl.col("_lag")
    window = ["asset", "_run"]
    hazard = (
        lag.abs().cum_max().over(window, order_by="pos")
        > (np.finfo(np.float64).max / 2.0)
    ).fill_null(False)
    has_hazard = frame.select(hazard.any().alias("_has_hazard")).item()
    simple = lag.ewm_mean(
        alpha=alpha, adjust=False, min_samples=int(min_periods),
        ignore_nulls=False,
    ).over(window, order_by="pos")
    if has_hazard:
        positive = pl.when(lag.is_null()).then(None).when(lag > 0).then(lag).otherwise(0.0)
        negative = pl.when(lag.is_null()).then(None).when(lag < 0).then(lag).otherwise(0.0)
        stable = (
            positive.ewm_mean(
                alpha=alpha, adjust=False, min_samples=int(min_periods),
                ignore_nulls=False,
            ).over(window, order_by="pos")
            + negative.ewm_mean(
                alpha=alpha, adjust=False, min_samples=int(min_periods),
                ignore_nulls=False,
            ).over(window, order_by="pos")
        )
        out = pl.when(hazard).then(stable).otherwise(simple)
    else:
        out = simple
    frame = frame.with_columns(out.alias("_out"))
    return frame.select("pos", pl.col("_out").alias("event_decay")).sort("pos")
