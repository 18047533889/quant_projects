"""Native Polars primitives for key-preserving long-panel EWMA."""
from __future__ import annotations

import math
import numbers

import numpy as np


def collect_lagged_ewma(positions, group_codes, values, *,
                        alpha: float, min_periods: int):
    """Return prior-row per-asset EWM keyed by original positional identity."""
    import polars as pl

    if (isinstance(alpha, (bool, np.bool_))
            or not isinstance(alpha, numbers.Real)
            or not math.isfinite(float(alpha))
            or not 0.0 < float(alpha) <= 1.0):
        raise ValueError("alpha must be finite and in (0, 1]")
    if (isinstance(min_periods, (bool, np.bool_))
            or not isinstance(min_periods, numbers.Integral)
            or min_periods < 0):
        raise ValueError("min_periods must be a non-negative integer")

    positions = np.asarray(positions, dtype=np.int64)
    group_codes = np.asarray(group_codes, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    if (positions.ndim != 1 or group_codes.ndim != 1 or values.ndim != 1
            or positions.size != group_codes.size or values.size != positions.size):
        raise ValueError("positions, group_codes and values must be equal-length vectors")

    base = pl.DataFrame({
        "ts": positions,
        "inst": group_codes,
        "_v": values,
    }).lazy().with_columns(pl.col("_v").fill_nan(None))
    by_asset = lambda expr: expr.over("inst", order_by="ts")
    lagged = by_asset(pl.col("_v").shift(1))
    plan = base.with_columns(lagged.alias("_lag"))
    ewm = pl.col("_lag").ewm_mean(
        alpha=float(alpha), adjust=False, min_samples=max(1, int(min_periods)),
        ignore_nulls=False,
    )
    plan = plan.with_columns(
        ewm.forward_fill().over("inst", order_by="ts").alias("_ema")
    )
    return plan.select("ts", "inst", "_ema").collect()
