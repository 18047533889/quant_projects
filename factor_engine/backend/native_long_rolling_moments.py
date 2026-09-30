"""Native Polars primitives for key-preserving long-panel rolling moments.

This module intentionally avoids the generic FE expression compiler: rolling
moments from one source panel remain one LazyFrame with no key joins.
"""
from __future__ import annotations


def collect_lagged_moments(
    positions,
    group_codes,
    values,
    valid,
    *,
    window: int,
    min_periods: int,
    ddof: int,
):
    """Return lagged mean/std keyed by the original unique row positions."""
    import polars as pl

    base = pl.DataFrame({
        "ts": positions,
        "inst": group_codes,
        "_v": values,
        "_valid": valid,
    }).lazy().with_columns(pl.col("_v").fill_nan(None))
    def by_asset(expr):
        return expr.over("inst", order_by="ts")
    lagged = by_asset(pl.col("_v").shift(1))
    lagged_valid = by_asset(pl.col("_valid").shift(1))
    plan = base.with_columns(
        lagged.alias("_lag"),
        lagged_valid.alias("_lag_valid"),
    ).with_columns(
        by_asset(pl.col("_lag").rolling_mean(
            window_size=window, min_samples=max(1, min_periods)
        )).alias("_mean_raw"),
        by_asset(pl.col("_lag").rolling_std(
            window_size=window, min_samples=1, ddof=0
        )).alias("_std_raw"),
        by_asset(pl.col("_lag").rolling_min(
            window_size=window, min_samples=1
        )).alias("_min"),
        by_asset(pl.col("_lag").rolling_max(
            window_size=window, min_samples=1
        )).alias("_max"),
        by_asset(pl.col("_lag_valid").rolling_sum(
            window_size=window, min_samples=1
        )).alias("_count"),
    )
    plan = plan.with_columns(
        pl.when(pl.col("_min") == pl.col("_max"))
        .then(0.0).otherwise(pl.col("_std_raw")).alias("_std_pop")
    )
    enough = pl.col("_count") >= max(1, min_periods)
    enough_dof = pl.col("_count") > ddof
    safe_denominator = pl.when(enough_dof).then(
        pl.col("_count") - ddof
    ).otherwise(1.0)
    adjusted_std = pl.col("_std_pop") * (
        pl.col("_count") / safe_denominator
    ).sqrt()
    plan = plan.with_columns(
        pl.when(enough).then(
            pl.when(pl.col("_min") == pl.col("_max"))
            .then(pl.col("_min")).otherwise(pl.col("_mean_raw"))
        ).otherwise(None).alias("_mean"),
        pl.when(enough & enough_dof).then(adjusted_std).otherwise(None).alias("_std"),
    ).select("ts", "inst", "_mean", "_std")
    return plan.collect()
