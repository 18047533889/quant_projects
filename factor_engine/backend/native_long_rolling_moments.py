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
    include_mean: bool = True,
    include_std: bool = True,
    current_values=None,
):
    """Return lagged mean/std keyed by the original unique row positions."""
    import polars as pl

    data = {
        "ts": positions,
        "inst": group_codes,
        "_v": values,
        "_valid": valid,
    }
    if current_values is not None:
        data["_current"] = current_values
    base = pl.DataFrame(data).lazy().with_columns(pl.col("_v").fill_nan(None))
    def by_asset(expr):
        return expr.over("inst", order_by="ts")
    lagged = by_asset(pl.col("_v").shift(1))
    lagged_valid = by_asset(pl.col("_valid").shift(1))
    plan = base.with_columns(
        lagged.alias("_lag"),
        lagged_valid.alias("_lag_valid"),
    )
    moment_exprs = [
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
    ]
    needs_mean = include_mean or current_values is not None
    if needs_mean:
        moment_exprs.insert(0, by_asset(pl.col("_lag").rolling_mean(
            window_size=window, min_samples=max(1, min_periods)
        )).alias("_mean_raw"))
    plan = plan.with_columns(*moment_exprs)
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
    if needs_mean:
        plan = plan.with_columns(
            pl.when(enough).then(
                pl.when(pl.col("_min") == pl.col("_max"))
                .then(pl.col("_min")).otherwise(pl.col("_mean_raw"))
            ).otherwise(None).alias("_mean"),
        )
    plan = plan.with_columns(
        pl.when(enough & enough_dof).then(adjusted_std).otherwise(None).alias("_std"),
    )
    if current_values is not None:
        plan = plan.with_columns(
            ((pl.col("_current") - pl.col("_mean")) / pl.col("_std")).alias("_zscore")
        )
    columns = ["ts", "inst"]
    if include_mean:
        columns.append("_mean")
    if include_std:
        columns.append("_std")
    if current_values is not None:
        columns.append("_zscore")
    plan = plan.select(*columns)
    return plan.collect()


def _stable_group_rows(group_codes):
    """Yield stable int64 row-index views and codes without Python row lists."""
    import numpy as np

    codes = np.asarray(group_codes)
    if (codes.ndim != 1 or codes.dtype.kind not in "iu"
            or (codes.size and (codes.min() < 0 or codes.max() >= codes.size))):
        raise ValueError("group_codes must be a one-dimensional nonnegative factor-code array")
    if codes.size == 0:
        return

    # pandas.factorize supplies dense nonnegative codes. One stable int64 row
    # index keeps source order within each asset, which is the chronology used
    # by the prior-row z-score contract. Group offsets add O(number of assets)
    # storage; no Python int/list is retained for every panel row.
    order = np.argsort(codes, kind="stable").astype(np.int64, copy=False)
    counts = np.bincount(codes.astype(np.int64, copy=False))
    ends = counts.cumsum(dtype=np.int64)
    start = 0
    for code, end in enumerate(ends):
        stop = int(end)
        if stop > start:
            yield code, order[start:stop]
        start = stop


def collect_lagged_zscores_stable(
    positions, group_codes, values, valid, current_values, *,
    window: int, min_periods: int, ddof: int,
    max_working_pairs: int = 250_000,
):
    """Compute stable z-scores with bounded per-asset native interval joins.

    A tile contains at most ``max_working_pairs`` target/history matches,
    except that one target necessarily needs its whole ``window`` of history.
    Polars computes all numeric reductions; Python only partitions row indices.
    """
    import numpy as np
    import polars as pl

    positions = np.asarray(positions)
    group_codes = np.asarray(group_codes)
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    current_values = np.asarray(current_values, dtype=np.float64)
    if positions.size == 0:
        return pl.DataFrame({
            "ts": np.asarray([], dtype=np.int64),
            "inst": np.asarray([], dtype=group_codes.dtype),
            "_zscore": np.asarray([], dtype=np.float64),
        })

    out_positions = []
    out_groups = []
    out_zscores = []
    for code, row_ids in _stable_group_rows(group_codes):
        tile_rows = max(1, int(max_working_pairs) // max(1, int(window)))
        for start in range(0, len(row_ids), tile_rows):
            stop = min(len(row_ids), start + tile_rows)
            history_start = max(0, start - int(window))
            history_ids = row_ids[history_start:stop]
            target_ids = row_ids[start:stop]

            targets = pl.DataFrame({
                "_tid": np.arange(start, stop, dtype=np.int64),
                "_pos": positions[target_ids],
                "_current": current_values[target_ids],
            }).lazy()
            finite_mask = valid[history_ids]
            finite_history_ids = history_ids[finite_mask]
            history = pl.DataFrame({
                "_hidx": np.arange(history_start, stop, dtype=np.int64)[finite_mask],
                "_hv": values[finite_history_ids],
            }).lazy()
            pairs = targets.join_where(
                history,
                pl.col("_hidx") < pl.col("_tid"),
                pl.col("_hidx") >= pl.col("_tid") - int(window),
            )
            stats = pairs.group_by("_tid").agg(
                pl.col("_hv").abs().max().alias("_scale"),
                pl.col("_hv").sort_by("_hidx", descending=True)
                .first().alias("_anchor"),
                pl.len().alias("_count"),
            )
            safe_scale = pl.when(pl.col("_scale") > 0.0).then(
                pl.col("_scale")
            ).otherwise(1.0)
            same_sign = (pl.col("_hv") < 0.0) == (pl.col("_anchor") < 0.0)
            delta = pl.when(same_sign).then(
                (pl.col("_hv") - pl.col("_anchor")) / safe_scale
            ).otherwise(
                pl.col("_hv") / safe_scale
                - pl.col("_anchor") / safe_scale
            ).alias("_delta")
            centered_pairs = pairs.join(stats, on="_tid", how="inner").with_columns(delta)
            delta_means = centered_pairs.group_by("_tid").agg(
                pl.col("_delta").mean().alias("_delta_mean")
            )
            centered = centered_pairs.join(delta_means, on="_tid", how="inner")
            variance_sums = centered.group_by("_tid").agg(
                (pl.col("_delta") - pl.col("_delta_mean"))
                .pow(2).sum().alias("_variance_sum")
            )
            moments = stats.join(delta_means, on="_tid", how="left").join(
                variance_sums, on="_tid", how="left"
            )
            result = targets.join(moments, on="_tid", how="left").with_columns(
                pl.when(pl.col("_scale").is_not_null() & (pl.col("_scale") > 0.0))
                .then(pl.col("_scale")).otherwise(1.0).alias("_safe_scale")
            )
            safe_ddof = pl.when(pl.col("_count") > int(ddof)).then(
                pl.col("_count") - int(ddof)
            ).otherwise(1.0)
            same_current_sign = (
                (pl.col("_current") < 0.0) == (pl.col("_anchor") < 0.0)
            )
            current_delta = pl.when(same_current_sign).then(
                (pl.col("_current") - pl.col("_anchor")) / pl.col("_safe_scale")
            ).otherwise(
                pl.col("_current") / pl.col("_safe_scale")
                - pl.col("_anchor") / pl.col("_safe_scale")
            )
            std = (pl.col("_variance_sum") / safe_ddof).sqrt()
            enough = (
                (pl.col("_count") >= max(1, int(min_periods)))
                & (pl.col("_count") > int(ddof))
            )
            result = result.with_columns(
                pl.when(enough).then(
                    (current_delta - pl.col("_delta_mean")) / std
                ).otherwise(None).alias("_zscore")
            ).select("_pos", "_zscore").collect()
            out_positions.append(result.get_column("_pos").to_numpy())
            out_groups.append(np.full(stop - start, code, dtype=group_codes.dtype))
            out_zscores.append(result.get_column("_zscore").to_numpy())

    return pl.DataFrame({
        "ts": np.concatenate(out_positions),
        "inst": np.concatenate(out_groups),
        "_zscore": np.concatenate(out_zscores),
    }).sort("ts")
