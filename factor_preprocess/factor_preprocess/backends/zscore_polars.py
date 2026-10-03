"""Native Polars kernel for the standalone v2 cross-sectional z-score."""
from __future__ import annotations

from typing import Any


def finite_anchor_centered_zscore_polars(
    frame: Any,
    *,
    value_col: str,
    group_col: str,
    ddof: float,
    constant_value: float,
):
    """Compute grouped stable z-scores using native Polars expressions only.

    Values cast to Float64 before moments, matching the standalone NumPy policy
    for integer inputs. NaN/null outputs stay missing; groups containing Inf
    retain FP's legacy Inf-participates/constant-fill result. This helper is not
    the FactorEngine native operator and makes no FE equivalence claim.
    """
    import polars as pl

    reserved = set(frame.columns)

    def fresh(stem: str) -> str:
        name = stem
        suffix = 0
        while name in reserved:
            suffix += 1
            name = f"{stem}_{suffix}"
        reserved.add(name)
        return name

    g, x = fresh("__zv2_group"), fresh("__zv2_value")
    finite_name, lo, hi = fresh("__zv2_finite"), fresh("__zv2_lo"), fresh("__zv2_hi")
    anchor, delta, scale = fresh("__zv2_anchor"), fresh("__zv2_delta"), fresh("__zv2_scale")
    unit, mean, centered, std = (
        fresh("__zv2_unit"), fresh("__zv2_mean"),
        fresh("__zv2_centered"), fresh("__zv2_std"),
    )
    has_inf, missing = fresh("__zv2_has_inf"), fresh("__zv2_missing")

    work = frame.select(
        pl.col(group_col).alias(g),
        pl.col(value_col).cast(pl.Float64, strict=False).alias(x),
    )
    group = pl.col(g)
    raw = pl.col(x)
    finite = raw.is_finite().fill_null(False)
    work = work.with_columns([
        finite.alias(finite_name),
        raw.is_infinite().fill_null(False).any().over(group).alias(has_inf),
        (raw.is_null() | raw.is_nan().fill_null(False)).alias(missing),
        pl.when(finite).then(raw).otherwise(None).min().over(group).alias(lo),
        pl.when(finite).then(raw).otherwise(None).max().over(group).alias(hi),
    ])
    work = work.with_columns(
        (pl.col(lo) / 2.0 + pl.col(hi) / 2.0).fill_null(0.0).alias(anchor)
    )
    work = work.with_columns(
        pl.when(pl.col(finite_name)).then(raw - pl.col(anchor))
        .otherwise(0.0).alias(delta)
    )
    work = work.with_columns(
        pl.col(delta).abs().max().over(group).fill_null(0.0).alias(scale)
    )
    work = work.with_columns(
        pl.when(pl.col(scale) > 0.0).then(pl.col(delta) / pl.col(scale))
        .otherwise(0.0).alias(unit)
    )
    work = work.with_columns(
        pl.when(pl.col(finite_name)).then(pl.col(unit)).otherwise(None)
        .mean().over(group).fill_null(0.0).alias(mean)
    )
    work = work.with_columns(
        pl.when(pl.col(finite_name)).then(pl.col(unit) - pl.col(mean))
        .otherwise(0.0).alias(centered)
    )
    work = work.with_columns(
        pl.when(pl.col(finite_name)).then(pl.col(centered)).otherwise(None)
        .std(ddof=ddof).over(group).alias(std)
    )
    positive_std = (pl.col(std) > 0.0) & pl.col(std).is_finite().fill_null(False)
    normalized = (
        pl.when(pl.col(has_inf)).then(pl.lit(constant_value))
        .when(positive_std).then(pl.col(centered) / pl.col(std))
        .otherwise(pl.lit(constant_value))
    )
    return work.with_columns(
        pl.when(pl.col(missing)).then(pl.lit(float("nan")))
        .otherwise(normalized).alias("_zscore")
    ).select("_zscore").to_series()
