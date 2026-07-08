# -*- coding: utf-8 -*-
"""分组算子 Polars 实现（P0 高频：group_rank/mean/zscore/std 等）。"""
from __future__ import annotations

import numpy as np

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator

_SKIP = frozenset({"date", "stock_code"})


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _group_rowwise(x: pl.DataFrame, group: pl.DataFrame | None, fn):
    cols = _numeric_cols(x)
    x_arr = x.select(cols).to_numpy()
    if group is None:
        g_arr = None
    else:
        g_arr = group.select(cols).to_numpy()
    out = np.full_like(x_arr, np.nan, dtype=float)
    for i in range(len(x_arr)):
        out[i] = fn(x_arr[i], g_arr[i] if g_arr is not None else None)
    result = pl.DataFrame(out, schema=cols)
    if "date" in x.columns:
        result = result.with_columns(x["date"])
    return result


def _demean_row(row_x, row_g):
    mask = ~np.isnan(row_x)
    if not np.any(mask):
        return row_x
    if row_g is None or np.all(np.isnan(row_g)):
        m = np.nanmean(row_x[mask])
        out = row_x.copy()
        out[mask] = row_x[mask] - m
        return out
    out = row_x.copy()
    for g in np.unique(row_g[~np.isnan(row_g)]):
        m = row_g == g
        gm = m & mask
        if np.any(gm):
            mu = np.nanmean(row_x[gm])
            out[gm] = row_x[gm] - mu
    return out


def _mean_row(row_x, row_g):
    out = np.full_like(row_x, np.nan)
    mask = ~np.isnan(row_x)
    if not np.any(mask):
        return out
    if row_g is None or np.all(np.isnan(row_g)):
        mu = np.nanmean(row_x[mask])
        out[mask] = mu
        return out
    for g in np.unique(row_g[~np.isnan(row_g)]):
        gm = (row_g == g) & mask
        if np.any(gm):
            out[gm] = np.nanmean(row_x[gm])
    return out


def _std_row(row_x, row_g):
    out = np.full_like(row_x, np.nan)
    mask = ~np.isnan(row_x)
    if row_g is None or np.all(np.isnan(row_g)) or not np.any(mask):
        return out
    for g in np.unique(row_g[~np.isnan(row_g)]):
        gm = (row_g == g) & mask
        if np.sum(gm) > 1:
            out[gm] = np.nanstd(row_x[gm], ddof=1)
    return out


def _zscore_row(row_x, row_g):
    out = row_x.copy()
    mu = _mean_row(row_x, row_g)
    sd = _std_row(row_x, row_g)
    valid = ~np.isnan(row_x) & (sd > 0)
    out[valid] = (row_x[valid] - mu[valid]) / sd[valid]
    return out


def _rank_row(row_x, row_g):
    out = np.full_like(row_x, np.nan)
    mask = ~np.isnan(row_x)

    def _rank_vals(vals):
        order = np.argsort(vals, kind="mergesort")
        ranks = np.empty(len(vals), dtype=float)
        ranks[order] = np.arange(1, len(vals) + 1, dtype=float)
        return ranks / len(vals)

    if not np.any(mask):
        return out
    if row_g is None or np.all(np.isnan(row_g)):
        out[mask] = _rank_vals(row_x[mask])
        return out
    for g in np.unique(row_g[~np.isnan(row_g)]):
        gm = (row_g == g) & mask
        if np.any(gm):
            out[gm] = _rank_vals(row_x[gm])
    return out


@register_operator(name="group_mean", category="cross_sectional", business_category="group_neutralization", canonical="group_mean", source="factor_dsl_polars")
class GroupMeanPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="group_mean", category="cross_sectional", description="组内均值",
        param_names=["x", "group"], return_type="series", tags=["cross_sectional", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame = None, **kwargs) -> pl.DataFrame:
        return _group_rowwise(x, group, _mean_row)


@register_operator(name="group_std", category="cross_sectional", business_category="group_neutralization", canonical="group_std", source="factor_dsl_polars")
class GroupStdPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="group_std", category="cross_sectional", description="组内标准差",
        param_names=["x", "group"], return_type="series", tags=["cross_sectional", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame = None, **kwargs) -> pl.DataFrame:
        return _group_rowwise(x, group, _std_row)


@register_operator(name="group_zscore", category="cross_sectional", business_category="group_neutralization", canonical="group_zscore", source="factor_dsl_polars")
class GroupZscorePolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="group_zscore", category="cross_sectional", description="组内 zscore",
        param_names=["x", "group"], return_type="series", tags=["cross_sectional", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame = None, **kwargs) -> pl.DataFrame:
        return _group_rowwise(x, group, _zscore_row)


@register_operator(name="group_rank", category="cross_sectional", business_category="group_neutralization", canonical="group_rank", source="factor_dsl_polars")
class GroupRankPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="group_rank", category="cross_sectional", description="组内排名 pct",
        param_names=["x", "group"], return_type="series", tags=["cross_sectional", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame = None, **kwargs) -> pl.DataFrame:
        return _group_rowwise(x, group, _rank_row)


@register_operator(name="group_normalize", category="cross_sectional", business_category="group_neutralization", canonical="group_normalize", source="factor_dsl_polars")
class GroupNormalizePolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="group_normalize", category="cross_sectional", description="组内 [0,1] 归一化",
        param_names=["x", "group"], return_type="series", tags=["cross_sectional", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame = None, **kwargs) -> pl.DataFrame:
        def _norm(row_x, row_g):
            out = np.full_like(row_x, np.nan)
            mask = ~np.isnan(row_x)
            if not np.any(mask):
                return out

            def _scale(vals):
                lo, hi = np.nanmin(vals), np.nanmax(vals)
                rng = hi - lo
                if rng == 0 or np.isnan(rng):
                    return np.full_like(vals, 0.5)
                return (vals - lo) / rng

            if row_g is None or np.all(np.isnan(row_g)):
                out[mask] = _scale(row_x[mask])
                return out
            for g in np.unique(row_g[~np.isnan(row_g)]):
                gm = (row_g == g) & mask
                if np.any(gm):
                    out[gm] = _scale(row_x[gm])
            return out

        return _group_rowwise(x, group, _norm)
