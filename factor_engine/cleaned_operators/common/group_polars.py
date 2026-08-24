# -*- coding: utf-8 -*-
"""分组算子 Polars 实现（P0 高频：group_rank/mean/zscore/std 等）。

实现模式
--------
截面/分组算子在 Polars 宽表上 **按行**（每个交易日一行多标的）处理：
``_group_rowwise`` 将每行转为 NumPy 向量，在组标签内聚合后再写回。
复杂组内逻辑（winsorize、decay_linear）与 pandas 版语义保持一致。
"""
from __future__ import annotations

import numpy as np

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

from factor_engine.cleaned_operators.base import ParamRole, ParamSpec
from factor_engine.cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator

_SKIP = frozenset({"date", "stock_code"})


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _group_rowwise(x: pl.DataFrame, group: pl.DataFrame | None, fn):
    """对每个交易日截面调用 ``fn(row_x, row_group) -> row_out``。"""
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


def _aggregate_row(row_x, row_g, reducer):
    """Broadcast a group aggregate while preserving missing input positions."""
    out = np.full_like(row_x, np.nan, dtype=float)
    mask = ~np.isnan(row_x)
    if row_g is None or np.all(np.isnan(row_g)):
        value = float(reducer(row_x[mask])) if np.any(mask) else float(reducer(row_x[mask]))
        out[:] = value
        return out
    for g in np.unique(row_g[~np.isnan(row_g)]):
        gm = (row_g == g) & mask
        if np.any(gm):
            out[gm] = float(reducer(row_x[gm]))
    return out


def _sum_row(row_x, row_g):
    return _aggregate_row(row_x, row_g, np.sum)


def _min_row(row_x, row_g):
    return _aggregate_row(row_x, row_g, lambda values: np.min(values) if len(values) else np.nan)


def _max_row(row_x, row_g):
    return _aggregate_row(row_x, row_g, lambda values: np.max(values) if len(values) else np.nan)


def _count_row(row_x, row_g):
    return _aggregate_row(row_x, row_g, lambda values: float(len(values)))


def _std_row(row_x, row_g):
    out = np.full_like(row_x, np.nan)
    mask = ~np.isnan(row_x)
    if row_g is None or np.all(np.isnan(row_g)) or not np.any(mask):
        return out
    for g in np.unique(row_g[~np.isnan(row_g)]):
        gm = (row_g == g) & mask
        n = int(np.sum(gm))
        if n == 1:
            out[gm] = 0.0
        elif n > 1:
            std = np.nanstd(row_x[gm], ddof=1)
            out[gm] = 0.0 if std != std else std
    return out


def _zscore_row(row_x, row_g):
    """对齐 pandas ``group_zscore``：零/缺失标准差或单元素组 → 0。"""
    out = np.full_like(row_x, np.nan)
    mask = ~np.isnan(row_x)
    if not np.any(mask):
        return out

    def _apply(indices: np.ndarray) -> None:
        vals = row_x[indices]
        if len(vals) == 0:
            return
        if len(vals) == 1:
            out[indices] = 0.0
            return
        mean = np.nanmean(vals)
        std = np.nanstd(vals, ddof=1)
        if std != 0 and not np.isnan(std):
            out[indices] = (vals - mean) / std
        else:
            out[indices] = 0.0

    if row_g is None or np.all(np.isnan(row_g)):
        _apply(np.where(mask)[0])
        return out

    for g in np.unique(row_g[~np.isnan(row_g)]):
        gm = (row_g == g) & mask
        if np.any(gm):
            _apply(np.where(gm)[0])
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
    """Polars 组内均值"""
    metadata = OperatorMetadata(
        name="group_mean", category="cross_sectional", description="组内均值",
        param_names=["x", "group", "fallback_policy"], return_type="series", tags=["cross_sectional", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame = None, fallback_policy: str = "nan", **kwargs) -> pl.DataFrame:
        return _group_rowwise(x, group, _mean_row)


@register_operator(name="group_sum", category="cross_sectional", business_category="group_neutralization", canonical="group_sum", source="factor_dsl_polars")
class GroupSumPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="group_sum", category="cross_sectional", description="组内求和",
        param_names=["x", "group", "fallback_policy"], return_type="series", tags=["cross_sectional", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame = None, fallback_policy: str = "nan", **kwargs) -> pl.DataFrame:
        return _group_rowwise(x, group, _sum_row)


@register_operator(name="group_min", category="cross_sectional", business_category="group_neutralization", canonical="group_min", source="factor_dsl_polars")
class GroupMinPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="group_min", category="cross_sectional", description="组内最小值",
        param_names=["x", "group", "fallback_policy"], return_type="series", tags=["cross_sectional", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame = None, fallback_policy: str = "nan", **kwargs) -> pl.DataFrame:
        return _group_rowwise(x, group, _min_row)


@register_operator(name="group_max", category="cross_sectional", business_category="group_neutralization", canonical="group_max", source="factor_dsl_polars")
class GroupMaxPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="group_max", category="cross_sectional", description="组内最大值",
        param_names=["x", "group", "fallback_policy"], return_type="series", tags=["cross_sectional", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame = None, fallback_policy: str = "nan", **kwargs) -> pl.DataFrame:
        return _group_rowwise(x, group, _max_row)


@register_operator(name="group_count", category="cross_sectional", business_category="group_neutralization", canonical="group_count", source="factor_dsl_polars")
class GroupCountPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="group_count", category="cross_sectional", description="组内非空计数",
        param_names=["x", "group", "fallback_policy"], return_type="series", tags=["cross_sectional", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame = None, fallback_policy: str = "nan", **kwargs) -> pl.DataFrame:
        return _group_rowwise(x, group, _count_row)


@register_operator(name="group_std", category="cross_sectional", business_category="group_neutralization", canonical="group_std", source="factor_dsl_polars")
class GroupStdPolars(SeriesOperator):
    """Polars 组内标准差"""
    metadata = OperatorMetadata(
        name="group_std", category="cross_sectional", description="组内标准差",
        param_names=["x", "group", "fallback_policy"], return_type="series", tags=["cross_sectional", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame = None, fallback_policy: str = "nan", **kwargs) -> pl.DataFrame:
        return _group_rowwise(x, group, _std_row)


@register_operator(
    name="group_zscore",
    category="cross_sectional",
    business_category="group_neutralization",
    canonical="group_zscore",
    source="factor_dsl_polars",
    backend="polars",
)
class GroupZscorePolars(SeriesOperator):
    """Polars 组内 zscore"""
    metadata = OperatorMetadata(
        name="group_zscore", category="cross_sectional", description="组内 zscore",
        param_names=["x", "group", "fallback_policy"], return_type="series", tags=["cross_sectional", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame = None, fallback_policy: str = "nan", **kwargs) -> pl.DataFrame:
        return _group_rowwise(x, group, _zscore_row)


@register_operator(name="group_rank", category="cross_sectional", business_category="group_neutralization", canonical="group_rank", source="factor_dsl_polars")
class GroupRankPolars(SeriesOperator):
    """Polars 组内排名 pct"""
    metadata = OperatorMetadata(
        name="group_rank", category="cross_sectional", description="组内排名 pct",
        param_names=["x", "group", "fallback_policy"], return_type="series", tags=["cross_sectional", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame = None, fallback_policy: str = "nan", **kwargs) -> pl.DataFrame:
        return _group_rowwise(x, group, _rank_row)


@register_operator(name="group_normalize", category="cross_sectional", business_category="group_neutralization", canonical="group_normalize", source="factor_dsl_polars")
class GroupNormalizePolars(SeriesOperator):
    """Polars 组内 [0,1] 归一化"""
    metadata = OperatorMetadata(
        name="group_normalize", category="cross_sectional", description="组内 [0,1] 归一化",
        param_names=["x", "group", "fallback_policy"], return_type="series", tags=["cross_sectional", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame = None, fallback_policy: str = "nan", **kwargs) -> pl.DataFrame:
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


@register_operator(name="group_winsorize", category="cross_sectional", business_category="group_neutralization", canonical="group_winsorize", source="factor_dsl_polars")
class GroupWinsorizePolars(SeriesOperator):
    """Polars 组内缩尾"""
    metadata = OperatorMetadata(
        name="group_winsorize", category="cross_sectional", description="组内缩尾",
        param_names=["x", "group", "a"], return_type="series", tags=["cross_sectional", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame = None, a: float = 0.05, **kwargs) -> pl.DataFrame:
        alpha = float(kwargs.get("p", a))

        def _winsor(row_x, row_g):
            out = row_x.copy()
            mask = ~np.isnan(row_x)
            if not np.any(mask):
                return out

            def _clip(vals):
                lo = np.nanquantile(vals, alpha)
                hi = np.nanquantile(vals, 1.0 - alpha)
                return np.clip(vals, lo, hi)

            if row_g is None or np.all(np.isnan(row_g)):
                out[mask] = _clip(row_x[mask])
                return out
            for g in np.unique(row_g[~np.isnan(row_g)]):
                gm = (row_g == g) & mask
                if np.any(gm):
                    out[gm] = _clip(row_x[gm])
            return out

        return _group_rowwise(x, group, _winsor)


@register_operator(name="group_percentile", category="cross_sectional", business_category="group_neutralization", canonical="group_percentile", source="factor_dsl_polars")
class GroupPercentilePolars(SeriesOperator):
    """Polars 组内分位数"""
    metadata = OperatorMetadata(
        name="group_percentile", category="cross_sectional", description="组内分位数",
        param_names=["x", "group", "p", "side", "missing_group_policy"],
        return_type="series", tags=["cross_sectional", "polars"],
    )

    def _calculate_series(
        self,
        x: pl.DataFrame,
        group: pl.DataFrame = None,
        p: float = 0.5,
        side: str = "top",
        missing_group_policy: str = "raise",
        **kwargs,
    ) -> pl.DataFrame:
        q = float(p)
        if not 0.0 < q <= 1.0:
            raise ValueError("p must satisfy 0 < p <= 1")
        if side not in {"top", "bottom"}:
            raise ValueError("side must be 'top' or 'bottom'")
        if missing_group_policy not in {"raise", "null", "global"}:
            raise ValueError(
                "missing_group_policy must be 'raise', 'null', or 'global'"
            )

        def _pct(row_x, row_g):
            out = np.full_like(row_x, np.nan, dtype=float)
            mask = ~np.isnan(row_x)
            if not np.any(mask):
                return out

            def _indicator(vals: np.ndarray) -> np.ndarray:
                if len(vals) == 0:
                    return np.array([], dtype=float)
                order = np.argsort(vals, kind="mergesort")
                ranks = np.empty(len(vals), dtype=float)
                i = 0
                while i < len(vals):
                    j = i
                    while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
                        j += 1
                    ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
                    i = j + 1
                if side == "bottom":
                    ranks = len(vals) + 1.0 - ranks
                return (ranks / len(vals) <= q).astype(float)

            if row_g is None or np.all(np.isnan(row_g)):
                if missing_group_policy == "raise":
                    raise ValueError("group labels are missing")
                if missing_group_policy == "null":
                    return out
                out[mask] = _indicator(row_x[mask])
                return out
            for g in np.unique(row_g[~np.isnan(row_g)]):
                gm = (row_g == g) & mask
                if np.any(gm):
                    out[gm] = _indicator(row_x[gm])
            return out

        return _group_rowwise(x, group, _pct)


def _average_ranks_1d(vals: np.ndarray) -> np.ndarray:
    """平均排名（等价 pandas ``rank(method='average')``）。

    并列值取平均名次 —— 保证同一组内相同数值获得相同权重（audit item 7），
    不依赖列位置。返回值总和恒为 ``n(n+1)/2``。
    """
    order = np.argsort(vals, kind="mergesort")
    n = len(vals)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def _decay_linear_row(row_x: np.ndarray, row_g: np.ndarray | None) -> np.ndarray:
    """组内按**平均排名**线性加权（CS rank-weighted value）。

    与 ``group.py`` 中 ``GroupRankWeightedValue`` 一致：``rank(method='average')``
    后权重 ∝ 平均排名、组内归一化，再 ``x * weight``；并列值得到相同权重
    （不依赖列位置）。``window`` 参数不参与计算。时间衰减见 ``group_ts_decay_linear``。
    """
    out = np.full_like(row_x, np.nan, dtype=float)
    mask = ~np.isnan(row_x)
    if not np.any(mask):
        return out

    def _apply(vals: np.ndarray) -> np.ndarray:
        ranks = _average_ranks_1d(vals)
        w = ranks / ranks.sum()
        return vals * w

    if row_g is None or np.all(np.isnan(row_g)):
        out[mask] = _apply(row_x[mask])
        return out
    for g in np.unique(row_g[~np.isnan(row_g)]):
        gm = (row_g == g) & mask
        if np.any(gm):
            out[gm] = _apply(row_x[gm])
    return out


@register_operator(
    name="group_rank_weighted_value",
    category="cross_sectional",
    business_category="group_neutralization",
    canonical="group_rank_weighted_value",
    source="factor_dsl_polars",
)
class GroupRankWeightedValuePolars(SeriesOperator):
    """Polars 组内平均排名线性加权（CS rank-weighted value）"""
    metadata = OperatorMetadata(
        name="group_rank_weighted_value",
        category="cross_sectional",
        description="组内平均排名线性加权（CS rank-weighted value；非时间衰减）",
        param_names=["x", "group", "fallback_policy"],
        return_type="series",
        tags=["cross_sectional", "polars"],
    )

    def _calculate_series(
        self, x: pl.DataFrame, group: pl.DataFrame = None, **kwargs
    ) -> pl.DataFrame:
        return _group_rowwise(x, group, _decay_linear_row)


@register_operator(
    name="group_decay_linear",
    category="cross_sectional",
    business_category="group_neutralization",
    canonical="group_decay_linear",
    source="factor_dsl_polars",
)
class GroupDecayLinearPolars(SeriesOperator):
    """Polars 组内按排名线性加权（兼容名；window 不参与计算）"""
    metadata = OperatorMetadata(
        name="group_decay_linear",
        category="cross_sectional",
        description="组内按排名线性加权（兼容名；window 不参与计算，诚实名称 group_rank_weighted_value）",
        param_names=["x", "group", "window", "fallback_policy"],
        return_type="series",
        tags=["cross_sectional", "polars"],
        param_specs={"window": ParamSpec(dtype=int, searchable=False, param_role=ParamRole.POLICY)},
    )

    def _calculate_series(
        self, x: pl.DataFrame, group: pl.DataFrame = None, window: int = 5,
        fallback_policy: str = "nan", **kwargs
    ) -> pl.DataFrame:
        return _group_rowwise(x, group, _decay_linear_row)
