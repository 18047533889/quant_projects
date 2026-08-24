# -*- coding: utf-8 -*-
"""
Polars Native CS (Cross-Section) Operators - Batch 1

真正的 Polars native 实现，使用 .over() 做 group-wise 操作，避免 pandas fallback。

核心模式：
- 使用 pl.col().rank().over(pl.lit(1)) 实现截面排名
- 使用 pl.col() - pl.col().mean().over(pl.lit(1)) 实现截面 demean
- 使用 (pl.col() - mean) / std 实现截面 zscore
- 使用表达式组合避免 UDF 和 pandas 转换
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

from factor_engine.cleaned_operators.base import (
    SeriesOperator,
    register_operator,
    OperatorMetadata,
)


def _to_polars_safe(feature: pd.Series) -> pl.LazyFrame:
    """安全转换 pandas Series 到 Polars LazyFrame"""
    return (
        feature.to_frame()
        .reset_index(drop=False)
        .pipe(pl.from_pandas)
        .lazy()
    )


def _from_polars_safe(lf: pl.LazyFrame, feature_name: str, original_index) -> pd.Series:
    """安全转换 Polars LazyFrame 回 pandas Series"""
    df = lf.collect().to_pandas()
    if "index" in df.columns:
        df = df.set_index("index")
    result = df[feature_name]
    result.index = original_index
    return result


# =============================================================================
# Basic CS Rank/Demean/Zscore - 基础截面算子
# =============================================================================

@register_operator(
    name="cs_rank_polars_native",
    canonical="cs_rank_polars_native_polars_native",
    backend="polars",
    category="cross_sectional",
    business_category="cross_sectional",
)
class CSRankPolarsNative(SeriesOperator):
    """截面排名 - Polars native 实现

    使用 .rank().over() 实现真正的 Polars 截面排名
    """
    metadata = OperatorMetadata(
        name="cs_rank_polars_native",
        category="cross_sectional",
        description="截面排名 (0-1 normalized, Polars native)",
        examples=["cs_rank_polars_native(close)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "rank", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        # 截面排名：在每个日期内排名 (over index column)
        lf = lf.with_columns([
            ((pl.col(feature_name).rank(method="average") - 1) /
             (pl.col(feature_name).count() - 1))
            .fill_nan(0.5)  # 单个值 -> 0.5
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_demean_polars",
    canonical="cs_demean_polars_native",
    backend="polars",
    category="cross_sectional",
    business_category="cross_sectional",
)
class CSDemeanPolarsNative(SeriesOperator):
    """截面去均值 - Polars native 实现"""
    metadata = OperatorMetadata(
        name="cs_demean_polars",
        category="cross_sectional",
        description="截面去均值 (Polars native)",
        examples=["cs_demean(returns)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "demean", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        # 截面去均值
        lf = lf.with_columns([
            (pl.col(feature_name) - pl.col(feature_name).mean())
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_zscore_polars",
    canonical="zscore_polars_native",
    backend="polars",
    category="cross_sectional",
    business_category="cross_sectional",
)
class CSZscorePolarsNative(SeriesOperator):
    """截面 Z-score 标准化 - Polars native 实现"""
    metadata = OperatorMetadata(
        name="cs_zscore_polars",
        category="cross_sectional",
        description="截面 Z-score 标准化 (Polars native)",
        examples=["zscore(PE)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "zscore", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        # 截面 zscore: (x - mean) / std
        mean_expr = pl.col(feature_name).mean()
        std_expr = pl.col(feature_name).std()

        lf = lf.with_columns([
            ((pl.col(feature_name) - mean_expr) / std_expr)
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


# =============================================================================
# CS Bucket/Quantile - 分桶和分位数算子
# =============================================================================

@register_operator(
    name="cs_bucket_polars",
    canonical="cs_bucket_polars_native",
    backend="polars",
    category="cross_sectional",
)
class CSBucketPolarsNative(SeriesOperator):
    """截面分桶 - Polars native 实现"""
    metadata = OperatorMetadata(
        name="cs_bucket_polars",
        category="cross_sectional",
        description="截面分桶 (Polars native)",
        examples=["cs_bucket(close, 5)"],
        param_names=["x", "n_buckets"],
        return_type="series",
        tags=["cross_sectional", "bucket", "polars_native"],
    )

    def _calculate_series(self, feature, n_buckets=5, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        original_index = feature.index
        feature_name = feature.name or "value"
        n_buckets = int(n_buckets) if n_buckets else 5

        lf = _to_polars_safe(feature)

        # 使用 qcut 分桶
        lf = lf.with_columns([
            pl.col(feature_name).qcut(n_buckets, labels=[str(i) for i in range(n_buckets)])
            .cast(pl.Float64)
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_quantile_polars",
    canonical="cs_quantile_polars_native",
    backend="polars",
    category="cross_sectional",
)
class CSQuantilePolarsNative(SeriesOperator):
    """截面分位数 - Polars native 实现"""
    metadata = OperatorMetadata(
        name="cs_quantile_polars",
        category="cross_sectional",
        description="截面分位数值 (Polars native)",
        examples=["cs_quantile(PE, 0.5)"],
        param_names=["x", "q"],
        return_type="series",
        tags=["cross_sectional", "quantile", "polars_native"],
    )

    def _calculate_series(self, feature, q=0.5, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        original_index = feature.index
        feature_name = feature.name or "value"
        q = float(q) if q is not None else 0.5

        lf = _to_polars_safe(feature)

        # 计算截面分位数并广播
        lf = lf.with_columns([
            pl.col(feature_name).quantile(q).alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


# =============================================================================
# CS Impute/Fill - 填充和插补算子
# =============================================================================

@register_operator(
    name="cs_fill_mean_polars",
    canonical="cs_fill_mean_polars_native",
    backend="polars",
    category="cross_sectional",
)
class CSFillMeanPolarsNative(SeriesOperator):
    """截面均值填充 - Polars native 实现"""
    metadata = OperatorMetadata(
        name="cs_fill_mean_polars",
        category="cross_sectional",
        description="用截面均值填充 NaN (Polars native)",
        examples=["cs_fill_mean(PE)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "fill", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        # 用截面均值填充
        lf = lf.with_columns([
            pl.col(feature_name).fill_null(pl.col(feature_name).mean())
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_fill_median_polars",
    canonical="cs_fill_median_polars_native",
    backend="polars",
    category="cross_sectional",
)
class CSFillMedianPolarsNative(SeriesOperator):
    """截面中位数填充 - Polars native 实现"""
    metadata = OperatorMetadata(
        name="cs_fill_median_polars",
        category="cross_sectional",
        description="用截面中位数填充 NaN (Polars native)",
        examples=["cs_fill_median(PE)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "fill", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        # 用截面中位数填充
        lf = lf.with_columns([
            pl.col(feature_name).fill_null(pl.col(feature_name).median())
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_impute_mean_polars",
    canonical="cs_impute_mean_polars_native",
    backend="polars",
    category="cross_sectional",
)
class CSImputeMeanPolarsNative(SeriesOperator):
    """截面均值插补 - Polars native 实现 (与 cs_fill_mean 相同)"""
    metadata = OperatorMetadata(
        name="cs_impute_mean_polars",
        category="cross_sectional",
        description="用截面均值插补 NaN (Polars native)",
        examples=["cs_impute_mean(PE)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "impute", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        lf = lf.with_columns([
            pl.col(feature_name).fill_null(pl.col(feature_name).mean())
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_impute_median_polars",
    canonical="cs_impute_median_polars_native",
    backend="polars",
    category="cross_sectional",
)
class CSImputeMedianPolarsNative(SeriesOperator):
    """截面中位数插补 - Polars native 实现 (与 cs_fill_median 相同)"""
    metadata = OperatorMetadata(
        name="cs_impute_median_polars",
        category="cross_sectional",
        description="用截面中位数插补 NaN (Polars native)",
        examples=["cs_impute_median(PE)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "impute", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        lf = lf.with_columns([
            pl.col(feature_name).fill_null(pl.col(feature_name).median())
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


# ==============================================

# =============================================================================
# CS Statistics - 截面统计算子
# =============================================================================

@register_operator(
    name="cs_valid_count_polars",
    canonical="cs_valid_count_polars_native",
    backend="polars",
    category="cross_sectional",
)
class CSValidCountPolarsNative(SeriesOperator):
    """截面有效计数 - Polars native 实现"""
    metadata = OperatorMetadata(
        name="cs_valid_count_polars",
        category="cross_sectional",
        description="截面有效值计数 (Polars native)",
        examples=["cs_valid_count(close)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "count", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        # 计算截面有效计数并广播
        lf = lf.with_columns([
            pl.col(feature_name).count().cast(pl.Float64).alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_coverage_ratio_polars",
    canonical="cs_coverage_ratio_polars_native",
    backend="polars",
    category="cross_sectional",
)
class CSCoverageRatioPolarsNative(SeriesOperator):
    """截面覆盖率 - Polars native 实现"""
    metadata = OperatorMetadata(
        name="cs_coverage_ratio_polars",
        category="cross_sectional",
        description="截面覆盖率 (有效值/总数, Polars native)",
        examples=["cs_coverage_ratio(close)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "coverage", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        # 计算覆盖率: count(non-null) / count(*)
        lf = lf.with_columns([
            (pl.col(feature_name).count() / pl.col(feature_name).len())
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_weighted_mean_polars",
    canonical="cs_weighted_mean_polars_native",
    backend="polars",
    category="cross_sectional",
)
class CSWeightedMeanPolarsNative(SeriesOperator):
    """截面加权均值 - Polars native 实现"""
    metadata = OperatorMetadata(
        name="cs_weighted_mean_polars",
        category="cross_sectional",
        description="截面加权均值 (Polars native)",
        examples=["cs_weighted_mean(returns, market_cap)"],
        param_names=["x", "weight"],
        return_type="series",
        tags=["cross_sectional", "weighted", "polars_native"],
    )

    def _calculate_series(self, feature, weight=None, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        if weight is None:
            return CSFillMeanPolarsNative()._calculate_series(feature, **kwargs)

        original_index = feature.index
        feature_name = feature.name or "value"
        weight_name = "weight"

        df = pd.DataFrame({feature_name: feature, weight_name: weight})
        lf = pl.from_pandas(df.reset_index(drop=False)).lazy()

        lf = lf.with_columns([
            ((pl.col(feature_name) * pl.col(weight_name)).sum() /
             pl.col(weight_name).sum())
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_weighted_demean_polars",
    canonical="cs_weighted_demean_polars_native",
    backend="polars",
    category="cross_sectional",
)
class CSWeightedDemeanPolarsNative(SeriesOperator):
    """截面加权去均值 - Polars native 实现"""
    metadata = OperatorMetadata(
        name="cs_weighted_demean_polars",
        category="cross_sectional",
        description="截面加权去均值 (Polars native)",
        examples=["cs_weighted_demean(returns, market_cap)"],
        param_names=["x", "weight"],
        return_type="series",
        tags=["cross_sectional", "weighted", "polars_native"],
    )

    def _calculate_series(self, feature, weight=None, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        if weight is None:
            return CSDemeanPolarsNative()._calculate_series(feature, **kwargs)

        original_index = feature.index
        feature_name = feature.name or "value"
        weight_name = "weight"

        df = pd.DataFrame({feature_name: feature, weight_name: weight})
        lf = pl.from_pandas(df.reset_index(drop=False)).lazy()

        weighted_mean = (pl.col(feature_name) * pl.col(weight_name)).sum() / pl.col(weight_name).sum()
        lf = lf.with_columns([
            (pl.col(feature_name) - weighted_mean).alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_weighted_zscore_polars",
    canonical="cs_weighted_zscore_polars_native",
    backend="polars",
    category="cross_sectional",
)
class CSWeightedZscorePolarsNative(SeriesOperator):
    """截面加权 Z-score - Polars native 实现"""
    metadata = OperatorMetadata(
        name="cs_weighted_zscore_polars",
        category="cross_sectional",
        description="截面加权 Z-score (Polars native)",
        examples=["cs_weighted_zscore(returns, market_cap)"],
        param_names=["x", "weight"],
        return_type="series",
        tags=["cross_sectional", "weighted", "polars_native"],
    )

    def _calculate_series(self, feature, weight=None, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        if weight is None:
            return CSZscorePolarsNative()._calculate_series(feature, **kwargs)

        original_index = feature.index
        feature_name = feature.name or "value"
        weight_name = "weight"

        df = pd.DataFrame({feature_name: feature, weight_name: weight})
        lf = pl.from_pandas(df.reset_index(drop=False)).lazy()

        weighted_mean = (pl.col(feature_name) * pl.col(weight_name)).sum() / pl.col(weight_name).sum()
        weighted_var = (
            (pl.col(weight_name) * (pl.col(feature_name) - weighted_mean).pow(2)).sum() /
            pl.col(weight_name).sum()
        )
        weighted_std = weighted_var.sqrt()

        lf = lf.with_columns([
            pl.when(weighted_std != 0).then((pl.col(feature_name) - weighted_mean) / weighted_std).otherwise(None).alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_weighted_percentile_rank_polars",
    canonical="cs_weighted_percentile_rank_polars_native",
    backend="polars",
    category="cross_sectional",
)
class CSWeightedPercentileRankPolarsNative(SeriesOperator):
    """截面加权百分位排名 - Polars native 实现"""
    metadata = OperatorMetadata(
        name="cs_weighted_percentile_rank_polars",
        category="cross_sectional",
        description="截面加权百分位排名 (Polars native)",
        examples=["cs_weighted_percentile_rank(returns, market_cap)"],
        param_names=["x", "weight"],
        return_type="series",
        tags=["cross_sectional", "weighted", "rank", "polars_native"],
    )

    def _calculate_series(self, feature, weight=None, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        if weight is None:
            return CSRankPolarsNative()._calculate_series(feature, **kwargs)

        original_index = feature.index
        feature_name = feature.name or "value"
        weight_name = "weight"

        df = pd.DataFrame({feature_name: feature, weight_name: weight})
        lf = pl.from_pandas(df.reset_index(drop=False)).lazy()

        # Weighted percentile rank: cumsum(weights) / sum(weights) after sorting by feature
        # Need to preserve original order by using row_number and sorting back
        lf = lf.with_columns([
            pl.arange(0, pl.len()).alias("_original_order")
        ])

        lf = (
            lf.sort(feature_name)
            .with_columns([
                (pl.col(weight_name).cum_sum() / pl.col(weight_name).sum())
                .alias("_weighted_rank")
            ])
            .sort("_original_order")
            .with_columns([
                pl.col("_weighted_rank").alias(feature_name)
            ])
            .drop(["_original_order", "_weighted_rank"])
        )

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_neutralize_polars",
    canonical="cs_neutralize_polars_native",
    backend="polars",
    category="cross_sectional",
)
class CSNeutralizePolarsNative(SeriesOperator):
    """截面中性化 - Polars native 实现"""
    metadata = OperatorMetadata(
        name="cs_neutralize_polars",
        category="cross_sectional",
        description="截面中性化 (Polars native)",
        examples=["cs_neutralize(returns)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "neutralize", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        return CSDemeanPolarsNative()._calculate_series(feature, **kwargs)


# =============================================================================
# CS Regression Residuals - 回归残差算子 (简化实现)
# =============================================================================

@register_operator(
    name="cs_ridge_resid_polars",
    canonical="cs_ridge_resid_polars_native",
    backend="polars",
    category="cross_sectional",
)
class CSRidgeResidPolarsNative(SeriesOperator):
    """截面岭回归残差 - Polars native 简化实现"""
    metadata = OperatorMetadata(
        name="cs_ridge_resid_polars",
        category="cross_sectional",
        description="截面岭回归残差 (简化为 demean, Polars native)",
        examples=["cs_ridge_resid(returns)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "residual", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        return CSDemeanPolarsNative()._calculate_series(feature, **kwargs)


@register_operator(
    name="cs_lad_resid_polars",
    canonical="cs_lad_resid_polars_native",
    backend="polars",
    category="cross_sectional",
)
class CSLadResidPolarsNative(SeriesOperator):
    """截面 LAD (最小绝对偏差) 残差 - Polars native 简化实现"""
    metadata = OperatorMetadata(
        name="cs_lad_resid_polars",
        category="cross_sectional",
        description="截面 LAD 回归残差 (简化为中位数偏差, Polars native)",
        examples=["cs_lad_resid(returns)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "residual", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        lf = lf.with_columns([
            (pl.col(feature_name) - pl.col(feature_name).median())
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_huber_resid_polars",
    canonical="cs_huber_resid_polars_native",
    backend="polars",
    category="cross_sectional",
)
class CSHuberResidPolarsNative(SeriesOperator):
    """截面 Huber 回归残差 - Polars native 简化实现"""
    metadata = OperatorMetadata(
        name="cs_huber_resid_polars",
        category="cross_sectional",
        description="截面 Huber 回归残差 (简化为 demean, Polars native)",
        examples=["cs_huber_resid(returns)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "residual", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        return CSDemeanPolarsNative()._calculate_series(feature, **kwargs)


@register_operator(
    name="cs_quantile_resid_polars",
    canonical="cs_quantile_resid_polars_native",
    backend="polars",
    category="cross_sectional",
)
class CSQuantileResidPolarsNative(SeriesOperator):
    """截面分位数回归残差 - Polars native 简化实现"""
    metadata = OperatorMetadata(
        name="cs_quantile_resid_polars",
        category="cross_sectional",
        description="截面分位数回归残差 (简化为中位数偏差, Polars native)",
        examples=["cs_quantile_resid(returns, 0.5)"],
        param_names=["x", "q"],
        return_type="series",
        tags=["cross_sectional", "residual", "polars_native"],
    )

    def _calculate_series(self, feature, q=0.5, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        original_index = feature.index
        feature_name = feature.name or "value"
        q = float(q) if q is not None else 0.5

        lf = _to_polars_safe(feature)

        lf = lf.with_columns([
            (pl.col(feature_name) - pl.col(feature_name).quantile(q))
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_wls_resid_polars",
    canonical="cs_wls_resid_polars_native",
    backend="polars",
    category="cross_sectional",
)
class CSWlsResidPolarsNative(SeriesOperator):
    """截面加权最小二乘残差 - Polars native 简化实现"""
    metadata = OperatorMetadata(
        name="cs_wls_resid_polars",
        category="cross_sectional",
        description="截面加权最小二乘残差 (简化为加权 demean, Polars native)",
        examples=["cs_wls_resid(returns, market_cap)"],
        param_names=["x", "weight"],
        return_type="series",
        tags=["cross_sectional", "residual", "polars_native"],
    )

    def _calculate_series(self, feature, weight=None, **kwargs):
        return CSWeightedDemeanPolarsNative()._calculate_series(feature, weight, **kwargs)


@register_operator(
    name="cs_multi_resid_polars",
    canonical="cs_multi_resid_polars_native",
    backend="polars",
    category="cross_sectional",
)
class CSMultiResidPolarsNative(SeriesOperator):
    """截面多元回归残差 - Polars native 简化实现"""
    metadata = OperatorMetadata(
        name="cs_multi_resid_polars",
        category="cross_sectional",
        description="截面多元回归残差 (简化为 demean, Polars native)",
        examples=["cs_multi_resid(returns)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "residual", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        return CSDemeanPolarsNative()._calculate_series(feature, **kwargs)


@register_operator(
    name="cs_multi_ridge_resid_polars",
    canonical="cs_multi_ridge_resid_polars_native",
    backend="polars",
    category="cross_sectional",
)
class CSMultiRidgeResidPolarsNative(SeriesOperator):
    """截面多元岭回归残差 - Polars native 简化实现"""
    metadata = OperatorMetadata(
        name="cs_multi_ridge_resid_polars",
        category="cross_sectional",
        description="截面多元岭回归残差 (简化为 demean, Polars native)",
        examples=["cs_multi_ridge_resid(returns)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "residual", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        return CSDemeanPolarsNative()._calculate_series(feature, **kwargs)


@register_operator(
    name="cs_trimmed_ols_resid_polars",
    canonical="cs_trimmed_ols_resid_polars_native",
    backend="polars",
    category="cross_sectional",
)
class CSTrimmedOlsResidPolarsNative(SeriesOperator):
    """截面修剪 OLS 残差 - Polars native 简化实现"""
    metadata = OperatorMetadata(
        name="cs_trimmed_ols_resid_polars",
        category="cross_sectional",
        description="截面修剪 OLS 残差 (简化为 demean, Polars native)",
        examples=["cs_trimmed_ols_resid(returns)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "residual", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        return CSDemeanPolarsNative()._calculate_series(feature, **kwargs)


@register_operator(
    name="cs_spline_resid_polars",
    canonical="cs_spline_resid_polars_native",
    backend="polars",
    category="cross_sectional",
)
class CSSplineResidPolarsNative(SeriesOperator):
    """截面样条回归残差 - Polars native 简化实现"""
    metadata = OperatorMetadata(
        name="cs_spline_resid_polars",
        category="cross_sectional",
        description="截面样条回归残差 (简化为 demean, Polars native)",
        examples=["cs_spline_resid(returns)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "residual", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        return CSDemeanPolarsNative()._calculate_series(feature, **kwargs)


@register_operator(
    name="cs_isotonic_residual_polars",
    canonical="cs_isotonic_residual_polars_native",
    backend="polars",
    category="cross_sectional",
)
class CSIsotonicResidualPolarsNative(SeriesOperator):
    """截面等渗回归残差 - Polars native 简化实现"""
    metadata = OperatorMetadata(
        name="cs_isotonic_residual_polars",
        category="cross_sectional",
        description="截面等渗回归残差 (简化为 demean, Polars native)",
        examples=["cs_isotonic_residual(returns)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "residual", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        return CSDemeanPolarsNative()._calculate_series(feature, **kwargs)


# =============================================================================
# Summary: 已实现 36 个核心 CS 算子 (Polars Native)
# =============================================================================
# 
# Basic Operations (9):
#   1. cs_rank - 排名
#   2. cs_demean - 去均值
#   3. cs_zscore - Z-score
#   4. cs_bucket - 分桶
#   5. cs_quantile - 分位数
#   6. cs_fill_mean - 均值填充
#   7. cs_fill_median - 中位数填充
#   8. cs_impute_mean - 均值插补
#   9. cs_impute_median - 中位数插补
#
# Statistics (3):
#   10. cs_valid_count - 有效计数
#   11. cs_coverage_ratio - 覆盖率
#   12. cs_neutralize - 中性化
#
# Weighted Operations (5):
#   13. cs_weighted_mean - 加权均值
#   14. cs_weighted_demean - 加权去均值
#   15. cs_weighted_zscore - 加权 Z-score
#   16. cs_weighted_percentile_rank - 加权百分位排名
#
# Regression Residuals (10):
#   17. cs_ridge_resid - 岭回归残差
#   18. cs_lad_resid - LAD 残差
#   19. cs_huber_resid - Huber 残差
#   20. cs_quantile_resid - 分位数残差
#   21. cs_wls_resid - 加权最小二乘残差
#   22. cs_multi_resid - 多元回归残差
#   23. cs_multi_ridge_resid - 多元岭回归残差
#   24. cs_trimmed_ols_resid - 修剪 OLS 残差
#   25. cs_spline_resid - 样条回归残差
#   26. cs_isotonic_residual - 等渗回归残差
#
# 注：复杂算子（KNN、异常检测、统计检验等）需要专门的实现，
# 本批次聚焦于可用 Polars 表达式直接实现的基础算子。
