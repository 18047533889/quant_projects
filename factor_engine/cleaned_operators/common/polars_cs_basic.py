# -*- coding: utf-8 -*-
"""Basic cross-sectional operators - Polars native implementations.

All operators use pure Polars expressions with unpivot/pivot pattern.
"""
from __future__ import annotations

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.base import ParamRole, ParamSpec

_SKIP = frozenset({"date", "stock_code"})
_SRC = "factor_dsl_polars_native"


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _with_meta(result: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
    if "date" in source.columns and "date" not in result.columns:
        result = result.with_columns(source["date"])
    return result


def _cs_long_transform(
    x: pl.DataFrame,
    transform,
) -> pl.DataFrame:
    """Row-wise cross-section via unpivot → window expr → pivot (no NumPy)."""
    cols = _numeric_cols(x)
    if not cols:
        return x
    long = (
        x.select(cols)
        .with_row_index("_r")
        .unpivot(index="_r", on=cols, variable_name="_c", value_name="_v")
        .with_columns(pl.col("_v").fill_nan(None))
    )
    long = transform(long)
    wide = (
        long.pivot(values="_v", index="_r", on="_c", aggregate_function="first")
        .sort("_r")
        .drop("_r")
        .select(cols)
    )
    return _with_meta(wide, x)


# ---------------------------------------------------------------------------
# Basic cross-sectional transformations
# ---------------------------------------------------------------------------


@register_operator(
    name="cs_rank",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_rank",
    source=_SRC,
    backend="polars")
class CSRankNative(SeriesOperator):
    """Cross-sectional percentile rank."""

    metadata = OperatorMetadata(
        name="cs_rank",
        category="cross_sectional",
        description="截面排名",
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return _cs_long_transform(
            x,
            lambda long: long.with_columns(
                (
                    pl.col("_v").rank(method="average").over("_r")
                    / pl.col("_v").is_not_null().cast(pl.Float64).sum().over("_r")
                ).alias("_v")
            ),
        )


@register_operator(
    name="cs_zscore",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_zscore",
    source=_SRC,
    backend="polars")
class CSZscoreNative(SeriesOperator):
    """Cross-sectional z-score."""

    metadata = OperatorMetadata(
        name="cs_zscore",
        category="cross_sectional",
        description="截面标准化",
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        def _xform(long: pl.DataFrame) -> pl.DataFrame:
            mean = pl.col("_v").mean().over("_r")
            std = pl.col("_v").std().over("_r")
            return long.with_columns(
                pl.when(std.is_null() | (std == 0))
                .then(None)
                .otherwise((pl.col("_v") - mean) / std)
                .alias("_v")
            )

        return _cs_long_transform(x, _xform)


@register_operator(
    name="cs_demean",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_demean",
    source=_SRC,
    backend="polars")
class CSDemeanNative(SeriesOperator):
    """Cross-sectional demean."""

    metadata = OperatorMetadata(
        name="cs_demean",
        category="cross_sectional",
        description="截面去均值",
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return _cs_long_transform(
            x,
            lambda long: long.with_columns(
                (pl.col("_v") - pl.col("_v").mean().over("_r")).alias("_v")
            ),
        )


@register_operator(
    name="cs_scale",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_scale",
    source=_SRC,
    backend="polars")
class CSScaleNative(SeriesOperator):
    """Cross-sectional scale to unit sum of absolute values."""

    metadata = OperatorMetadata(
        name="cs_scale",
        category="cross_sectional",
        description="截面归一化",
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        def _xform(long: pl.DataFrame) -> pl.DataFrame:
            abs_sum = pl.col("_v").abs().sum().over("_r")
            return long.with_columns(
                pl.when(abs_sum.is_null() | (abs_sum == 0))
                .then(None)
                .otherwise(pl.col("_v") / abs_sum)
                .alias("_v")
            )

        return _cs_long_transform(x, _xform)


@register_operator(
    name="cs_quantile",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_quantile",
    source=_SRC,
    backend="polars")
class CSQuantileNative(SeriesOperator):
    """Cross-sectional quantile (broadcast)."""

    metadata = OperatorMetadata(
        name="cs_quantile",
        category="cross_sectional",
        description="截面分位数",
        param_names=["x", "q"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
        param_specs={
            "q": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.5, searchable=False, param_role=ParamRole.THRESHOLD),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, q: float = 0.5, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar

        quantile = strict_finite_scalar(q, "q", minimum=0.0, maximum=1.0)
        return _cs_long_transform(
            x,
            lambda long: long.with_columns(
                pl.col("_v").quantile(quantile, interpolation="linear").over("_r").alias("_v")
            ),
        )


# ---------------------------------------------------------------------------
# Bucketing operators
# ---------------------------------------------------------------------------


@register_operator(
    name="cs_bucket",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_bucket",
    source=_SRC,
    backend="polars")
class CSBucketNative(SeriesOperator):
    """Cross-sectional bucketing into n bins."""

    metadata = OperatorMetadata(
        name="cs_bucket",
        category="cross_sectional",
        description="截面分桶",
        param_names=["x", "n"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
        param_specs={
            "n": ParamSpec(dtype=int, min=2, default=5, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, n: int = 5, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        bins = strict_integer(n, "n", minimum=2)

        def _xform(long: pl.DataFrame) -> pl.DataFrame:
            rank = pl.col("_v").rank(method="average").over("_r")
            count = pl.col("_v").is_not_null().cast(pl.Float64).sum().over("_r")
            return long.with_columns(
                pl.when(pl.col("_v").is_null())
                .then(None)
                .otherwise(((rank - 1) / count * bins).floor().clip(0, bins - 1) + 1)
                .alias("_v")
            )

        return _cs_long_transform(x, _xform)


@register_operator(
    name="cs_bucket_fixed",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_bucket_fixed",
    source=_SRC,
    backend="polars")
class CSBucketFixedNative(SeriesOperator):
    """Bucket by fixed thresholds."""

    metadata = OperatorMetadata(
        name="cs_bucket_fixed",
        category="cross_sectional",
        description="固定阈值分桶",
        param_names=["x", "thresholds"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, thresholds: list[float] | None = None, **kwargs) -> pl.DataFrame:
        if thresholds is None:
            thresholds = [-1.0, 0.0, 1.0]
        th = sorted(thresholds)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            bucket_expr = pl.lit(0.0)
            for i, t in enumerate(th):
                bucket_expr = pl.when(pl.col(c) > t).then(pl.lit(float(i + 1))).otherwise(bucket_expr)
            exprs.append(bucket_expr.alias(c))
        return x.lazy().with_columns(exprs).collect()


@register_operator(
    name="cs_bucket_historical",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_bucket_historical",
    source=_SRC,
    backend="polars")
class CSBucketHistoricalNative(SeriesOperator):
    """Bucket using historical quantiles."""

    metadata = OperatorMetadata(
        name="cs_bucket_historical",
        category="cross_sectional",
        description="历史分位分桶",
        param_names=["x", "n", "lookback"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
        param_specs={
            "n": ParamSpec(dtype=int, min=2, default=5, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
            "lookback": ParamSpec(dtype=int, min=20, default=252, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, n: int = 5, lookback: int = 252, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        bins = strict_integer(n, "n", minimum=2)
        window = strict_integer(lookback, "lookback", minimum=20)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            quantiles = [pl.col(c).rolling_quantile(i / bins, window_size=window) for i in range(1, bins)]
            bucket_expr = pl.lit(0.0)
            for i, q in enumerate(quantiles):
                bucket_expr = pl.when(pl.col(c) > q).then(pl.lit(float(i + 2))).otherwise(bucket_expr)
            exprs.append(bucket_expr.alias(c))
        return x.lazy().with_columns(exprs).collect()


# ---------------------------------------------------------------------------
# Imputation and filling
# ---------------------------------------------------------------------------


@register_operator(
    name="cs_fill_mean",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_fill_mean",
    source=_SRC,
    backend="polars")
class CSFillMeanNative(SeriesOperator):
    """Fill nulls with cross-sectional mean."""

    metadata = OperatorMetadata(
        name="cs_fill_mean",
        category="cross_sectional",
        description="截面均值填充",
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return _cs_long_transform(
            x,
            lambda long: long.with_columns(
                pl.col("_v").fill_null(pl.col("_v").mean().over("_r")).alias("_v")
            ),
        )


@register_operator(
    name="cs_fill_median",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_fill_median",
    source=_SRC,
    backend="polars")
class CSFillMedianNative(SeriesOperator):
    """Fill nulls with cross-sectional median."""

    metadata = OperatorMetadata(
        name="cs_fill_median",
        category="cross_sectional",
        description="截面中位数填充",
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return _cs_long_transform(
            x,
            lambda long: long.with_columns(
                pl.col("_v").fill_null(pl.col("_v").median().over("_r")).alias("_v")
            ),
        )


@register_operator(
    name="cs_impute_mean",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_impute_mean",
    source=_SRC,
    backend="polars")
class CSImputeMeanNative(SeriesOperator):
    """Alias for cs_fill_mean."""

    metadata = OperatorMetadata(
        name="cs_impute_mean",
        category="cross_sectional",
        description="截面均值插补",
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return _cs_long_transform(
            x,
            lambda long: long.with_columns(
                pl.col("_v").fill_null(pl.col("_v").mean().over("_r")).alias("_v")
            ),
        )


@register_operator(
    name="cs_impute_median",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_impute_median",
    source=_SRC,
    backend="polars")
class CSImputeMedianNative(SeriesOperator):
    """Alias for cs_fill_median."""

    metadata = OperatorMetadata(
        name="cs_impute_median",
        category="cross_sectional",
        description="截面中位数插补",
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return _cs_long_transform(
            x,
            lambda long: long.with_columns(
                pl.col("_v").fill_null(pl.col("_v").median().over("_r")).alias("_v")
            ),
        )


# ---------------------------------------------------------------------------
# Neutralization and weighted operations
# ---------------------------------------------------------------------------


@register_operator(
    name="cs_neutralize",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_neutralize",
    source=_SRC,
    backend="polars")
class CSNeutralizeNative(SeriesOperator):
    """Cross-sectional demean (alias)."""

    metadata = OperatorMetadata(
        name="cs_neutralize",
        category="cross_sectional",
        description="截面中性化",
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return _cs_long_transform(
            x,
            lambda long: long.with_columns(
                (pl.col("_v") - pl.col("_v").mean().over("_r")).alias("_v")
            ),
        )


@register_operator(
    name="cs_weighted_demean",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_weighted_demean",
    source=_SRC,
    backend="polars")
class CSWeightedDemeanNative(SeriesOperator):
    """Weighted cross-sectional demean."""

    metadata = OperatorMetadata(
        name="cs_weighted_demean",
        category="cross_sectional",
        description="加权截面去均值",
        param_names=["x", "weights"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, weights: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            if c not in weights.columns:
                exprs.append(pl.lit(None).alias(c))
                continue
            w = weights[c].fill_nan(None)
            weighted_sum = (pl.col(c) * w).sum()
            weight_sum = w.sum()
            weighted_mean = pl.when(weight_sum != 0).then(weighted_sum / weight_sum).otherwise(None)
            exprs.append((pl.col(c) - weighted_mean).alias(c))
        return x.lazy().with_columns(exprs).collect()


@register_operator(
    name="cs_weighted_mean",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_weighted_mean",
    source=_SRC,
    backend="polars")
class CSWeightedMeanNative(SeriesOperator):
    """Weighted cross-sectional mean (broadcast)."""

    metadata = OperatorMetadata(
        name="cs_weighted_mean",
        category="cross_sectional",
        description="加权截面均值",
        param_names=["x", "weights"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, weights: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            if c not in weights.columns:
                exprs.append(pl.lit(None).alias(c))
                continue
            w = weights[c].fill_nan(None)
            weighted_sum = (pl.col(c) * w).sum()
            weight_sum = w.sum()
            exprs.append((weighted_sum / weight_sum).alias(c))
        return x.lazy().with_columns(exprs).collect()


@register_operator(
    name="cs_weighted_zscore",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_weighted_zscore",
    source=_SRC,
    backend="polars")
class CSWeightedZscoreNative(SeriesOperator):
    """Weighted cross-sectional z-score."""

    metadata = OperatorMetadata(
        name="cs_weighted_zscore",
        category="cross_sectional",
        description="加权截面标准化",
        param_names=["x", "weights"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, weights: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            if c not in weights.columns:
                exprs.append(pl.lit(None).alias(c))
                continue
            w = weights[c].fill_nan(None)
            weighted_sum = (pl.col(c) * w).sum()
            weight_sum = w.sum()
            weighted_mean = pl.when(weight_sum != 0).then(weighted_sum / weight_sum).otherwise(None)
            weighted_var = pl.when(weight_sum != 0).then(((pl.col(c) - weighted_mean) ** 2 * w).sum() / weight_sum).otherwise(None)
            weighted_std = weighted_var.sqrt()
            exprs.append(
                pl.when(weighted_std.is_null() | (weighted_std == 0))
                .then(None)
                .otherwise((pl.col(c) - weighted_mean) / weighted_std)
                .alias(c)
            )
        return x.lazy().with_columns(exprs).collect()


@register_operator(
    name="cs_weighted_percentile_rank",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_weighted_percentile_rank",
    source=_SRC,
    backend="polars")
class CSWeightedPercentileRankNative(SeriesOperator):
    """Weighted percentile rank."""

    metadata = OperatorMetadata(
        name="cs_weighted_percentile_rank",
        category="cross_sectional",
        description="加权百分位排名",
        param_names=["x", "weights"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, weights: pl.DataFrame, **kwargs) -> pl.DataFrame:
        # Simplified: use unweighted rank as approximation
        return _cs_long_transform(
            x,
            lambda long: long.with_columns(
                (
                    pl.col("_v").rank(method="average").over("_r")
                    / pl.col("_v").is_not_null().cast(pl.Float64).sum().over("_r")
                ).alias("_v")
            ),
        )


# ---------------------------------------------------------------------------
# Coverage and validity metrics
# ---------------------------------------------------------------------------


@register_operator(
    name="cs_valid_count",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_valid_count",
    source=_SRC,
    backend="polars")
class CSValidCountNative(SeriesOperator):
    """Count of non-null values in cross-section (broadcast)."""

    metadata = OperatorMetadata(
        name="cs_valid_count",
        category="cross_sectional",
        description="截面有效计数",
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return _cs_long_transform(
            x,
            lambda long: long.with_columns(
                pl.col("_v").is_not_null().cast(pl.Float64).sum().over("_r").alias("_v")
            ),
        )


@register_operator(
    name="cs_coverage_ratio",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_coverage_ratio",
    source=_SRC,
    backend="polars")
class CSCoverageRatioNative(SeriesOperator):
    """Ratio of non-null values in cross-section."""

    metadata = OperatorMetadata(
        name="cs_coverage_ratio",
        category="cross_sectional",
        description="截面覆盖率",
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return _cs_long_transform(
            x,
            lambda long: long.with_columns(
                (
                    pl.col("_v").is_not_null().cast(pl.Float64).sum().over("_r")
                    / pl.col("_v").len().over("_r")
                ).alias("_v")
            ),
        )


@register_operator(
    name="cs_universe_coverage",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_universe_coverage",
    source=_SRC,
    backend="polars")
class CSUniverseCoverageNative(SeriesOperator):
    """Universe coverage ratio (alias)."""

    metadata = OperatorMetadata(
        name="cs_universe_coverage",
        category="cross_sectional",
        description="宇宙覆盖率",
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return _cs_long_transform(
            x,
            lambda long: long.with_columns(
                (
                    pl.col("_v").is_not_null().cast(pl.Float64).sum().over("_r")
                    / pl.col("_v").len().over("_r")
                ).alias("_v")
            ),
        )


@register_operator(
    name="cs_physical_panel_coverage",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_physical_panel_coverage",
    source=_SRC,
    backend="polars")
class CSPhysicalPanelCoverageNative(SeriesOperator):
    """Physical panel coverage (alias)."""

    metadata = OperatorMetadata(
        name="cs_physical_panel_coverage",
        category="cross_sectional",
        description="物理面板覆盖率",
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return _cs_long_transform(
            x,
            lambda long: long.with_columns(
                (
                    pl.col("_v").is_not_null().cast(pl.Float64).sum().over("_r")
                    / pl.col("_v").len().over("_r")
                ).alias("_v")
            ),
        )


# ---------------------------------------------------------------------------
# Advanced transformations
# ---------------------------------------------------------------------------


@register_operator(
    name="cs_rank_gaussian",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_rank_gaussian",
    source=_SRC,
    backend="polars")
class CSRankGaussianNative(SeriesOperator):
    """Gaussian rank normalization (rank to normal quantiles)."""

    metadata = OperatorMetadata(
        name="cs_rank_gaussian",
        category="cross_sectional",
        description="高斯排名归一化",
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        # Approximate: map percentile rank to [-3, 3]
        def _xform(long: pl.DataFrame) -> pl.DataFrame:
            rank = pl.col("_v").rank(method="average").over("_r")
            count = pl.col("_v").is_not_null().cast(pl.Float64).sum().over("_r")
            pct = pl.when(count + 1 != 0).then(rank / (count + 1)).otherwise(None)
            # Approximate inverse normal: 6 * (pct - 0.5)
            return long.with_columns((6.0 * (pct - 0.5)).alias("_v"))

        return _cs_long_transform(x, _xform)


@register_operator(
    name="cs_residual_percentile",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_residual_percentile",
    source=_SRC,
    backend="polars")
class CSResidualPercentileNative(SeriesOperator):
    """Percentile of residuals (after demean)."""

    metadata = OperatorMetadata(
        name="cs_residual_percentile",
        category="cross_sectional",
        description="残差百分位",
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        def _xform(long: pl.DataFrame) -> pl.DataFrame:
            mean = pl.col("_v").mean().over("_r")
            resid = pl.col("_v") - mean
            rank = resid.rank(method="average").over("_r")
            count = resid.is_not_null().cast(pl.Float64).sum().over("_r")
            pct = pl.when(count != 0).then(rank / count).otherwise(None)
            return long.with_columns(pct.alias("_v"))

        return _cs_long_transform(x, _xform)


@register_operator(
    name="cs_tail_breadth",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_tail_breadth",
    source=_SRC,
    backend="polars")
class CSTailBreadthNative(SeriesOperator):
    """Fraction of values in tails (beyond ±threshold std)."""

    metadata = OperatorMetadata(
        name="cs_tail_breadth",
        category="cross_sectional",
        description="尾部广度",
        param_names=["x", "threshold"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
        param_specs={
            "threshold": ParamSpec(dtype=float, min=0.0, default=2.0, searchable=False, param_role=ParamRole.THRESHOLD),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, threshold: float = 2.0, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar

        th = strict_finite_scalar(threshold, "threshold", minimum=0.0)

        def _xform(long: pl.DataFrame) -> pl.DataFrame:
            mean = pl.col("_v").mean().over("_r")
            std = pl.col("_v").std().over("_r")
            zscore = pl.when(std > 1e-10).then((pl.col("_v") - mean) / std).otherwise(None)
            in_tail = (zscore.abs() > th).cast(pl.Float64)
            return long.with_columns(in_tail.mean().over("_r").alias("_v"))

        return _cs_long_transform(x, _xform)


@register_operator(
    name="cs_tail_retention",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_tail_retention",
    source=_SRC,
    backend="polars")
class CSTailRetentionNative(SeriesOperator):
    """Retain only values in tails, set others to null."""

    metadata = OperatorMetadata(
        name="cs_tail_retention",
        category="cross_sectional",
        description="尾部保留",
        param_names=["x", "threshold"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
        param_specs={
            "threshold": ParamSpec(dtype=float, min=0.0, default=2.0, searchable=False, param_role=ParamRole.THRESHOLD),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, threshold: float = 2.0, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar

        th = strict_finite_scalar(threshold, "threshold", minimum=0.0)

        def _xform(long: pl.DataFrame) -> pl.DataFrame:
            mean = pl.col("_v").mean().over("_r")
            std = pl.col("_v").std().over("_r")
            zscore = pl.when(std > 1e-10).then((pl.col("_v") - mean) / std).otherwise(None)
            return long.with_columns(
                pl.when(zscore.abs() > th).then(pl.col("_v")).otherwise(None).alias("_v")
            )

        return _cs_long_transform(x, _xform)


# ---------------------------------------------------------------------------
# Robust regression residuals
# ---------------------------------------------------------------------------


@register_operator(
    name="cs_huber_resid",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_huber_resid",
    source=_SRC,
    backend="polars")
class CSHuberResidNative(SeriesOperator):
    """Huber regression residual (simplified: use demean as proxy)."""

    metadata = OperatorMetadata(
        name="cs_huber_resid",
        category="cross_sectional",
        description="Huber回归残差",
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        # Simplified: demean as proxy
        return _cs_long_transform(
            x,
            lambda long: long.with_columns(
                (pl.col("_v") - pl.col("_v").mean().over("_r")).alias("_v")
            ),
        )


@register_operator(
    name="cs_lad_resid",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_lad_resid",
    source=_SRC,
    backend="polars")
class CSLadResidNative(SeriesOperator):
    """LAD (L1) regression residual (use median)."""

    metadata = OperatorMetadata(
        name="cs_lad_resid",
        category="cross_sectional",
        description="LAD回归残差",
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return _cs_long_transform(
            x,
            lambda long: long.with_columns(
                (pl.col("_v") - pl.col("_v").median().over("_r")).alias("_v")
            ),
        )


@register_operator(
    name="cs_ridge_resid",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_ridge_resid",
    source=_SRC,
    backend="polars")
class CSRidgeResidNative(SeriesOperator):
    """Ridge regression residual (simplified: demean)."""

    metadata = OperatorMetadata(
        name="cs_ridge_resid",
        category="cross_sectional",
        description="Ridge回归残差",
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return _cs_long_transform(
            x,
            lambda long: long.with_columns(
                (pl.col("_v") - pl.col("_v").mean().over("_r")).alias("_v")
            ),
        )


@register_operator(
    name="cs_quantile_resid",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_quantile_resid",
    source=_SRC,
    backend="polars")
class CSQuantileResidNative(SeriesOperator):
    """Quantile regression residual (use quantile center)."""

    metadata = OperatorMetadata(
        name="cs_quantile_resid",
        category="cross_sectional",
        description="分位数回归残差",
        param_names=["x", "q"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
        param_specs={
            "q": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.5, searchable=False, param_role=ParamRole.THRESHOLD),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, q: float = 0.5, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar

        quantile = strict_finite_scalar(q, "q", minimum=0.0, maximum=1.0)
        return _cs_long_transform(
            x,
            lambda long: long.with_columns(
                (
                    pl.col("_v")
                    - pl.col("_v").quantile(quantile, interpolation="linear").over("_r")
                ).alias("_v")
            ),
        )


@register_operator(
    name="cs_spline_resid",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_spline_resid",
    source=_SRC,
    backend="polars")
class CSSplineResidNative(SeriesOperator):
    """Spline regression residual (simplified: demean)."""

    metadata = OperatorMetadata(
        name="cs_spline_resid",
        category="cross_sectional",
        description="样条回归残差",
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return _cs_long_transform(
            x,
            lambda long: long.with_columns(
                (pl.col("_v") - pl.col("_v").mean().over("_r")).alias("_v")
            ),
        )

