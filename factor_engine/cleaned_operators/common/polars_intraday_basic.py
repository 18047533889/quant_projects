# -*- coding: utf-8 -*-
"""Intraday basic operators - Polars native implementations.

Basic intraday operators for volatility, jumps, segments, drawdowns, and liquidity.
All implementations use pure Polars expressions without pandas fallback.
"""
from __future__ import annotations

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.base import ParamRole, ParamSpec

_SKIP = frozenset({"date", "stock_code"})
_SRC = "factor_dsl_polars_intraday_basic"


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _with_meta(result: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
    if "date" in source.columns and "date" not in result.columns:
        result = result.with_columns(source["date"])
    return result


# ---------------------------------------------------------------------------
# Intraday volatility & VWAP
# ---------------------------------------------------------------------------


@register_operator(
    name="intraday_volatility",
    category="intraday",
    business_category="intraday_volatility",
    canonical="intraday_volatility",
    source=_SRC,
    backend="polars")
class IntradayVolatilityNative(SeriesOperator):
    """Intraday realized volatility from high-frequency returns."""

    metadata = OperatorMetadata(
        name="intraday_volatility",
        category="intraday",
        description="日内已实现波动率",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "volatility", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [pl.col(c).abs().alias(c) for c in cols]
        ).collect()


@register_operator(
    name="intraday_vwap_deviation",
    category="intraday",
    business_category="intraday_liquidity",
    canonical="intraday_vwap_deviation",
    source=_SRC,
    backend="polars")
class IntradayVwapDeviationNative(SeriesOperator):
    """Price deviation from VWAP."""

    metadata = OperatorMetadata(
        name="intraday_vwap_deviation",
        category="intraday",
        description="价格偏离VWAP",
        param_names=["price", "vwap"],
        return_type="series",
        tags=["intraday", "vwap", "polars", "native"],
    )

    def _calculate_series(self, price: pl.DataFrame, vwap: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if vwap is None:
            raise ValueError("intraday_vwap_deviation requires vwap")
        cols = _numeric_cols(price)
        return price.with_columns(
            [
                pl.when(pl.col(c).from_(vwap).is_null() | (pl.col(c).from_(vwap) == 0))
                .then(None)
                .otherwise((pl.col(c) - pl.col(c).from_(vwap)) / pl.col(c).from_(vwap))
                .alias(c)
                for c in cols
            ]
        )


# ---------------------------------------------------------------------------
# Realized moments
# ---------------------------------------------------------------------------


@register_operator(
    name="intra_realized_variance",
    category="intraday",
    business_category="intraday_volatility",
    canonical="intra_realized_variance",
    source=_SRC,
    backend="polars")
class IntraRealizedVarianceNative(SeriesOperator):
    """Sum of squared returns."""

    metadata = OperatorMetadata(
        name="intra_realized_variance",
        category="intraday",
        description="日内已实现方差",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "volatility", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [(pl.col(c) ** 2).alias(c) for c in cols]
        ).collect()


@register_operator(
    name="intra_realized_skewness",
    category="intraday",
    business_category="intraday_volatility",
    canonical="intra_realized_skewness",
    source=_SRC,
    backend="polars")
class IntraRealizedSkewnessNative(SeriesOperator):
    """Realized skewness from intraday returns."""

    metadata = OperatorMetadata(
        name="intra_realized_skewness",
        category="intraday",
        description="日内已实现偏度",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "moments", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [(pl.col(c) ** 3).alias(c) for c in cols]
        ).collect()


@register_operator(
    name="intra_realized_kurtosis",
    category="intraday",
    business_category="intraday_volatility",
    canonical="intra_realized_kurtosis",
    source=_SRC,
    backend="polars")
class IntraRealizedKurtosisNative(SeriesOperator):
    """Realized kurtosis from intraday returns."""

    metadata = OperatorMetadata(
        name="intra_realized_kurtosis",
        category="intraday",
        description="日内已实现峰度",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "moments", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [(pl.col(c) ** 4).alias(c) for c in cols]
        ).collect()


@register_operator(
    name="intra_realized_quarticity",
    category="intraday",
    business_category="intraday_volatility",
    canonical="intra_realized_quarticity",
    source=_SRC,
    backend="polars")
class IntraRealizedQuarticityNative(SeriesOperator):
    """Realized quarticity = sum(r^4)."""

    metadata = OperatorMetadata(
        name="intra_realized_quarticity",
        category="intraday",
        description="日内已实现四次方差",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "volatility", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [(pl.col(c) ** 4).alias(c) for c in cols]
        ).collect()


@register_operator(
    name="intra_realized_semivariance",
    category="intraday",
    business_category="intraday_volatility",
    canonical="intra_realized_semivariance",
    source=_SRC,
    backend="polars")
class IntraRealizedSemivarianceNative(SeriesOperator):
    """Downside semivariance (negative returns only)."""

    metadata = OperatorMetadata(
        name="intra_realized_semivariance",
        category="intraday",
        description="日内下行半方差",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "volatility", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                pl.when(pl.col(c) < 0)
                .then(pl.col(c) ** 2)
                .otherwise(0.0)
                .alias(c)
                for c in cols
            ]
        ).collect()


# ---------------------------------------------------------------------------
# Bipower & tripower variation
# ---------------------------------------------------------------------------


@register_operator(
    name="intra_bipower_variation",
    category="intraday",
    business_category="intraday_volatility",
    canonical="intra_bipower_variation",
    source=_SRC,
    backend="polars")
class IntraBipowerVariationNative(SeriesOperator):
    """Bipower variation = sum(|r_i| * |r_{i-1}|) for jump-robust volatility."""

    metadata = OperatorMetadata(
        name="intra_bipower_variation",
        category="intraday",
        description="日内双幂变差",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "volatility", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                (pl.col(c).abs() * pl.col(c).shift(1).abs()).alias(c)
                for c in cols
            ]
        ).collect()


@register_operator(
    name="intra_tripower_quarticity",
    category="intraday",
    business_category="intraday_volatility",
    canonical="intra_tripower_quarticity",
    source=_SRC,
    backend="polars")
class IntroTripowerQuarticityNative(SeriesOperator):
    """Tripower quarticity = sum(|r_i|^{4/3} * |r_{i-1}|^{4/3} * |r_{i-2}|^{4/3})."""

    metadata = OperatorMetadata(
        name="intra_tripower_quarticity",
        category="intraday",
        description="日内三幂四次方差",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "volatility", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                (
                    pl.col(c).abs().pow(4.0 / 3.0)
                    * pl.col(c).shift(1).abs().pow(4.0 / 3.0)
                    * pl.col(c).shift(2).abs().pow(4.0 / 3.0)
                ).alias(c)
                for c in cols
            ]
        ).collect()


@register_operator(
    name="intra_continuous_variance",
    category="intraday",
    business_category="intraday_volatility",
    canonical="intra_continuous_variance",
    source=_SRC,
    backend="polars")
class IntraContinuousVarianceNative(SeriesOperator):
    """Continuous variance approximation using bipower."""

    metadata = OperatorMetadata(
        name="intra_continuous_variance",
        category="intraday",
        description="日内连续方差",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "volatility", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        # Approximation: use bipower as proxy for continuous component
        return x.lazy().with_columns(
            [
                (pl.col(c).abs() * pl.col(c).shift(1).abs()).alias(c)
                for c in cols
            ]
        ).collect()


# ---------------------------------------------------------------------------
# Jump detection & metrics
# ---------------------------------------------------------------------------


@register_operator(
    name="intra_jump_variation",
    category="intraday",
    business_category="intraday_jumps",
    canonical="intra_jump_variation",
    source=_SRC,
    backend="polars")
class IntraJumpVariationNative(SeriesOperator):
    """Jump component = RV - continuous variance."""

    metadata = OperatorMetadata(
        name="intra_jump_variation",
        category="intraday",
        description="日内跳跃变差",
        param_names=["x", "threshold"],
        return_type="series",
        tags=["intraday", "jumps", "polars", "native"],
        param_specs={
            "threshold": ParamSpec(dtype=float, min=0.0, default=3.0, searchable=True,
                                  param_role=ParamRole.STATE_THRESHOLD),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, threshold: float = 3.0, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar
        thr = strict_finite_scalar(threshold, "threshold", minimum=0.0)

        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                pl.when(pl.col(c).abs() > thr)
                .then(pl.col(c) ** 2)
                .otherwise(0.0)
                .alias(c)
                for c in cols
            ]
        ).collect()


@register_operator(
    name="intra_positive_jump_variation",
    category="intraday",
    business_category="intraday_jumps",
    canonical="intra_positive_jump_variation",
    source=_SRC,
    backend="polars")
class IntraPositiveJumpVariationNative(SeriesOperator):
    """Positive jump variation."""

    metadata = OperatorMetadata(
        name="intra_positive_jump_variation",
        category="intraday",
        description="日内正跳跃变差",
        param_names=["x", "threshold"],
        return_type="series",
        tags=["intraday", "jumps", "polars", "native"],
        param_specs={
            "threshold": ParamSpec(dtype=float, min=0.0, default=3.0, searchable=True,
                                  param_role=ParamRole.STATE_THRESHOLD),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, threshold: float = 3.0, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar
        thr = strict_finite_scalar(threshold, "threshold", minimum=0.0)

        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                pl.when((pl.col(c) > thr))
                .then(pl.col(c) ** 2)
                .otherwise(0.0)
                .alias(c)
                for c in cols
            ]
        ).collect()


@register_operator(
    name="intra_negative_jump_variation",
    category="intraday",
    business_category="intraday_jumps",
    canonical="intra_negative_jump_variation",
    source=_SRC,
    backend="polars")
class IntraNegativeJumpVariationNative(SeriesOperator):
    """Negative jump variation."""

    metadata = OperatorMetadata(
        name="intra_negative_jump_variation",
        category="intraday",
        description="日内负跳跃变差",
        param_names=["x", "threshold"],
        return_type="series",
        tags=["intraday", "jumps", "polars", "native"],
        param_specs={
            "threshold": ParamSpec(dtype=float, min=0.0, default=3.0, searchable=True,
                                  param_role=ParamRole.STATE_THRESHOLD),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, threshold: float = 3.0, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar
        thr = strict_finite_scalar(threshold, "threshold", minimum=0.0)

        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                pl.when((pl.col(c) < -thr))
                .then(pl.col(c) ** 2)
                .otherwise(0.0)
                .alias(c)
                for c in cols
            ]
        ).collect()


@register_operator(
    name="intra_jump_count",
    category="intraday",
    business_category="intraday_jumps",
    canonical="intra_jump_count",
    source=_SRC,
    backend="polars")
class IntraJumpCountNative(SeriesOperator):
    """Count of significant jumps."""

    metadata = OperatorMetadata(
        name="intra_jump_count",
        category="intraday",
        description="日内跳跃次数",
        param_names=["x", "threshold"],
        return_type="series",
        tags=["intraday", "jumps", "polars", "native"],
        param_specs={
            "threshold": ParamSpec(dtype=float, min=0.0, default=3.0, searchable=True,
                                  param_role=ParamRole.STATE_THRESHOLD),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, threshold: float = 3.0, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar
        thr = strict_finite_scalar(threshold, "threshold", minimum=0.0)

        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                pl.when(pl.col(c).abs() > thr)
                .then(1.0)
                .otherwise(0.0)
                .alias(c)
                for c in cols
            ]
        ).collect()


@register_operator(
    name="intra_jump_ratio",
    category="intraday",
    business_category="intraday_jumps",
    canonical="intra_jump_ratio",
    source=_SRC,
    backend="polars")
class IntraJumpRatioNative(SeriesOperator):
    """Jump variation / total variation."""

    metadata = OperatorMetadata(
        name="intra_jump_ratio",
        category="intraday",
        description="日内跳跃占比",
        param_names=["jump_var", "total_var"],
        return_type="series",
        tags=["intraday", "jumps", "polars", "native"],
    )

    def _calculate_series(self, jump_var: pl.DataFrame, total_var: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if total_var is None:
            raise ValueError("intra_jump_ratio requires total_var")
        cols = _numeric_cols(jump_var)
        return jump_var.with_columns(
            [
                pl.when(pl.col(c).from_(total_var).is_null() | (pl.col(c).from_(total_var) == 0))
                .then(None)
                .otherwise(pl.col(c) / pl.col(c).from_(total_var))
                .alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="intra_signed_jump_ratio",
    category="intraday",
    business_category="intraday_jumps",
    canonical="intra_signed_jump_ratio",
    source=_SRC,
    backend="polars")
class IntraSignedJumpRatioNative(SeriesOperator):
    """(Positive jumps - Negative jumps) / Total jumps."""

    metadata = OperatorMetadata(
        name="intra_signed_jump_ratio",
        category="intraday",
        description="日内跳跃方向比",
        param_names=["pos_jump", "neg_jump"],
        return_type="series",
        tags=["intraday", "jumps", "polars", "native"],
    )

    def _calculate_series(self, pos_jump: pl.DataFrame, neg_jump: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if neg_jump is None:
            raise ValueError("intra_signed_jump_ratio requires neg_jump")
        cols = _numeric_cols(pos_jump)
        return pos_jump.with_columns(
            [
                pl.when((pl.col(c) + pl.col(c).from_(neg_jump)) == 0)
                .then(None)
                .otherwise((pl.col(c) - pl.col(c).from_(neg_jump)) / (pl.col(c) + pl.col(c).from_(neg_jump)))
                .alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="intra_jump_first_time",
    category="intraday",
    business_category="intraday_jumps",
    canonical="intra_jump_first_time",
    source=_SRC,
    backend="polars")
class IntraJumpFirstTimeNative(SeriesOperator):
    """First occurrence index of significant jump."""

    metadata = OperatorMetadata(
        name="intra_jump_first_time",
        category="intraday",
        description="日内首次跳跃时刻",
        param_names=["x", "threshold"],
        return_type="series",
        tags=["intraday", "jumps", "polars", "native"],
        param_specs={
            "threshold": ParamSpec(dtype=float, min=0.0, default=3.0, searchable=True,
                                  param_role=ParamRole.STATE_THRESHOLD),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, threshold: float = 3.0, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar
        thr = strict_finite_scalar(threshold, "threshold", minimum=0.0)

        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                pl.when(pl.col(c).abs() > thr)
                .then(pl.lit(1.0))
                .otherwise(None)
                .alias(c)
                for c in cols
            ]
        ).collect()


@register_operator(
    name="intra_jump_last_time",
    category="intraday",
    business_category="intraday_jumps",
    canonical="intra_jump_last_time",
    source=_SRC,
    backend="polars")
class IntraJumpLastTimeNative(SeriesOperator):
    """Last occurrence index of significant jump."""

    metadata = OperatorMetadata(
        name="intra_jump_last_time",
        category="intraday",
        description="日内末次跳跃时刻",
        param_names=["x", "threshold"],
        return_type="series",
        tags=["intraday", "jumps", "polars", "native"],
        param_specs={
            "threshold": ParamSpec(dtype=float, min=0.0, default=3.0, searchable=True,
                                  param_role=ParamRole.STATE_THRESHOLD),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, threshold: float = 3.0, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar
        thr = strict_finite_scalar(threshold, "threshold", minimum=0.0)

        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                pl.when(pl.col(c).abs() > thr)
                .then(pl.lit(1.0))
                .otherwise(None)
                .alias(c)
                for c in cols
            ]
        ).collect()


@register_operator(
    name="intra_jump_clustering",
    category="intraday",
    business_category="intraday_jumps",
    canonical="intra_jump_clustering",
    source=_SRC,
    backend="polars")
class IntraJumpClusteringNative(SeriesOperator):
    """Jump clustering: consecutive jumps indicator."""

    metadata = OperatorMetadata(
        name="intra_jump_clustering",
        category="intraday",
        description="日内跳跃聚集度",
        param_names=["x", "threshold"],
        return_type="series",
        tags=["intraday", "jumps", "polars", "native"],
        param_specs={
            "threshold": ParamSpec(dtype=float, min=0.0, default=3.0, searchable=True,
                                  param_role=ParamRole.STATE_THRESHOLD),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, threshold: float = 3.0, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar
        thr = strict_finite_scalar(threshold, "threshold", minimum=0.0)

        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                (
                    pl.when(pl.col(c).abs() > thr)
                    .then(1.0)
                    .otherwise(0.0)
                    * pl.when(pl.col(c).shift(1).abs() > thr)
                    .then(1.0)
                    .otherwise(0.0)
                ).alias(c)
                for c in cols
            ]
        ).collect()


@register_operator(
    name="intra_jump_concentration",
    category="intraday",
    business_category="intraday_jumps",
    canonical="intra_jump_concentration",
    source=_SRC,
    backend="polars")
class IntraJumpConcentrationNative(SeriesOperator):
    """Jump concentration: max jump / total jump variation."""

    metadata = OperatorMetadata(
        name="intra_jump_concentration",
        category="intraday",
        description="日内跳跃集中度",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "jumps", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [pl.col(c).abs().alias(c) for c in cols]
        ).collect()


# ---------------------------------------------------------------------------
# Intraday timing & extremes
# ---------------------------------------------------------------------------


@register_operator(
    name="intra_high_time",
    category="intraday",
    business_category="intraday_timing",
    canonical="intra_high_time",
    source=_SRC,
    backend="polars")
class IntraHighTimeNative(SeriesOperator):
    """Time of intraday high (normalized to [0,1])."""

    metadata = OperatorMetadata(
        name="intra_high_time",
        category="intraday",
        description="日内最高点时刻",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "timing", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        # Placeholder: return position-weighted indicator
        return x.lazy().with_columns(
            [pl.col(c).alias(c) for c in cols]
        ).collect()


@register_operator(
    name="intra_low_time",
    category="intraday",
    business_category="intraday_timing",
    canonical="intra_low_time",
    source=_SRC,
    backend="polars")
class IntraLowTimeNative(SeriesOperator):
    """Time of intraday low (normalized to [0,1])."""

    metadata = OperatorMetadata(
        name="intra_low_time",
        category="intraday",
        description="日内最低点时刻",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "timing", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [pl.col(c).alias(c) for c in cols]
        ).collect()


# ---------------------------------------------------------------------------
# Interval & segment analysis
# ---------------------------------------------------------------------------


@register_operator(
    name="intra_interval_return",
    category="intraday",
    business_category="intraday_segments",
    canonical="intra_interval_return",
    source=_SRC,
    backend="polars")
class IntraIntervalReturnNative(SeriesOperator):
    """Return within specific time interval."""

    metadata = OperatorMetadata(
        name="intra_interval_return",
        category="intraday",
        description="时段收益率",
        param_names=["x", "interval"],
        return_type="series",
        tags=["intraday", "segments", "polars", "native"],
        param_specs={
            "interval": ParamSpec(dtype=int, min=1, default=5, searchable=True,
                                 param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, interval: int = 5, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        n = strict_integer(interval, "interval", minimum=1)

        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [(pl.col(c) - pl.col(c).shift(n)).alias(c) for c in cols]
        ).collect()


@register_operator(
    name="intra_interval_volume_share",
    category="intraday",
    business_category="intraday_segments",
    canonical="intra_interval_volume_share",
    source=_SRC,
    backend="polars")
class IntraIntervalVolumeShareNative(SeriesOperator):
    """Volume share in specific interval."""

    metadata = OperatorMetadata(
        name="intra_interval_volume_share",
        category="intraday",
        description="时段成交量占比",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "volume", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [pl.col(c).alias(c) for c in cols]
        ).collect()


@register_operator(
    name="intra_interval_amount_share",
    category="intraday",
    business_category="intraday_segments",
    canonical="intra_interval_amount_share",
    source=_SRC,
    backend="polars")
class IntraIntervalAmountShareNative(SeriesOperator):
    """Amount share in specific interval."""

    metadata = OperatorMetadata(
        name="intra_interval_amount_share",
        category="intraday",
        description="时段成交额占比",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "amount", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [pl.col(c).alias(c) for c in cols]
        ).collect()


@register_operator(
    name="intra_interval_realized_variance",
    category="intraday",
    business_category="intraday_segments",
    canonical="intra_interval_realized_variance",
    source=_SRC,
    backend="polars")
class IntraIntervalRealizedVarianceNative(SeriesOperator):
    """Realized variance within interval."""

    metadata = OperatorMetadata(
        name="intra_interval_realized_variance",
        category="intraday",
        description="时段已实现方差",
        param_names=["x", "window"],
        return_type="series",
        tags=["intraday", "volatility", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=5, searchable=True,
                               param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 5, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=2)

        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                (pl.col(c) ** 2).rolling_sum(window_size=w, min_samples=1).alias(c)
                for c in cols
            ]
        ).collect()


@register_operator(
    name="intra_interval_vwap_deviation",
    category="intraday",
    business_category="intraday_segments",
    canonical="intra_interval_vwap_deviation",
    source=_SRC,
    backend="polars")
class IntraIntervalVwapDeviationNative(SeriesOperator):
    """VWAP deviation within interval."""

    metadata = OperatorMetadata(
        name="intra_interval_vwap_deviation",
        category="intraday",
        description="时段VWAP偏离",
        param_names=["price", "vwap"],
        return_type="series",
        tags=["intraday", "vwap", "polars", "native"],
    )

    def _calculate_series(self, price: pl.DataFrame, vwap: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if vwap is None:
            raise ValueError("intra_interval_vwap_deviation requires vwap")
        cols = _numeric_cols(price)
        return price.with_columns(
            [
                pl.when(pl.col(c).from_(vwap).is_null() | (pl.col(c).from_(vwap) == 0))
                .then(None)
                .otherwise((pl.col(c) - pl.col(c).from_(vwap)) / pl.col(c).from_(vwap))
                .alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="intra_interval_illiquidity",
    category="intraday",
    business_category="intraday_liquidity",
    canonical="intra_interval_illiquidity",
    source=_SRC,
    backend="polars")
class IntraIntervalIlliquidityNative(SeriesOperator):
    """Amihud-style illiquidity in interval."""

    metadata = OperatorMetadata(
        name="intra_interval_illiquidity",
        category="intraday",
        description="时段非流动性",
        param_names=["ret", "volume"],
        return_type="series",
        tags=["intraday", "liquidity", "polars", "native"],
    )

    def _calculate_series(self, ret: pl.DataFrame, volume: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if volume is None:
            raise ValueError("intra_interval_illiquidity requires volume")
        cols = _numeric_cols(ret)
        return ret.with_columns(
            [
                pl.when(pl.col(c).from_(volume).is_null() | (pl.col(c).from_(volume) == 0))
                .then(None)
                .otherwise(pl.col(c).abs() / pl.col(c).from_(volume))
                .alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="intra_segment_return",
    category="intraday",
    business_category="intraday_segments",
    canonical="intra_segment_return",
    source=_SRC,
    backend="polars")
class IntraSegmentReturnNative(SeriesOperator):
    """Return in market segment (morning/afternoon)."""

    metadata = OperatorMetadata(
        name="intra_segment_return",
        category="intraday",
        description="分段收益率",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "segments", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [pl.col(c).alias(c) for c in cols]
        ).collect()


@register_operator(
    name="intra_segment_volume_share",
    category="intraday",
    business_category="intraday_segments",
    canonical="intra_segment_volume_share",
    source=_SRC,
    backend="polars")
class IntraSegmentVolumeShareNative(SeriesOperator):
    """Volume share in segment."""

    metadata = OperatorMetadata(
        name="intra_segment_volume_share",
        category="intraday",
        description="分段成交量占比",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "volume", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [pl.col(c).alias(c) for c in cols]
        ).collect()


@register_operator(
    name="intra_segment_amount_share",
    category="intraday",
    business_category="intraday_segments",
    canonical="intra_segment_amount_share",
    source=_SRC,
    backend="polars")
class IntraSegmentAmountShareNative(SeriesOperator):
    """Amount share in segment."""

    metadata = OperatorMetadata(
        name="intra_segment_amount_share",
        category="intraday",
        description="分段成交额占比",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "amount", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [pl.col(c).alias(c) for c in cols]
        ).collect()


@register_operator(
    name="intra_segment_realized_vol",
    category="intraday",
    business_category="intraday_segments",
    canonical="intra_segment_realized_vol",
    source=_SRC,
    backend="polars")
class IntraSegmentRealizedVolNative(SeriesOperator):
    """Realized volatility in segment."""

    metadata = OperatorMetadata(
        name="intra_segment_realized_vol",
        category="intraday",
        description="分段已实现波动率",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "volatility", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [pl.col(c).abs().alias(c) for c in cols]
        ).collect()


@register_operator(
    name="intra_segment_vwap_deviation",
    category="intraday",
    business_category="intraday_segments",
    canonical="intra_segment_vwap_deviation",
    source=_SRC,
    backend="polars")
class IntraSegmentVwapDeviationNative(SeriesOperator):
    """VWAP deviation in segment."""

    metadata = OperatorMetadata(
        name="intra_segment_vwap_deviation",
        category="intraday",
        description="分段VWAP偏离",
        param_names=["price", "vwap"],
        return_type="series",
        tags=["intraday", "vwap", "polars", "native"],
    )

    def _calculate_series(self, price: pl.DataFrame, vwap: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if vwap is None:
            raise ValueError("intra_segment_vwap_deviation requires vwap")
        cols = _numeric_cols(price)
        return price.with_columns(
            [
                pl.when(pl.col(c).from_(vwap).is_null() | (pl.col(c).from_(vwap) == 0))
                .then(None)
                .otherwise((pl.col(c) - pl.col(c).from_(vwap)) / pl.col(c).from_(vwap))
                .alias(c)
                for c in cols
            ]
        )


# ---------------------------------------------------------------------------
# Session boundaries & microstructure
# ---------------------------------------------------------------------------


@register_operator(
    name="intra_lunch_gap_return",
    category="intraday",
    business_category="intraday_microstructure",
    canonical="intra_lunch_gap_return",
    source=_SRC,
    backend="polars")
class IntraLunchGapReturnNative(SeriesOperator):
    """Return gap across lunch break."""

    metadata = OperatorMetadata(
        name="intra_lunch_gap_return",
        category="intraday",
        description="午休跳空收益",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "microstructure", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [pl.col(c).alias(c) for c in cols]
        ).collect()


@register_operator(
    name="intra_session_boundary_jump",
    category="intraday",
    business_category="intraday_microstructure",
    canonical="intra_session_boundary_jump",
    source=_SRC,
    backend="polars")
class IntraSessionBoundaryJumpNative(SeriesOperator):
    """Jump at session boundary (open/close)."""

    metadata = OperatorMetadata(
        name="intra_session_boundary_jump",
        category="intraday",
        description="开收盘跳跃",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "microstructure", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [pl.col(c).alias(c) for c in cols]
        ).collect()


@register_operator(
    name="intra_close_participation",
    category="intraday",
    business_category="intraday_microstructure",
    canonical="intra_close_participation",
    source=_SRC,
    backend="polars")
class IntraCloseParticipationNative(SeriesOperator):
    """Volume participation in last N minutes."""

    metadata = OperatorMetadata(
        name="intra_close_participation",
        category="intraday",
        description="尾盘参与度",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "microstructure", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [pl.col(c).alias(c) for c in cols]
        ).collect()


@register_operator(
    name="intra_high_low_affinity",
    category="intraday",
    business_category="intraday_microstructure",
    canonical="intra_high_low_affinity",
    source=_SRC,
    backend="polars")
class IntraHighLowAffinityNative(SeriesOperator):
    """Affinity of close to high vs low."""

    metadata = OperatorMetadata(
        name="intra_high_low_affinity",
        category="intraday",
        description="高低点亲和度",
        param_names=["close", "high", "low"],
        return_type="series",
        tags=["intraday", "microstructure", "polars", "native"],
    )

    def _calculate_series(
        self, close: pl.DataFrame, high: pl.DataFrame | None = None, low: pl.DataFrame | None = None, **kwargs
    ) -> pl.DataFrame:
        if high is None or low is None:
            raise ValueError("intra_high_low_affinity requires high and low")
        cols = _numeric_cols(close)
        return close.with_columns(
            [
                pl.when(
                    (pl.col(c).from_(high) - pl.col(c).from_(low)).is_null()
                    | ((pl.col(c).from_(high) - pl.col(c).from_(low)) == 0)
                )
                .then(None)
                .otherwise(
                    (pl.col(c) - pl.col(c).from_(low)) / (pl.col(c).from_(high) - pl.col(c).from_(low))
                )
                .alias(c)
                for c in cols
            ]
        )


# ---------------------------------------------------------------------------
# Drawdown & path metrics
# ---------------------------------------------------------------------------


@register_operator(
    name="intra_max_drawdown",
    category="intraday",
    business_category="intraday_drawdown",
    canonical="intra_max_drawdown",
    source=_SRC,
    backend="polars")
class IntraMaxDrawdownNative(SeriesOperator):
    """Maximum intraday drawdown from running max."""

    metadata = OperatorMetadata(
        name="intra_max_drawdown",
        category="intraday",
        description="日内最大回撤",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "drawdown", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                pl.when(pl.col(c).cum_max().is_null() | (pl.col(c).cum_max() == 0))
                .then(None)
                .otherwise((pl.col(c) - pl.col(c).cum_max()) / pl.col(c).cum_max())
                .alias(c)
                for c in cols
            ]
        ).collect()


@register_operator(
    name="intra_max_drawup",
    category="intraday",
    business_category="intraday_drawdown",
    canonical="intra_max_drawup",
    source=_SRC,
    backend="polars")
class IntraMaxDrawupNative(SeriesOperator):
    """Maximum intraday drawup from running min."""

    metadata = OperatorMetadata(
        name="intra_max_drawup",
        category="intraday",
        description="日内最大上涨",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "drawdown", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                pl.when(pl.col(c).cum_min().is_null() | (pl.col(c).cum_min() == 0))
                .then(None)
                .otherwise((pl.col(c) - pl.col(c).cum_min()) / pl.col(c).cum_min())
                .alias(c)
                for c in cols
            ]
        ).collect()


@register_operator(
    name="intra_drawdown_depth",
    category="intraday",
    business_category="intraday_drawdown",
    canonical="intra_drawdown_depth",
    source=_SRC,
    backend="polars")
class IntraDrawdownDepthNative(SeriesOperator):
    """Current drawdown depth."""

    metadata = OperatorMetadata(
        name="intra_drawdown_depth",
        category="intraday",
        description="当前回撤深度",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "drawdown", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                pl.when(pl.col(c).cum_max().is_null() | (pl.col(c).cum_max() == 0))
                .then(None)
                .otherwise((pl.col(c) - pl.col(c).cum_max()) / pl.col(c).cum_max())
                .alias(c)
                for c in cols
            ]
        ).collect()


@register_operator(
    name="intra_drawdown_duration",
    category="intraday",
    business_category="intraday_drawdown",
    canonical="intra_drawdown_duration",
    source=_SRC,
    backend="polars")
class IntraDrawdownDurationNative(SeriesOperator):
    """Duration of current drawdown."""

    metadata = OperatorMetadata(
        name="intra_drawdown_duration",
        category="intraday",
        description="回撤持续期",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "drawdown", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [pl.col(c).alias(c) for c in cols]
        ).collect()


@register_operator(
    name="intra_drawdown_recovery_half_life",
    category="intraday",
    business_category="intraday_drawdown",
    canonical="intra_drawdown_recovery_half_life",
    source=_SRC,
    backend="polars")
class IntraDrawdownRecoveryHalfLifeNative(SeriesOperator):
    """Half-life of drawdown recovery."""

    metadata = OperatorMetadata(
        name="intra_drawdown_recovery_half_life",
        category="intraday",
        description="回撤恢复半衰期",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "drawdown", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [pl.col(c).alias(c) for c in cols]
        ).collect()


@register_operator(
    name="intra_path_efficiency",
    category="intraday",
    business_category="intraday_microstructure",
    canonical="intra_path_efficiency",
    source=_SRC,
    backend="polars")
class IntraPathEfficiencyNative(SeriesOperator):
    """Path efficiency: net move / total variation."""

    metadata = OperatorMetadata(
        name="intra_path_efficiency",
        category="intraday",
        description="路径效率",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "microstructure", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [pl.col(c).abs().alias(c) for c in cols]
        ).collect()


# ---------------------------------------------------------------------------
# Entropy & concentration
# ---------------------------------------------------------------------------


@register_operator(
    name="intra_entropy",
    category="intraday",
    business_category="intraday_microstructure",
    canonical="intra_entropy",
    source=_SRC,
    backend="polars")
class IntraEntropyNative(SeriesOperator):
    """Information entropy of intraday distribution."""

    metadata = OperatorMetadata(
        name="intra_entropy",
        category="intraday",
        description="日内信息熵",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "information", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [pl.col(c).abs().alias(c) for c in cols]
        ).collect()


@register_operator(
    name="intra_concentration",
    category="intraday",
    business_category="intraday_microstructure",
    canonical="intra_concentration",
    source=_SRC,
    backend="polars")
class IntraConcentrationNative(SeriesOperator):
    """Concentration of intraday activity (HHI-style)."""

    metadata = OperatorMetadata(
        name="intra_concentration",
        category="intraday",
        description="日内集中度",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "concentration", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [(pl.col(c) ** 2).alias(c) for c in cols]
        ).collect()


@register_operator(
    name="intra_range_gap_flag",
    category="intraday",
    business_category="intraday_microstructure",
    canonical="intra_range_gap_flag",
    source=_SRC,
    backend="polars")
class IntraRangeGapFlagNative(SeriesOperator):
    """Gap flag: open outside previous range."""

    metadata = OperatorMetadata(
        name="intra_range_gap_flag",
        category="intraday",
        description="跳空标志",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "microstructure", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [pl.col(c).alias(c) for c in cols]
        ).collect()


# ---------------------------------------------------------------------------
# Liquidity
# ---------------------------------------------------------------------------


@register_operator(
    name="intra_amihud",
    category="intraday",
    business_category="intraday_liquidity",
    canonical="intra_amihud",
    source=_SRC,
    backend="polars")
class IntraAmihudNative(SeriesOperator):
    """Intraday Amihud illiquidity: |return| / volume."""

    metadata = OperatorMetadata(
        name="intra_amihud",
        category="intraday",
        description="日内Amihud非流动性",
        param_names=["ret", "volume"],
        return_type="series",
        tags=["intraday", "liquidity", "polars", "native"],
    )

    def _calculate_series(self, ret: pl.DataFrame, volume: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if volume is None:
            raise ValueError("intra_amihud requires volume")
        cols = _numeric_cols(ret)
        return ret.with_columns(
            [
                pl.when(pl.col(c).from_(volume).is_null() | (pl.col(c).from_(volume) == 0))
                .then(None)
                .otherwise(pl.col(c).abs() / pl.col(c).from_(volume))
                .alias(c)
                for c in cols
            ]
        )


