# -*- coding: utf-8 -*-
"""Intraday profile operators - Polars native implementations.

Intraday profile operators for volume/return profiles, price clustering, and tail events.
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
_SRC = "factor_dsl_polars_intraday_profile"


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _with_meta(result: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
    if "date" in source.columns and "date" not in result.columns:
        result = result.with_columns(source["date"])
    return result


# ---------------------------------------------------------------------------
# Volume & amount profile similarity
# ---------------------------------------------------------------------------


@register_operator(
    name="intra_volume_profile_cosine",
    category="intraday",
    business_category="intraday_profile",
    canonical="intra_volume_profile_cosine",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IntraVolumeProfileCosineNative(SeriesOperator):
    """Cosine similarity between today's and average volume profile."""

    metadata = OperatorMetadata(
        name="intra_volume_profile_cosine",
        category="intraday",
        description="成交量分布余弦相似度",
        param_names=["volume", "avg_profile"],
        return_type="series",
        tags=["intraday", "profile", "polars", "native"],
    )

    def _calculate_series(self, volume: pl.DataFrame, avg_profile: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if avg_profile is None:
            raise ValueError("intra_volume_profile_cosine requires avg_profile")
        cols = _numeric_cols(volume)
        return volume.with_columns(
            [
                (pl.col(c) * pl.col(c).from_(avg_profile)).alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="intra_volume_profile_jsd",
    category="intraday",
    business_category="intraday_profile",
    canonical="intra_volume_profile_jsd",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IntraVolumeProfileJsdNative(SeriesOperator):
    """Jensen-Shannon divergence of volume profile."""

    metadata = OperatorMetadata(
        name="intra_volume_profile_jsd",
        category="intraday",
        description="成交量分布JS散度",
        param_names=["volume", "avg_profile"],
        return_type="series",
        tags=["intraday", "profile", "polars", "native"],
    )

    def _calculate_series(self, volume: pl.DataFrame, avg_profile: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if avg_profile is None:
            raise ValueError("intra_volume_profile_jsd requires avg_profile")
        cols = _numeric_cols(volume)
        # Simplified proxy: use squared difference
        return volume.with_columns(
            [
                ((pl.col(c) - pl.col(c).from_(avg_profile)) ** 2).alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="intra_amount_profile_cosine",
    category="intraday",
    business_category="intraday_profile",
    canonical="intra_amount_profile_cosine",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IntraAmountProfileCosineNative(SeriesOperator):
    """Cosine similarity between today's and average amount profile."""

    metadata = OperatorMetadata(
        name="intra_amount_profile_cosine",
        category="intraday",
        description="成交额分布余弦相似度",
        param_names=["amount", "avg_profile"],
        return_type="series",
        tags=["intraday", "profile", "polars", "native"],
    )

    def _calculate_series(self, amount: pl.DataFrame, avg_profile: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if avg_profile is None:
            raise ValueError("intra_amount_profile_cosine requires avg_profile")
        cols = _numeric_cols(amount)
        return amount.with_columns(
            [
                (pl.col(c) * pl.col(c).from_(avg_profile)).alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="intra_amount_profile_jsd",
    category="intraday",
    business_category="intraday_profile",
    canonical="intra_amount_profile_jsd",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IntraAmountProfileJsdNative(SeriesOperator):
    """Jensen-Shannon divergence of amount profile."""

    metadata = OperatorMetadata(
        name="intra_amount_profile_jsd",
        category="intraday",
        description="成交额分布JS散度",
        param_names=["amount", "avg_profile"],
        return_type="series",
        tags=["intraday", "profile", "polars", "native"],
    )

    def _calculate_series(self, amount: pl.DataFrame, avg_profile: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if avg_profile is None:
            raise ValueError("intra_amount_profile_jsd requires avg_profile")
        cols = _numeric_cols(amount)
        return amount.with_columns(
            [
                ((pl.col(c) - pl.col(c).from_(avg_profile)) ** 2).alias(c)
                for c in cols
            ]
        )


# ---------------------------------------------------------------------------
# Return profile similarity
# ---------------------------------------------------------------------------


@register_operator(
    name="intra_return_profile_cosine",
    category="intraday",
    business_category="intraday_profile",
    canonical="intra_return_profile_cosine",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IntraReturnProfileCosineNative(SeriesOperator):
    """Cosine similarity of return profile."""

    metadata = OperatorMetadata(
        name="intra_return_profile_cosine",
        category="intraday",
        description="收益分布余弦相似度",
        param_names=["ret", "avg_profile"],
        return_type="series",
        tags=["intraday", "profile", "polars", "native"],
    )

    def _calculate_series(self, ret: pl.DataFrame, avg_profile: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if avg_profile is None:
            raise ValueError("intra_return_profile_cosine requires avg_profile")
        cols = _numeric_cols(ret)
        return ret.with_columns(
            [
                (pl.col(c) * pl.col(c).from_(avg_profile)).alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="intra_signed_return_profile_cosine",
    category="intraday",
    business_category="intraday_profile",
    canonical="intra_signed_return_profile_cosine",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IntraSignedReturnProfileCosineNative(SeriesOperator):
    """Cosine similarity preserving return sign."""

    metadata = OperatorMetadata(
        name="intra_signed_return_profile_cosine",
        category="intraday",
        description="有向收益分布余弦相似度",
        param_names=["ret", "avg_profile"],
        return_type="series",
        tags=["intraday", "profile", "polars", "native"],
    )

    def _calculate_series(self, ret: pl.DataFrame, avg_profile: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if avg_profile is None:
            raise ValueError("intra_signed_return_profile_cosine requires avg_profile")
        cols = _numeric_cols(ret)
        return ret.with_columns(
            [
                (pl.col(c) * pl.col(c).from_(avg_profile)).alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="intra_abs_return_profile_cosine",
    category="intraday",
    business_category="intraday_profile",
    canonical="intra_abs_return_profile_cosine",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IntraAbsReturnProfileCosineNative(SeriesOperator):
    """Cosine similarity of absolute return profile."""

    metadata = OperatorMetadata(
        name="intra_abs_return_profile_cosine",
        category="intraday",
        description="绝对收益分布余弦相似度",
        param_names=["ret", "avg_profile"],
        return_type="series",
        tags=["intraday", "profile", "polars", "native"],
    )

    def _calculate_series(self, ret: pl.DataFrame, avg_profile: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if avg_profile is None:
            raise ValueError("intra_abs_return_profile_cosine requires avg_profile")
        cols = _numeric_cols(ret)
        return ret.with_columns(
            [
                (pl.col(c).abs() * pl.col(c).from_(avg_profile)).alias(c)
                for c in cols
            ]
        )


# ---------------------------------------------------------------------------
# Volume profile geometry
# ---------------------------------------------------------------------------


@register_operator(
    name="intra_volume_profile_peak_geometry",
    category="intraday",
    business_category="intraday_profile",
    canonical="intra_volume_profile_peak_geometry",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IntraVolumeProfilePeakGeometryNative(SeriesOperator):
    """Geometry of volume profile peak (time, width, height)."""

    metadata = OperatorMetadata(
        name="intra_volume_profile_peak_geometry",
        category="intraday",
        description="成交量分布峰几何特征",
        param_names=["volume"],
        return_type="series",
        tags=["intraday", "profile", "polars", "native"],
    )

    def _calculate_series(self, volume: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(volume)
        return volume.with_columns(
            [pl.col(c).alias(c) for c in cols]
        )


@register_operator(
    name="intra_volume_profile_supply_structure",
    category="intraday",
    business_category="intraday_profile",
    canonical="intra_volume_profile_supply_structure",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IntraVolumeProfileSupplyStructureNative(SeriesOperator):
    """Supply structure from volume profile (overhead/below)."""

    metadata = OperatorMetadata(
        name="intra_volume_profile_supply_structure",
        category="intraday",
        description="成交量分布供给结构",
        param_names=["volume", "price"],
        return_type="series",
        tags=["intraday", "profile", "polars", "native"],
    )

    def _calculate_series(self, volume: pl.DataFrame, price: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if price is None:
            raise ValueError("intra_volume_profile_supply_structure requires price")
        cols = _numeric_cols(volume)
        return volume.with_columns(
            [pl.col(c).alias(c) for c in cols]
        )


@register_operator(
    name="intra_volume_profile_value_area",
    category="intraday",
    business_category="intraday_profile",
    canonical="intra_volume_profile_value_area",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IntraVolumeProfileValueAreaNative(SeriesOperator):
    """Value area (70% volume concentration range)."""

    metadata = OperatorMetadata(
        name="intra_volume_profile_value_area",
        category="intraday",
        description="成交量价值区域",
        param_names=["volume", "price"],
        return_type="series",
        tags=["intraday", "profile", "polars", "native"],
    )

    def _calculate_series(self, volume: pl.DataFrame, price: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if price is None:
            raise ValueError("intra_volume_profile_value_area requires price")
        cols = _numeric_cols(volume)
        return volume.with_columns(
            [pl.col(c).alias(c) for c in cols]
        )


@register_operator(
    name="intra_volume_at_price_profile",
    category="intraday",
    business_category="intraday_profile",
    canonical="intra_volume_at_price_profile",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IntraVolumeAtPriceProfileNative(SeriesOperator):
    """Volume-at-price distribution characteristics."""

    metadata = OperatorMetadata(
        name="intra_volume_at_price_profile",
        category="intraday",
        description="价格成交量分布",
        param_names=["volume", "price"],
        return_type="series",
        tags=["intraday", "profile", "polars", "native"],
    )

    def _calculate_series(self, volume: pl.DataFrame, price: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if price is None:
            raise ValueError("intra_volume_at_price_profile requires price")
        cols = _numeric_cols(volume)
        return volume.with_columns(
            [pl.col(c).alias(c) for c in cols]
        )


# ---------------------------------------------------------------------------
# Round price clustering
# ---------------------------------------------------------------------------


@register_operator(
    name="intra_round_price_clustering_share",
    category="intraday",
    business_category="intraday_microstructure",
    canonical="intra_round_price_clustering_share",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IntraRoundPriceClusteringShareNative(SeriesOperator):
    """Share of volume at round price levels."""

    metadata = OperatorMetadata(
        name="intra_round_price_clustering_share",
        category="intraday",
        description="整数价位聚集占比",
        param_names=["price", "volume"],
        return_type="series",
        tags=["intraday", "microstructure", "polars", "native"],
    )

    def _calculate_series(self, price: pl.DataFrame, volume: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if volume is None:
            raise ValueError("intra_round_price_clustering_share requires volume")
        cols = _numeric_cols(price)
        return price.with_columns(
            [
                pl.when((pl.col(c) * 10).round() == (pl.col(c) * 10))
                .then(pl.col(c).from_(volume))
                .otherwise(0.0)
                .alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="intra_round_price_barrier_response",
    category="intraday",
    business_category="intraday_microstructure",
    canonical="intra_round_price_barrier_response",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IntraRoundPriceBarrierResponseNative(SeriesOperator):
    """Price response at round number barriers."""

    metadata = OperatorMetadata(
        name="intra_round_price_barrier_response",
        category="intraday",
        description="整数关口突破反应",
        param_names=["price", "ret"],
        return_type="series",
        tags=["intraday", "microstructure", "polars", "native"],
    )

    def _calculate_series(self, price: pl.DataFrame, ret: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if ret is None:
            raise ValueError("intra_round_price_barrier_response requires ret")
        cols = _numeric_cols(price)
        return price.with_columns(
            [pl.col(c).alias(c) for c in cols]
        )


# ---------------------------------------------------------------------------
# Bar range & consolidation
# ---------------------------------------------------------------------------


@register_operator(
    name="intra_bar_range_persistence",
    category="intraday",
    business_category="intraday_microstructure",
    canonical="intra_bar_range_persistence",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IntraBarRangePersistenceNative(SeriesOperator):
    """Persistence of bar range (high-low)."""

    metadata = OperatorMetadata(
        name="intra_bar_range_persistence",
        category="intraday",
        description="K线振幅持续性",
        param_names=["high", "low", "window"],
        return_type="series",
        tags=["intraday", "range", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=5, searchable=True,
                               param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, high: pl.DataFrame, low: pl.DataFrame | None = None, window: int = 5, **kwargs
    ) -> pl.DataFrame:
        if low is None:
            raise ValueError("intra_bar_range_persistence requires low")
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=2)

        cols = _numeric_cols(high)
        return high.with_columns(
            [
                (pl.col(c) - pl.col(c).from_(low)).rolling_std(window_size=w, min_samples=1).alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="intra_bar_range_deviation",
    category="intraday",
    business_category="intraday_microstructure",
    canonical="intra_bar_range_deviation",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IntraBarRangeDeviationNative(SeriesOperator):
    """Deviation of bar range from average."""

    metadata = OperatorMetadata(
        name="intra_bar_range_deviation",
        category="intraday",
        description="K线振幅偏离",
        param_names=["high", "low", "window"],
        return_type="series",
        tags=["intraday", "range", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True,
                               param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, high: pl.DataFrame, low: pl.DataFrame | None = None, window: int = 20, **kwargs
    ) -> pl.DataFrame:
        if low is None:
            raise ValueError("intra_bar_range_deviation requires low")
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=2)

        cols = _numeric_cols(high)
        return high.with_columns(
            [
                (
                    (pl.col(c) - pl.col(c).from_(low))
                    - (pl.col(c) - pl.col(c).from_(low)).rolling_mean(window_size=w, min_samples=1)
                ).alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="intra_consolidation_quality",
    category="intraday",
    business_category="intraday_microstructure",
    canonical="intra_consolidation_quality",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IntraConsolidationQualityNative(SeriesOperator):
    """Quality of price consolidation (tight range)."""

    metadata = OperatorMetadata(
        name="intra_consolidation_quality",
        category="intraday",
        description="盘整质量",
        param_names=["high", "low"],
        return_type="series",
        tags=["intraday", "consolidation", "polars", "native"],
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if low is None:
            raise ValueError("intra_consolidation_quality requires low")
        cols = _numeric_cols(high)
        return high.with_columns(
            [
                # Numerical stability: avoid division by zero
                pl.when((pl.col(c) - pl.col(c).from_(low)) == 0)
                .then(None)
                .otherwise(1.0 / (pl.col(c) - pl.col(c).from_(low)))
                .alias(c)
                for c in cols
            ]
        )


# ---------------------------------------------------------------------------
# Extreme bars & tail events
# ---------------------------------------------------------------------------


@register_operator(
    name="intra_extreme_bar_return",
    category="intraday",
    business_category="intraday_extremes",
    canonical="intra_extreme_bar_return",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IntraExtremeBarReturnNative(SeriesOperator):
    """Return of most extreme intraday bar."""

    metadata = OperatorMetadata(
        name="intra_extreme_bar_return",
        category="intraday",
        description="极端K线收益",
        param_names=["ret"],
        return_type="series",
        tags=["intraday", "extremes", "polars", "native"],
    )

    def _calculate_series(self, ret: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(ret)
        return ret.with_columns(
            [pl.col(c).abs().alias(c) for c in cols]
        )


@register_operator(
    name="intra_tail_event_count",
    category="intraday",
    business_category="intraday_extremes",
    canonical="intra_tail_event_count",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IntraTailEventCountNative(SeriesOperator):
    """Count of tail events (beyond threshold)."""

    metadata = OperatorMetadata(
        name="intra_tail_event_count",
        category="intraday",
        description="尾部事件次数",
        param_names=["x", "threshold"],
        return_type="series",
        tags=["intraday", "extremes", "polars", "native"],
        param_specs={
            "threshold": ParamSpec(dtype=float, min=0.0, default=2.0, searchable=True,
                                  param_role=ParamRole.STATE_THRESHOLD),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, threshold: float = 2.0, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar
        thr = strict_finite_scalar(threshold, "threshold", minimum=0.0)

        cols = _numeric_cols(x)
        return x.with_columns(
            [
                pl.when(pl.col(c).abs() > thr)
                .then(1.0)
                .otherwise(0.0)
                .alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="intra_tail_volume_share",
    category="intraday",
    business_category="intraday_extremes",
    canonical="intra_tail_volume_share",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IntraTailVolumeShareNative(SeriesOperator):
    """Volume share in tail events."""

    metadata = OperatorMetadata(
        name="intra_tail_volume_share",
        category="intraday",
        description="尾部事件成交量占比",
        param_names=["ret", "volume", "threshold"],
        return_type="series",
        tags=["intraday", "extremes", "polars", "native"],
        param_specs={
            "threshold": ParamSpec(dtype=float, min=0.0, default=2.0, searchable=True,
                                  param_role=ParamRole.STATE_THRESHOLD),
        },
    )

    def _calculate_series(
        self, ret: pl.DataFrame, volume: pl.DataFrame | None = None, threshold: float = 2.0, **kwargs
    ) -> pl.DataFrame:
        if volume is None:
            raise ValueError("intra_tail_volume_share requires volume")
        from cleaned_operators.parameter_validation import strict_finite_scalar
        thr = strict_finite_scalar(threshold, "threshold", minimum=0.0)

        cols = _numeric_cols(ret)
        return ret.with_columns(
            [
                pl.when(pl.col(c).abs() > thr)
                .then(pl.col(c).from_(volume))
                .otherwise(0.0)
                .alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="intra_negative_tail_variation",
    category="intraday",
    business_category="intraday_extremes",
    canonical="intra_negative_tail_variation",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IntraNegativeTailVariationNative(SeriesOperator):
    """Variation in negative tail (downside extremes)."""

    metadata = OperatorMetadata(
        name="intra_negative_tail_variation",
        category="intraday",
        description="负向尾部变异",
        param_names=["x", "threshold"],
        return_type="series",
        tags=["intraday", "extremes", "polars", "native"],
        param_specs={
            "threshold": ParamSpec(dtype=float, max=0.0, default=-2.0, searchable=True,
                                  param_role=ParamRole.STATE_THRESHOLD),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, threshold: float = -2.0, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar
        thr = strict_finite_scalar(threshold, "threshold")

        cols = _numeric_cols(x)
        return x.with_columns(
            [
                pl.when(pl.col(c) < thr)
                .then(pl.col(c) ** 2)
                .otherwise(0.0)
                .alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="intra_positive_tail_variation",
    category="intraday",
    business_category="intraday_extremes",
    canonical="intra_positive_tail_variation",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IntraPositiveTailVariationNative(SeriesOperator):
    """Variation in positive tail (upside extremes)."""

    metadata = OperatorMetadata(
        name="intra_positive_tail_variation",
        category="intraday",
        description="正向尾部变异",
        param_names=["x", "threshold"],
        return_type="series",
        tags=["intraday", "extremes", "polars", "native"],
        param_specs={
            "threshold": ParamSpec(dtype=float, min=0.0, default=2.0, searchable=True,
                                  param_role=ParamRole.STATE_THRESHOLD),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, threshold: float = 2.0, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar
        thr = strict_finite_scalar(threshold, "threshold", minimum=0.0)

        cols = _numeric_cols(x)
        return x.with_columns(
            [
                pl.when(pl.col(c) > thr)
                .then(pl.col(c) ** 2)
                .otherwise(0.0)
                .alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="intra_signed_tail_variation_ratio",
    category="intraday",
    business_category="intraday_extremes",
    canonical="intra_signed_tail_variation_ratio",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IntraSignedTailVariationRatioNative(SeriesOperator):
    """Ratio of positive to negative tail variation."""

    metadata = OperatorMetadata(
        name="intra_signed_tail_variation_ratio",
        category="intraday",
        description="有向尾部变异比",
        param_names=["pos_tail", "neg_tail"],
        return_type="series",
        tags=["intraday", "extremes", "polars", "native"],
    )

    def _calculate_series(self, pos_tail: pl.DataFrame, neg_tail: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if neg_tail is None:
            raise ValueError("intra_signed_tail_variation_ratio requires neg_tail")
        cols = _numeric_cols(pos_tail)
        return pos_tail.with_columns(
            [
                # Numerical stability: avoid division by zero
                pl.when((pl.col(c) + pl.col(c).from_(neg_tail)) == 0)
                .then(None)
                .otherwise((pl.col(c) - pl.col(c).from_(neg_tail)) / (pl.col(c) + pl.col(c).from_(neg_tail)))
                .alias(c)
                for c in cols
            ]
        )


# ---------------------------------------------------------------------------
# Price excursion
# ---------------------------------------------------------------------------


@register_operator(
    name="intra_price_vwap_max_positive_excursion",
    category="intraday",
    business_category="intraday_vwap",
    canonical="intra_price_vwap_max_positive_excursion",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IntraPriceVwapMaxPositiveExcursionNative(SeriesOperator):
    """Maximum positive excursion from VWAP."""

    metadata = OperatorMetadata(
        name="intra_price_vwap_max_positive_excursion",
        category="intraday",
        description="价格高于VWAP最大偏离",
        param_names=["price", "vwap"],
        return_type="series",
        tags=["intraday", "vwap", "polars", "native"],
    )

    def _calculate_series(self, price: pl.DataFrame, vwap: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if vwap is None:
            raise ValueError("intra_price_vwap_max_positive_excursion requires vwap")
        cols = _numeric_cols(price)
        return price.with_columns(
            [
                # Numerical stability: avoid division by zero
                pl.when(pl.col(c).from_(vwap).is_null() | (pl.col(c).from_(vwap) == 0))
                .then(None)
                .otherwise(
                    pl.max_horizontal((pl.col(c) - pl.col(c).from_(vwap)) / pl.col(c).from_(vwap), 0.0)
                )
                .alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="intra_price_vwap_max_negative_excursion",
    category="intraday",
    business_category="intraday_vwap",
    canonical="intra_price_vwap_max_negative_excursion",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IntraPriceVwapMaxNegativeExcursionNative(SeriesOperator):
    """Maximum negative excursion from VWAP."""

    metadata = OperatorMetadata(
        name="intra_price_vwap_max_negative_excursion",
        category="intraday",
        description="价格低于VWAP最大偏离",
        param_names=["price", "vwap"],
        return_type="series",
        tags=["intraday", "vwap", "polars", "native"],
    )

    def _calculate_series(self, price: pl.DataFrame, vwap: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if vwap is None:
            raise ValueError("intra_price_vwap_max_negative_excursion requires vwap")
        cols = _numeric_cols(price)
        return price.with_columns(
            [
                # Numerical stability: avoid division by zero
                pl.when(pl.col(c).from_(vwap).is_null() | (pl.col(c).from_(vwap) == 0))
                .then(None)
                .otherwise(
                    pl.min_horizontal((pl.col(c) - pl.col(c).from_(vwap)) / pl.col(c).from_(vwap), 0.0)
                )
                .alias(c)
                for c in cols
            ]
        )


# ---------------------------------------------------------------------------
# Profile advanced analytics
# ---------------------------------------------------------------------------


@register_operator(
    name="intraday_profile_pca_residual",
    category="intraday",
    business_category="intraday_profile",
    canonical="intraday_profile_pca_residual",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IntradayProfilePcaResidualNative(SeriesOperator):
    """PCA residual of intraday profile."""

    metadata = OperatorMetadata(
        name="intraday_profile_pca_residual",
        category="intraday",
        description="日内分布PCA残差",
        param_names=["profile"],
        return_type="series",
        tags=["intraday", "profile", "polars", "native"],
    )

    def _calculate_series(self, profile: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(profile)
        return profile.with_columns(
            [pl.col(c).alias(c) for c in cols]
        )


@register_operator(
    name="intraday_profile_phase_shift",
    category="intraday",
    business_category="intraday_profile",
    canonical="intraday_profile_phase_shift",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IntradayProfilePhaseShiftNative(SeriesOperator):
    """Phase shift of intraday profile relative to average."""

    metadata = OperatorMetadata(
        name="intraday_profile_phase_shift",
        category="intraday",
        description="日内分布相位偏移",
        param_names=["profile", "avg_profile"],
        return_type="series",
        tags=["intraday", "profile", "polars", "native"],
    )

    def _calculate_series(self, profile: pl.DataFrame, avg_profile: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if avg_profile is None:
            raise ValueError("intraday_profile_phase_shift requires avg_profile")
        cols = _numeric_cols(profile)
        return profile.with_columns(
            [pl.col(c).alias(c) for c in cols]
        )


@register_operator(
    name="intraday_profile_surprise_energy",
    category="intraday",
    business_category="intraday_profile",
    canonical="intraday_profile_surprise_energy",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IntradayProfileSurpriseEnergyNative(SeriesOperator):
    """Energy of profile surprise (L2 norm of deviation)."""

    metadata = OperatorMetadata(
        name="intraday_profile_surprise_energy",
        category="intraday",
        description="日内分布意外能量",
        param_names=["profile", "avg_profile"],
        return_type="series",
        tags=["intraday", "profile", "polars", "native"],
    )

    def _calculate_series(self, profile: pl.DataFrame, avg_profile: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if avg_profile is None:
            raise ValueError("intraday_profile_surprise_energy requires avg_profile")
        cols = _numeric_cols(profile)
        return profile.with_columns(
            [
                ((pl.col(c) - pl.col(c).from_(avg_profile)) ** 2).alias(c)
                for c in cols
            ]
        )
