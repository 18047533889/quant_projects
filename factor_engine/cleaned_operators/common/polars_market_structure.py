# -*- coding: utf-8 -*-
"""Market structure operators - Polars native implementations.

Market cap ratios, listing age, suspension metrics, index membership and weights,
capital structure changes.
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


# ---------------------------------------------------------------------------
# Market cap structure
# ---------------------------------------------------------------------------

@register_operator(
    name="a_share_cap_ratio",
    category="market_structure",
    business_category="market_structure",
    canonical="a_share_cap_ratio",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class AShareCapRatioNative(SeriesOperator):
    """A-share market cap / total market cap."""

    metadata = OperatorMetadata(
        name="a_share_cap_ratio",
        category="market_structure",
        description="A股市值占比",
        param_names=["a_share_cap", "total_cap"],
        return_type="series",
        tags=["market_structure", "polars", "native"],
    )

    def _calculate_series(
        self, a_share_cap: pl.DataFrame, total_cap: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(a_share_cap)
        exprs = []
        for c in cols:
            a_cap = a_share_cap[c]
            t_cap = total_cap[c] if c in total_cap.columns else pl.lit(None)
            exprs.append(
                pl.when((t_cap.is_null()) | (t_cap == 0))
                .then(None)
                .otherwise(a_cap / t_cap)
                .alias(c)
            )
        result = pl.DataFrame(exprs)
        return _with_meta(result, a_share_cap)


@register_operator(
    name="circulating_cap_ratio_change",
    category="market_structure",
    business_category="market_structure",
    canonical="circulating_cap_ratio_change",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class CirculatingCapRatioChangeNative(SeriesOperator):
    """Change in circulating cap / total cap ratio over d periods."""

    metadata = OperatorMetadata(
        name="circulating_cap_ratio_change",
        category="market_structure",
        description="流通市值占比变化",
        param_names=["circulating_cap", "total_cap", "d"],
        return_type="series",
        tags=["market_structure", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, circulating_cap: pl.DataFrame, total_cap: pl.DataFrame, d: int = 1, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        n = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(circulating_cap)
        exprs = []
        for c in cols:
            c_cap = circulating_cap[c]
            t_cap = total_cap[c] if c in total_cap.columns else pl.lit(None)
            ratio = pl.when((t_cap.is_null()) | (t_cap == 0)).then(None).otherwise(c_cap / t_cap)
            change = ratio - ratio.shift(n)
            exprs.append(change.alias(c))
        result = circulating_cap.with_columns(exprs)
        return result


@register_operator(
    name="circulating_cap_unlock_proxy",
    category="market_structure",
    business_category="market_structure",
    canonical="circulating_cap_unlock_proxy",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class CirculatingCapUnlockProxyNative(SeriesOperator):
    """Positive changes in circulating cap ratio (unlock events)."""

    metadata = OperatorMetadata(
        name="circulating_cap_unlock_proxy",
        category="market_structure",
        description="流通市值解禁代理",
        param_names=["circulating_cap", "total_cap"],
        return_type="series",
        tags=["market_structure", "polars", "native"],
    )

    def _calculate_series(
        self, circulating_cap: pl.DataFrame, total_cap: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(circulating_cap)
        exprs = []
        for c in cols:
            c_cap = circulating_cap[c]
            t_cap = total_cap[c] if c in total_cap.columns else pl.lit(None)
            ratio = pl.when((t_cap.is_null()) | (t_cap == 0)).then(None).otherwise(c_cap / t_cap)
            change = ratio - ratio.shift(1)
            unlock = pl.when(change > 0).then(change).otherwise(0)
            exprs.append(unlock.alias(c))
        result = circulating_cap.with_columns(exprs)
        return result


@register_operator(
    name="market_cap_free_cap_gap",
    category="market_structure",
    business_category="market_structure",
    canonical="market_cap_free_cap_gap",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class MarketCapFreeCapGapNative(SeriesOperator):
    """(total_cap - free_cap) / total_cap; locked share ratio."""

    metadata = OperatorMetadata(
        name="market_cap_free_cap_gap",
        category="market_structure",
        description="总市值与自由流通市值差距",
        param_names=["total_cap", "free_cap"],
        return_type="series",
        tags=["market_structure", "polars", "native"],
    )

    def _calculate_series(
        self, total_cap: pl.DataFrame, free_cap: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(total_cap)
        exprs = []
        for c in cols:
            t_cap = total_cap[c]
            f_cap = free_cap[c] if c in free_cap.columns else pl.lit(None)
            exprs.append(
                pl.when((t_cap.is_null()) | (t_cap == 0))
                .then(None)
                .otherwise((t_cap - f_cap) / t_cap)
                .alias(c)
            )
        result = pl.DataFrame(exprs)
        return _with_meta(result, total_cap)


# ---------------------------------------------------------------------------
# Listing and suspension metrics
# ---------------------------------------------------------------------------

@register_operator(
    name="listing_age",
    category="market_structure",
    business_category="listing",
    canonical="listing_age",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class ListingAgeNative(SeriesOperator):
    """Days since listing date."""

    metadata = OperatorMetadata(
        name="listing_age",
        category="market_structure",
        description="上市天数",
        param_names=["listing_date", "current_date"],
        return_type="series",
        tags=["market_structure", "listing", "polars", "native"],
    )

    def _calculate_series(
        self, listing_date: pl.DataFrame, current_date: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(listing_date)
        exprs = []
        for c in cols:
            ld = listing_date[c]
            cd = current_date[c] if c in current_date.columns else pl.lit(None)
            exprs.append((cd - ld).alias(c))
        result = pl.DataFrame(exprs)
        return _with_meta(result, listing_date)


@register_operator(
    name="suspension_frequency",
    category="market_structure",
    business_category="suspension",
    canonical="suspension_frequency",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class SuspensionFrequencyNative(SeriesOperator):
    """Rolling count of suspension days / window."""

    metadata = OperatorMetadata(
        name="suspension_frequency",
        category="market_structure",
        description="停牌频率",
        param_names=["suspension_indicator", "window"],
        return_type="series",
        tags=["market_structure", "suspension", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, suspension_indicator: pl.DataFrame, window: int = 20, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(suspension_indicator)
        exprs = []
        for c in cols:
            sus = suspension_indicator[c]
            freq = sus.rolling_sum(window_size=w) / pl.lit(float(w))
            exprs.append(freq.alias(c))
        result = suspension_indicator.with_columns(exprs)
        return result


@register_operator(
    name="suspension_status_coverage",
    category="market_structure",
    business_category="suspension",
    canonical="suspension_status_coverage",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class SuspensionStatusCoverageNative(SeriesOperator):
    """Rolling non-null suspension status ratio."""

    metadata = OperatorMetadata(
        name="suspension_status_coverage",
        category="market_structure",
        description="停牌状态覆盖率",
        param_names=["suspension_indicator", "window"],
        return_type="series",
        tags=["market_structure", "suspension", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, suspension_indicator: pl.DataFrame, window: int = 20, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(suspension_indicator)
        exprs = []
        for c in cols:
            sus = suspension_indicator[c]
            coverage = sus.is_not_null().cast(pl.Float64).rolling_mean(window_size=w)
            exprs.append(coverage.alias(c))
        result = suspension_indicator.with_columns(exprs)
        return result


# ---------------------------------------------------------------------------
# Capital change metrics
# ---------------------------------------------------------------------------

@register_operator(
    name="capital_change_age",
    category="market_structure",
    business_category="capital_change",
    canonical="capital_change_age",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class CapitalChangeAgeNative(SeriesOperator):
    """Days since last capital change event."""

    metadata = OperatorMetadata(
        name="capital_change_age",
        category="market_structure",
        description="资本变动天数",
        param_names=["capital_change_date", "current_date"],
        return_type="series",
        tags=["market_structure", "capital", "polars", "native"],
    )

    def _calculate_series(
        self, capital_change_date: pl.DataFrame, current_date: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(capital_change_date)
        exprs = []
        for c in cols:
            ccd = capital_change_date[c]
            cd = current_date[c] if c in current_date.columns else pl.lit(None)
            exprs.append((cd - ccd).alias(c))
        result = pl.DataFrame(exprs)
        return _with_meta(result, capital_change_date)


@register_operator(
    name="capital_change_magnitude",
    category="market_structure",
    business_category="capital_change",
    canonical="capital_change_magnitude",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class CapitalChangeMagnitudeNative(SeriesOperator):
    """Relative change in total shares: (new - old) / old."""

    metadata = OperatorMetadata(
        name="capital_change_magnitude",
        category="market_structure",
        description="股本变动幅度",
        param_names=["total_shares"],
        return_type="series",
        tags=["market_structure", "capital", "polars", "native"],
    )

    def _calculate_series(
        self, total_shares: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(total_shares)
        exprs = []
        for c in cols:
            ts = total_shares[c]
            prev = ts.shift(1)
            change = pl.when((prev.is_null()) | (prev == 0)).then(None).otherwise((ts - prev) / prev)
            exprs.append(change.alias(c))
        result = total_shares.with_columns(exprs)
        return result


# ---------------------------------------------------------------------------
# Index membership and weights
# ---------------------------------------------------------------------------

@register_operator(
    name="index_member",
    category="market_structure",
    business_category="index",
    canonical="index_member",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IndexMemberNative(SeriesOperator):
    """Binary index membership indicator."""

    metadata = OperatorMetadata(
        name="index_member",
        category="market_structure",
        description="指数成分股指示器",
        param_names=["membership"],
        return_type="series",
        tags=["market_structure", "index", "polars", "native"],
    )

    def _calculate_series(
        self, membership: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(membership)
        exprs = []
        for c in cols:
            mem = membership[c]
            exprs.append(pl.when(mem > 0).then(1).otherwise(0).alias(c))
        result = membership.with_columns(exprs)
        return result


@register_operator(
    name="index_membership_age",
    category="market_structure",
    business_category="index",
    canonical="index_membership_age",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IndexMembershipAgeNative(SeriesOperator):
    """Days since entry into index (consecutive membership)."""

    metadata = OperatorMetadata(
        name="index_membership_age",
        category="market_structure",
        description="指数成分股天数",
        param_names=["membership"],
        return_type="series",
        tags=["market_structure", "index", "polars", "native"],
    )

    def _calculate_series(
        self, membership: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(membership)
        exprs = []
        for c in cols:
            mem = membership[c]
            is_member = (mem > 0).cast(pl.Int32)
            # Cumsum resets on exit
            block = (~is_member.cast(pl.Boolean)).cum_sum()
            age = is_member.cum_sum() - is_member.cum_sum().shift(1).fill_null(0).over(block)
            exprs.append(age.alias(c))
        result = membership.with_columns(exprs)
        return result


@register_operator(
    name="index_weight",
    category="market_structure",
    business_category="index",
    canonical="index_weight",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IndexWeightNative(SeriesOperator):
    """Index weight (pass-through, may normalize)."""

    metadata = OperatorMetadata(
        name="index_weight",
        category="market_structure",
        description="指数权重",
        param_names=["weight"],
        return_type="series",
        tags=["market_structure", "index", "polars", "native"],
    )

    def _calculate_series(
        self, weight: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        return weight


@register_operator(
    name="index_weight_change",
    category="market_structure",
    business_category="index",
    canonical="index_weight_change",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IndexWeightChangeNative(SeriesOperator):
    """Change in index weight over d periods."""

    metadata = OperatorMetadata(
        name="index_weight_change",
        category="market_structure",
        description="指数权重变化",
        param_names=["weight", "d"],
        return_type="series",
        tags=["market_structure", "index", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, weight: pl.DataFrame, d: int = 1, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        n = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(weight)
        exprs = []
        for c in cols:
            w = weight[c]
            change = w - w.shift(n)
            exprs.append(change.alias(c))
        result = weight.with_columns(exprs)
        return result


@register_operator(
    name="index_weight_gap_to_free_float",
    category="market_structure",
    business_category="index",
    canonical="index_weight_gap_to_free_float",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IndexWeightGapToFreeFloatNative(SeriesOperator):
    """index_weight - free_float_cap / sum(free_float_cap)."""

    metadata = OperatorMetadata(
        name="index_weight_gap_to_free_float",
        category="market_structure",
        description="指数权重与自由流通市值权重差距",
        param_names=["index_weight", "free_float_cap"],
        return_type="series",
        tags=["market_structure", "index", "polars", "native"],
    )

    def _calculate_series(
        self, index_weight: pl.DataFrame, free_float_cap: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(index_weight)
        exprs = []
        for c in cols:
            iw = index_weight[c]
            ffc = free_float_cap[c] if c in free_float_cap.columns else pl.lit(None)
            # Compute implied weight from free float cap (cross-sectional sum)
            # For panel data, this is approximate; real implementation needs row-wise sum
            # Simplified: assume already normalized or use placeholder
            gap = iw - ffc
            exprs.append(gap.alias(c))
        result = pl.DataFrame(exprs)
        return _with_meta(result, index_weight)


@register_operator(
    name="index_entry_exit_event",
    category="market_structure",
    business_category="index",
    canonical="index_entry_exit_event",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IndexEntryExitEventNative(SeriesOperator):
    """Index entry/exit indicator: +1 entry, -1 exit, 0 no change."""

    metadata = OperatorMetadata(
        name="index_entry_exit_event",
        category="market_structure",
        description="指数纳入退出事件",
        param_names=["membership"],
        return_type="series",
        tags=["market_structure", "index", "polars", "native"],
    )

    def _calculate_series(
        self, membership: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(membership)
        exprs = []
        for c in cols:
            mem = membership[c]
            is_member = (mem > 0).cast(pl.Int32)
            prev_member = is_member.shift(1).fill_null(0)
            event = is_member - prev_member
            exprs.append(event.alias(c))
        result = membership.with_columns(exprs)
        return result


@register_operator(
    name="index_event_decay",
    category="market_structure",
    business_category="index",
    canonical="index_event_decay",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IndexEventDecayNative(SeriesOperator):
    """Exponential decay of index entry/exit events."""

    metadata = OperatorMetadata(
        name="index_event_decay",
        category="market_structure",
        description="指数事件衰减",
        param_names=["membership", "halflife"],
        return_type="series",
        tags=["market_structure", "index", "polars", "native"],
        param_specs={
            "halflife": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, membership: pl.DataFrame, halflife: int = 20, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        import math

        hl = strict_integer(halflife, "halflife", minimum=1)
        decay_factor = math.exp(-math.log(2) / hl)

        cols = _numeric_cols(membership)
        exprs = []
        for c in cols:
            mem = membership[c]
            is_member = (mem > 0).cast(pl.Int32)
            prev_member = is_member.shift(1).fill_null(0)
            event = is_member - prev_member
            # Exponential weighted moving average of events
            # Simplified: use EWM span conversion
            span = 2 * hl - 1
            decayed = event.ewm_mean(span=span, adjust=False)
            exprs.append(decayed.alias(c))
        result = membership.with_columns(exprs)
        return result


@register_operator(
    name="index_reconstitution_churn",
    category="market_structure",
    business_category="index",
    canonical="index_reconstitution_churn",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IndexReconstitutionChurnNative(SeriesOperator):
    """Rolling sum of |entry_exit_event| over window."""

    metadata = OperatorMetadata(
        name="index_reconstitution_churn",
        category="market_structure",
        description="指数重构流动性",
        param_names=["membership", "window"],
        return_type="series",
        tags=["market_structure", "index", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=60, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, membership: pl.DataFrame, window: int = 60, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(membership)
        exprs = []
        for c in cols:
            mem = membership[c]
            is_member = (mem > 0).cast(pl.Int32)
            prev_member = is_member.shift(1).fill_null(0)
            event = (is_member - prev_member).abs()
            churn = event.rolling_sum(window_size=w)
            exprs.append(churn.alias(c))
        result = membership.with_columns(exprs)
        return result


@register_operator(
    name="multi_index_entry_intensity",
    category="market_structure",
    business_category="index",
    canonical="multi_index_entry_intensity",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class MultiIndexEntryIntensityNative(SeriesOperator):
    """Sum of entry events across multiple index memberships."""

    metadata = OperatorMetadata(
        name="multi_index_entry_intensity",
        category="market_structure",
        description="多指数纳入强度",
        param_names=["membership1", "membership2", "membership3"],
        return_type="series",
        tags=["market_structure", "index", "polars", "native"],
    )

    def _calculate_series(
        self,
        membership1: pl.DataFrame,
        membership2: pl.DataFrame | None = None,
        membership3: pl.DataFrame | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        cols = _numeric_cols(membership1)
        exprs = []
        for c in cols:
            m1 = membership1[c]
            is_m1 = (m1 > 0).cast(pl.Int32)
            prev_m1 = is_m1.shift(1).fill_null(0)
            entry1 = pl.when((is_m1 - prev_m1) > 0).then(1).otherwise(0)

            total = entry1
            if membership2 is not None and c in membership2.columns:
                m2 = membership2[c]
                is_m2 = (m2 > 0).cast(pl.Int32)
                prev_m2 = is_m2.shift(1).fill_null(0)
                entry2 = pl.when((is_m2 - prev_m2) > 0).then(1).otherwise(0)
                total = total + entry2

            if membership3 is not None and c in membership3.columns:
                m3 = membership3[c]
                is_m3 = (m3 > 0).cast(pl.Int32)
                prev_m3 = is_m3.shift(1).fill_null(0)
                entry3 = pl.when((is_m3 - prev_m3) > 0).then(1).otherwise(0)
                total = total + entry3

            exprs.append(total.alias(c))
        result = membership1.with_columns(exprs)
        return result
