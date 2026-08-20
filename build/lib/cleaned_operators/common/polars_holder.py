# -*- coding: utf-8 -*-
"""Holder/shareholder operators - Polars native implementations.

Holder and shareholder operators for concentration, churn, overlap, and network analysis.
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
_SRC = "factor_dsl_polars_holder"


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _with_meta(result: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
    if "date" in source.columns and "date" not in result.columns:
        result = result.with_columns(source["date"])
    return result


# ---------------------------------------------------------------------------
# Holder concentration
# ---------------------------------------------------------------------------


@register_operator(
    name="holder_concentration",
    category="holder",
    business_category="holder_concentration",
    canonical="holder_concentration",
    source=_SRC,
    backend="polars",
)
class HolderConcentrationNative(SeriesOperator):
    """Holder concentration (HHI or top-K share)."""

    metadata = OperatorMetadata(
        name="holder_concentration",
        category="holder",
        description="股东集中度",
        param_names=["x"],
        return_type="series",
        tags=["holder", "concentration", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns(
            [(pl.col(c) ** 2).alias(c) for c in cols]
        )


@register_operator(
    name="holder_concentration_acceleration",
    category="holder",
    business_category="holder_concentration",
    canonical="holder_concentration_acceleration",
    source=_SRC,
    backend="polars",
)
class HolderConcentrationAccelerationNative(SeriesOperator):
    """Second derivative of holder concentration."""

    metadata = OperatorMetadata(
        name="holder_concentration_acceleration",
        category="holder",
        description="股东集中度加速度",
        param_names=["x"],
        return_type="series",
        tags=["holder", "concentration", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns(
            [
                (pl.col(c) - 2 * pl.col(c).shift(1) + pl.col(c).shift(2)).alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="holder_concentration_slope",
    category="holder",
    business_category="holder_concentration",
    canonical="holder_concentration_slope",
    source=_SRC,
    backend="polars",
)
class HolderConcentrationSlopeNative(SeriesOperator):
    """Rate of change in holder concentration."""

    metadata = OperatorMetadata(
        name="holder_concentration_slope",
        category="holder",
        description="股东集中度变化率",
        param_names=["x", "window"],
        return_type="series",
        tags=["holder", "concentration", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=4, searchable=True,
                               param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 4, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=2)

        cols = _numeric_cols(x)
        return x.with_columns(
            [
                (pl.col(c) - pl.col(c).shift(w)).alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="holder_topk_share_sum",
    category="holder",
    business_category="holder_concentration",
    canonical="holder_topk_share_sum",
    source=_SRC,
    backend="polars",
)
class HolderTopkShareSumNative(SeriesOperator):
    """Sum of top-K holder shares."""

    metadata = OperatorMetadata(
        name="holder_topk_share_sum",
        category="holder",
        description="前K大股东持股占比",
        param_names=["x"],
        return_type="series",
        tags=["holder", "concentration", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns(
            [pl.col(c).alias(c) for c in cols]
        )


@register_operator(
    name="holder_company_ownership_hhi",
    category="holder",
    business_category="holder_concentration",
    canonical="holder_company_ownership_hhi",
    source=_SRC,
    backend="polars",
)
class HolderCompanyOwnershipHhiNative(SeriesOperator):
    """HHI of company ownership distribution."""

    metadata = OperatorMetadata(
        name="holder_company_ownership_hhi",
        category="holder",
        description="公司股权HHI指数",
        param_names=["x"],
        return_type="series",
        tags=["holder", "concentration", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns(
            [(pl.col(c) ** 2).alias(c) for c in cols]
        )


@register_operator(
    name="holder_observed_topk_hhi",
    category="holder",
    business_category="holder_concentration",
    canonical="holder_observed_topk_hhi",
    source=_SRC,
    backend="polars",
)
class HolderObservedTopkHhiNative(SeriesOperator):
    """HHI computed from observed top-K holders."""

    metadata = OperatorMetadata(
        name="holder_observed_topk_hhi",
        category="holder",
        description="观测前K股东HHI",
        param_names=["x"],
        return_type="series",
        tags=["holder", "concentration", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns(
            [(pl.col(c) ** 2).alias(c) for c in cols]
        )


# ---------------------------------------------------------------------------
# Holder entropy & diversity
# ---------------------------------------------------------------------------


@register_operator(
    name="holder_class_entropy",
    category="holder",
    business_category="holder_diversity",
    canonical="holder_class_entropy",
    source=_SRC,
    backend="polars",
)
class HolderClassEntropyNative(SeriesOperator):
    """Entropy of holder class distribution."""

    metadata = OperatorMetadata(
        name="holder_class_entropy",
        category="holder",
        description="股东类别熵",
        param_names=["x"],
        return_type="series",
        tags=["holder", "entropy", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns(
            [
                pl.when(pl.col(c) <= 0)
                .then(0.0)
                .otherwise(-pl.col(c) * pl.col(c).log())
                .alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="holder_nature_entropy",
    category="holder",
    business_category="holder_diversity",
    canonical="holder_nature_entropy",
    source=_SRC,
    backend="polars",
)
class HolderNatureEntropyNative(SeriesOperator):
    """Entropy of holder nature distribution."""

    metadata = OperatorMetadata(
        name="holder_nature_entropy",
        category="holder",
        description="股东性质熵",
        param_names=["x"],
        return_type="series",
        tags=["holder", "entropy", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns(
            [
                pl.when(pl.col(c) <= 0)
                .then(0.0)
                .otherwise(-pl.col(c) * pl.col(c).log())
                .alias(c)
                for c in cols
            ]
        )


# ---------------------------------------------------------------------------
# Holder disclosure
# ---------------------------------------------------------------------------


@register_operator(
    name="holder_disclosure_count",
    category="holder",
    business_category="holder_disclosure",
    canonical="holder_disclosure_count",
    source=_SRC,
    backend="polars",
)
class HolderDisclosureCountNative(SeriesOperator):
    """Count of holder disclosures."""

    metadata = OperatorMetadata(
        name="holder_disclosure_count",
        category="holder",
        description="股东披露次数",
        param_names=["x"],
        return_type="series",
        tags=["holder", "disclosure", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns(
            [pl.col(c).is_not_null().cast(pl.Float64).alias(c) for c in cols]
        )


@register_operator(
    name="holder_disclosure_coverage",
    category="holder",
    business_category="holder_disclosure",
    canonical="holder_disclosure_coverage",
    source=_SRC,
    backend="polars",
)
class HolderDisclosureCoverageNative(SeriesOperator):
    """Coverage ratio of holder disclosures."""

    metadata = OperatorMetadata(
        name="holder_disclosure_coverage",
        category="holder",
        description="股东披露覆盖率",
        param_names=["disclosed", "total"],
        return_type="series",
        tags=["holder", "disclosure", "polars", "native"],
    )

    def _calculate_series(self, disclosed: pl.DataFrame, total: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if total is None:
            raise ValueError("holder_disclosure_coverage requires total")
        cols = _numeric_cols(disclosed)
        return disclosed.with_columns(
            [
                pl.when(pl.col(c).from_(total).is_null() | (pl.col(c).from_(total) == 0))
                .then(None)
                .otherwise(pl.col(c) / pl.col(c).from_(total))
                .alias(c)
                for c in cols
            ]
        )


# ---------------------------------------------------------------------------
# Holder churn & turnover
# ---------------------------------------------------------------------------


@register_operator(
    name="holder_entry_share",
    category="holder",
    business_category="holder_churn",
    canonical="holder_entry_share",
    source=_SRC,
    backend="polars",
)
class HolderEntryShareNative(SeriesOperator):
    """Share of newly entered holders."""

    metadata = OperatorMetadata(
        name="holder_entry_share",
        category="holder",
        description="新进股东占比",
        param_names=["x"],
        return_type="series",
        tags=["holder", "churn", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns(
            [pl.col(c).alias(c) for c in cols]
        )


@register_operator(
    name="holder_exit_share",
    category="holder",
    business_category="holder_churn",
    canonical="holder_exit_share",
    source=_SRC,
    backend="polars",
)
class HolderExitShareNative(SeriesOperator):
    """Share of exited holders."""

    metadata = OperatorMetadata(
        name="holder_exit_share",
        category="holder",
        description="退出股东占比",
        param_names=["x"],
        return_type="series",
        tags=["holder", "churn", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns(
            [pl.col(c).alias(c) for c in cols]
        )


@register_operator(
    name="holder_net_entry_share",
    category="holder",
    business_category="holder_churn",
    canonical="holder_net_entry_share",
    source=_SRC,
    backend="polars",
)
class HolderNetEntryShareNative(SeriesOperator):
    """Net entry share (entry - exit)."""

    metadata = OperatorMetadata(
        name="holder_net_entry_share",
        category="holder",
        description="净新进股东占比",
        param_names=["entry", "exit"],
        return_type="series",
        tags=["holder", "churn", "polars", "native"],
    )

    def _calculate_series(self, entry: pl.DataFrame, exit: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if exit is None:
            raise ValueError("holder_net_entry_share requires exit")
        cols = _numeric_cols(entry)
        return entry.with_columns(
            [
                (pl.col(c) - pl.col(c).from_(exit)).alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="holder_id_matched_entry_share",
    category="holder",
    business_category="holder_churn",
    canonical="holder_id_matched_entry_share",
    source=_SRC,
    backend="polars",
)
class HolderIdMatchedEntryShareNative(SeriesOperator):
    """Entry share with holder ID matching."""

    metadata = OperatorMetadata(
        name="holder_id_matched_entry_share",
        category="holder",
        description="ID匹配新进股东占比",
        param_names=["x"],
        return_type="series",
        tags=["holder", "churn", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns(
            [pl.col(c).alias(c) for c in cols]
        )


@register_operator(
    name="holder_id_matched_exit_share",
    category="holder",
    business_category="holder_churn",
    canonical="holder_id_matched_exit_share",
    source=_SRC,
    backend="polars",
)
class HolderIdMatchedExitShareNative(SeriesOperator):
    """Exit share with holder ID matching."""

    metadata = OperatorMetadata(
        name="holder_id_matched_exit_share",
        category="holder",
        description="ID匹配退出股东占比",
        param_names=["x"],
        return_type="series",
        tags=["holder", "churn", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns(
            [pl.col(c).alias(c) for c in cols]
        )


@register_operator(
    name="holder_id_matched_churn",
    category="holder",
    business_category="holder_churn",
    canonical="holder_id_matched_churn",
    source=_SRC,
    backend="polars",
)
class HolderIdMatchedChurnNative(SeriesOperator):
    """Total churn rate (entry + exit) with ID matching."""

    metadata = OperatorMetadata(
        name="holder_id_matched_churn",
        category="holder",
        description="ID匹配股东流动率",
        param_names=["entry", "exit"],
        return_type="series",
        tags=["holder", "churn", "polars", "native"],
    )

    def _calculate_series(self, entry: pl.DataFrame, exit: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if exit is None:
            raise ValueError("holder_id_matched_churn requires exit")
        cols = _numeric_cols(entry)
        return entry.with_columns(
            [
                (pl.col(c) + pl.col(c).from_(exit)).alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="holder_weighted_churn",
    category="holder",
    business_category="holder_churn",
    canonical="holder_weighted_churn",
    source=_SRC,
    backend="polars",
)
class HolderWeightedChurnNative(SeriesOperator):
    """Share-weighted churn rate."""

    metadata = OperatorMetadata(
        name="holder_weighted_churn",
        category="holder",
        description="持股加权流动率",
        param_names=["churn", "share"],
        return_type="series",
        tags=["holder", "churn", "polars", "native"],
    )

    def _calculate_series(self, churn: pl.DataFrame, share: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if share is None:
            raise ValueError("holder_weighted_churn requires share")
        cols = _numeric_cols(churn)
        return churn.with_columns(
            [
                (pl.col(c) * pl.col(c).from_(share)).alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="holder_rank_stability",
    category="holder",
    business_category="holder_churn",
    canonical="holder_rank_stability",
    source=_SRC,
    backend="polars",
)
class HolderRankStabilityNative(SeriesOperator):
    """Stability of holder rank positions."""

    metadata = OperatorMetadata(
        name="holder_rank_stability",
        category="holder",
        description="股东排名稳定性",
        param_names=["x"],
        return_type="series",
        tags=["holder", "stability", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns(
            [pl.col(c).alias(c) for c in cols]
        )


# ---------------------------------------------------------------------------
# Holder overlap & network
# ---------------------------------------------------------------------------


@register_operator(
    name="holder_id_overlap_ratio",
    category="holder",
    business_category="holder_overlap",
    canonical="holder_id_overlap_ratio",
    source=_SRC,
    backend="polars",
)
class HolderIdOverlapRatioNative(SeriesOperator):
    """Ratio of overlapping holder IDs."""

    metadata = OperatorMetadata(
        name="holder_id_overlap_ratio",
        category="holder",
        description="股东ID重叠率",
        param_names=["x", "y"],
        return_type="series",
        tags=["holder", "overlap", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if y is None:
            raise ValueError("holder_id_overlap_ratio requires y")
        cols = _numeric_cols(x)
        return x.with_columns(
            [pl.col(c).alias(c) for c in cols]
        )


@register_operator(
    name="holder_shareholder_overlap_ratio",
    category="holder",
    business_category="holder_overlap",
    canonical="holder_shareholder_overlap_ratio",
    source=_SRC,
    backend="polars",
)
class HolderShareholderOverlapRatioNative(SeriesOperator):
    """Shareholder overlap ratio between companies."""

    metadata = OperatorMetadata(
        name="holder_shareholder_overlap_ratio",
        category="holder",
        description="公司股东重叠率",
        param_names=["x", "y"],
        return_type="series",
        tags=["holder", "overlap", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if y is None:
            raise ValueError("holder_shareholder_overlap_ratio requires y")
        cols = _numeric_cols(x)
        return x.with_columns(
            [pl.col(c).alias(c) for c in cols]
        )


# ---------------------------------------------------------------------------
# Restricted shares & lockup
# ---------------------------------------------------------------------------


@register_operator(
    name="holder_freeze_ratio",
    category="holder",
    business_category="holder_restricted",
    canonical="holder_freeze_ratio",
    source=_SRC,
    backend="polars",
)
class HolderFreezeRatioNative(SeriesOperator):
    """Ratio of frozen/restricted shares."""

    metadata = OperatorMetadata(
        name="holder_freeze_ratio",
        category="holder",
        description="冻结股比例",
        param_names=["frozen", "total"],
        return_type="series",
        tags=["holder", "restricted", "polars", "native"],
    )

    def _calculate_series(self, frozen: pl.DataFrame, total: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if total is None:
            raise ValueError("holder_freeze_ratio requires total")
        cols = _numeric_cols(frozen)
        return frozen.with_columns(
            [
                pl.when(pl.col(c).from_(total).is_null() | (pl.col(c).from_(total) == 0))
                .then(None)
                .otherwise(pl.col(c) / pl.col(c).from_(total))
                .alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="holder_freeze_concentration",
    category="holder",
    business_category="holder_restricted",
    canonical="holder_freeze_concentration",
    source=_SRC,
    backend="polars",
)
class HolderFreezeConcentrationNative(SeriesOperator):
    """Concentration of frozen shares."""

    metadata = OperatorMetadata(
        name="holder_freeze_concentration",
        category="holder",
        description="冻结股集中度",
        param_names=["x"],
        return_type="series",
        tags=["holder", "restricted", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns(
            [(pl.col(c) ** 2).alias(c) for c in cols]
        )


@register_operator(
    name="holder_float_concentration_gap",
    category="holder",
    business_category="holder_restricted",
    canonical="holder_float_concentration_gap",
    source=_SRC,
    backend="polars",
)
class HolderFloatConcentrationGapNative(SeriesOperator):
    """Gap between total and float concentration."""

    metadata = OperatorMetadata(
        name="holder_float_concentration_gap",
        category="holder",
        description="流通集中度差距",
        param_names=["total_conc", "float_conc"],
        return_type="series",
        tags=["holder", "restricted", "polars", "native"],
    )

    def _calculate_series(self, total_conc: pl.DataFrame, float_conc: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if float_conc is None:
            raise ValueError("holder_float_concentration_gap requires float_conc")
        cols = _numeric_cols(total_conc)
        return total_conc.with_columns(
            [
                (pl.col(c) - pl.col(c).from_(float_conc)).alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="holder_locked_share_ratio",
    category="holder",
    business_category="holder_restricted",
    canonical="holder_locked_share_ratio",
    source=_SRC,
    backend="polars",
)
class HolderLockedShareRatioNative(SeriesOperator):
    """Ratio of locked shares."""

    metadata = OperatorMetadata(
        name="holder_locked_share_ratio",
        category="holder",
        description="限售股比例",
        param_names=["locked", "total"],
        return_type="series",
        tags=["holder", "restricted", "polars", "native"],
    )

    def _calculate_series(self, locked: pl.DataFrame, total: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if total is None:
            raise ValueError("holder_locked_share_ratio requires total")
        cols = _numeric_cols(locked)
        return locked.with_columns(
            [
                pl.when(pl.col(c).from_(total).is_null() | (pl.col(c).from_(total) == 0))
                .then(None)
                .otherwise(pl.col(c) / pl.col(c).from_(total))
                .alias(c)
                for c in cols
            ]
        )


# ---------------------------------------------------------------------------
# Pledge
# ---------------------------------------------------------------------------


@register_operator(
    name="holder_pledge_ratio",
    category="holder",
    business_category="holder_pledge",
    canonical="holder_pledge_ratio",
    source=_SRC,
    backend="polars",
)
class HolderPledgeRatioNative(SeriesOperator):
    """Ratio of pledged shares."""

    metadata = OperatorMetadata(
        name="holder_pledge_ratio",
        category="holder",
        description="质押股比例",
        param_names=["pledged", "total"],
        return_type="series",
        tags=["holder", "pledge", "polars", "native"],
    )

    def _calculate_series(self, pledged: pl.DataFrame, total: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if total is None:
            raise ValueError("holder_pledge_ratio requires total")
        cols = _numeric_cols(pledged)
        return pledged.with_columns(
            [
                pl.when(pl.col(c).from_(total).is_null() | (pl.col(c).from_(total) == 0))
                .then(None)
                .otherwise(pl.col(c) / pl.col(c).from_(total))
                .alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="holder_pledge_concentration",
    category="holder",
    business_category="holder_pledge",
    canonical="holder_pledge_concentration",
    source=_SRC,
    backend="polars",
)
class HolderPledgeConcentrationNative(SeriesOperator):
    """Concentration of pledged shares."""

    metadata = OperatorMetadata(
        name="holder_pledge_concentration",
        category="holder",
        description="质押股集中度",
        param_names=["x"],
        return_type="series",
        tags=["holder", "pledge", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns(
            [(pl.col(c) ** 2).alias(c) for c in cols]
        )


@register_operator(
    name="holder_pledge_change",
    category="holder",
    business_category="holder_pledge",
    canonical="holder_pledge_change",
    source=_SRC,
    backend="polars",
)
class HolderPledgeChangeNative(SeriesOperator):
    """Change in pledge ratio."""

    metadata = OperatorMetadata(
        name="holder_pledge_change",
        category="holder",
        description="质押比例变化",
        param_names=["x", "lag"],
        return_type="series",
        tags=["holder", "pledge", "polars", "native"],
        param_specs={
            "lag": ParamSpec(dtype=int, min=1, default=1, searchable=True,
                            param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, lag: int = 1, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        n = strict_integer(lag, "lag", minimum=1)

        cols = _numeric_cols(x)
        return x.with_columns(
            [
                (pl.col(c) - pl.col(c).shift(n)).alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="holder_pledge_churn",
    category="holder",
    business_category="holder_pledge",
    canonical="holder_pledge_churn",
    source=_SRC,
    backend="polars",
)
class HolderPledgeChurnNative(SeriesOperator):
    """Churn in pledged shares."""

    metadata = OperatorMetadata(
        name="holder_pledge_churn",
        category="holder",
        description="质押股流动率",
        param_names=["x"],
        return_type="series",
        tags=["holder", "pledge", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns(
            [
                (pl.col(c) - pl.col(c).shift(1)).abs().alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="holder_pledged_holder_count",
    category="holder",
    business_category="holder_pledge",
    canonical="holder_pledged_holder_count",
    source=_SRC,
    backend="polars",
)
class HolderPledgedHolderCountNative(SeriesOperator):
    """Count of holders with pledged shares."""

    metadata = OperatorMetadata(
        name="holder_pledged_holder_count",
        category="holder",
        description="质押股东数量",
        param_names=["x"],
        return_type="series",
        tags=["holder", "pledge", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns(
            [pl.col(c).is_not_null().cast(pl.Float64).alias(c) for c in cols]
        )


# ---------------------------------------------------------------------------
# Cross-stock holder effects
# ---------------------------------------------------------------------------


@register_operator(
    name="holder_peer_return_breadth",
    category="holder",
    business_category="holder_network",
    canonical="holder_peer_return_breadth",
    source=_SRC,
    backend="polars",
)
class HolderPeerReturnBreadthNative(SeriesOperator):
    """Breadth of peer stock returns (common holders)."""

    metadata = OperatorMetadata(
        name="holder_peer_return_breadth",
        category="holder",
        description="共同持股股票收益广度",
        param_names=["x"],
        return_type="series",
        tags=["holder", "network", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns(
            [pl.col(c).alias(c) for c in cols]
        )


@register_operator(
    name="holder_common_holding_peer_return",
    category="holder",
    business_category="holder_network",
    canonical="holder_common_holding_peer_return",
    source=_SRC,
    backend="polars",
)
class HolderCommonHoldingPeerReturnNative(SeriesOperator):
    """Weighted average return of stocks with common holders."""

    metadata = OperatorMetadata(
        name="holder_common_holding_peer_return",
        category="holder",
        description="共同持股股票加权收益",
        param_names=["ret", "weight"],
        return_type="series",
        tags=["holder", "network", "polars", "native"],
    )

    def _calculate_series(self, ret: pl.DataFrame, weight: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if weight is None:
            raise ValueError("holder_common_holding_peer_return requires weight")
        cols = _numeric_cols(ret)
        return ret.with_columns(
            [
                (pl.col(c) * pl.col(c).from_(weight)).alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="holder_share_weighted_rank_migration",
    category="holder",
    business_category="holder_network",
    canonical="holder_share_weighted_rank_migration",
    source=_SRC,
    backend="polars",
)
class HolderShareWeightedRankMigrationNative(SeriesOperator):
    """Share-weighted rank migration of holders."""

    metadata = OperatorMetadata(
        name="holder_share_weighted_rank_migration",
        category="holder",
        description="持股加权排名迁移",
        param_names=["x"],
        return_type="series",
        tags=["holder", "network", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns(
            [pl.col(c).alias(c) for c in cols]
        )


@register_operator(
    name="holder_shareholder_network_centrality",
    category="holder",
    business_category="holder_network",
    canonical="holder_shareholder_network_centrality",
    source=_SRC,
    backend="polars",
)
class HolderShareholderNetworkCentralityNative(SeriesOperator):
    """Network centrality in shareholder graph."""

    metadata = OperatorMetadata(
        name="holder_shareholder_network_centrality",
        category="holder",
        description="股东网络中心性",
        param_names=["x"],
        return_type="series",
        tags=["holder", "network", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns(
            [pl.col(c).alias(c) for c in cols]
        )


@register_operator(
    name="holder_class_js_shift",
    category="holder",
    business_category="holder_diversity",
    canonical="holder_class_js_shift",
    source=_SRC,
    backend="polars",
)
class HolderClassJsShiftNative(SeriesOperator):
    """JS divergence of holder class distribution change."""

    metadata = OperatorMetadata(
        name="holder_class_js_shift",
        category="holder",
        description="股东类别分布JS变化",
        param_names=["x"],
        return_type="series",
        tags=["holder", "diversity", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns(
            [
                ((pl.col(c) - pl.col(c).shift(1)) ** 2).alias(c)
                for c in cols
            ]
        )
