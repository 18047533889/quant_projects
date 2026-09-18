# -*- coding: utf-8 -*-
"""Holder/shareholder operators - Polars native implementations.

Holder and shareholder operators for concentration, churn, overlap, and network analysis.
Simple kernels use native Polars expressions; complex contract-repaired kernels
delegate explicitly to the authoritative pandas implementation.
"""
from __future__ import annotations

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore

from factor_engine.cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.cleaned_operators.base import ParamRole, ParamSpec
from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
from factor_engine.cleaned_operators.common.polars_share_ratio import bounded_share_ratio

_SKIP = frozenset({"date", "stock_code"})
_SRC = "factor_dsl_polars_holder"
_ID_PARAMS = [
    *(f"s{i}" for i in range(1, 11)), *(f"sid{i}" for i in range(1, 11)),
    *(f"p{i}" for i in range(1, 11)), *(f"psid{i}" for i in range(1, 11)),
]
_DISCLOSURE_PARAMS = _ID_PARAMS[:20]


def _delegate(canonical: str, panels: tuple, kwargs: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.rolling_pack import _call_pandas_delegate
    return _call_pandas_delegate(canonical, panels, kwargs)


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
        param_names=["top_holder_shares", "total_shares"],
        return_type="series",
        tags=["holder", "concentration", "polars", "delegate:pandas_numpy"],
    )

    def _calculate_series(self, top_holder_shares, total_shares, **kwargs) -> pl.DataFrame:
        return _delegate(
            "holder_concentration", (top_holder_shares, total_shares), kwargs
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
        param_names=["concentration", "window", "snapshot_date"],
        return_type="series",
        tags=["holder", "concentration", "polars", "delegate:pandas_numpy"],
    )

    def _calculate_series(self, concentration, window=8, snapshot_date=None, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.rolling_pack import _call_pandas_delegate
        return _call_pandas_delegate(
            "holder_concentration_acceleration", (concentration,),
            {"window": window, "snapshot_date": snapshot_date, **kwargs},
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
        param_names=["concentration", "window", "snapshot_date"],
        return_type="series",
        tags=["holder", "concentration", "polars", "delegate:pandas_numpy"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=8, searchable=True,
                               param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, concentration, window=8, snapshot_date=None, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.rolling_pack import _call_pandas_delegate
        return _call_pandas_delegate(
            "holder_concentration_slope", (concentration,),
            {"window": window, "snapshot_date": snapshot_date, **kwargs},
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
        param_names=_DISCLOSURE_PARAMS,
        return_type="series",
        tags=["holder", "concentration", "polars", "delegate:pandas_numpy"],
    )

    def _calculate_series(self, *panels, **kwargs) -> pl.DataFrame:
        return _delegate("holder_topk_share_sum", panels, kwargs)


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
        param_names=[f"s{i}" for i in range(1, 11)],
        return_type="series",
        tags=["holder", "concentration", "polars", "delegate:pandas_numpy"],
    )

    def _calculate_series(self, s1,s2,s3,s4,s5,s6,s7,s8,s9,s10, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.rolling_pack import _call_pandas_delegate
        return _call_pandas_delegate("holder_company_ownership_hhi", (s1,s2,s3,s4,s5,s6,s7,s8,s9,s10), kwargs)


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
        param_names=[*(f"s{i}" for i in range(1, 11)), "missing_semantic"],
        return_type="series",
        tags=["holder", "concentration", "polars", "delegate:pandas_numpy"],
    )

    def _calculate_series(self, s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,
                          missing_semantic="outside_top_k", **kwargs) -> pl.DataFrame:
        return _delegate(
            "holder_observed_topk_hhi", (s1,s2,s3,s4,s5,s6,s7,s8,s9,s10),
            {"missing_semantic": missing_semantic, **kwargs},
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
        param_names=["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"],
        return_type="series",
        tags=["holder", "entropy", "polars", "delegate:pandas_numpy"],
    )

    def _calculate_series(self, s1, s2, s3, s4, s5, s6, s7, s8, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.rolling_pack import _call_pandas_delegate
        return _call_pandas_delegate(
            "holder_class_entropy", (s1, s2, s3, s4, s5, s6, s7, s8), kwargs
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
        param_names=["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"],
        return_type="series",
        tags=["holder", "entropy", "polars", "delegate:pandas_numpy"],
    )

    def _calculate_series(self, s1, s2, s3, s4, s5, s6, s7, s8, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.rolling_pack import _call_pandas_delegate
        return _call_pandas_delegate("holder_nature_entropy", (s1,s2,s3,s4,s5,s6,s7,s8), kwargs)


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
        param_names=_DISCLOSURE_PARAMS,
        return_type="series",
        tags=["holder", "disclosure", "polars", "delegate:pandas_numpy"],
    )

    def _calculate_series(self, *panels, **kwargs) -> pl.DataFrame:
        return _delegate("holder_disclosure_count", panels, kwargs)


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
        param_names=_DISCLOSURE_PARAMS,
        return_type="series",
        tags=["holder", "disclosure", "polars", "delegate:pandas_numpy"],
    )

    def _calculate_series(self, *panels, **kwargs) -> pl.DataFrame:
        return _delegate("holder_disclosure_coverage", panels, kwargs)


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
        param_names=_ID_PARAMS,
        return_type="series",
        tags=["holder", "churn", "polars", "delegate:pandas_numpy"],
    )

    def _calculate_series(self, *panels, **kwargs) -> pl.DataFrame:
        return _delegate("holder_entry_share", panels, kwargs)


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
        param_names=_ID_PARAMS,
        return_type="series",
        tags=["holder", "churn", "polars", "delegate:pandas_numpy"],
    )

    def _calculate_series(self, *panels, **kwargs) -> pl.DataFrame:
        return _delegate("holder_exit_share", panels, kwargs)


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
        param_names=_ID_PARAMS,
        return_type="series",
        tags=["holder", "churn", "polars", "delegate:pandas_numpy"],
    )

    def _calculate_series(self, *panels, **kwargs) -> pl.DataFrame:
        return _delegate("holder_net_entry_share", panels, kwargs)


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
        param_names=_ID_PARAMS,
        return_type="series",
        tags=["holder", "churn", "polars", "native"],
    )

    def _calculate_series(self, *panels, **kwargs) -> pl.DataFrame:
        return _delegate("holder_id_matched_entry_share", panels, kwargs)


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
        param_names=_ID_PARAMS,
        return_type="series",
        tags=["holder", "churn", "polars", "native"],
    )

    def _calculate_series(self, *panels, **kwargs) -> pl.DataFrame:
        return _delegate("holder_id_matched_exit_share", panels, kwargs)


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
        param_names=_ID_PARAMS,
        return_type="series",
        tags=["holder", "churn", "polars", "native"],
    )

    def _calculate_series(self, *panels, **kwargs) -> pl.DataFrame:
        return _delegate("holder_id_matched_churn", panels, kwargs)


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
        param_names=_ID_PARAMS,
        return_type="series",
        tags=["holder", "churn", "polars", "native"],
    )

    def _calculate_series(self, *panels, **kwargs) -> pl.DataFrame:
        return _delegate("holder_weighted_churn", panels, kwargs)


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
        param_names=_ID_PARAMS,
        return_type="series",
        tags=["holder", "stability", "polars", "native"],
    )

    def _calculate_series(self, *panels, **kwargs) -> pl.DataFrame:
        return _delegate("holder_rank_stability", panels, kwargs)


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
        param_names=_ID_PARAMS,
        return_type="series",
        tags=["holder", "overlap", "polars", "native"],
    )

    def _calculate_series(self, *panels, **kwargs) -> pl.DataFrame:
        return _delegate("holder_id_overlap_ratio", panels, kwargs)


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
        param_names=["shared_holders", "total_holders"],
        return_type="series",
        tags=["holder", "overlap", "polars", "delegate:pandas_numpy"],
    )

    def _calculate_series(self, shared_holders, total_holders, **kwargs) -> pl.DataFrame:
        return _delegate(
            "holder_shareholder_overlap_ratio",
            (shared_holders, total_holders), kwargs,
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
        param_names=["freeze_shares", "total_capital"],
        return_type="series",
        tags=["holder", "restricted", "polars", "native"],
    )

    def _calculate_series(self, freeze_shares: pl.DataFrame, total_capital: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if total_capital is None:
            raise ValueError("holder_freeze_ratio requires total")
        return bounded_share_ratio(freeze_shares, total_capital)


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
        param_names=[f"s{i}" for i in range(1, 11)],
        return_type="series",
        tags=["holder", "restricted", "polars", "delegate:pandas_numpy"],
    )

    def _calculate_series(self, s1,s2,s3,s4,s5,s6,s7,s8,s9,s10, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.rolling_pack import _call_pandas_delegate
        return _call_pandas_delegate("holder_freeze_concentration", (s1,s2,s3,s4,s5,s6,s7,s8,s9,s10), kwargs)


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
        param_names=["top10_concentration", "top10_float_concentration"],
        return_type="series",
        tags=["holder", "restricted", "polars", "delegate:pandas_numpy"],
    )

    def _calculate_series(self, top10_concentration, top10_float_concentration, **kwargs) -> pl.DataFrame:
        return _delegate(
            "holder_float_concentration_gap",
            (top10_concentration, top10_float_concentration), kwargs,
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
        param_names=["locked_shares", "total_capital"],
        return_type="series",
        tags=["holder", "restricted", "polars", "native"],
    )

    def _calculate_series(self, locked_shares: pl.DataFrame, total_capital: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if total_capital is None:
            raise ValueError("holder_locked_share_ratio requires total")
        return bounded_share_ratio(locked_shares, total_capital)


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
        param_names=["pledge_shares", "total_capital"],
        return_type="series",
        tags=["holder", "pledge", "polars", "native"],
    )

    def _calculate_series(self, pledge_shares: pl.DataFrame, total_capital: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if total_capital is None:
            raise ValueError("holder_pledge_ratio requires total")
        return bounded_share_ratio(pledge_shares, total_capital)


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
        param_names=[f"s{i}" for i in range(1, 11)],
        return_type="series",
        tags=["holder", "pledge", "polars", "delegate:pandas_numpy"],
    )

    def _calculate_series(self, s1,s2,s3,s4,s5,s6,s7,s8,s9,s10, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.rolling_pack import _call_pandas_delegate
        return _call_pandas_delegate("holder_pledge_concentration", (s1,s2,s3,s4,s5,s6,s7,s8,s9,s10), kwargs)


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
        from factor_engine.cleaned_operators.parameter_validation import strict_integer
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
        param_names=[*(f"s{i}" for i in range(1, 11)), *(f"p{i}" for i in range(1, 11))],
        return_type="series",
        tags=["holder", "pledge", "polars", "delegate:pandas_numpy"],
    )

    def _calculate_series(self, *panels, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.rolling_pack import _call_pandas_delegate
        return _call_pandas_delegate("holder_pledge_churn", panels, kwargs)


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
        param_names=[f"s{i}" for i in range(1, 11)],
        return_type="series",
        tags=["holder", "pledge", "polars", "delegate:pandas_numpy"],
    )

    def _calculate_series(self, s1,s2,s3,s4,s5,s6,s7,s8,s9,s10, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.rolling_pack import _call_pandas_delegate
        return _call_pandas_delegate("holder_pledged_holder_count", (s1,s2,s3,s4,s5,s6,s7,s8,s9,s10), kwargs)


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
        param_names=["peer_return", "own_return", "overlap"],
        return_type="series",
        tags=["holder", "network", "polars", "delegate:pandas_numpy"],
    )

    def _calculate_series(self, peer_return, own_return, overlap, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.rolling_pack import _call_pandas_delegate
        return _call_pandas_delegate(
            "holder_common_holding_peer_return",
            (peer_return, own_return, overlap), kwargs,
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
        param_names=_ID_PARAMS,
        return_type="series",
        tags=["holder", "network", "polars", "native"],
    )

    def _calculate_series(self, *panels, **kwargs) -> pl.DataFrame:
        return _delegate("holder_share_weighted_rank_migration", panels, kwargs)


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
        param_names=["degree", "total"],
        return_type="series",
        tags=["holder", "network", "polars", "delegate:pandas_numpy"],
    )

    def _calculate_series(self, degree, total, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.rolling_pack import _call_pandas_delegate
        return _call_pandas_delegate("holder_shareholder_network_centrality", (degree, total), kwargs)


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
        param_names=["s1", "s2", "s3", "s4", "s5", "ps1", "ps2", "ps3", "ps4", "ps5"],
        return_type="series",
        tags=["holder", "diversity", "polars", "delegate:pandas_numpy"],
    )

    def _calculate_series(
        self, s1, s2, s3, s4, s5, ps1, ps2, ps3, ps4, ps5, **kwargs
    ) -> pl.DataFrame:
        from factor_engine.cleaned_operators.rolling_pack import _call_pandas_delegate

        return _call_pandas_delegate(
            "holder_class_js_shift",
            (s1, s2, s3, s4, s5, ps1, ps2, ps3, ps4, ps5),
            kwargs,
        )


_PANDAS_DELEGATE_CLASSES = (
    HolderConcentrationNative,
    HolderConcentrationAccelerationNative,
    HolderConcentrationSlopeNative,
    HolderTopkShareSumNative,
    HolderCompanyOwnershipHhiNative,
    HolderObservedTopkHhiNative,
    HolderClassEntropyNative,
    HolderNatureEntropyNative,
    HolderDisclosureCountNative,
    HolderDisclosureCoverageNative,
    HolderEntryShareNative,
    HolderExitShareNative,
    HolderNetEntryShareNative,
    HolderIdMatchedEntryShareNative,
    HolderIdMatchedExitShareNative,
    HolderIdMatchedChurnNative,
    HolderWeightedChurnNative,
    HolderRankStabilityNative,
    HolderIdOverlapRatioNative,
    HolderShareholderOverlapRatioNative,
    HolderFloatConcentrationGapNative,
    HolderFreezeConcentrationNative,
    HolderPledgeConcentrationNative,
    HolderPledgeChurnNative,
    HolderPledgedHolderCountNative,
    HolderCommonHoldingPeerReturnNative,
    HolderShareWeightedRankMigrationNative,
    HolderShareholderNetworkCentralityNative,
    HolderClassJsShiftNative,
)
for _delegate_cls in _PANDAS_DELEGATE_CLASSES:
    _delegate_cls._physical_spec = PhysicalImplementationSpec(
        canonical=_delegate_cls.metadata.name,
        backend="polars",
        execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,
        supports_lazy=False,
        supports_streaming=False,
        materializes_full_panel=True,
        supports_nulls=True,
        supports_nan=True,
        supports_inf=True,
    )
