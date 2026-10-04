"""Immutable schemas for the fixed source-profile report families."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


REPORT_SHAPE = (2586, 5461, 48)
REPORT_TILE_CAP = 16
REPORT_RUN_ORDER = ("cpu", "cuda_strict", "cuda_strict", "cpu")
REPORT_RUN_BACKENDS = ("cpu", "cuda", "cuda", "cpu")
PEARSON_METRICS = (
    "pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir",
)
LINEAR_SHAPE_METRICS = (
    "quantile_curvature", "quantile_tail_asymmetry", "quantile_adjacent_spread",
    "quantile_extreme_cliff", "top_quantile_cliff", "bottom_quantile_cliff",
)
PEARSON_ORACLE = "independent scipy/decimal Pearson chain"
LINEAR_SHAPE_ORACLE = "independent factor-only quantile/linear shape chain"

F61_REPORT_SHAPE = (2586, 5461, 61)
F61_ORACLE = "independent all-source metric chain"
F61_ALL15_METRICS = (
    "rank_ic", "rank_ic_series", "ic_ir", "ic_std", "ic_median",
    "pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir",
    "coverage", "quantile_spread", "quantile_monotonicity",
    "daily_quantile_monotonicity_rate", "turnover", "factor_turnover_rate",
)
F61_ALL24_METRICS = (
    "rank_ic", "rank_ic_series", "ic_ir", "ic_std", "ic_median",
    "pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir",
    "coverage", "quantile_spread", "quantile_monotonicity",
    "daily_quantile_monotonicity_rate", "turnover", "factor_turnover_rate",
    "quantile_curvature", "quantile_tail_asymmetry", "quantile_adjacent_spread",
    "quantile_extreme_cliff", "top_quantile_cliff", "bottom_quantile_cliff",
    "rank_ic_positive_ratio", "recent_3m_rank_ic", "rolling_rank_ic_ir",
)


@dataclass(frozen=True, slots=True)
class SourceProfileReportSchema:
    """One closed report contract; no field accepts caller-selected domains."""

    kind: str
    metric_ids: tuple[str, ...]
    oracle: str
    default_auto_required: bool
    shape: tuple[int, int, int] = REPORT_SHAPE
    requested_tile_cap: int = REPORT_TILE_CAP
    auto_values_must_match_profile: bool = False
    full_auto_identity_required: bool = False


PEARSON_V1_SCHEMA = SourceProfileReportSchema(
    "real_cos_profile_abba_f48_pearson.v1", PEARSON_METRICS, PEARSON_ORACLE, False)
PEARSON_V2_SCHEMA = SourceProfileReportSchema(
    "real_cos_profile_abba_f48_pearson.v2", PEARSON_METRICS, PEARSON_ORACLE, True)
LINEAR_SHAPE_SCHEMA = SourceProfileReportSchema(
    "real_cos_profile_abba_f48_linear_shape.v1", LINEAR_SHAPE_METRICS,
    LINEAR_SHAPE_ORACLE, True, auto_values_must_match_profile=True)
F61_ALL15_SCHEMA = SourceProfileReportSchema(
    "real_cos_profile_abba_f61_all15.v1", F61_ALL15_METRICS, F61_ORACLE, True,
    shape=F61_REPORT_SHAPE, auto_values_must_match_profile=True,
    full_auto_identity_required=True)
F61_ALL24_SCHEMA = SourceProfileReportSchema(
    "real_cos_profile_abba_f61_all24.v1", F61_ALL24_METRICS, F61_ORACLE, True,
    shape=F61_REPORT_SHAPE, auto_values_must_match_profile=True,
    full_auto_identity_required=True)
REPORT_SCHEMAS = MappingProxyType({
    schema.kind: schema
    for schema in (PEARSON_V1_SCHEMA, PEARSON_V2_SCHEMA, LINEAR_SHAPE_SCHEMA,
                   F61_ALL15_SCHEMA, F61_ALL24_SCHEMA)
})

REPORT_KIND = PEARSON_V1_SCHEMA.kind
REPORT_KIND_V2 = PEARSON_V2_SCHEMA.kind
LINEAR_SHAPE_REPORT_KIND = LINEAR_SHAPE_SCHEMA.kind
REPORT_STATUS = "complete"

__all__ = (
    "F61_ALL15_METRICS", "F61_ALL24_METRICS", "F61_ALL15_SCHEMA",
    "F61_ALL24_SCHEMA", "F61_ORACLE", "F61_REPORT_SHAPE",
    "LINEAR_SHAPE_METRICS", "LINEAR_SHAPE_ORACLE", "LINEAR_SHAPE_REPORT_KIND",
    "LINEAR_SHAPE_SCHEMA", "PEARSON_METRICS", "PEARSON_ORACLE",
    "PEARSON_V1_SCHEMA", "PEARSON_V2_SCHEMA", "REPORT_KIND", "REPORT_KIND_V2",
    "REPORT_RUN_BACKENDS", "REPORT_RUN_ORDER", "REPORT_SCHEMAS", "REPORT_SHAPE",
    "REPORT_STATUS", "REPORT_TILE_CAP", "SourceProfileReportSchema",
)
