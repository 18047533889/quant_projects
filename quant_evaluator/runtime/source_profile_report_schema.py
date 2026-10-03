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


PEARSON_V1_SCHEMA = SourceProfileReportSchema(
    "real_cos_profile_abba_f48_pearson.v1", PEARSON_METRICS, PEARSON_ORACLE, False)
PEARSON_V2_SCHEMA = SourceProfileReportSchema(
    "real_cos_profile_abba_f48_pearson.v2", PEARSON_METRICS, PEARSON_ORACLE, True)
LINEAR_SHAPE_SCHEMA = SourceProfileReportSchema(
    "real_cos_profile_abba_f48_linear_shape.v1", LINEAR_SHAPE_METRICS,
    LINEAR_SHAPE_ORACLE, True, auto_values_must_match_profile=True)
REPORT_SCHEMAS = MappingProxyType({
    schema.kind: schema
    for schema in (PEARSON_V1_SCHEMA, PEARSON_V2_SCHEMA, LINEAR_SHAPE_SCHEMA)
})

REPORT_KIND = PEARSON_V1_SCHEMA.kind
REPORT_KIND_V2 = PEARSON_V2_SCHEMA.kind
LINEAR_SHAPE_REPORT_KIND = LINEAR_SHAPE_SCHEMA.kind
REPORT_STATUS = "complete"

__all__ = (
    "LINEAR_SHAPE_METRICS", "LINEAR_SHAPE_ORACLE", "LINEAR_SHAPE_REPORT_KIND",
    "LINEAR_SHAPE_SCHEMA", "PEARSON_METRICS", "PEARSON_ORACLE",
    "PEARSON_V1_SCHEMA", "PEARSON_V2_SCHEMA", "REPORT_KIND", "REPORT_KIND_V2",
    "REPORT_RUN_BACKENDS", "REPORT_RUN_ORDER", "REPORT_SCHEMAS", "REPORT_SHAPE",
    "REPORT_STATUS", "REPORT_TILE_CAP", "SourceProfileReportSchema",
)
