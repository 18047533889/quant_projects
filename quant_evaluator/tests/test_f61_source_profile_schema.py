"""Closed report-schema contract for the F61 all-source profiles.

A missing registry entry or a broadened/misordered metric tuple can make the
reader accept a report that no independent all-source measurement produced.
"""

from quant_evaluator.runtime import source_profile_report_schema as schemas


F61_REPORT_SHAPE = (2586, 5461, 61)
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
F61_ORACLE = "independent all-source metric chain"
F61_ALL15_KIND = "real_cos_profile_abba_f61_all15.v1"
F61_ALL24_KIND = "real_cos_profile_abba_f61_all24.v1"
_MISSING = object()


def _required_module_value(name):
    value = getattr(schemas, name, _MISSING)
    assert value is not _MISSING, f"source profile schema must expose {name}"
    return value


def test_f61_all_source_schema_contract_and_legacy_schema_stability():
    all15_schema = schemas.REPORT_SCHEMAS.get(F61_ALL15_KIND)
    all24_schema = schemas.REPORT_SCHEMAS.get(F61_ALL24_KIND)
    assert all15_schema is not None, "F61 all15 report kind must be registered"
    assert all24_schema is not None, "F61 all24 report kind must be registered"

    assert _required_module_value("F61_REPORT_SHAPE") == F61_REPORT_SHAPE
    assert _required_module_value("F61_ALL15_METRICS") == F61_ALL15_METRICS
    assert _required_module_value("F61_ALL24_METRICS") == F61_ALL24_METRICS
    assert _required_module_value("F61_ORACLE") == F61_ORACLE
    assert _required_module_value("F61_ALL15_SCHEMA") is all15_schema
    assert _required_module_value("F61_ALL24_SCHEMA") is all24_schema

    for schema, kind, metrics in (
        (all15_schema, F61_ALL15_KIND, F61_ALL15_METRICS),
        (all24_schema, F61_ALL24_KIND, F61_ALL24_METRICS),
    ):
        assert schema.kind == kind
        assert schema.metric_ids == metrics
        assert schema.oracle == F61_ORACLE
        assert schema.shape == F61_REPORT_SHAPE
        assert schema.requested_tile_cap == 16
        assert schema.default_auto_required is True
        assert schema.auto_values_must_match_profile is True
        assert getattr(schema, "full_auto_identity_required", False) is True

    # The F61 additions must not silently redefine any existing F48 report.
    assert schemas.PEARSON_V1_SCHEMA.kind == "real_cos_profile_abba_f48_pearson.v1"
    assert schemas.PEARSON_V1_SCHEMA.metric_ids == (
        "pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir",
    )
    assert schemas.PEARSON_V1_SCHEMA.oracle == "independent scipy/decimal Pearson chain"
    assert schemas.PEARSON_V1_SCHEMA.default_auto_required is False
    assert schemas.PEARSON_V2_SCHEMA.kind == "real_cos_profile_abba_f48_pearson.v2"
    assert schemas.PEARSON_V2_SCHEMA.metric_ids == schemas.PEARSON_V1_SCHEMA.metric_ids
    assert schemas.PEARSON_V2_SCHEMA.oracle == schemas.PEARSON_V1_SCHEMA.oracle
    assert schemas.PEARSON_V2_SCHEMA.default_auto_required is True
    assert schemas.LINEAR_SHAPE_SCHEMA.kind == "real_cos_profile_abba_f48_linear_shape.v1"
    assert schemas.LINEAR_SHAPE_SCHEMA.metric_ids == (
        "quantile_curvature", "quantile_tail_asymmetry", "quantile_adjacent_spread",
        "quantile_extreme_cliff", "top_quantile_cliff", "bottom_quantile_cliff",
    )
    assert schemas.LINEAR_SHAPE_SCHEMA.oracle == (
        "independent factor-only quantile/linear shape chain"
    )
    assert schemas.LINEAR_SHAPE_SCHEMA.default_auto_required is True
    assert schemas.LINEAR_SHAPE_SCHEMA.auto_values_must_match_profile is True
    assert tuple(schemas.REPORT_SCHEMAS) == (
        schemas.PEARSON_V1_SCHEMA.kind,
        schemas.PEARSON_V2_SCHEMA.kind,
        schemas.LINEAR_SHAPE_SCHEMA.kind,
        F61_ALL15_KIND,
        F61_ALL24_KIND,
    )
