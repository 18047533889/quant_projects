from __future__ import annotations

import pytest


def _ops(node):
    output = [node.op]
    for child in node.inputs:
        output.extend(_ops(child))
    return output


def test_extended_surface_includes_daily_primitives_before_macro_expansion():
    from api.operator_registry import build_dsl_allowlist

    allow = build_dsl_allowlist(surface="extended")
    for name in ("add", "subtract", "multiply", "divide", "ts_mean", "NATR"):
        assert name in allow


def test_composite_technical_names_lower_to_primitive_dag():
    from api.dsl_parser import parse_factor
    from ir.analyzer import Analyzer

    cases = {
        "NATR(high,low,close,14)": "NATR",
        "PPO(close,12,26)": "PPO",
        "KeltnerPosition(high,low,close,20,14,2.0)": "KeltnerPosition",
        "DEMA(close,20)": "DEMA",
        "ichimoku_cloud_width(high,low,9,26,52)": "ichimoku_cloud_width",
        "ts_breakout_high(close,20)": "ts_breakout_high",
    }
    for formula, composite in cases.items():
        factor = parse_factor(
            formula, name=f"macro::{composite}", surface="extended"
        )
        ir = Analyzer().lower(factor.expr).ir
        names = set(_ops(ir))
        assert composite not in names, (formula, names)
        assert len(names) > 1


def test_intraday_daily_dsl_builds_auditable_source_ref():
    from api.dsl_parser import parse_factor
    from api.source_ref import decode_source_ref
    from ir.analyzer import Analyzer
    from runtime.quality.pit_audit import audit_ir

    factor = parse_factor(
        "intraday_realized_vol(bar_minutes=5, cutoff_time='15:50', min_coverage=0.9)",
        name="intraday::rv",
        surface="extended",
    )
    analysis = Analyzer().lower(factor.expr)
    assert analysis.ir.op == "column"
    specification = decode_source_ref(analysis.ir.attrs["name"])
    assert specification is not None
    assert specification.table == "StockMinuteBar"
    assert specification.transform == "intraday_feature"
    params = specification.transform_params_dict()
    assert params["feature"] == "realized_vol"
    assert params["bar_minutes"] == 5
    assert params["cutoff_time"] == "15:50"
    assert params["min_coverage"] == 0.9
    assert params["min_bars"] == 2
    assert params["timestamp_convention"] == "bar_end"
    assert params["history_days"] == 0
    report = audit_ir(analysis.ir)
    assert report.passed, report.violations


def test_intraday_profile_requires_bounded_history():
    from api.dsl_parser import parse_factor
    from ir.analyzer import Analyzer
    from runtime.quality.pit_audit import audit_ir

    factor = parse_factor(
        "intraday_profile_zscore(bar_minutes=5, history_days=20, cutoff_time='session_close', min_coverage=0.9)",
        name="intraday::profile",
        surface="extended",
    )
    report = audit_ir(Analyzer().lower(factor.expr).ir)
    assert report.passed, report.violations


def test_intraday_authoring_rejects_invalid_static_contracts():
    from api.intraday_daily import intraday_realized_vol

    with pytest.raises(ValueError, match="timestamp_convention"):
        intraday_realized_vol(timestamp_convention="ambiguous")
    with pytest.raises(ValueError, match="min_bars"):
        intraday_realized_vol(min_bars=1)
    with pytest.raises(ValueError, match="min_coverage"):
        intraday_realized_vol(min_coverage=1.1)
