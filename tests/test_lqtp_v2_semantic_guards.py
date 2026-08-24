from __future__ import annotations

import pytest


def _analysis(formula: str):
    from factor_engine.api.dsl_parser import parse_expr
    from factor_engine.ir.analyzer import Analyzer
    return Analyzer().lower(parse_expr(formula, surface="lqtp"))


def test_known_derived_fields_become_source_refs_not_daily_columns() -> None:
    from factor_engine.api.source_ref import decode_source_ref
    analysis = _analysis("enterprise_value / ebitda_approx")
    specs = [decode_source_ref(name) for name in analysis.referenced_columns]
    specs = [spec for spec in specs if spec is not None]
    assert {(spec.table, spec.field) for spec in specs} == {
        ("DerivedField", "enterprise_value"),
        ("DerivedField", "ebitda_approx"),
    }


def test_missing_derived_field_definition_fails_closed(monkeypatch) -> None:
    from factor_engine.runtime.derived_field_registry import load_derived_field_definition
    monkeypatch.delenv("FACTOR_ENGINE_DERIVED_FIELD_REGISTRY", raising=False)
    with pytest.raises(RuntimeError, match="no configured definition"):
        load_derived_field_definition("enterprise_value", path="/definitely/missing/derived_fields.yaml")


def test_industry_neutralize_injects_industry_source() -> None:
    from factor_engine.api.source_ref import decode_source_ref
    analysis = _analysis("industry_neutralize(close)")
    assert "close" in analysis.referenced_columns
    specs = [decode_source_ref(name) for name in analysis.referenced_columns]
    assert any(
        spec is not None
        and spec.table == "IndustryDaily"
        and spec.field == "IndustryCode"
        for spec in specs
    )


def test_size_neutralize_injects_size_source() -> None:
    from factor_engine.api.source_ref import decode_source_ref
    analysis = _analysis("size_neutralize(close)")
    specs = [decode_source_ref(name) for name in analysis.referenced_columns]
    assert any(
        spec is not None
        and spec.table == "SizeDaily"
        and spec.field == "MarketCap"
        for spec in specs
    )


def test_industry_size_neutralize_injects_both_sources() -> None:
    from factor_engine.api.source_ref import decode_source_ref
    analysis = _analysis("industry_size_neutralize(close)")
    specs = [s for s in (decode_source_ref(n) for n in analysis.referenced_columns) if s]
    tables = {(s.table, s.field) for s in specs}
    assert ("IndustryDaily", "IndustryCode") in tables
    assert ("SizeDaily", "MarketCap") in tables


def test_legacy_neutralize_aliases_redirect_to_stable_names() -> None:
    from factor_engine.api.source_ref import decode_source_ref
    # market_cap_neutralize → size_neutralize
    analysis = _analysis("market_cap_neutralize(close)")
    specs = [decode_source_ref(name) for name in analysis.referenced_columns]
    assert any(s is not None and s.table == "SizeDaily" for s in specs)
    # size_industry_neutralize → industry_size_neutralize
    analysis = _analysis("size_industry_neutralize(close)")
    specs = [s for s in (decode_source_ref(n) for n in analysis.referenced_columns) if s]
    tables = {s.table for s in specs}
    assert "IndustryDaily" in tables and "SizeDaily" in tables


def test_one_arg_neutralize_fails_closed() -> None:
    from factor_engine.api.dsl_parser import DSLParseError, parse_expr
    with pytest.raises(DSLParseError, match="stable"):
        parse_expr("neutralize(close)", surface="lqtp")


def test_default_market_return_uses_registered_return_field() -> None:
    from factor_engine.api.source_ref import decode_source_ref
    analysis = _analysis("market_ret")
    specs = [decode_source_ref(name) for name in analysis.referenced_columns]
    specs = [spec for spec in specs if spec is not None]
    assert len(specs) == 1
    assert specs[0].table == "BenchmarkIndexDailyBar"
    assert specs[0].field == "Return"
    assert specs[0].params_dict() == {"index": "000985.SH"}
